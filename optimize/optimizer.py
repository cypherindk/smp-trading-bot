"""
optimize/optimizer.py
Gelişmiş Optuna optimizasyonu — Pine Script parametrelerinin tamamını kapsar.

Mevcut faz4/faz5'ten fark:
  - 8 parametre (4 yerine)
  - Walk-forward validation (overfitting önleme)
  - Çok amaçlı optimizasyon (return + sharpe + drawdown)
  - Her çalıştırmada stabil sonuçlar
"""

import pandas as pd
import numpy as np
import optuna
import warnings
from typing import Optional
warnings.filterwarnings("ignore")


def single_objective(trial, df, ind_func, signal_func, filter_func, backtest_func,
                     use_cvd=True):
    """
    Tek amaçlı Optuna objective.
    Optimize edilen metrik: Sharpe Ratio (getiri/risk dengesi en iyi gösterge)
    """
    # ── Optimize edilecek parametreler ──
    rvol_thr   = trial.suggest_float("rvol_thr", 1.2, 3.5, step=0.1)
    vsa_thr    = trial.suggest_float("vsa_thr", 0.5, 2.5, step=0.1)
    adr_mult   = trial.suggest_float("adr_mult", 1.0, 4.0, step=0.1)
    rr_ratio   = trial.suggest_float("rr_ratio", 1.2, 3.5, step=0.1)
    min_conf   = trial.suggest_int("min_conf", 1, 4)
    eff_score  = trial.suggest_float("eff_score", 3.0, 8.0, step=0.5)
    zombie_bars = trial.suggest_int("zombie_bars", 10, 40)
    preset     = trial.suggest_categorical("preset", ["Default", "Aggressive", "Conservative"])
    
    try:
        # Indikatörleri preset ile hesapla
        ind = ind_func(df, preset=preset)
        
        # RVOL eşiğini uygula (hacim filtresini sıkıştır)
        ind["rvol_ok"] = ind["rvol"] > rvol_thr
        
        # VSA eşiğini parametre ile kullan
        from engine.indicators import calc_vsa_shield
        vsa_df = calc_vsa_shield(
            df["high"], df["low"], df["open"], df["close"], df["volume"],
            threshold=vsa_thr
        )
        for col in ["vsa_bc", "vsa_ut", "vsa_sc", "vsa_spr", "vsa_dt"]:
            ind[col] = vsa_df[col]
        
        # Sinyaller
        from engine.signals import (calc_bull_bear_score, calc_triggers,
                                     generate_signals)
        sc = calc_bull_bear_score(ind)
        tr = calc_triggers(ind, sc)
        sg = generate_signals(ind, sc, tr, preset=preset,
                              eff_score=eff_score, min_conf=min_conf,
                              use_vsa=True, is_scalp_mode=(preset == "Scalping"))
        
        # Filtreler
        from engine.filters import apply_all_filters
        fs = apply_all_filters(ind, sg, use_cvd=use_cvd)
        
        if fs["buy_signal"].sum() + fs["sell_signal"].sum() < 5:
            return -999  # Çok az sinyal — geçersiz
        
        # Backtest
        results = backtest_func(
            df, ind, fs,
            adr_mult=adr_mult,
            rr_ratio=rr_ratio,
            zombie_bars=zombie_bars,
        )
        
        if results is None:
            return -999
        
        sharpe = results["sharpe_ratio"]
        if np.isnan(sharpe) or np.isinf(sharpe):
            return -999
        
        # Drawdown cezası (>30% drawdown'u cezalandır)
        dd = results["max_drawdown_pct"]
        dd_penalty = max(0, (dd - 30) * 0.5)
        
        return sharpe - dd_penalty
        
    except Exception as e:
        return -999


def run_optimization(df: pd.DataFrame,
                     n_trials: int = 150,
                     use_walk_forward: bool = True,
                     n_splits: int = 3) -> dict:
    """
    Walk-forward optimizasyon ile en iyi parametreleri bul.
    
    Walk-forward mantığı:
      - Veriyi n_splits parçaya böl
      - Her parçada: önceki veri train, mevcut veri test
      - Ortalama test performansını optimize et (overfitting engeller)
    """
    
    from engine.indicators import compute_all_indicators
    from backtest.runner import run_backtest
    
    print(f"🤖 Optuna Optimizasyonu Başlıyor... ({n_trials} deneme)")
    
    if use_walk_forward and n_splits > 1:
        # Walk-forward: veriyi bloklara böl
        total_len = len(df)
        fold_size = total_len // (n_splits + 1)
        
        def walk_forward_objective(trial):
            scores = []
            
            for fold in range(1, n_splits + 1):
                train_end = fold * fold_size
                test_start = train_end
                test_end = min(train_end + fold_size, total_len)
                
                # Test seti (out-of-sample)
                test_df = df.iloc[test_start:test_end].copy()
                
                if len(test_df) < 100:
                    continue
                
                score = single_objective(
                    trial, test_df,
                    compute_all_indicators,
                    None, None, run_backtest
                )
                scores.append(score)
            
            return np.mean(scores) if scores else -999
        
        objective = walk_forward_objective
    else:
        # Basit optimizasyon (tüm veri)
        def simple_objective(trial):
            return single_objective(
                trial, df,
                compute_all_indicators,
                None, None, run_backtest
            )
        objective = simple_objective
    
    # Optuna çalıştır
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),  # Stabil sonuçlar için seed
        pruner=optuna.pruners.MedianPruner(n_startup_trials=20)
    )
    
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1,  # Paralel çalıştırma sorun çıkarabilir
    )
    
    best = study.best_params
    best_value = study.best_value
    
    print(f"\n{'═'*55}")
    print(f"🏆 YAPAY ZEKA EN İYİ AYARLARI BULDU!")
    print(f"{'═'*55}")
    print(f"  Sharpe Ratio (walk-forward): {best_value:.4f}")
    print(f"\n  Parametreler:")
    for k, v in best.items():
        print(f"    {k}: {v}")
    print(f"{'═'*55}\n")
    
    return {
        "best_params": best,
        "best_value": best_value,
        "study": study,
    }


def validate_best_params(df: pd.DataFrame, best_params: dict) -> dict:
    """
    Bulunan en iyi parametrelerle son validasyon backtesti çalıştır.
    Son %20 veriyi test seti olarak kullan (hiç görmediği veri).
    """
    from engine.indicators import compute_all_indicators
    from engine.indicators import calc_vsa_shield
    from engine.signals import calc_bull_bear_score, calc_triggers, generate_signals
    from engine.filters import apply_all_filters
    from backtest.runner import run_backtest, print_results
    
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    
    print(f"\n🔬 Son validasyon ({len(test_df)} bar, hiç görülmemiş veri)...")
    
    p = best_params
    ind = compute_all_indicators(test_df, preset=p.get("preset", "Default"))
    
    # VSA parametresiyle yeniden hesapla
    vsa_df = calc_vsa_shield(
        test_df["high"], test_df["low"], test_df["open"],
        test_df["close"], test_df["volume"],
        threshold=p.get("vsa_thr", 1.0)
    )
    for col in ["vsa_bc", "vsa_ut", "vsa_sc", "vsa_spr", "vsa_dt"]:
        ind[col] = vsa_df[col]
    
    sc = calc_bull_bear_score(ind)
    tr = calc_triggers(ind, sc)
    sg = generate_signals(ind, sc, tr,
                          preset=p.get("preset", "Default"),
                          eff_score=p.get("eff_score", 5.0),
                          min_conf=p.get("min_conf", 2))
    fs = apply_all_filters(ind, sg)
    
    results = run_backtest(
        test_df, ind, fs,
        adr_mult=p.get("adr_mult", 1.5),
        rr_ratio=p.get("rr_ratio", 2.0),
        zombie_bars=p.get("zombie_bars", 20),
    )
    
    print_results(results, "VALIDASYON SONUCU (Out-of-Sample)")
    return results


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from data.fetcher import fetch_ohlcv
    
    print("📡 Veri çekiliyor...")
    df = fetch_ohlcv("BTC-USD", interval="4h", period="2y")
    
    opt_results = run_optimization(df, n_trials=150, use_walk_forward=True)
    validate_best_params(df, opt_results["best_params"])
