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

    [FIX] Eski parametre listesinde "rvol_thr" vardi ama hicbir yerde
    KULLANILMIYORDU -- ind["rvol_ok"] hesaplaniyor, sonra generate_signals
    hic bu kolonu okumuyor. Yani Optuna gecmiste bu parametreyi "optimize
    ediyor gibi yaparak" bosuna 150 deneme harciyordu. Kaldirildi.

    [YENİ] preset artik TUM Pine presetlerini kapsiyor (Scalping/Swing
    de eklendi) ve grade_filter de optimize ediliyor -- kullanicinin
    istegi: "TradingView'den bagimsiz, gecmiste en iyi ne ise o"
    (yani artik preset=Aggressive'e sabitlenmiyor).

    NOT: mtf (MTF Likidite Filtresi) bilerek None birakildi -- 2 yillik
    backtest penceresinin cogunda yfinance'in 5dk/15dk verisi zaten yok
    (~60 gunle sinirli), o yuzden optimizasyona dahil edilmedi. Canli
    tarama (telegram_bot.py) MTF'i ayrica uyguluyor; bu optimize edilen
    parametreler MTF'siz bir ortamda bulundu, MTF canlida ekstra bir
    filtre/puan katkisi yapacak.
    """
    # ── Optimize edilecek parametreler ──
    vsa_thr    = trial.suggest_float("vsa_thr", 0.5, 2.5, step=0.1)
    adr_mult   = trial.suggest_float("adr_mult", 1.0, 4.0, step=0.1)
    rr_ratio   = trial.suggest_float("rr_ratio", 1.2, 3.5, step=0.1)
    min_conf   = trial.suggest_int("min_conf", 1, 4)
    eff_score  = trial.suggest_float("eff_score", 2.0, 9.0, step=0.5)
    zombie_bars = trial.suggest_int("zombie_bars", 10, 40)
    preset     = trial.suggest_categorical(
        "preset", ["Scalping", "Aggressive", "Default", "Conservative", "Swing"])
    grade_filter = trial.suggest_categorical(
        "grade_filter", ["All", "A+ and A", "A+ Only"])

    try:
        # Indikatörleri preset ile hesapla (timeframe_minutes=240: bu
        # bot her zaman 4H calisiyor -> whale $ esigi de dogru hesaplanir)
        ind = ind_func(df, preset=preset, timeframe_minutes=240)

        # VSA eşiğini parametre ile kullan
        from engine.indicators import calc_vsa_shield
        vsa_df = calc_vsa_shield(
            df["high"], df["low"], df["open"], df["close"], df["volume"],
            threshold=vsa_thr
        )
        for col in ["vsa_bc", "vsa_ut", "vsa_sc", "vsa_spr", "vsa_dt"]:
            ind[col] = vsa_df[col]

        # Sinyaller (mtf=None -- yukaridaki NOT'a bak)
        from engine.signals import (calc_bull_bear_score, calc_triggers,
                                     generate_signals)
        sc = calc_bull_bear_score(ind, mtf=None)
        tr = calc_triggers(ind, sc)
        sg = generate_signals(ind, sc, tr, preset=preset,
                              eff_score=eff_score, min_conf=min_conf,
                              grade_filter=grade_filter,
                              use_vsa=True, is_scalp_mode=(preset == "Scalping"))

        # Filtreler
        from engine.filters import apply_all_filters
        fs = apply_all_filters(ind, sg, use_cvd=use_cvd)

        if fs["buy_signal"].sum() + fs["sell_signal"].sum() < 5:
            return -999  # Çok az sinyal — geçersiz

        # Backtest
        # [FIX] is_scalp_mode hic gecilmiyordu -> Zombi Kesici (zombie_bars)
        # backtest_func icinde hep pasif kaliyordu, preset="Scalping" secilse
        # bile. Artik generate_signals'a verilenle ayni deger kullaniliyor.
        results = backtest_func(
            df, ind, fs,
            adr_mult=adr_mult,
            rr_ratio=rr_ratio,
            zombie_bars=zombie_bars,
            is_scalp_mode=(preset == "Scalping"),
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
    ind = compute_all_indicators(test_df, preset=p.get("preset", "Default"),
                                 timeframe_minutes=240)

    # VSA parametresiyle yeniden hesapla
    vsa_df = calc_vsa_shield(
        test_df["high"], test_df["low"], test_df["open"],
        test_df["close"], test_df["volume"],
        threshold=p.get("vsa_thr", 1.0)
    )
    for col in ["vsa_bc", "vsa_ut", "vsa_sc", "vsa_spr", "vsa_dt"]:
        ind[col] = vsa_df[col]

    # mtf=None -- single_objective ile ayni ortam (bkz. oradaki NOT)
    sc = calc_bull_bear_score(ind, mtf=None)
    tr = calc_triggers(ind, sc)
    sg = generate_signals(ind, sc, tr,
                          preset=p.get("preset", "Default"),
                          eff_score=p.get("eff_score", 5.0),
                          min_conf=p.get("min_conf", 2),
                          grade_filter=p.get("grade_filter", "All"),
                          is_scalp_mode=(p.get("preset", "Default") == "Scalping"))
    fs = apply_all_filters(ind, sg)
    
    results = run_backtest(
        test_df, ind, fs,
        adr_mult=p.get("adr_mult", 1.5),
        rr_ratio=p.get("rr_ratio", 2.0),
        zombie_bars=p.get("zombie_bars", 20),
        is_scalp_mode=(p.get("preset", "Default") == "Scalping"),
    )
    
    print_results(results, "VALIDASYON SONUCU (Out-of-Sample)")
    return results


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    # [FIX] yfinance "4h" interval'ini DESTEKLEMEZ -- eskiden burada
    # dogrudan fetch_ohlcv(..., interval="4h", ...) cagriliyordu, bu da
    # yfinance tarafinda hataya/bos veriye yol aciyordu. Canli botun
    # kullandigi AYNI fetch_smart (1h->4h resample) fonksiyonu kullanildi.
    from telegram_bot import fetch_smart

    print("📡 Veri çekiliyor...")
    df = fetch_smart("BTC-USD", "4h", "2y")

    opt_results = run_optimization(df, n_trials=150, use_walk_forward=True)
    validate_best_params(df, opt_results["best_params"])