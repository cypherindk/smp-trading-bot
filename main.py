import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.fetcher import fetch_ohlcv
from engine.indicators import compute_all_indicators
from engine.signals import calc_bull_bear_score, calc_triggers, generate_signals
from engine.filters import apply_all_filters
from backtest.runner import run_backtest, print_results

# 4H parametreleri (mevcut en iyi)
COIN_PARAMS_4H = {
    "BTC-USD": {
        "preset": "Default", "eff_score": 5.0, "min_conf": 2,
        "adr_mult": 1.5, "rr_ratio": 2.0, "priority": 1,
        "symbol": "BTC/USDT",
    },
    "ETH-USD": {
        "preset": "Conservative", "eff_score": 4.5, "min_conf": 1,
        "adr_mult": 2.8, "rr_ratio": 3.5, "priority": 2,
        "symbol": "ETH/USDT",
    },
    "SOL-USD": {
        "preset": "Default", "eff_score": 7.0, "min_conf": 1,
        "adr_mult": 1.9, "rr_ratio": 1.8, "priority": 3,
        "symbol": "SOL/USDT",
    },
}

# 1H parametreleri (optimize edilecek, simdilik scalping preset)
COIN_PARAMS_1H = {
    "BTC-USD": {
        "preset": "Scalping", "eff_score": 4.0, "min_conf": 2,
        "adr_mult": 1.2, "rr_ratio": 1.5, "priority": 1,
        "symbol": "BTC/USDT",
    },
    "ETH-USD": {
        "preset": "Scalping", "eff_score": 4.0, "min_conf": 2,
        "adr_mult": 1.5, "rr_ratio": 2.0, "priority": 2,
        "symbol": "ETH/USDT",
    },
    "SOL-USD": {
        "preset": "Scalping", "eff_score": 5.0, "min_conf": 2,
        "adr_mult": 1.3, "rr_ratio": 1.8, "priority": 3,
        "symbol": "SOL/USDT",
    },
}


def run_test():
    print("\n HIZLI TEST MODU")
    print("-" * 40)
    for tf in ["4h", "1h"]:
        df  = fetch_ohlcv("BTC-USD", interval=tf, period="3mo")
        ind = compute_all_indicators(df, preset="Default")
        sc  = calc_bull_bear_score(ind)
        tr  = calc_triggers(ind, sc)
        sg  = generate_signals(ind, sc, tr)
        fs  = apply_all_filters(ind, sg)
        print(f"\n BTC {tf.upper()}:")
        print(f"   Veri: {len(df)} bar")
        print(f"   BUY: {fs['buy_signal'].sum()}  SELL: {fs['sell_signal'].sum()}")
        last_buy  = df[fs["buy_signal"]].index.tolist()
        last_sell = df[fs["sell_signal"]].index.tolist()
        if last_buy:
            print(f"   Son BUY: {last_buy[-1]}")
        if last_sell:
            print(f"   Son SELL: {last_sell[-1]}")


def run_full_backtest():
    print("\n TAM BACKTEST — BTC 4H ve 1H")
    print("-" * 40)

    for tf, params in [("4h", COIN_PARAMS_4H), ("1h", COIN_PARAMS_1H)]:
        p  = params["BTC-USD"]
        period = "2y" if tf == "4h" else "2y"
        df  = fetch_ohlcv("BTC-USD", interval=tf, period=period)
        ind = compute_all_indicators(df, preset=p["preset"])
        sc  = calc_bull_bear_score(ind)
        tr  = calc_triggers(ind, sc)
        sg  = generate_signals(ind, sc, tr, preset=p["preset"],
                               eff_score=p["eff_score"], min_conf=p["min_conf"])
        fs  = apply_all_filters(ind, sg, use_cvd=True)
        print(f"\nBTC {tf.upper()} — Sinyaller: BUY={fs['buy_signal'].sum()} SELL={fs['sell_signal'].sum()}")
        results = run_backtest(df, ind, fs, initial_capital=10000, risk_pct=2.0,
                               adr_mult=p["adr_mult"], rr_ratio=p["rr_ratio"])
        print_results(results, f"SMP V3.1 - BTC {tf.upper()}")


def run_multi_coin():
    print("\n MULTI-COIN TEST — 4H ve 1H")
    print("=" * 65)

    for tf, params, period in [("4h", COIN_PARAMS_4H, "2y"), ("1h", COIN_PARAMS_1H, "2y")]:
        print(f"\n{'─'*65}")
        print(f"  {tf.upper()} ZAMAN DİLİMİ")
        print(f"{'─'*65}")
        print(f"{'Coin':<12} {'Getiri':>8} {'Win%':>7} {'Islem':>7} {'MaxDD':>8} {'Sharpe':>8}")
        print("-" * 55)
        for coin, p in params.items():
            try:
                df  = fetch_ohlcv(coin, interval=tf, period=period)
                ind = compute_all_indicators(df, preset=p["preset"])
                sc  = calc_bull_bear_score(ind)
                tr  = calc_triggers(ind, sc)
                sg  = generate_signals(ind, sc, tr, preset=p["preset"],
                                       eff_score=p["eff_score"], min_conf=p["min_conf"])
                fs  = apply_all_filters(ind, sg, use_cvd=True)
                r   = run_backtest(df, ind, fs, initial_capital=10000, risk_pct=2.0,
                                   adr_mult=p["adr_mult"], rr_ratio=p["rr_ratio"])
                if r:
                    print(f"{coin:<12} {r['total_return_pct']:>7.1f}%"
                          f" {r['win_rate_pct']:>6.1f}%"
                          f" {r['total_trades']:>7}"
                          f" {r['max_drawdown_pct']:>7.1f}%"
                          f" {r['sharpe_ratio']:>8.2f}")
            except Exception as e:
                print(f"{coin:<12} HATA: {e}")
        print("-" * 55)


def run_optimization_mode():
    from optimize.optimizer import run_optimization
    print("\n OPTIMIZASYON MODU — 4H ve 1H, Tum Coinler")
    print("-" * 40)

    coins = ["BTC-USD", "ETH-USD", "SOL-USD"]
    results = {}

    for tf, period in [("4h", "2y"), ("1h", "2y")]:
        results[tf] = {}
        for coin in coins:
            print(f"\n{'='*40}\n{coin} {tf.upper()} optimize ediliyor...\n{'='*40}")
            df  = fetch_ohlcv(coin, interval=tf, period=period)
            opt = run_optimization(df, n_trials=100, use_walk_forward=True)
            results[tf][coin] = opt["best_params"]

    print("\n\n TUM SONUCLAR:")
    print("=" * 55)
    for tf in ["4h", "1h"]:
        print(f"\n--- {tf.upper()} ---")
        for coin, params in results[tf].items():
            print(f"\n{coin}:")
            for k, v in params.items():
                print(f"  {k}: {v}")

    return results


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    if mode == "test":
        run_test()
    elif mode == "backtest":
        run_full_backtest()
    elif mode == "multicoin":
        run_multi_coin()
    elif mode == "optimize":
        run_optimization_mode()
    else:
        print(f"Bilinmeyen mod: {mode}")
        print("Kullanim: python main.py [test|backtest|multicoin|optimize]")