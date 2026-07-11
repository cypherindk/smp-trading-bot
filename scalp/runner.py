"""
scalp/runner.py
SCALP PRO V1.0 — Backtest Motoru
"""

import sys
import os
import pandas as pd
import numpy as np
import vectorbt as vbt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetcher import fetch_ohlcv
from scalp.indicators import compute_scalp_indicators
from scalp.signals import generate_scalp_signals

COINS = ["BTC-USD", "ETH-USD", "SOL-USD"]

TF_PARAMS = {
    "5m":  {"period": "60d",  "zombie": 10, "sl_m": 2.0, "tp1_m": 2.0, "tp2_m": 4.0, "rvol": 1.5},
    "15m": {"period": "60d",  "zombie": 15, "sl_m": 1.5, "tp1_m": 1.5, "tp2_m": 3.0, "rvol": 1.2},
    "1h":  {"period": "730d", "zombie": 12, "sl_m": 1.2, "tp1_m": 1.2, "tp2_m": 2.4, "rvol": 1.0},
}


def run_scalp_backtest(coin, tf="15m",
                       initial_capital=10000,
                       risk_pct=1.0,
                       commission_pct=0.05):
    p = TF_PARAMS[tf]
    df  = fetch_ohlcv(coin, interval=tf, period=p["period"])
    ind = compute_scalp_indicators(df, tf=tf)
    sg  = generate_scalp_signals(ind, tf=tf, rvol_thr=p["rvol"], require_macd=False)

    entries_long  = sg["buy_signal"]
    entries_short = sg["sell_signal"]

    sl_pct  = ind["sl_pct"] / 100
    tp1_pct = ind["tp1_pct"] / 100

    try:
        pf = vbt.Portfolio.from_signals(
            close=df["close"],
            entries=entries_long,
            exits=entries_short,
            short_entries=entries_short,
            short_exits=entries_long,
            sl_stop=sl_pct,
            tp_stop=tp1_pct,
            init_cash=initial_capital,
            fees=commission_pct / 100,
            freq=tf,
            upon_opposite_entry="Reverse",
        )
        stats = pf.stats()
        return {
            "coin":       coin,
            "tf":         tf,
            "trades":     int(stats.get("Total Trades", 0)),
            "return_pct": float(stats.get("Total Return [%]", 0)),
            "win_rate":   float(stats.get("Win Rate [%]", 0)),
            "pf":         float(stats.get("Profit Factor", 0)),
            "max_dd":     float(stats.get("Max Drawdown [%]", 0)),
            "sharpe":     float(stats.get("Sharpe Ratio", 0)),
        }
    except Exception as e:
        print(f"  Hata ({coin} {tf}): {e}")
        return None


def run_all():
    print(f"\n{'='*70}")
    print(f"  SCALP PRO V1.0 — TAM BACKTEST")
    print(f"{'='*70}")

    for tf in ["15m", "1h"]:
        print(f"\n{'─'*70}")
        print(f"  {tf.upper()} ZAMAN DİLİMİ")
        print(f"{'─'*70}")
        print(f"{'Coin':<12} {'Getiri':>8} {'Win%':>7} {'Islem':>7} {'MaxDD':>8} {'Sharpe':>8} {'PF':>6}")
        print("-" * 60)

        for coin in COINS:
            r = run_scalp_backtest(coin, tf=tf)
            if r:
                print(f"{coin:<12} {r['return_pct']:>7.1f}%"
                      f" {r['win_rate']:>6.1f}%"
                      f" {r['trades']:>7}"
                      f" {r['max_dd']:>7.1f}%"
                      f" {r['sharpe']:>8.2f}"
                      f" {r['pf']:>6.2f}")
        print("-" * 60)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "single"
    if mode == "single":
        r = run_scalp_backtest("BTC-USD", tf="15m")
        if r:
            print(f"\n{'='*50}")
            print(f"  BTC 15M SCALP BACKTEST")
            print(f"{'='*50}")
            print(f"  Getiri:    %{r['return_pct']:.2f}")
            print(f"  Win Rate:  %{r['win_rate']:.1f}")
            print(f"  Islem:     {r['trades']}")
            print(f"  Max DD:    %{r['max_dd']:.2f}")
            print(f"  Sharpe:    {r['sharpe']:.3f}")
            print(f"  PF:        {r['pf']:.3f}")
    elif mode == "all":
        run_all()