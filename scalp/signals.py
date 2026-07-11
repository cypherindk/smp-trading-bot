"""
scalp/signals.py
SCALP PRO V1.0 — Sinyal Motoru
"""

import pandas as pd
import numpy as np


def generate_scalp_signals(ind, tf="15m",
                           rvol_thr=None,
                           require_macd=False,
                           require_ema200=True):
    if rvol_thr is None:
        rvol_map = {"5m": 1.5, "15m": 1.2, "1h": 1.0}
        rvol_thr = rvol_map.get(tf, 1.2)

    trend_bull = ind["close"] > ind["ema200"] if require_ema200 else pd.Series(True, index=ind.index)
    trend_bear = ind["close"] < ind["ema200"] if require_ema200 else pd.Series(True, index=ind.index)

    vol_ok = ind["rvol"] > rvol_thr

    if require_macd:
        macd_bull = (ind["macd_hist"] > 0) & (ind["macd_line"] > ind["macd_signal"])
        macd_bear = (ind["macd_hist"] < 0) & (ind["macd_line"] < ind["macd_signal"])
    else:
        macd_bull = pd.Series(True, index=ind.index)
        macd_bear = pd.Series(True, index=ind.index)

    trigger_bull = (
        ind["trigger_pac"] |
        ind["trigger_expansion_bull"] |
        ind["era_long"]
    )

    trigger_bear = (
        ind["trigger_expansion_bear"] |
        ind["era_short"]
    )

    raw_buy = (
        trend_bull &
        vol_ok &
        macd_bull &
        trigger_bull &
        (ind["rsi14"] < 75) &
        (ind["rsi14"] > 25)
    )

    raw_sell = (
        trend_bear &
        vol_ok &
        macd_bear &
        trigger_bear &
        (ind["rsi14"] > 25) &
        (ind["rsi14"] < 75)
    )

    buy_final  = pd.Series(False, index=ind.index)
    sell_final = pd.Series(False, index=ind.index)
    last_dir   = 0

    for i in range(len(ind)):
        if raw_buy.iloc[i] and last_dir != 1:
            buy_final.iloc[i] = True
            last_dir = 1
        elif raw_sell.iloc[i] and last_dir != -1:
            sell_final.iloc[i] = True
            last_dir = -1

    return pd.DataFrame({
        "buy_signal":  buy_final,
        "sell_signal": sell_final,
        "raw_buy":     raw_buy,
        "raw_sell":    raw_sell,
    })