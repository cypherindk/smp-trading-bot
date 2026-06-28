"""
kraken/indicators.py
Project KRAKEN V16 — DUZELTILMIS
Trap tolerans suresi (trapTo) eksikti, eklendi.
"""

import pandas as pd
import numpy as np


def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def calc_yang_zhang_atr(open_, high, low, close, period=20):
    prev_close = close.shift(1)

    yz_or = np.log(open_ / prev_close.replace(0, np.nan)).fillna(0)
    yz_co = np.log(close / open_.replace(0, np.nan)).fillna(0)
    yz_ho = np.log(high  / open_.replace(0, np.nan)).fillna(0)
    yz_hc = np.log(high  / close.replace(0, np.nan)).fillna(0)
    yz_lo = np.log(low   / open_.replace(0, np.nan)).fillna(0)
    yz_lc = np.log(low   / close.replace(0, np.nan)).fillna(0)

    sq_or = yz_or.rolling(period).var()
    sq_co = yz_co.rolling(period).var()
    sq_rs = (yz_ho * yz_hc + yz_lo * yz_lc).rolling(period).mean()

    k = 0.34 / (1.34 + (period + 1.0) / max(period - 1.0, 1.0))
    sq = sq_or.fillna(0) + k * sq_co.fillna(0) + (1.0 - k) * sq_rs.fillna(0)
    sigma = np.sqrt(sq.clip(lower=0)).clip(lower=1e-10)

    return close * sigma


def calc_cmf(high, low, close, volume, period=21):
    hl_range = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / hl_range
    mfm = mfm.fillna(0)
    mfv = volume * mfm
    cmf = mfv.rolling(period).sum() / volume.rolling(period).sum().replace(0, np.nan)
    return cmf.fillna(0)


def calc_pivot_high(high, length=5):
    result = pd.Series(np.nan, index=high.index)
    for i in range(length, len(high) - length):
        window = high.iloc[i-length:i+length+1]
        if high.iloc[i] == window.max() and (window == high.iloc[i]).sum() == 1:
            result.iloc[i] = high.iloc[i]
    return result


def calc_pivot_low(low, length=5):
    result = pd.Series(np.nan, index=low.index)
    for i in range(length, len(low) - length):
        window = low.iloc[i-length:i+length+1]
        if low.iloc[i] == window.min() and (window == low.iloc[i]).sum() == 1:
            result.iloc[i] = low.iloc[i]
    return result


def calc_kmeans_supertrend(yz_atr, train_data=100):
    upper_vol = yz_atr.rolling(train_data).max()
    lower_vol = yz_atr.rolling(train_data).min()

    amean = lower_vol + (upper_vol - lower_vol) * 0.75
    bmean = lower_vol + (upper_vol - lower_vol) * 0.50
    cmean = lower_vol + (upper_vol - lower_vol) * 0.25

    dist_a = (yz_atr - amean).abs()
    dist_b = (yz_atr - bmean).abs()
    dist_c = (yz_atr - cmean).abs()

    min_dist = pd.concat([dist_a, dist_b, dist_c], axis=1).min(axis=1)
    assigned = pd.Series(np.where(
        min_dist == dist_a, amean,
        np.where(min_dist == dist_b, bmean, cmean)
    ), index=yz_atr.index)

    return assigned


def calc_supertrend_bands(hl2, close, assigned_centroid, mult=3.0):
    upper_band = hl2 + mult * assigned_centroid
    lower_band = hl2 - mult * assigned_centroid

    final_lower = lower_band.copy()
    final_upper = upper_band.copy()

    for i in range(1, len(close)):
        if lower_band.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
            final_lower.iloc[i] = lower_band.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i-1]

        if upper_band.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
            final_upper.iloc[i] = upper_band.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i-1]

    direction = pd.Series(1, index=close.index)
    supertrend = pd.Series(np.nan, index=close.index)

    for i in range(1, len(close)):
        prev_st = supertrend.iloc[i-1] if not pd.isna(supertrend.iloc[i-1]) else final_upper.iloc[i-1]

        if prev_st == final_upper.iloc[i-1]:
            direction.iloc[i] = -1 if close.iloc[i] > final_upper.iloc[i] else 1
        else:
            direction.iloc[i] = 1 if close.iloc[i] < final_lower.iloc[i] else -1

        supertrend.iloc[i] = final_lower.iloc[i] if direction.iloc[i] == -1 else final_upper.iloc[i]

    return pd.DataFrame({"supertrend": supertrend, "direction": direction})


def compute_kraken_indicators(df, yz_len=20, piv_len=5, cmf_len=21,
                               train_data=100, st_mult=3.0,
                               ema_len=200, bb_len=20, trap_tolerance=8,
                               fvg_mult=0.5):
    """
    DUZELTME: trap_tolerance eklendi.
    Pine: trapBullActive 8 bar boyunca true kalir (barsSinceBullTrap <= trapTo)
    """
    out = df.copy()

    out["yz_atr"] = calc_yang_zhang_atr(df["open"], df["high"], df["low"], df["close"], yz_len)

    out["ema_macro"] = calc_ema(df["close"], ema_len)
    out["is_uptrend"]   = df["close"] > out["ema_macro"]
    out["is_downtrend"] = df["close"] < out["ema_macro"]

    basis = df["close"].rolling(bb_len).mean()
    dev   = 2.0 * df["close"].rolling(bb_len).std()
    kc_dev = 1.5 * out["yz_atr"]
    out["is_squeeze"] = (basis + dev < basis + kc_dev) & (basis - dev > basis - kc_dev)

    out["cmf"] = calc_cmf(df["high"], df["low"], df["close"], df["volume"], cmf_len)

    out["pivot_high"] = calc_pivot_high(df["high"], piv_len)
    out["pivot_low"]  = calc_pivot_low(df["low"], piv_len)

    last_pl = out["pivot_low"].ffill()
    last_ph = out["pivot_high"].ffill()

    out["is_bull_sweep"] = (df["low"] < last_pl) & (df["close"] > last_pl)
    out["is_bear_sweep"] = (df["high"] > last_ph) & (df["close"] < last_ph)

    out["is_bull_fvg"] = (df["low"] > df["high"].shift(2)) & \
                         ((df["low"] - df["high"].shift(2)) > (out["yz_atr"] * fvg_mult))
    out["is_bear_fvg"] = (df["high"] < df["low"].shift(2)) & \
                         ((df["low"].shift(2) - df["high"]) > (out["yz_atr"] * fvg_mult))

    # === DUZELTME: Trap tolerans suresi mantigi ===
    cmf_rising  = out["cmf"] > out["cmf"].shift(1)
    cmf_falling = out["cmf"] < out["cmf"].shift(1)

    trap_bull_trigger = out["is_bull_sweep"] & cmf_rising
    trap_bear_trigger = out["is_bear_sweep"] & cmf_falling

    trap_bull_active = pd.Series(False, index=df.index)
    trap_bear_active = pd.Series(False, index=df.index)

    bars_since_bull_trap = 999
    bars_since_bear_trap = 999

    for i in range(len(df)):
        if trap_bull_trigger.iloc[i]:
            bars_since_bull_trap = 0
        else:
            bars_since_bull_trap += 1

        if trap_bear_trigger.iloc[i]:
            bars_since_bear_trap = 0
        else:
            bars_since_bear_trap += 1

        trap_bull_active.iloc[i] = bars_since_bull_trap <= trap_tolerance
        trap_bear_active.iloc[i] = bars_since_bear_trap <= trap_tolerance

    out["trap_bull_active"] = trap_bull_active
    out["trap_bear_active"] = trap_bear_active

    # Final sinyaller — artik trap aktifken FVG cikarsa yakalanir
    out["buy_signal"]  = out["trap_bull_active"] & out["is_bull_fvg"]
    out["sell_signal"] = out["trap_bear_active"] & out["is_bear_fvg"]

    hl2 = (df["high"] + df["low"]) / 2
    assigned = calc_kmeans_supertrend(out["yz_atr"], train_data)
    st_df = calc_supertrend_bands(hl2, df["close"], assigned, st_mult)
    out = pd.concat([out, st_df], axis=1)

    return out


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from data.fetcher import fetch_ohlcv

    df = fetch_ohlcv("EREGL.IS", interval="1d", period="2y")
    ind = compute_kraken_indicators(df)
    print(f"BUY: {ind['buy_signal'].sum()}, SELL: {ind['sell_signal'].sum()}")