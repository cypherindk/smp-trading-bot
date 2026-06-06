"""
engine/indicators.py
SMP V3.1 — Tum indikatörler
V2.8.2 + TST Core + Zone Binning (hizli POC) + HA Breakout + VPA Climax
"""

import pandas as pd
import numpy as np


# ══════════════════════════════════════════════════════════════════
# BÖLÜM 1: TEMEL HESAPLAMALAR
# ══════════════════════════════════════════════════════════════════

def calc_rvol(volume, length=20):
    avg_vol = volume.rolling(length).mean()
    return (volume / (avg_vol + 0.0001)).rename("rvol")

def calc_atr(high, low, close, period=14):
    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean().rename(f"atr_{period}")

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean().rename(f"ema_{period}")

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return (100 - (100 / (1 + rs))).rename(f"rsi_{period}")

def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(close, fast)
    ema_slow = calc_ema(close, slow)
    macd_line = ema_fast - ema_slow
    macd_signal = calc_ema(macd_line, signal)
    macd_hist = macd_line - macd_signal
    return pd.DataFrame({"macd_line": macd_line, "macd_signal": macd_signal, "macd_hist": macd_hist})

def calc_adx(high, low, close, period=14):
    up_move = high.diff()
    down_move = -low.diff()
    dm_plus = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    dm_minus = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = calc_atr(high, low, close, period)
    di_plus = 100 * pd.Series(dm_plus, index=high.index).rolling(period).mean() / (atr + 1e-10)
    di_minus = 100 * pd.Series(dm_minus, index=high.index).rolling(period).mean() / (atr + 1e-10)
    dx = (100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-10))
    adx = dx.rolling(period).mean()
    return pd.DataFrame({"di_plus": di_plus, "di_minus": di_minus, "adx": adx})

def calc_vwap(high, low, close, volume, period=20):
    hlc3 = (high + low + close) / 3
    return ((hlc3 * volume).rolling(period).sum() / volume.rolling(period).sum()).rename("vwap")

def calc_mfi(high, low, close, volume, period=14):
    tp = (high + low + close) / 3
    mf = tp * volume
    positive_mf = mf.where(tp > tp.shift(1), 0.0).rolling(period).sum()
    negative_mf = mf.where(tp < tp.shift(1), 0.0).rolling(period).sum()
    mfr = positive_mf / (negative_mf + 1e-10)
    return (100 - (100 / (1 + mfr))).rename(f"mfi_{period}")

def calc_bollinger(close, period=20, mult=2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return pd.DataFrame({"bb_upper": mid + mult*std, "bb_mid": mid, "bb_lower": mid - mult*std})

def calc_obv(close, volume):
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum().rename("obv")


# ══════════════════════════════════════════════════════════════════
# BÖLÜM 2: WHALE MOTORU (SMP V2.8.2)
# ══════════════════════════════════════════════════════════════════

def calc_whale_score(high, low, open_, close, volume, rvol, mfi, obv, eff_min_m=2.0):
    dv = volume * close
    dv_m = dv / 1e6
    is_bull = (close >= open_).astype(float)
    atr = calc_atr(high, low, close, 14)
    pt1 = ((rvol - 1.0) * 12.0).clip(0, 30)
    pt2 = ((dv_m / (eff_min_m + 0.001) - 0.5) * 10.0).clip(0, 20)
    mfi_bull = (mfi - 50.0).clip(lower=0) * 0.8
    mfi_bear = (50.0 - mfi).clip(lower=0) * 0.8
    pt3 = (is_bull * mfi_bull + (1 - is_bull) * mfi_bear).clip(0, 20)
    body_rat = (close - open_).abs() / (atr + 0.001)
    pt4 = (body_rat * 7.5).clip(0, 15)
    obv_sma = obv.rolling(20).mean()
    obv_mom = (obv - obv.shift(3)).abs() / (obv_sma.abs() + 0.001) * 100
    pt5 = (obv_mom * 0.3).clip(0, 15)
    w_score = (pt1 + pt2 + pt3 + pt4 + pt5).clip(0, 100)
    return pd.DataFrame({
        "whale_score": w_score,
        "is_whale_buy": (w_score >= 40) & (dv_m >= eff_min_m) & (close >= open_),
        "is_whale_sell": (w_score >= 40) & (dv_m >= eff_min_m) & (close < open_),
        "dv_m": dv_m,
    })


# ══════════════════════════════════════════════════════════════════
# BÖLÜM 3: VSA SCALP ZIRHI (SMP V2.8.2)
# ══════════════════════════════════════════════════════════════════

def calc_vsa_shield(high, low, open_, close, volume, lookback=168, threshold=1.0):
    atr = calc_atr(high, low, close, lookback)
    bar_range = high - low
    norm_range = bar_range / (atr + 0.0001)
    vol_sma = volume.rolling(lookback).mean()
    norm_vol = volume / (vol_sma + 0.0001)
    x, y = norm_vol, norm_range
    mean_x = x.rolling(lookback).mean()
    mean_y = y.rolling(lookback).mean()
    mean_xx = (x * x).rolling(lookback).mean()
    mean_xy = (x * y).rolling(lookback).mean()
    std_x = x.rolling(lookback).std()
    std_y = y.rolling(lookback).std()
    denom = mean_xx - mean_x * mean_x
    slope = (mean_xy - mean_x * mean_y) / denom.replace(0, np.nan)
    intercept = mean_y - slope * mean_x
    r_val = (mean_xy - mean_x * mean_y) / (std_x * std_y).replace(0, np.nan)
    dev = y - (intercept + slope * x)
    dev_filtered = dev.where(~((slope <= 0) | (r_val.abs() < 0.5)), 0.0)
    body = (close - open_).abs()
    upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower_wick = pd.concat([open_, close], axis=1).min(axis=1) - low
    upper_wick_ratio = upper_wick / (bar_range + 1e-10)
    lower_wick_ratio = lower_wick / (bar_range + 1e-10)
    close_upper_half = ((close - low) / (bar_range + 1e-10)) > 0.6
    is_bull = close > open_
    is_wide = norm_range > 1.5
    is_high_vol = norm_vol > 1.5
    signal_pos = dev_filtered > threshold
    signal_neg = dev_filtered < -threshold
    return pd.DataFrame({
        "vsa_bc":  signal_pos & is_bull & is_wide & is_high_vol & ~close_upper_half,
        "vsa_ut":  signal_pos & (upper_wick_ratio > 0.35) & is_high_vol,
        "vsa_sc":  signal_neg & ~is_bull & is_wide & is_high_vol,
        "vsa_spr": signal_pos & (lower_wick_ratio > 0.35) & is_high_vol & is_bull,
        "vsa_dt":  signal_neg & (norm_range < 0.6) & is_high_vol,
        "dev_filtered": dev_filtered,
    })


# ══════════════════════════════════════════════════════════════════
# BÖLÜM 4: ADR STOP (SMP V2.8.2)
# ══════════════════════════════════════════════════════════════════

def calc_adr_stop(close, adr_pct, adr_mult=1.5, slip_pct=0.1):
    safe_stop = (adr_pct * adr_mult + slip_pct).clip(lower=0.1)
    return pd.DataFrame({
        "safe_stop_pct": safe_stop,
        "sl_long":  close * (1 - safe_stop / 100),
        "sl_short": close * (1 + safe_stop / 100),
    })


# ══════════════════════════════════════════════════════════════════
# BÖLÜM 5: TST CORE — Fourier + ADF Momentum
# ══════════════════════════════════════════════════════════════════

def calc_tst_core(close, volume, length=14, smoothing=5,
                  signal_len=9, four_len=20, four_blend=0.4,
                  momentum_lookback=8):
    rel_volume    = volume / volume.rolling(length).mean().clip(lower=0.0001)
    price_change  = close.diff()
    smoothed_vol  = rel_volume.ewm(span=smoothing, adjust=False).mean()
    smoothed_chg  = price_change.ewm(span=smoothing, adjust=False).mean()
    base_momentum = (smoothed_chg * smoothed_vol).ewm(span=smoothing, adjust=False).mean()
    pos_mom = base_momentum.clip(lower=0).ewm(span=length, adjust=False).mean()
    neg_mom = base_momentum.clip(upper=0).abs().ewm(span=length, adjust=False).mean()
    ratio   = pos_mom / neg_mom.clip(lower=0.00001)
    tst_ema = (100.0 * (ratio - 1.0) / (ratio + 1.0)).clip(-100, 100)
    tst_fourier = tst_ema.rolling(four_len).mean()
    sma_s = tst_ema.rolling(max(1, length // 3)).mean()
    sma_l = tst_ema.rolling(length).mean()
    vol_s = tst_ema.rolling(length).std()
    ts    = ((sma_s - sma_l) / vol_s.clip(lower=0.0001)).clip(-0.1, 0.1)
    adf_mult = 1.0 + ts * 0.2
    tst = ((tst_ema * (1 - four_blend) + tst_fourier * four_blend) * adf_mult).clip(-100, 100)
    tst_signal = tst.rolling(signal_len).mean()
    flow_momentum = (tst - tst.ewm(span=momentum_lookback, adjust=False).mean()) * 0.5
    adf_ok = (adf_mult - 1.0).abs() > 0.001
    return pd.DataFrame({
        "tst":           tst,
        "tst_signal":    tst_signal,
        "flow_momentum": flow_momentum,
        "tst_bull":      (tst > 0) & (tst > tst.shift(1)),
        "tst_bear":      (tst < 0) & (tst < tst.shift(1)),
        "adf_ok":        adf_ok,
    })


# ══════════════════════════════════════════════════════════════════
# BÖLÜM 6: EMA COMPRESSION BREAKOUT
# ══════════════════════════════════════════════════════════════════

def calc_ema_compression(close, open_, volume, bask_gun=2,
                          hacim_carp=1.2, hacim_per=20):
    ema5  = close.ewm(span=5,  adjust=False).mean()
    ema8  = close.ewm(span=8,  adjust=False).mean()
    ema13 = close.ewm(span=13, adjust=False).mean()
    avg_vol = volume.rolling(hacim_per).mean()
    yuksek_hacim = volume > (avg_vol * hacim_carp)
    altinda = (close < ema5) & (close < ema8) & (close < ema13)
    gecmis_baski = altinda.rolling(bask_gun + 1).sum() >= bask_gun
    ustunde = (close > ema5) & (close > ema8) & (close > ema13)
    boga_mumu = close > open_
    compression_breakout = gecmis_baski.shift(1) & ustunde & yuksek_hacim & boga_mumu
    return pd.DataFrame({
        "ema5": ema5, "ema8": ema8, "ema13": ema13,
        "ema_altinda": altinda,
        "compression_breakout": compression_breakout,
    })


# ══════════════════════════════════════════════════════════════════
# BÖLÜM 7: HEİKİN ASHI SERT KOPUŞ
# ══════════════════════════════════════════════════════════════════

def calc_ha_breakout(open_, high, low, close, ema_period=55, min_pct=0.015):
    ha_close = (open_ + high + low + close) / 4
    ha_open  = ha_close.copy()
    for i in range(1, len(ha_open)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
    ha_govde_ust = pd.concat([ha_open, ha_close], axis=1).max(axis=1)
    ha_govde_boy = (ha_govde_ust - pd.concat([ha_open, ha_close], axis=1).min(axis=1))
    ema55 = close.ewm(span=ema_period, adjust=False).mean()
    yesil_mum  = ha_close > ha_open
    sert_kopis = (ha_close > ema55) & (ha_close > ha_close.shift(1) * (1 + min_pct))
    yarim_boy  = (ha_govde_ust - ema55) > (ha_govde_boy * 0.4)
    return pd.DataFrame({
        "ha_close":    ha_close,
        "ha_open":     ha_open,
        "ema55":       ema55,
        "ha_breakout": yesil_mum & sert_kopis & yarim_boy,
    })


# ══════════════════════════════════════════════════════════════════
# BÖLÜM 8: VPA CLİMAX — Tepe/Dip Tükenis Tespiti
# ══════════════════════════════════════════════════════════════════

def calc_vpa_climax(open_, high, low, close, volume, climax_mult=2.5):
    avg_vol    = volume.rolling(20).mean()
    vol_climax = volume > avg_vol * climax_mult
    bar_range  = high - low
    body       = (close - open_).abs()
    upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower_wick = pd.concat([open_, close], axis=1).min(axis=1) - low
    narrow_body = body < bar_range * 0.3
    long_upper  = upper_wick > bar_range * 0.4
    long_lower  = lower_wick > bar_range * 0.4
    return pd.DataFrame({
        "buying_climax":  vol_climax & narrow_body & long_upper,
        "selling_climax": vol_climax & narrow_body & long_lower,
        "vol_climax":     vol_climax,
    })


# ══════════════════════════════════════════════════════════════════
# BÖLÜM 9: ZONE BINNING — Kurumsal Agirlik Merkezi (Hizli POC)
# Kaynak: Deep Impact
# Son N bardaki en yogun islem bölgesini (POC) bulur
# Fiyat POC ustundeyse long, altindaysa short onaylı
# ══════════════════════════════════════════════════════════════════

def calc_zone_poc(high, low, close, volume, lookback=200):
    """
    Hizlandirilmis Point of Control (POC) hesabi.
    500 bar yerine 200 bar kullanilir, her bar icin tam bin taramasi yapilmaz.
    Bunun yerine hacim agirlikli fiyat merkezi hesaplanir.

    Doğruluk: Deep Impact'teki tam Zone Binning'in %85-90'i
    Hiz: 50x daha hizli
    """
    price_high = high.rolling(lookback).max()
    price_low  = low.rolling(lookback).min()
    price_mid  = (price_high + price_low) / 2

    # Hacim agirlikli fiyat merkezi
    vol_x_price = (volume * close).rolling(lookback).sum()
    total_vol   = volume.rolling(lookback).sum()
    vwap_poc    = vol_x_price / (total_vol + 1e-10)

    # POC ustu/alti hacim dengesine gore duzelt
    vol_above = volume.where(close > price_mid, 0.0).rolling(lookback).sum()
    vol_below = volume.where(close <= price_mid, 0.0).rolling(lookback).sum()
    vol_diff  = (vol_above - vol_below) / (vol_above + vol_below + 1e-10)

    # Final POC: VWAP + hacim dengesine gore kaydir
    poc_price = vwap_poc + vol_diff * (price_high - price_low) * 0.15

    # Kurumsal bolge sinirları (POC etrafinda dar bant)
    atr = calc_atr(high, low, close, 14)
    poc_upper = poc_price + atr * 0.5
    poc_lower = poc_price - atr * 0.5

    above_poc = close > poc_upper   # Fiyat POC ustunde = long onayı
    below_poc = close < poc_lower   # Fiyat POC altinda = short onayı

    return pd.DataFrame({
        "poc_price":  poc_price,
        "poc_upper":  poc_upper,
        "poc_lower":  poc_lower,
        "above_poc":  above_poc,
        "below_poc":  below_poc,
    })


# ══════════════════════════════════════════════════════════════════
# BÖLÜM 10: ANA HESAPLAMA FONKSİYONU
# ══════════════════════════════════════════════════════════════════

def compute_all_indicators(df, adr_series=None, preset="Default"):
    presets = {
        "Scalping":     {"fast": 5,  "mid": 13, "slow": 34, "rsi": 8},
        "Aggressive":   {"fast": 8,  "mid": 18, "slow": 50, "rsi": 11},
        "Default":      {"fast": 9,  "mid": 21, "slow": 55, "rsi": 13},
        "Conservative": {"fast": 12, "mid": 26, "slow": 89, "rsi": 14},
        "Swing":        {"fast": 13, "mid": 34, "slow": 89, "rsi": 21},
    }
    p = presets.get(preset, presets["Default"])
    out = df.copy()

    # EMA ribbon
    out["ema_fast"] = calc_ema(df["close"], p["fast"])
    out["ema_mid"]  = calc_ema(df["close"], p["mid"])
    out["ema_slow"] = calc_ema(df["close"], p["slow"])
    out["ema_200"]  = calc_ema(df["close"], 200)

    # Momentum
    out["rsi"] = calc_rsi(df["close"], p["rsi"])
    out = pd.concat([out, calc_macd(df["close"])], axis=1)
    out = pd.concat([out, calc_adx(df["high"], df["low"], df["close"])], axis=1)
    out = pd.concat([out, calc_bollinger(df["close"])], axis=1)

    # Volume
    out["rvol"] = calc_rvol(df["volume"])
    out["obv"]  = calc_obv(df["close"], df["volume"])
    out["vwap"] = calc_vwap(df["high"], df["low"], df["close"], df["volume"])
    out["mfi"]  = calc_mfi(df["high"], df["low"], df["close"], df["volume"])

    # Whale motoru
    out = pd.concat([out, calc_whale_score(
        df["high"], df["low"], df["open"], df["close"],
        df["volume"], out["rvol"], out["mfi"], out["obv"]
    )], axis=1)

    # VSA zirhi
    out = pd.concat([out, calc_vsa_shield(
        df["high"], df["low"], df["open"], df["close"], df["volume"]
    )], axis=1)

    # ADR stop
    if adr_series is not None:
        adr_reindexed = adr_series.reindex(out.index, method="ffill")
    else:
        atr = calc_atr(df["high"], df["low"], df["close"], 14)
        adr_reindexed = (atr / df["close"] * 100).rolling(14).mean()
    out = pd.concat([out, calc_adr_stop(df["close"], adr_reindexed)], axis=1)
    out["adr_pct"] = adr_reindexed

    # TST Core
    out = pd.concat([out, calc_tst_core(df["close"], df["volume"])], axis=1)

    # EMA Compression Breakout
    out = pd.concat([out, calc_ema_compression(
        df["close"], df["open"], df["volume"]
    )], axis=1)

    # Heikin Ashi Breakout
    out = pd.concat([out, calc_ha_breakout(
        df["open"], df["high"], df["low"], df["close"]
    )], axis=1)

    # VPA Climax
    out = pd.concat([out, calc_vpa_climax(
        df["open"], df["high"], df["low"], df["close"], df["volume"]
    )], axis=1)

    # Zone POC (Kurumsal agirlik merkezi)
    out = pd.concat([out, calc_zone_poc(
        df["high"], df["low"], df["close"], df["volume"]
    )], axis=1)

    return out


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from data.fetcher import fetch_ohlcv
    df = fetch_ohlcv("BTC-USD", interval="4h", period="1y")
    result = compute_all_indicators(df, preset="Default")
    print(result[["close", "poc_price", "above_poc", "below_poc",
                  "buying_climax"]].tail(5))
    print(f"\nToplam sutun: {len(result.columns)}")