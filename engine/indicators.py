"""
engine/indicators.py
Pine Script SMP V2.8.2 indikatörlerinin birebir Python karşılıkları.
Her fonksiyon Pine Script'teki ilgili bölüme referans verir.
"""

import pandas as pd
import numpy as np
from scipy import stats


# ══════════════════════════════════════════════════════════════════
# BÖLÜM 1: RVOL & TEMEL HESAPLAMALAR
# Pine Script: "RVOL & GLOBAL TARİHSEL HESAPLAMALAR"
# ══════════════════════════════════════════════════════════════════

def calc_rvol(volume: pd.Series, length: int = 20) -> pd.Series:
    """
    Relative Volume (Göreceli Hacim)
    Pine: rvol = volume / (avgVol + 0.0001)
    """
    avg_vol = volume.rolling(length).mean()
    rvol = volume / (avg_vol + 0.0001)
    return rvol.rename("rvol")


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ATR — ta.atr() karşılığı"""
    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr.rename(f"atr_{period}")


def calc_ema(series: pd.Series, period: int) -> pd.Series:
    """EMA — ta.ema() karşılığı"""
    return series.ewm(span=period, adjust=False).mean().rename(f"ema_{period}")


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI — ta.rsi() karşılığı"""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.rename(f"rsi_{period}")


def calc_macd(close: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    """
    MACD — ta.macd() karşılığı
    Returns: DataFrame with macd_line, macd_signal, macd_hist
    """
    ema_fast = calc_ema(close, fast)
    ema_slow = calc_ema(close, slow)
    macd_line = ema_fast - ema_slow
    macd_signal = calc_ema(macd_line, signal)
    macd_hist = macd_line - macd_signal
    return pd.DataFrame({
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
    })


def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series,
             period: int = 14) -> pd.DataFrame:
    """
    ADX + DI+/DI- — ta.dmi() karşılığı
    Pine: [diPlus, diMinus, adxVal] = ta.dmi(14, 14)
    """
    up_move = high.diff()
    down_move = -low.diff()
    
    dm_plus = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    dm_minus = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    atr = calc_atr(high, low, close, period)
    
    di_plus = 100 * pd.Series(dm_plus, index=high.index).rolling(period).mean() / (atr + 1e-10)
    di_minus = 100 * pd.Series(dm_minus, index=high.index).rolling(period).mean() / (atr + 1e-10)
    
    dx = (100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-10))
    adx = dx.rolling(period).mean()
    
    return pd.DataFrame({
        "di_plus": di_plus,
        "di_minus": di_minus,
        "adx": adx,
    })


def calc_vwap(high: pd.Series, low: pd.Series, close: pd.Series,
              volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Rolling VWAP (ta.vwap() karşılığı — Pine'da session bazlı ama biz rolling kullanıyoruz)
    """
    hlc3 = (high + low + close) / 3
    vwap = (hlc3 * volume).rolling(period).sum() / volume.rolling(period).sum()
    return vwap.rename("vwap")


def calc_mfi(high: pd.Series, low: pd.Series, close: pd.Series,
             volume: pd.Series, period: int = 14) -> pd.Series:
    """
    Money Flow Index — Pine Script MFI hesabı ile birebir
    Pine: pmfSum / nmfSum mantığı
    """
    tp = (high + low + close) / 3
    mf = tp * volume
    
    positive_mf = mf.where(tp > tp.shift(1), 0.0).rolling(period).sum()
    negative_mf = mf.where(tp < tp.shift(1), 0.0).rolling(period).sum()
    
    # Pine: 100 - (100 / (1 + mfr)) — nmfSum=0 durumunda mfr=100
    mfr = positive_mf / (negative_mf + 1e-10)
    mfi = 100 - (100 / (1 + mfr))
    return mfi.rename(f"mfi_{period}")


def calc_bollinger(close: pd.Series, period: int = 20, mult: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands — ta.bb() karşılığı"""
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + mult * std
    lower = mid - mult * std
    return pd.DataFrame({"bb_upper": upper, "bb_mid": mid, "bb_lower": lower})


# ══════════════════════════════════════════════════════════════════
# BÖLÜM 2: WHALE MOTORU
# Pine Script: "COLD/HOT WHALE MOTORU"
# ══════════════════════════════════════════════════════════════════

def calc_whale_score(high: pd.Series, low: pd.Series,
                     open_: pd.Series, close: pd.Series,
                     volume: pd.Series, rvol: pd.Series,
                     mfi: pd.Series, obv: pd.Series,
                     eff_min_m: float = 2.0,
                     v_len: int = 20) -> pd.DataFrame:
    """
    Whale güven skoru (0-100)
    Pine Script'teki wPt1..wPt5 mantığının birebir karşılığı.
    """
    dv = volume * close
    dv_m = dv / 1e6
    is_bull = (close >= open_).astype(float)
    
    atr = calc_atr(high, low, close, 14)
    
    # wPt1: RVOL primi (max 30)
    pt1 = ((rvol - 1.0) * 12.0).clip(0, 30)
    
    # wPt2: İşlem büyüklüğü (max 20)
    pt2 = ((dv_m / (eff_min_m + 0.001) - 0.5) * 10.0).clip(0, 20)
    
    # wPt3: MFI yön skoru (max 20)
    mfi_bull = (mfi - 50.0).clip(lower=0) * 0.8
    mfi_bear = (50.0 - mfi).clip(lower=0) * 0.8
    mfi_score = is_bull * mfi_bull + (1 - is_bull) * mfi_bear
    pt3 = mfi_score.clip(0, 20)
    
    # wPt4: Mum gövde oranı (max 15)
    body_rat = (close - open_).abs() / (atr + 0.001)
    pt4 = (body_rat * 7.5).clip(0, 15)
    
    # wPt5: OBV momentum (max 15)
    obv_sma = obv.rolling(20).mean()
    obv_mom = (obv - obv.shift(3)).abs() / (obv_sma.abs() + 0.001) * 100
    pt5 = (obv_mom * 0.3).clip(0, 15)
    
    w_score = (pt1 + pt2 + pt3 + pt4 + pt5).clip(0, 100)
    
    is_whale_buy = (w_score >= 40) & (dv_m >= eff_min_m) & (close >= open_)
    is_whale_sell = (w_score >= 40) & (dv_m >= eff_min_m) & (close < open_)
    
    return pd.DataFrame({
        "whale_score": w_score,
        "is_whale_buy": is_whale_buy,
        "is_whale_sell": is_whale_sell,
        "dv_m": dv_m,
    })


# ══════════════════════════════════════════════════════════════════
# BÖLÜM 3: VSA SCALP ZIRHI
# Pine Script: "VSA (VOLUME SPREAD ANALYSIS) SCALP ZIRHI"
# ══════════════════════════════════════════════════════════════════

def calc_vsa_shield(high: pd.Series, low: pd.Series,
                    open_: pd.Series, close: pd.Series,
                    volume: pd.Series,
                    lookback: int = 168,
                    threshold: float = 1.0) -> pd.DataFrame:
    """
    VSA Scalp Zırhı — Pine Script'teki sıfır-lag regresyon tabanlı anomali tespiti.
    
    Dönüş: vsa_bc (Buying Climax), vsa_ut (Upthrust), vsa_sc (Selling Climax),
           vsa_spr (Spring), vsa_dt (No Demand / Distribution Test)
    """
    atr = calc_atr(high, low, close, lookback)
    bar_range = high - low
    
    norm_range = bar_range / (atr + 0.0001)
    vol_sma = volume.rolling(lookback).mean()
    norm_vol = volume / (vol_sma + 0.0001)
    
    # Kayan lineer regresyon (slope + intercept)
    # Pine: denom = mean_xx - mean_x*mean_x
    x = norm_vol
    y = norm_range
    
    mean_x = x.rolling(lookback).mean()
    mean_y = y.rolling(lookback).mean()
    mean_xx = (x * x).rolling(lookback).mean()
    mean_xy = (x * y).rolling(lookback).mean()
    std_x = x.rolling(lookback).std()
    std_y = y.rolling(lookback).std()
    
    denom = mean_xx - mean_x * mean_x
    slope = (mean_xy - mean_x * mean_y) / denom.replace(0, np.nan)
    intercept = mean_y - slope * mean_x
    
    r_denom = std_x * std_y
    r_val = (mean_xy - mean_x * mean_y) / r_denom.replace(0, np.nan)
    
    pred_range = intercept + slope * x
    dev = y - pred_range
    
    # Pine: slope <= 0 veya |r| < 0.5 ise dev=0
    bad_fit = (slope <= 0) | (r_val.abs() < 0.5)
    dev_filtered = dev.where(~bad_fit, 0.0)
    
    # Mum yapısı
    vsa_range = high - low
    body = (close - open_).abs()
    upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower_wick = pd.concat([open_, close], axis=1).min(axis=1) - low
    
    upper_wick_ratio = upper_wick / (vsa_range + 1e-10)
    lower_wick_ratio = lower_wick / (vsa_range + 1e-10)
    close_upper_half = ((close - low) / (vsa_range + 1e-10)) > 0.6
    
    is_bull = close > open_
    is_wide = norm_range > 1.5
    is_high_vol = norm_vol > 1.5
    
    signal_pos = dev_filtered > threshold
    signal_neg = dev_filtered < -threshold
    
    # Pine: vsa_BC, vsa_UT, vsa_SC, vsa_SPR, vsa_DT
    vsa_bc  = signal_pos & is_bull & is_wide & is_high_vol & ~close_upper_half
    vsa_ut  = signal_pos & (upper_wick_ratio > 0.35) & is_high_vol
    vsa_sc  = signal_neg & ~is_bull & is_wide & is_high_vol
    vsa_spr = signal_pos & (lower_wick_ratio > 0.35) & is_high_vol & is_bull
    vsa_dt  = signal_neg & (norm_range < 0.6) & is_high_vol
    
    return pd.DataFrame({
        "vsa_bc": vsa_bc,
        "vsa_ut": vsa_ut,
        "vsa_sc": vsa_sc,
        "vsa_spr": vsa_spr,
        "vsa_dt": vsa_dt,
        "dev_filtered": dev_filtered,
    })


# ══════════════════════════════════════════════════════════════════
# BÖLÜM 4: ADR VE DİNAMİK RİSK
# Pine Script: "ADR VE DİNAMİK RİSK (POSITION SIZING) MOTORU"
# ══════════════════════════════════════════════════════════════════

def calc_adr_stop(close: pd.Series,
                  adr_pct: pd.Series,
                  adr_mult: float = 1.5,
                  slip_pct: float = 0.1) -> pd.DataFrame:
    """
    ADR tabanlı dinamik stop mesafesi hesabı.
    Pine: safe_stop_dist_pct = max(adr_pct * adr_mult + slip_pct, 0.1)
    """
    safe_stop = (adr_pct * adr_mult + slip_pct).clip(lower=0.1)
    sl_long = close * (1 - safe_stop / 100)
    sl_short = close * (1 + safe_stop / 100)
    
    return pd.DataFrame({
        "safe_stop_pct": safe_stop,
        "sl_long": sl_long,
        "sl_short": sl_short,
    })


def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """OBV (On-Balance Volume) — ta.obv karşılığı"""
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    return obv.rename("obv")


# ══════════════════════════════════════════════════════════════════
# BÖLÜM 5: HEPSI BİR ARADA — ANA HESAPLAMA FONKSİYONU
# ══════════════════════════════════════════════════════════════════

def compute_all_indicators(df: pd.DataFrame,
                           adr_series: pd.Series = None,
                           preset: str = "Default") -> pd.DataFrame:
    """
    Tüm indikatörleri hesapla ve df'e ekle.
    adr_series: günlük ADR verisi (farklı TF'den gelir, reindex edilir)
    preset: "Scalping", "Default", "Aggressive", "Swing", "Conservative"
    """
    
    # Preset'e göre parametreler (Pine Script ile birebir)
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
    macd_df    = calc_macd(df["close"])
    out        = pd.concat([out, macd_df], axis=1)
    
    # ADX
    adx_df = calc_adx(df["high"], df["low"], df["close"])
    out    = pd.concat([out, adx_df], axis=1)
    
    # Bollinger
    bb_df = calc_bollinger(df["close"])
    out   = pd.concat([out, bb_df], axis=1)
    
    # Volume
    out["rvol"]  = calc_rvol(df["volume"])
    out["obv"]   = calc_obv(df["close"], df["volume"])
    out["vwap"]  = calc_vwap(df["high"], df["low"], df["close"], df["volume"])
    out["mfi"]   = calc_mfi(df["high"], df["low"], df["close"], df["volume"])
    
    # Whale motoru
    whale_df = calc_whale_score(
        df["high"], df["low"], df["open"], df["close"],
        df["volume"], out["rvol"], out["mfi"], out["obv"]
    )
    out = pd.concat([out, whale_df], axis=1)
    
    # VSA zırhı
    vsa_df = calc_vsa_shield(
        df["high"], df["low"], df["open"], df["close"], df["volume"]
    )
    out = pd.concat([out, vsa_df], axis=1)
    
    # ADR stop (günlük veri varsa kullan, yoksa basit ATR bazlı hesapla)
    if adr_series is not None:
        adr_reindexed = adr_series.reindex(out.index, method="ffill")
    else:
        # Fallback: ATR bazlı basit tahmin
        atr = calc_atr(df["high"], df["low"], df["close"], 14)
        adr_reindexed = (atr / df["close"] * 100).rolling(14).mean()
    
    adr_df = calc_adr_stop(df["close"], adr_reindexed)
    out    = pd.concat([out, adr_df], axis=1)
    out["adr_pct"] = adr_reindexed
    
    return out


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from data.fetcher import fetch_ohlcv
    
    df = fetch_ohlcv("BTC-USD", interval="4h", period="1y")
    result = compute_all_indicators(df, preset="Default")
    print(result[["close", "ema_fast", "ema_mid", "ema_slow",
                  "rsi", "adx", "rvol", "whale_score", "dev_filtered"]].tail(10))
    print(f"\nToplam sütun: {len(result.columns)}")
    print(f"NaN satır sayısı (son 100): {result.tail(100).isna().any(axis=1).sum()}")
