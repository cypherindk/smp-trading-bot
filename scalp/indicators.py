"""
scalp/indicators.py
SCALP PRO V1.0 — Kantitatif Scalp Indikatörleri
Gemini Raporu + Claude Python Portu

Modüller:
  1. PAC Pullback & Recovery Engine
  2. ATR Gövde Patlamasi (Body Expansion)
  3. ERA Hizli Momentum Tetigi
  4. TST Flow Momentum (Trend Filtresi)
  5. RVOL (Hacim Onayi)
  6. Risk Yonetimi (ATR Stop/TP)
"""

import pandas as pd
import numpy as np


def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def calc_atr(high, low, close, period=14):
    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calc_rvol(volume, length=20):
    avg_vol = volume.rolling(length).mean()
    return volume / (avg_vol + 0.0001)


def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast   = calc_ema(close, fast)
    ema_slow   = calc_ema(close, slow)
    macd_line  = ema_fast - ema_slow
    macd_sig   = calc_ema(macd_line, signal)
    macd_hist  = macd_line - macd_sig
    return pd.DataFrame({
        "macd_line":   macd_line,
        "macd_signal": macd_sig,
        "macd_hist":   macd_hist,
    })


# ══════════════════════════════════════════════════════════════════
# MODÜL 1: PAC PULLBACK & RECOVERY ENGINE
# Kaynak: Scalping PullBack Tool R1.1
# Fiyat dinamik kanalin altina çekildikten sonra
# tekrar üstüne kapanirsa = long tetigi
# ══════════════════════════════════════════════════════════════════

def calc_pac_trigger(high, low, close, open_,
                     tf="15m", length=None):
    """
    PAC (Price Action Channel) Pullback & Recovery

    Pine:
      pacH = ema(high, 34)
      pacC = ema(close, 34)
      pulledBackLong  = barssince(close < pacC) <= 3
      triggerPAC = pulledBackLong and (open < pacH and close > pacH)
    """
    # TF'e gore PAC length
    if length is None:
        length_map = {"5m": 21, "15m": 34, "1h": 55}
        length = length_map.get(tf, 34)

    pac_h = calc_ema(high,  length)
    pac_c = calc_ema(close, length)

    # Son 4 barda close < pac_c oldu mu?
    below_pac   = close < pac_c
    pulled_back = below_pac.rolling(4).max().astype(bool)

    # Recovery: open < pac_h AND close > pac_h
    recovery = (open_ < pac_h) & (close > pac_h)

    trigger_pac = pulled_back & recovery

    return pd.DataFrame({
        "pac_h":       pac_h,
        "pac_c":       pac_c,
        "pulled_back": pulled_back,
        "trigger_pac": trigger_pac,
    })


# ══════════════════════════════════════════════════════════════════
# MODÜL 2: ATR GÖVDE PATLAMASI (Body Expansion)
# Kaynak: Super Scalper 5Min 15Min
# Mumun govdesi ATR'yi asarsa = hacimli kopus
# ══════════════════════════════════════════════════════════════════

def calc_expansion_trigger(high, low, close, open_,
                           atr_period=14, atr_mult=1.0):
    """
    ATR Body Expansion Trigger

    Pine:
      atr = ta.atr(14)
      bodyExpansion = (close - open) > (atr * 1.0)
    """
    atr  = calc_atr(high, low, close, atr_period)
    body = close - open_  # Pozitif = bullish, negatif = bearish

    trigger_bull = body > (atr * atr_mult)   # Guclu bullish mum
    trigger_bear = body < -(atr * atr_mult)  # Guclu bearish mum

    return pd.DataFrame({
        "atr":                 atr,
        "body":                body,
        "trigger_expansion_bull": trigger_bull,
        "trigger_expansion_bear": trigger_bear,
    })


# ══════════════════════════════════════════════════════════════════
# MODÜL 3: ERA HIZLI MOMENTUM TETİĞİ
# Kaynak: ERA Scalper
# RSI(3) asiri satim bölgesinden çikis + fiyat onayı
# ══════════════════════════════════════════════════════════════════

def calc_era_trigger(high, low, close,
                     rsi_period=3, oversold=20, overbought=80):
    """
    ERA (EMA RSI ADX) Hizli Momentum Tetigi

    Pine:
      rsi3 = ta.rsi(close, 3)
      eraTriggerLong  = (rsi3[1] <= 20 or rsi3[2] <= 20) and close > high[1]
      eraTriggerShort = (rsi3[1] >= 80 or rsi3[2] >= 80) and close < low[1]
    """
    rsi3 = calc_rsi(close, rsi_period)

    # Long: Son 2 barda RSI asiri satim + simdi yuksek kirdi
    era_long  = ((rsi3.shift(1) <= oversold) | (rsi3.shift(2) <= oversold)) & \
                (close > high.shift(1))

    # Short: Son 2 barda RSI asiri alim + simdi dusuk kirdi
    era_short = ((rsi3.shift(1) >= overbought) | (rsi3.shift(2) >= overbought)) & \
                (close < low.shift(1))

    return pd.DataFrame({
        "rsi3":      rsi3,
        "era_long":  era_long,
        "era_short": era_short,
    })


# ══════════════════════════════════════════════════════════════════
# MODÜL 4: RISK YÖNETİMİ
# ATR bazli Stop Loss ve TP hedefleri
# ══════════════════════════════════════════════════════════════════

def calc_scalp_risk(high, low, close,
                    tf="15m",
                    sl_mult=None, tp1_mult=None, tp2_mult=None):
    """
    TF'e gore ATR bazli Stop/TP hesabi

    Parametre tablosu (Gemini raporu):
      5M:  sl=2.0x, tp1=2.0x, tp2=4.0x
      15M: sl=1.5x, tp1=1.5x, tp2=3.0x
      1H:  sl=1.2x, tp1=1.2x, tp2=2.4x
    """
    mult_map = {
        "5m":  {"sl": 2.0, "tp1": 2.0, "tp2": 4.0},
        "15m": {"sl": 1.5, "tp1": 1.5, "tp2": 3.0},
        "1h":  {"sl": 1.2, "tp1": 1.2, "tp2": 2.4},
    }
    m = mult_map.get(tf, mult_map["15m"])

    sl_m  = sl_mult  or m["sl"]
    tp1_m = tp1_mult or m["tp1"]
    tp2_m = tp2_mult or m["tp2"]

    atr = calc_atr(high, low, close, 14)

    return pd.DataFrame({
        "scalp_atr":       atr,
        "sl_long":         close - atr * sl_m,
        "tp1_long":        close + atr * tp1_m,
        "tp2_long":        close + atr * tp2_m,
        "sl_short":        close + atr * sl_m,
        "tp1_short":       close - atr * tp1_m,
        "tp2_short":       close - atr * tp2_m,
        "sl_pct":          (atr * sl_m / close * 100),
        "tp1_pct":         (atr * tp1_m / close * 100),
        "tp2_pct":         (atr * tp2_m / close * 100),
    })


# ══════════════════════════════════════════════════════════════════
# ANA FONKSİYON: Tüm scalp indikatörlerini hesapla
# ══════════════════════════════════════════════════════════════════

def compute_scalp_indicators(df, tf="15m"):
    """
    Tüm scalp indikatörlerini hesapla ve birlestir.

    df: OHLCV DataFrame
    tf: "5m", "15m", "1h"
    """
    out = df.copy()

    # Temel filtreler
    out["ema200"] = calc_ema(df["close"], 200)
    out["ema50"]  = calc_ema(df["close"], 50)
    out["rvol"]   = calc_rvol(df["volume"])

    macd_df = calc_macd(df["close"])
    out = pd.concat([out, macd_df], axis=1)

    out["rsi14"] = calc_rsi(df["close"], 14)
    out["atr14"] = calc_atr(df["high"], df["low"], df["close"], 14)

    # PAC Pullback
    pac_df = calc_pac_trigger(df["high"], df["low"], df["close"], df["open"], tf=tf)
    out = pd.concat([out, pac_df], axis=1)

    # ATR Expansion
    exp_df = calc_expansion_trigger(df["high"], df["low"], df["close"], df["open"])
    out = pd.concat([out, exp_df], axis=1)

    # ERA Trigger
    era_df = calc_era_trigger(df["high"], df["low"], df["close"])
    out = pd.concat([out, era_df], axis=1)

    # Risk yonetimi
    risk_df = calc_scalp_risk(df["high"], df["low"], df["close"], tf=tf)
    out = pd.concat([out, risk_df], axis=1)

    return out


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from data.fetcher import fetch_ohlcv

    df  = fetch_ohlcv("BTC-USD", interval="15m", period="7d")
    ind = compute_scalp_indicators(df, tf="15m")

    print(ind[["close", "rvol", "trigger_pac",
               "trigger_expansion_bull", "era_long",
               "sl_long", "tp1_long", "tp2_long"]].tail(5))
    print(f"\nToplam sutun: {len(ind.columns)}")