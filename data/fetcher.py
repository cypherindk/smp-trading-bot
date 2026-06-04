"""
data/fetcher.py
Veri çekme modülü — Pine Script'in request.security() eşdeğeri.
Hem günlük (1d) hem de intraday (1h, 4h vb.) verileri destekler.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta


def fetch_ohlcv(symbol: str, interval: str = "1h", period: str = "2y") -> pd.DataFrame:
    """
    OHLCV verisi çek. yfinance interval formatı:
      "1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo"
    
    Returns: DatetimeIndex ile OHLCV DataFrame
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(interval=interval, period=period, auto_adjust=True)
    
    if df.empty:
        raise ValueError(f"{symbol} için veri çekilemedi. Sembolü kontrol edin.")
    
    # Sütun isimlerini Pine Script ile uyumlu hale getir
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index.name = "Date"
    
    # NaN temizle
    df.dropna(inplace=True)
    
    print(f"✅ {symbol} [{interval}] — {len(df)} bar yüklendi. "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    return df


def fetch_daily_for_adr(symbol: str, adr_period: int = 14) -> pd.Series:
    """
    ADR (Average Daily Range) hesabı için günlük veri çek.
    Pine Script'teki request.security(..., "D", ...) karşılığı.
    
    Returns: Tarih indeksli ADR yüzdesi serisi
    """
    df = fetch_ohlcv(symbol, interval="1d", period="3mo")
    daily_range_pct = ((df["high"] - df["low"]) / df["low"].clip(lower=0.0001)) * 100
    adr = daily_range_pct.rolling(adr_period).mean()
    adr.name = "adr_pct"
    return adr


def resample_to_htf(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """
    Alt TF veriyi üst TF'e resample et.
    Pine Script'teki request.security() ile HTF bias hesabı için kullanılır.
    
    target_tf örnekleri: "4h", "1D", "1W"
    """
    rule_map = {
        "4h": "4h", "4H": "4h",
        "1D": "1D", "1d": "1D",
        "1W": "1W", "1w": "1W",
    }
    rule = rule_map.get(target_tf, target_tf)
    
    htf = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    return htf


if __name__ == "__main__":
    # Test
    df = fetch_ohlcv("BTC-USD", interval="4h", period="2y")
    print(df.tail())
