"""
data/crypto_fetcher.py
Binance public REST (klines) ile intraday kripto OHLCV cekici.

Neden bu modul?
  - yfinance 5m/15m veriyi sadece ~60 gun geriye veriyor -> intraday
    backtest (200+ islem, walk-forward) icin YETERSIZ.
  - Binance'in public veri aynasi (data-api.binance.vision) API key
    istemeden, rate-limit derdi olmadan YILLARCA 5m/15m/1h bar veriyor.
  - Ekstra bagimlilik YOK: sadece requests (zaten kurulu).

Cikti semasi, data/fetcher.py:fetch_ohlcv ile AYNI: DatetimeIndex (UTC)
+ lowercase [open, high, low, close, volume] float kolonlari. Yani
mevcut indikatör/backtest koduna drop-in uyumlu.

Onbellek: data/cache/{SYMBOL}_{interval}.csv — ikinci cagrida sadece
eksik kuyruk (son bardan bugune) cekilip eklenir (incremental).
"""

import os
import time
import requests
import pandas as pd

# Binance public veri aynasi (API key gerekmez, klines icin rate-limit yok).
# Erisilemezse ana API'ye dus.
_BASES = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
]

# interval -> milisaniye (pagination adimi icin)
_INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000,
    "4h": 14_400_000, "6h": 21_600_000, "12h": 43_200_000,
    "1d": 86_400_000,
}

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def to_binance_symbol(symbol: str) -> str:
    """
    'BTC-USD' -> 'BTCUSDT', 'ETH-USD' -> 'ETHUSDT', 'SOL/USDT' -> 'SOLUSDT'.
    Zaten Binance formatindaysa ('BTCUSDT') aynen birakir.
    """
    s = symbol.upper().replace("/", "").replace(" ", "")
    if s.endswith("-USD"):
        s = s[:-4] + "USDT"
    s = s.replace("-", "")
    if s.endswith("USD") and not s.endswith("USDT"):
        s = s + "T"  # ...USD -> ...USDT
    return s


def _klines_request(symbol: str, interval: str, start_ms: int,
                    end_ms: int, limit: int = 1000) -> list:
    """Tek bir klines cagrisi (max `limit` bar). Basari olan ilk base'i kullanir."""
    params = {
        "symbol": symbol, "interval": interval,
        "startTime": start_ms, "endTime": end_ms, "limit": limit,
    }
    last_err = None
    for base in _BASES:
        try:
            r = requests.get(base + "/api/v3/klines", params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
            last_err = f"HTTP {r.status_code}: {r.text[:120]}"
        except Exception as e:
            last_err = repr(e)[:120]
    raise RuntimeError(f"Binance klines cekilemedi ({symbol} {interval}): {last_err}")


def _raw_to_df(rows: list) -> pd.DataFrame:
    """Binance klines ham listesini OHLCV DataFrame'e cevir (UTC index)."""
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tbbav", "tbqav", "ignore",
    ])
    idx = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    out = df[["open", "high", "low", "close", "volume"]].astype(float)
    out.index = idx
    out.index.name = "Date"
    return out


def fetch_binance_ohlcv(symbol: str, interval: str = "15m",
                        days: int = 365, end: pd.Timestamp = None,
                        use_cache: bool = True, quiet: bool = False) -> pd.DataFrame:
    """
    `symbol` icin son `days` gunluk `interval` OHLCV verisini cek.

    symbol : 'BTC-USD' / 'BTCUSDT' / 'ETH-USD' ... (otomatik cevrilir)
    interval: '5m','15m','1h','4h','1d' ...
    days   : kac gun geriye (varsayilan 365)
    end    : bitis zamani (None -> simdi, UTC)
    use_cache: data/cache/*.csv incremental onbellek

    Returns: DatetimeIndex (UTC) + [open,high,low,close,volume] float
    """
    if interval not in _INTERVAL_MS:
        raise ValueError(f"Desteklenmeyen interval: {interval}")

    bsym = to_binance_symbol(symbol)
    step_ms = _INTERVAL_MS[interval]
    end_ts = pd.Timestamp.now(tz="UTC") if end is None else pd.Timestamp(end, tz="UTC")
    end_ms = int(end_ts.timestamp() * 1000)
    start_ms = end_ms - days * 86_400_000

    cache_path = os.path.join(_CACHE_DIR, f"{bsym}_{interval}.csv")
    cached = None
    if use_cache and os.path.exists(cache_path):
        try:
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            cached.index = pd.to_datetime(cached.index, utc=True)
            # Onbellekteki son bardan devam et (son bar guncelleniyor olabilir -> 1 bar geriden)
            last_ms = int(cached.index[-1].timestamp() * 1000)
            start_ms = max(start_ms, last_ms - step_ms)
        except Exception:
            cached = None

    rows_all = []
    cur = start_ms
    calls = 0
    try:
        while cur < end_ms:
            rows = _klines_request(bsym, interval, cur, end_ms, limit=1000)
            calls += 1
            if not rows:
                break
            rows_all.extend(rows)
            last_open = rows[-1][0]
            nxt = last_open + step_ms
            if nxt <= cur:      # ilerleme yoksa dur (sonsuz dongu korumasi)
                break
            cur = nxt
            if len(rows) < 1000:  # son sayfa
                break
            time.sleep(0.12)      # nazik ol
    except RuntimeError as e:
        # [FIX] Ag hatasinda cokme -> elde cache varsa onu kullan (dayaniklilik).
        if cached is not None and not cached.empty:
            if not quiet:
                print(f"[binance] {bsym} {interval}: ag hatasi, onbellek kullaniliyor ({repr(e)[:60]})")
        elif rows_all:
            if not quiet:
                print(f"[binance] {bsym} {interval}: ag kesildi, kismi veri kullaniliyor")
        else:
            raise

    fresh = _raw_to_df(rows_all)

    if cached is not None and not cached.empty:
        df = pd.concat([cached, fresh])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    else:
        df = fresh.sort_index()
        df = df[~df.index.duplicated(keep="last")]

    # [FIX] concat/cache okumasi object dtype uretebilir (bos fresh ile birlesince)
    # -> OHLCV'yi float'a zorla, aksi halde groupby.cumsum vs. patlar.
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])

    # Onbellegi guncelle
    if use_cache and not df.empty:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        df.to_csv(cache_path)

    # Istenen pencereye kirp
    win_start = end_ts - pd.Timedelta(days=days)
    df = df[df.index >= win_start]

    if not quiet and not df.empty:
        print(f"[binance] {bsym} [{interval}] {len(df)} bar "
              f"({df.index[0]} -> {df.index[-1]}) | {calls} API cagrisi")
    return df


if __name__ == "__main__":
    # Hizli dogrulama: BTC 15m son 120 gun
    df = fetch_binance_ohlcv("BTC-USD", interval="15m", days=120)
    print(df.tail(3))
    print("Toplam bar:", len(df))
    span = df.index[-1] - df.index[0]
    print("Kapsam:", span, "| beklenen ~", 120 * 96, "bar (15m gunde 96)")
