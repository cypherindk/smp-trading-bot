"""
engine/liquidity_mtf.py
Pine Script SMP V2.8.2 — "PhenLabs MTF Likidite Matrix" (Equal High/Low +
Sweep) esdegeri.

Pine mantigi (just_me.txt, satir ~540-670 ve ~833-843):
  1) 3 alt zaman diliminde (varsayilan 5dk / 15dk / 1sa) pivot high/low
     (6,6) tespit edilir.
  2) Ayni fiyat seviyesinde (%0.1 tolerans) tekrar eden pivotlar "equal
     level" olarak kayit altina alinir; hangi alt TF'lerde gorulduyse
     "confluence" puani artar (max 3).
  3) Ana grafik (4H) barlarinda bu seviyeler "sweep" edilirse (fitil
     seviyeyi gecip kapanis geri donerse) o yon icin bir sweep olayi
     tetiklenir ve seviye "harcanmis" sayilir.
  4) mtf_ok_bull / mtf_ok_bear   : son 50 barda o yonde sweep oldu mu
     (rawBuy/rawSell'i kapatan bir filtre).
  5) recent_demand_bounce / recent_supply_bounce : son 15 barda o yonde
     sweep oldu mu (Pine'da hem triggerBull/Bear'in HEM DE
     isAnaTrendBull/Bear'in icine giriyor, hem de bullScore/bearScore'a
     +1.0 puan katiyor).
  6) mtf_score_bull / mtf_score_bear : kapanis, harcanmamis bir seviyeye
     yakinsa (%0.2) confluence*0.5 puanlik bonus (esik/grade kontrolune
     bull_score/bear_score'un USTUNE eklenir).

NOT — bu yaklasik bir yeniden-uygulamadir:
  - Pine, request.security(..., lookahead_off) ile alt TF verisini
    "repaint etmeden" ana grafige tasir; burada da bir LTF pivotu ana
    grafikte ANCAK ilgili LTF barinin kapanis zamanindan SONRAKI ilk 4H
    barinda "biliniyor" sayiliyor (ayni mantik, farkli uygulama).
  - yfinance'in 5dk/15dk barlarda sagladigi gecmis veri sinirli
    (genelde ~60 gun) oldugu icin, cok eski donemler icin bu modul
    boyle veri yoksa devre disi kalir (mtf=None -> tum bayraklar
    varsayilan/acik davranisa duser, bkz. engine/signals.py).
"""

import numpy as np
import pandas as pd


def _pivot_series(high: pd.Series, low: pd.Series, left: int = 6, right: int = 6):
    """ta.pivothigh(high,6,6) / ta.pivotlow(low,6,6) esdegeri."""
    n = len(high)
    ph = pd.Series(np.nan, index=high.index)
    pl = pd.Series(np.nan, index=low.index)
    h = high.values
    l = low.values
    for i in range(left, n - right):
        wh = h[i - left:i + right + 1]
        if h[i] == wh.max() and np.sum(wh == h[i]) == 1:
            ph.iloc[i] = h[i]
        wl = l[i - left:i + right + 1]
        if l[i] == wl.min() and np.sum(wl == l[i]) == 1:
            pl.iloc[i] = l[i]
    return ph, pl


def _detect_equal_levels(high: pd.Series, low: pd.Series, left: int = 6, right: int = 6,
                         tol_pct: float = 0.1, buf_size: int = 4):
    """
    f_detectEqualLevels() esdegeri. Her yeni confirm olan pivotu, son
    buf_size pivotla karsilastirir; %tol_pct icinde esitse "equal level"
    olarak isaretler.

    Donen: [(confirm_bar_position, price), ...] listeleri (eqh, eql) —
    confirm_bar_position = pivot barinin index'i + right (Pine'da pivot
    ancak right bar sonra "biliniyor").
    """
    ph, pl = _pivot_series(high, low, left, right)
    n = len(ph)

    eqh_events, eql_events = [], []

    buf_h = []
    for pos in range(n):
        val = ph.iloc[pos]
        if not np.isnan(val):
            confirm_pos = min(pos + right, n - 1)
            for prev in buf_h:
                if prev and abs(val - prev) / val * 100 <= tol_pct:
                    eqh_events.append((confirm_pos, (val + prev) / 2))
                    break
            buf_h.append(val)
            if len(buf_h) > buf_size:
                buf_h.pop(0)

    buf_l = []
    for pos in range(n):
        val = pl.iloc[pos]
        if not np.isnan(val):
            confirm_pos = min(pos + right, n - 1)
            for prev in buf_l:
                if prev and abs(val - prev) / val * 100 <= tol_pct:
                    eql_events.append((confirm_pos, (val + prev) / 2))
                    break
            buf_l.append(val)
            if len(buf_l) > buf_size:
                buf_l.pop(0)

    return eqh_events, eql_events


def build_mtf_confirmation(df_4h: pd.DataFrame, ltf_map: dict,
                           level_buffer: int = 20,
                           sweep_lookback_bars: int = 50,
                           bounce_lookback_bars: int = 15,
                           level_tol_pct: float = 0.05,
                           score_tol_pct: float = 0.2) -> pd.DataFrame:
    """
    df_4h    : ana (4H) OHLC DataFrame (fetch_smart'tan gelen ham veri;
               "high","low","close" kolonlari olmali).
    ltf_map  : {"tf1": df_5m, "tf2": df_15m, "tf3": df_1h} — herhangi
               biri eksik/None olabilir, o TF sadece atlanir.

    Donen DataFrame, df_4h.index ile hizali, kolonlar:
      mtf_ok_bull, mtf_ok_bear, recent_demand_bounce, recent_supply_bounce,
      mtf_score_bull, mtf_score_bear
    """
    events = []  # (bar_pos, price, is_high, tf_idx)
    four_h_index = df_4h.index

    for tf_idx, key in ((1, "tf1"), (2, "tf2"), (3, "tf3")):
        ltf_df = ltf_map.get(key)
        if ltf_df is None or len(ltf_df) < (6 + 6 + 5):
            continue
        eqh_events, eql_events = _detect_equal_levels(ltf_df["high"], ltf_df["low"])
        for pos, price in eqh_events:
            ts = ltf_df.index[pos]
            bar_pos = four_h_index.searchsorted(ts, side="right") - 1
            if bar_pos >= 0:
                events.append((bar_pos, price, True, tf_idx))
        for pos, price in eql_events:
            ts = ltf_df.index[pos]
            bar_pos = four_h_index.searchsorted(ts, side="right") - 1
            if bar_pos >= 0:
                events.append((bar_pos, price, False, tf_idx))

    n = len(df_4h)
    high = df_4h["high"].values
    low = df_4h["low"].values
    close = df_4h["close"].values

    events_by_bar = {}
    for bar_pos, price, is_high, tf_idx in events:
        events_by_bar.setdefault(bar_pos, []).append((price, is_high, tf_idx))

    levels = []  # {"price","is_high","tfs":set,"confluence","swept"}

    def find_existing(price, is_high):
        for lvl in levels:
            if lvl["is_high"] == is_high and not lvl["swept"]:
                if abs(lvl["price"] - price) / price * 100 < level_tol_pct:
                    return lvl
        return None

    mtf_ok_bull = np.zeros(n, dtype=bool)
    mtf_ok_bear = np.zeros(n, dtype=bool)
    recent_demand_bounce = np.zeros(n, dtype=bool)
    recent_supply_bounce = np.zeros(n, dtype=bool)
    mtf_score_bull = np.zeros(n)
    mtf_score_bear = np.zeros(n)

    last_bull_sweep_bar = 0
    last_bear_sweep_bar = 0

    for i in range(n):
        for price, is_high, tf_idx in events_by_bar.get(i, []):
            existing = find_existing(price, is_high)
            if existing is not None:
                if tf_idx not in existing["tfs"]:
                    existing["tfs"].add(tf_idx)
                    existing["confluence"] += 1
            else:
                levels.append({
                    "price": price, "is_high": is_high,
                    "tfs": {tf_idx}, "confluence": 1, "swept": False,
                })
                if len(levels) > level_buffer:
                    levels.pop(0)

        any_sweep_bull = False
        any_sweep_bear = False
        h, l, c = high[i], low[i], close[i]
        h_prev = high[i - 1] if i > 0 else h
        l_prev = low[i - 1] if i > 0 else l

        for lvl in levels:
            if lvl["swept"]:
                continue
            price = lvl["price"]
            half = price * 0.05 / 100
            top = price + half
            bot = price - half
            if lvl["is_high"]:
                if (h > top and c < top) or (h_prev > top and c < price):
                    lvl["swept"] = True
                    any_sweep_bear = True
            else:
                if (l < bot and c > bot) or (l_prev < bot and c > price):
                    lvl["swept"] = True
                    any_sweep_bull = True

        if any_sweep_bull:
            last_bull_sweep_bar = i
        if any_sweep_bear:
            last_bear_sweep_bar = i

        mtf_ok_bull[i] = (i - last_bull_sweep_bar) <= sweep_lookback_bars
        mtf_ok_bear[i] = (i - last_bear_sweep_bar) <= sweep_lookback_bars
        recent_demand_bounce[i] = (i - last_bull_sweep_bar) <= bounce_lookback_bars
        recent_supply_bounce[i] = (i - last_bear_sweep_bar) <= bounce_lookback_bars

        sb, ss = 0.0, 0.0
        for lvl in levels:
            if lvl["swept"]:
                continue
            price = lvl["price"]
            if price and abs(c - price) / price < (score_tol_pct / 100):
                if not lvl["is_high"]:
                    sb = lvl["confluence"] * 0.5
                else:
                    ss = lvl["confluence"] * 0.5
        mtf_score_bull[i] = sb
        mtf_score_bear[i] = ss

    return pd.DataFrame({
        "mtf_ok_bull": mtf_ok_bull,
        "mtf_ok_bear": mtf_ok_bear,
        "recent_demand_bounce": recent_demand_bounce,
        "recent_supply_bounce": recent_supply_bounce,
        "mtf_score_bull": mtf_score_bull,
        "mtf_score_bear": mtf_score_bear,
    }, index=df_4h.index)


def fetch_mtf_frame(symbol: str, df_4h: pd.DataFrame, fetch_ohlcv_fn,
                    tf1: str = "5m", tf2: str = "15m", tf3: str = "1h",
                    period: str = "60d"):
    """
    3 alt zaman dilimini cek (basarisiz olani atla) ve build_mtf_confirmation
    ile birlestir. Hicbiri cekilemezse None doner -> cagiran taraf bunu
    "MTF filtresi devre disi" (tum bayraklar acik/varsayilan) olarak
    yorumlamali (bkz. engine/signals.py, mtf=None yolu).
    """
    import time
    ltf_map = {}
    for key, interval in (("tf1", tf1), ("tf2", tf2), ("tf3", tf3)):
        try:
            ltf_map[key] = fetch_ohlcv_fn(symbol, interval=interval, period=period)
        except Exception as e:
            print(f"    [mtf] {symbol} {interval} verisi cekilemedi: {e}")
        time.sleep(0.3)  # yfinance rate-limit riskini azalt

    if not ltf_map:
        return None
    try:
        return build_mtf_confirmation(df_4h, ltf_map)
    except Exception as e:
        print(f"    [mtf] {symbol} icin MTF hesaplanamadi: {e}")
        return None
