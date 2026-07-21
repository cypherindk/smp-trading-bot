"""
diagnose_history.py
SMP motorunun belirli hisseler icin GECMISTE hangi tarihlerde LONG/SHORT
sinyali uretmis oldugunu listeler. Telegram'a HICBIR SEY GONDERMEZ.

Not: Artik 4H zaman diliminde calisiyor (TradingView grafiginizle ayni
zaman dilimi), gunluk degil. yfinance "4h" desteklemedigi icin 1h veri
cekilip 4h'e resample ediliyor (telegram_bot.py'deki fetch_smart).

[FIX] PRESET/EFF_SCORE/GRADE_FILTER, telegram_bot.py ile ayni sekilde
TradingView BIST grafiginin gercek ayarlarina (Preset=Auto -> 4H'de
"Aggressive", Grade Filtresi=A+ Only) eslendi. Eskiden "Default" +
eff_score=5.0 + grade_filter=All kullaniliyordu, bu da bu script'in
TradingView'de hic gorunmeyen sinyalleri "sinyal var" diye raporlamasina
sebep oluyordu.

Kullanim:
  python diagnose_history.py
  python diagnose_history.py AKBNK.IS ISCTR.IS GARAN.IS
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from telegram_bot import fetch_smart
from data.fetcher import fetch_ohlcv
from engine.indicators import compute_all_indicators
from engine.signals import calc_bull_bear_score, calc_triggers, generate_signals
from engine.filters import apply_all_filters
from engine.liquidity_mtf import fetch_mtf_frame
from engine.smart_money_flow import calc_smart_money_droplet

USE_MTF = True  # telegram_bot.py'deki BIST_USE_MTF ile ayni mantik

PRESET = "Aggressive"
EFF_SCORE = 3.0
GRADE_FILTER = "A+ Only"
MIN_CONF = 2


def diagnose(ticker: str):
    print(f"\n{'='*60}")
    print(f"  {ticker}  [4H]")
    print(f"{'='*60}")

    # telegram_bot.py ile ayni periyot ("60d" -- hiz nedeniyle 1y'dan
    # geri donuldu, bkz. scan_bist icindeki NOT).
    df = fetch_smart(ticker, "4h", "60d", resample_offset="2h")
    from telegram_bot import resolve_symbol_config
    cfg = resolve_symbol_config(ticker, {
        "preset": PRESET, "eff_score": EFF_SCORE,
        "min_conf": MIN_CONF, "grade_filter": GRADE_FILTER,
    })
    ind = compute_all_indicators(df, preset=cfg["preset"], timeframe_minutes=240)
    mtf_frame = fetch_mtf_frame(ticker, df, fetch_ohlcv) if USE_MTF else None
    sc = calc_bull_bear_score(ind, mtf=mtf_frame)
    tr = calc_triggers(ind, sc)
    sg = generate_signals(ind, sc, tr, preset=cfg["preset"],
                           eff_score=cfg["eff_score"], min_conf=cfg["min_conf"],
                           grade_filter=cfg["grade_filter"])
    fs = apply_all_filters(ind, sg, use_cvd=True)

    buy_dates = df.index[fs["buy_signal"]].tolist()
    sell_dates = df.index[fs["sell_signal"]].tolist()

    # HAM tetikleyiciler (filtre/skor esigi uygulanmadan once) -- bunlar
    # bull_score/bear_score veya apply_all_filters tarafindan elenmis
    # olabilir. Buy/Sell sinyali olmayip trigger_bull/bear=True olan
    # gunler, "motor tetigi gordu ama filtreledi" demektir.
    raw_trigger_bull_dates = df.index[tr["trigger_bull"]].tolist()
    raw_trigger_bear_dates = df.index[tr["trigger_bear"]].tolist()
    print(f"\n  HAM trigger_bull tarihleri (filtre oncesi): "
          f"{[d.strftime('%Y-%m-%d %H:%M') for d in raw_trigger_bull_dates]}")
    print(f"  HAM trigger_bear tarihleri (filtre oncesi): "
          f"{[d.strftime('%Y-%m-%d %H:%M') for d in raw_trigger_bear_dates]}")

    print(f"  Toplam bar: {len(df)}  |  Tarih araligi: "
          f"{df.index[0].strftime('%Y-%m-%d %H:%M')} -> "
          f"{df.index[-1].strftime('%Y-%m-%d %H:%M')}")
    print(f"  BUY sinyali sayisi:  {len(buy_dates)}")
    print(f"  SELL sinyali sayisi: {len(sell_dates)}")

    print("\n  --- BUY sinyalleri (tarih, kapanis fiyati, bull_score) ---")
    for d in buy_dates:
        price = df.loc[d, "close"]
        bull = sc.loc[d, "bull_score"]
        print(f"    {d.strftime('%Y-%m-%d %H:%M')}  |  {price:,.2f} TRY  |  bull_score={bull:.2f}")

    print("\n  --- SELL sinyalleri (tarih, kapanis fiyati, bear_score) ---")
    for d in sell_dates:
        price = df.loc[d, "close"]
        bear = sc.loc[d, "bear_score"]
        print(f"    {d.strftime('%Y-%m-%d %H:%M')}  |  {price:,.2f} TRY  |  bear_score={bear:.2f}")

    # Son 30 gunun gunluk skor/RSI/RVOL detayini da goster
    # (sinyal olmasa bile "neden esigi gecemedi" sorusuna isik tutar)
    droplet = calc_smart_money_droplet(ind)
    print("\n  --- Son 20 mumun (4H) detayi ---")
    print(f"  {'Tarih/Saat':<17}{'Kapanis':>10}{'Bull':>7}{'Bear':>7}"
          f"{'RSI':>7}{'RVOL':>7}{'TrigB':>7}{'TrigS':>7}{'BUY':>6}{'SELL':>6}{'DAMLA':>7}")
    tail_n = 20
    for d in df.index[-tail_n:]:
        print(f"  {d.strftime('%Y-%m-%d %H:%M'):<17}{df.loc[d,'close']:>10.2f}"
              f"{sc.loc[d,'bull_score']:>7.2f}{sc.loc[d,'bear_score']:>7.2f}"
              f"{ind.loc[d,'rsi']:>7.1f}{ind.loc[d,'rvol']:>7.2f}"
              f"{str(bool(tr.loc[d,'trigger_bull'])):>7}"
              f"{str(bool(tr.loc[d,'trigger_bear'])):>7}"
              f"{str(bool(fs.loc[d,'buy_signal'])):>6}"
              f"{str(bool(fs.loc[d,'sell_signal'])):>6}"
              f"{str(bool(droplet.loc[d,'smart_money_droplet'])):>7}")


if __name__ == "__main__":
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["AKBNK.IS", "ISCTR.IS"]
    for t in tickers:
        try:
            diagnose(t)
        except Exception as e:
            print(f"{t}: hata — {e}")
