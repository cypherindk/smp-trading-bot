"""
telegram_bot.py
SMP V3.1 — Kripto (BTC/ETH/SOL) + BIST100 birlesik tarama, ikisi de 4H.

Degisiklikler (onceki versiyona gore):
  1) yfinance "4h" interval'i DESTEKLEMIYOR. Bu yuzden hem kripto hem BIST
     icin artik 1h veri cekilip 4h'e resample ediliyor (fetch_smart).
     Onceki kod interval="4h" ile direkt cagirdigi icin muhtemelen hep
     hata aliyor ve hicbir sinyal Telegram'a gitmiyordu.
  2) BIST100 hisseleri ayni taramaya eklendi — kullanicinin TradingView
     grafiginde SMP indikatorunu 4 saatlik (4sa) zaman diliminde
     kullandigi teyit edildi, bu yuzden gunluk (1d) degil 4h kullanildi.
  3) State/dedup dosyasina artik gerek yok: kripto ile ayni mantik
     (son 2 mumda sinyal var mi kontrolu) BIST icin de yeterli.
"""

import sys
import os
import time
import requests
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.fetcher import fetch_ohlcv
from engine.indicators import compute_all_indicators
from engine.signals import calc_bull_bear_score, calc_triggers, generate_signals
from engine.filters import apply_all_filters
from bist100_tickers import BIST100_YF

# ── Telegram ayarlari ──
# ⚠️ Token'i GitHub Secrets'tan okuyun, koda hardcode ETMEYIN.
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

if not BOT_TOKEN or not CHAT_ID:
    print("UYARI: BOT_TOKEN veya CHAT_ID environment variable eksik.")

# ── Kripto parametreleri (mevcut, degismedi) ──
COINS = {
    "BTC-USD": {
        "preset": "Default", "eff_score": 5.0, "min_conf": 2,
        "adr_mult": 1.5, "rr_ratio": 2.0, "priority": 1,
        "symbol": "BTC/USDT",
    },
    "ETH-USD": {
        "preset": "Conservative", "eff_score": 4.5, "min_conf": 1,
        "adr_mult": 2.8, "rr_ratio": 3.5, "priority": 2,
        "symbol": "ETH/USDT",
    },
    "SOL-USD": {
        "preset": "Default", "eff_score": 7.0, "min_conf": 1,
        "adr_mult": 1.9, "rr_ratio": 1.8, "priority": 3,
        "symbol": "SOL/USDT",
    },
}

# ── BIST icin ortak parametreler (hisse bazli ayri optimize edilmedi) ──
BIST_PRESET = "Default"
BIST_EFF_SCORE = 5.0
BIST_MIN_CONF = 2
BIST_RR_RATIO = 2.0
BIST_REQUEST_DELAY = 0.6  # ardisik yfinance istekleri arasi bekleme
WHALE_SCORE_THRESHOLD = 80  # sadece bu skor ve ustundeki whale alert'ler gonderilir


# ───────────────────────── Veri cekme yardimcisi ─────────────────────────

def fetch_smart(symbol: str, timeframe: str, period: str, resample_offset: str = None):
    """
    yfinance "4h" interval'ini desteklemez. "4h" istenirse 1h veri cekip
    4h'e resample eder. Diger interval'ler (orn. "1d") direkt gecer.

    resample_offset: BIST icin "2h" verilir -- boylece 4h barlar varsayilan
    00:00/04:00/08:00... yerine 10:00-14:00 / 14:00-18:00 seans saatlerine
    hizalanir (BIST seansi 10:00-18:00 TR). Bu olmadan barlarin basi/sonu
    yarim seans veri iceriyordu, bu da TradingView'deki gercek 4h mumlardan
    farkli kesisim/sinyal tarihlerine yol aciyordu.
    """
    if timeframe in ("4h", "4H"):
        raw = fetch_ohlcv(symbol, interval="1h", period=period)
        resample_kwargs = {"offset": resample_offset} if resample_offset else {}
        df = raw.resample("4h", **resample_kwargs).agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
        # Hacmi 0 olan mumlar seans disi/hayalet barlardir (orn. BIST'te
        # 06:00-10:00 gibi islem olmayan bir saat araligindan gelen
        # yaniltici tek-tik veri) -- bunlari cikariyoruz.
        df = df[df["volume"] > 0]
        df.index.name = "Date"
        return df
    return fetch_ohlcv(symbol, interval=timeframe, period=period)


# ───────────────────────── Telegram ─────────────────────────

def send_message(text: str) -> bool:
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram gonderilemedi: BOT_TOKEN/CHAT_ID tanimli degil.")
        return False
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"Telegram hata: {e}")
        return False


def calc_grade(score: float) -> str:
    if score >= 8.0:
        return "A+"
    elif score >= 6.5:
        return "A"
    elif score >= 5.0:
        return "B"
    return "C"


def format_signal_message(opp: dict) -> str:
    direction = opp["direction"]
    grade = opp["grade"]
    score = opp["score"]
    price = opp["price"]
    sl = opp["sl"]
    tp1 = opp["tp1"]
    tp2 = opp["tp2"]
    stop_pct = opp["stop_pct"]
    tp1_pct = opp["tp1_pct"]
    tp2_pct = opp["tp2_pct"]
    rvol = opp["rvol"]
    rsi = opp["rsi"]
    rr = opp["rr"]
    label = opp["label"]
    currency = opp["currency"]

    grade_emoji = {"A+": "🏆", "A": "⭐", "B": "👍", "C": "⚠️"}.get(grade, "")
    dir_emoji = "📈 LONG" if direction == "LONG" else "📉 SHORT"
    asset_tag = "🪙 KRİPTO" if opp["asset_type"] == "crypto" else "🇹🇷 BIST"

    msg = f"""🚨 <b>SMP SİNYAL</b> — {asset_tag} 🚨
━━━━━━━━━━━━━━━━━━━━━
<b>{label}</b>  {dir_emoji}
Grade: {grade_emoji} <b>{grade}</b>  |  Skor: {score:.1f}/10

💰 <b>GİRİŞ:</b>  {price:,.4f} {currency}

🔴 <b>STOP LOSS:</b>  {sl:,.4f} {currency}
   (-%{stop_pct:.2f})

🟢 <b>TP1:</b>  {tp1:,.4f} {currency}
   (+%{tp1_pct:.2f})"""

    if tp2:
        msg += f"""

🟢 <b>TP2:</b>  {tp2:,.4f} {currency}
   (+%{tp2_pct:.2f})"""

    msg += f"""

━━━━━━━━━━━━━━━━━━━━━
⚖️ R/R Orani:  1:{rr:.1f}
📊 RVOL: {rvol:.2f}x  |  RSI: {rsi:.0f}
⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}
━━━━━━━━━━━━━━━━━━━━━
⚠️ Bu bir sinyal sistemidir.
Kendi analizinizi de yapin."""

    return msg.strip()


def format_whale_message(w: dict) -> str:
    side = "🟢 ALIM" if w["side"] == "BUY" else "🔴 SATIŞ"
    return f"""🐋 <b>WHALE ALERT</b> — {side}
━━━━━━━━━━━━━━━━━━━━━
<b>{w['label']}</b>
Whale Skoru: {w['whale_score']:.0f}%
İşlem Hacmi: {w['dv_m']:.2f}M {w['currency']}
Fiyat: {w['price']:,.4f} {w['currency']}
⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}
━━━━━━━━━━━━━━━━━━━━━
Bu, ana SMP sinyalinden bağımsız bir para akışı uyarısıdır.
Sadece takip listesi amaçlıdır."""


# ───────────────────────── Tarama mantigi ─────────────────────────

def scan_crypto():
    opportunities = []
    whale_events = []
    for coin, p in COINS.items():
        try:
            df = fetch_smart(coin, "4h", "60d")
            if len(df) < 60:
                print(f"  {coin}: yetersiz veri ({len(df)} bar)")
                continue

            ind = compute_all_indicators(df, preset=p["preset"])
            sc = calc_bull_bear_score(ind)
            tr = calc_triggers(ind, sc)
            sg = generate_signals(ind, sc, tr, preset=p["preset"],
                                   eff_score=p["eff_score"], min_conf=p["min_conf"])
            fs = apply_all_filters(ind, sg, use_cvd=True)

            # Whale alert (ana sinyalden bagimsiz, sadece son bar kontrol edilir,
            # sadece esik ustundeki gercekten hacimli olanlar gonderilir)
            whale_score_now = ind["whale_score"].iloc[-1]
            if (ind["is_whale_buy"].iloc[-1] or ind["is_whale_sell"].iloc[-1]) and whale_score_now >= WHALE_SCORE_THRESHOLD:
                whale_events.append({
                    "label": p["symbol"], "currency": "USD",
                    "side": "BUY" if ind["is_whale_buy"].iloc[-1] else "SELL",
                    "whale_score": whale_score_now,
                    "dv_m": ind["dv_m"].iloc[-1],
                    "price": df["close"].iloc[-1],
                })

            recent_buy = fs["buy_signal"].iloc[-2:].any()
            recent_sell = fs["sell_signal"].iloc[-2:].any()
            if not recent_buy and not recent_sell:
                print(f"  {coin}: sinyal yok")
                continue

            bull_score = sc["bull_score"].iloc[-1]
            bear_score = sc["bear_score"].iloc[-1]
            price = df["close"].iloc[-1]
            stop_pct = ind["safe_stop_pct"].iloc[-1]

            direction = "LONG" if (recent_buy and bull_score >= bear_score) else "SHORT"
            score = bull_score if direction == "LONG" else bear_score

            tp1_pct = stop_pct * 1.0
            tp2_pct = stop_pct * p["rr_ratio"]
            if direction == "LONG":
                sl = price * (1 - stop_pct / 100)
                tp1 = price * (1 + tp1_pct / 100)
                tp2 = price * (1 + tp2_pct / 100) if p["rr_ratio"] > 1.5 else None
            else:
                sl = price * (1 + stop_pct / 100)
                tp1 = price * (1 - tp1_pct / 100)
                tp2 = price * (1 - tp2_pct / 100) if p["rr_ratio"] > 1.5 else None

            opportunities.append({
                "asset_type": "crypto", "label": p["symbol"], "currency": "USD",
                "direction": direction, "score": score, "grade": calc_grade(score),
                "rvol": ind["rvol"].iloc[-1], "rsi": ind["rsi"].iloc[-1],
                "price": price, "sl": sl, "tp1": tp1, "tp2": tp2,
                "stop_pct": stop_pct, "tp1_pct": tp1_pct, "tp2_pct": tp2_pct,
                "rr": tp2_pct / stop_pct if stop_pct > 0 else 0,
                "priority": p["priority"], "dedup_key": None,
            })
            print(f"  {coin}: {direction} sinyali (skor={score:.1f})")

        except Exception as e:
            print(f"  {coin}: hata — {e}")

    return opportunities, whale_events


def scan_bist():
    opportunities = []
    whale_events = []
    for ticker in BIST100_YF:
        try:
            df = fetch_smart(ticker, "4h", "60d", resample_offset="2h")
            if len(df) < 60:
                print(f"  {ticker}: yetersiz veri ({len(df)} bar)")
                continue

            ind = compute_all_indicators(df, preset=BIST_PRESET)
            sc = calc_bull_bear_score(ind)
            tr = calc_triggers(ind, sc)
            sg = generate_signals(ind, sc, tr, preset=BIST_PRESET,
                                   eff_score=BIST_EFF_SCORE, min_conf=BIST_MIN_CONF)
            fs = apply_all_filters(ind, sg, use_cvd=True)

            label = ticker.replace(".IS", "")

            # Whale alert (ana sinyalden bagimsiz, sadece son bar kontrol edilir,
            # sadece esik ustundeki gercekten hacimli olanlar gonderilir)
            whale_score_now = ind["whale_score"].iloc[-1]
            if (ind["is_whale_buy"].iloc[-1] or ind["is_whale_sell"].iloc[-1]) and whale_score_now >= WHALE_SCORE_THRESHOLD:
                whale_events.append({
                    "label": label, "currency": "TRY",
                    "side": "BUY" if ind["is_whale_buy"].iloc[-1] else "SELL",
                    "whale_score": whale_score_now,
                    "dv_m": ind["dv_m"].iloc[-1],
                    "price": df["close"].iloc[-1],
                })

            recent_buy = fs["buy_signal"].iloc[-2:].any()
            recent_sell = fs["sell_signal"].iloc[-2:].any()
            if not recent_buy and not recent_sell:
                continue

            bull_score = sc["bull_score"].iloc[-1]
            bear_score = sc["bear_score"].iloc[-1]
            price = df["close"].iloc[-1]
            stop_pct = ind["safe_stop_pct"].iloc[-1]

            direction = "LONG" if (recent_buy and bull_score >= bear_score) else "SHORT"

            score = bull_score if direction == "LONG" else bear_score
            tp1_pct = stop_pct * 1.0
            tp2_pct = stop_pct * BIST_RR_RATIO
            if direction == "LONG":
                sl = price * (1 - stop_pct / 100)
                tp1 = price * (1 + tp1_pct / 100)
                tp2 = price * (1 + tp2_pct / 100) if BIST_RR_RATIO > 1.5 else None
            else:
                sl = price * (1 + stop_pct / 100)
                tp1 = price * (1 - tp1_pct / 100)
                tp2 = price * (1 - tp2_pct / 100) if BIST_RR_RATIO > 1.5 else None

            opportunities.append({
                "asset_type": "bist", "label": label, "currency": "TRY",
                "direction": direction, "score": score, "grade": calc_grade(score),
                "rvol": ind["rvol"].iloc[-1], "rsi": ind["rsi"].iloc[-1],
                "price": price, "sl": sl, "tp1": tp1, "tp2": tp2,
                "stop_pct": stop_pct, "tp1_pct": tp1_pct, "tp2_pct": tp2_pct,
                "rr": tp2_pct / stop_pct if stop_pct > 0 else 0,
                "priority": 9, "dedup_key": None,
            })
            print(f"  {ticker}: {direction} sinyali (skor={score:.1f})")

        except Exception as e:
            print(f"  {ticker}: hata — {e}")

        time.sleep(BIST_REQUEST_DELAY)

    return opportunities, whale_events


def scan_and_notify():
    print(f"\n[{datetime.now().strftime('%H:%M')}] SMP Tarama basladi "
          f"(3 kripto + {len(BIST100_YF)} BIST hissesi)...")

    crypto_opps, crypto_whales = scan_crypto()
    bist_opps, bist_whales = scan_bist()

    opportunities = crypto_opps + bist_opps
    whale_events = crypto_whales + bist_whales

    if opportunities:
        opportunities.sort(key=lambda x: (x["priority"], -x["score"]))
        for opp in opportunities:
            msg = format_signal_message(opp)
            ok = send_message(msg)
            print(f"  {opp['label']} mesaji {'gonderildi' if ok else 'gonderilemedi'}")
            time.sleep(1)
    else:
        print("  Aktif sinyal yok.")

    if whale_events:
        whale_events.sort(key=lambda x: -x["whale_score"])
        for w in whale_events:
            msg = format_whale_message(w)
            ok = send_message(msg)
            print(f"  {w['label']} whale alert {'gonderildi' if ok else 'gonderilemedi'}")
            time.sleep(1)
    else:
        print("  Whale alert yok.")

    return len(opportunities) + len(whale_events)


def send_test_message():
    msg = f"""🤖 <b>SMP Bot Aktif!</b>

Kripto: BTC/USDT, ETH/USDT, SOL/USDT (4H)
BIST100: {len(BIST100_YF)} hisse (4H)

Baslangic: {datetime.now().strftime('%d.%m.%Y %H:%M')}"""
    ok = send_message(msg)
    print("Test mesaji gonderildi." if ok else "Mesaj gonderilemedi.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "scan"
    if mode == "test":
        send_test_message()
    elif mode == "scan":
        scan_and_notify()
    else:
        print("Kullanim: python telegram_bot.py [test|scan]")