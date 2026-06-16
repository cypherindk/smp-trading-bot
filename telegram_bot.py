"""
telegram_bot.py
SMP V3.1 FINAL — VPA Climax + Zone POC bonus dahil
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

BOT_TOKEN = os.environ.get("BOT_TOKEN", "8220091488:AAGAZDtlqTr3MVW66RjEEdDDXdF0O1ZAZTI")
CHAT_ID   = os.environ.get("CHAT_ID", "959954532")

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


def send_message(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"Telegram hata: {e}")
        return False


def calc_grade(score):
    if score >= 8.0:
        return "A+"
    elif score >= 6.5:
        return "A"
    elif score >= 5.0:
        return "B"
    return "C"


def format_signal_message(opp):
    direction = opp["direction"]
    grade     = opp["grade"]
    score     = opp["score"]
    price     = opp["price"]
    sl        = opp["sl"]
    tp1       = opp["tp1"]
    tp2       = opp["tp2"]
    stop_pct  = opp["stop_pct"]
    tp1_pct   = opp["tp1_pct"]
    tp2_pct   = opp["tp2_pct"]
    rvol      = opp["rvol"]
    rsi       = opp["rsi"]
    rr        = opp["rr"]
    symbol    = COINS[opp["coin"]]["symbol"]

    grade_emoji = {"A+": "🏆", "A": "⭐", "B": "👍", "C": "⚠️"}.get(grade, "")
    dir_emoji   = "📈 LONG" if direction == "LONG" else "📉 SHORT"

    msg = f"""🚨 <b>SMP V3.1 SİNYAL</b> 🚨
━━━━━━━━━━━━━━━━━━━━━
<b>{symbol}</b>  {dir_emoji}
Grade: {grade_emoji} <b>{grade}</b>  |  Skor: {score:.1f}/10

💰 <b>GİRİŞ:</b>  ${price:,.4f}

🔴 <b>STOP LOSS:</b>  ${sl:,.4f}
   (-%{stop_pct:.2f})

🟢 <b>TP1:</b>  ${tp1:,.4f}
   (+%{tp1_pct:.2f})"""

    if tp2:
        msg += f"""

🟢 <b>TP2:</b>  ${tp2:,.4f}
   (+%{tp2_pct:.2f})"""

    msg += f"""

📊 <b>TP3:</b>  Trend takibi (trailing stop)

━━━━━━━━━━━━━━━━━━━━━
⚖️ R/R Orani:  1:{rr:.1f}
📊 RVOL: {rvol:.2f}x  |  RSI: {rsi:.0f}
⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}
━━━━━━━━━━━━━━━━━━━━━
⚠️ Bu bir sinyal sistemidir.
Kendi analizinizi de yapin."""

    return msg.strip()


def scan_and_notify():
    print(f"\n[{datetime.now().strftime('%H:%M')}] SMP V3.1 Tarama basladi...")
    opportunities = []

    for coin, p in COINS.items():
        try:
            df  = fetch_ohlcv(coin, interval="4h", period="6mo")
            ind = compute_all_indicators(df, preset=p["preset"])
            sc  = calc_bull_bear_score(ind)
            tr  = calc_triggers(ind, sc)
            sg  = generate_signals(ind, sc, tr,
                                   preset=p["preset"],
                                   eff_score=p["eff_score"],
                                   min_conf=p["min_conf"])
            fs  = apply_all_filters(ind, sg, use_cvd=True)

            recent_buy  = fs["buy_signal"].iloc[-2:].any()
            recent_sell = fs["sell_signal"].iloc[-2:].any()

            if not recent_buy and not recent_sell:
                print(f"  {coin}: sinyal yok")
                continue

            bull_score = sc["bull_score"].iloc[-1]
            bear_score = sc["bear_score"].iloc[-1]
            rvol       = ind["rvol"].iloc[-1]
            rsi        = ind["rsi"].iloc[-1]
            price      = df["close"].iloc[-1]
            stop_pct   = ind["safe_stop_pct"].iloc[-1]

            direction = "LONG" if (recent_buy and bull_score >= bear_score) else "SHORT"
            score     = bull_score if direction == "LONG" else bear_score
            grade     = calc_grade(score)

            tp1_pct = stop_pct * 1.0
            tp2_pct = stop_pct * p["rr_ratio"]

            if direction == "LONG":
                sl  = price * (1 - stop_pct / 100)
                tp1 = price * (1 + tp1_pct / 100)
                tp2 = price * (1 + tp2_pct / 100) if p["rr_ratio"] > 1.5 else None
            else:
                sl  = price * (1 + stop_pct / 100)
                tp1 = price * (1 - tp1_pct / 100)
                tp2 = price * (1 - tp2_pct / 100) if p["rr_ratio"] > 1.5 else None

            rr = tp2_pct / stop_pct if stop_pct > 0 else 0

            opportunities.append({
                "coin": coin, "direction": direction, "score": score,
                "grade": grade, "rvol": rvol, "rsi": rsi, "price": price,
                "sl": sl, "tp1": tp1, "tp2": tp2,
                "stop_pct": stop_pct, "tp1_pct": tp1_pct, "tp2_pct": tp2_pct,
                "rr": rr, "priority": p["priority"],
            })
            print(f"  {coin}: {direction} sinyali (skor={score:.1f})")

        except Exception as e:
            print(f"  {coin}: hata — {e}")

    if not opportunities:
        print("  Aktif sinyal yok.")
        return 0

    opportunities.sort(key=lambda x: (x["priority"], -x["score"]))

    for opp in opportunities:
        msg = format_signal_message(opp)
        ok  = send_message(msg)
        print(f"  {opp['coin']} mesaji {'gonderildi' if ok else 'gonderilemedi'}")
        time.sleep(1)

    return len(opportunities)


def send_test_message():
    msg = f"""🤖 <b>SMP V3.1 Bot Aktif!</b>

Sinyal tarayici basariyla baglandi.
VPA Climax + Zone POC modülleri dahil.

Taranacak coinler:
• BTC/USDT (Oncelikli)
• ETH/USDT
• SOL/USDT

Her 4 saatte bir otomatik tarama.

Baslangic: {datetime.now().strftime('%d.%m.%Y %H:%M')}"""

    ok = send_message(msg)
    print("Test mesaji gonderildi." if ok else "Mesaj gonderilemedi.")


def start_auto_scan(interval_hours=4):
    print(f"SMP V3.1 Bot Basladi! Her {interval_hours} saatte bir tarama.")
    send_message(f"🤖 <b>SMP V3.1 Bot Basladi!</b>\nHer {interval_hours} saatte bir tarama.")
    while True:
        try:
            scan_and_notify()
            print(f"Sonraki tarama {interval_hours} saat sonra...")
            time.sleep(interval_hours * 3600)
        except KeyboardInterrupt:
            print("Bot durduruldu.")
            send_message("Bot durduruldu.")
            break
        except Exception as e:
            print(f"Hata: {e}")
            time.sleep(300)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "scan"
    if mode == "test":
        send_test_message()
    elif mode == "scan":
        scan_and_notify()
    elif mode == "start":
        start_auto_scan(interval_hours=4)
    else:
        print("Kullanim: python telegram_bot.py [test|scan|start]")