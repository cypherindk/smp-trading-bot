"""
telegram_bot.py
SMP V2.8.2 — Telegram Sinyal Botu

Kullanim:
  python telegram_bot.py          -> Bir kere tara ve sinyal varsa gonder
  python telegram_bot.py start    -> Her 4 saatte bir otomatik tara
  python telegram_bot.py test     -> Test mesaji gonder
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

# ── Telegram Ayarlari ──
import os
BOT_TOKEN = os.environ.get("BOT_TOKEN", "8220091488:AAGAZDtlqTr3MVW66RjEEdDDXdF0O1ZAZTI")
CHAT_ID   = os.environ.get("CHAT_ID", "959954532")

# ── Coin Parametreleri ──
COINS = {
    "BTC-USD": {
        "preset": "Default", "eff_score": 3.0, "min_conf": 1,
        "adr_mult": 2.6, "rr_ratio": 1.2, "priority": 1,
        "name": "Bitcoin", "symbol": "BTC/USDT",
    },
    "ETH-USD": {
        "preset": "Conservative", "eff_score": 4.5, "min_conf": 1,
        "adr_mult": 2.8, "rr_ratio": 3.5, "priority": 2,
        "name": "Ethereum", "symbol": "ETH/USDT",
    },
    "SOL-USD": {
        "preset": "Default", "eff_score": 7.0, "min_conf": 1,
        "adr_mult": 1.9, "rr_ratio": 1.8, "priority": 3,
        "name": "Solana", "symbol": "SOL/USDT",
    },
}


def send_message(text: str) -> bool:
    """Telegram'a mesaj gonder."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"Telegram hata: {e}")
        return False


def format_signal_message(opp: dict) -> str:
    """Sinyal mesajini formatla."""
    coin = opp["coin"]
    p    = COINS[coin]
    direction = opp["direction"]
    price     = opp["price"]
    sl        = opp["sl"]
    tp1       = opp["tp1"]
    tp2       = opp["tp2"]
    tp_trend  = opp["tp_trend"]
    stop_pct  = opp["stop_pct"]
    tp1_pct   = opp["tp1_pct"]
    tp2_pct   = opp["tp2_pct"]
    score     = opp["score"]
    rvol      = opp["rvol"]
    rsi       = opp["rsi"]
    grade     = opp["grade"]
    rr        = opp["rr"]

    # Yon emojisi
    if direction == "LONG":
        dir_emoji = "📈 LONG"
        sl_emoji  = "🔴"
        tp_emoji  = "🟢"
    else:
        dir_emoji = "📉 SHORT"
        sl_emoji  = "🔴"
        tp_emoji  = "🟢"

    # Grade emojisi
    grade_emoji = {"A+": "🏆", "A": "⭐", "B": "👍", "C": "⚠️"}.get(grade, "")

    msg = f"""
🚨 <b>SMP V2.8.2 SİNYAL</b> 🚨
━━━━━━━━━━━━━━━━━━━━━
<b>{p['symbol']}</b>  {dir_emoji}
Grade: {grade_emoji} <b>{grade}</b>  |  Skor: {score:.1f}/10

💰 <b>GİRİŞ:</b>  ${price:,.4f}

{sl_emoji} <b>STOP LOSS:</b>  ${sl:,.4f}
   (-%{stop_pct:.2f})

{tp_emoji} <b>TP1:</b>  ${tp1:,.4f}
   (+%{tp1_pct:.2f})"""

    if tp2:
        msg += f"""
{tp_emoji} <b>TP2:</b>  ${tp2:,.4f}
   (+%{tp2_pct:.2f})"""

    if tp_trend:
        msg += f"""
📊 <b>TP3:</b>  Trend takibi (trailing stop)"""

    msg += f"""

━━━━━━━━━━━━━━━━━━━━━
⚖️ R/R Oranı:  1:{rr:.1f}
📊 RVOL: {rvol:.2f}x  |  RSI: {rsi:.0f}
⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}
━━━━━━━━━━━━━━━━━━━━━
⚠️ Bu bir sinyal sistemidir.
Kendi analizinizi de yapın."""

    return msg.strip()


def calc_grade(score: float) -> str:
    if score >= 8.0:
        return "A+"
    elif score >= 6.5:
        return "A"
    elif score >= 5.0:
        return "B"
    return "C"


def scan_and_notify():
    """Tum coinleri tara, sinyal varsa Telegram'a gonder."""
    print(f"\n[{datetime.now().strftime('%H:%M')}] Tarama basladi...")

    opportunities = []

    for coin, p in COINS.items():
        try:
            df  = fetch_ohlcv(coin, interval="4h", period="3mo")
            ind = compute_all_indicators(df, preset=p["preset"])
            sc  = calc_bull_bear_score(ind)
            tr  = calc_triggers(ind, sc)
            sg  = generate_signals(ind, sc, tr,
                                   preset=p["preset"],
                                   eff_score=p["eff_score"],
                                   min_conf=p["min_conf"])
            fs  = apply_all_filters(ind, sg, use_cvd=True)

            # Son 2 barda sinyal var mi?
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

            # TP hesaplari
            # TP1: R/R 1:1
            tp1_pct = stop_pct * 1.0
            # TP2: optimize edilmis R/R
            tp2_pct = stop_pct * p["rr_ratio"]

            if direction == "LONG":
                sl   = price * (1 - stop_pct / 100)
                tp1  = price * (1 + tp1_pct / 100)
                tp2  = price * (1 + tp2_pct / 100) if p["rr_ratio"] > 1.5 else None
            else:
                sl   = price * (1 + stop_pct / 100)
                tp1  = price * (1 - tp1_pct / 100)
                tp2  = price * (1 - tp2_pct / 100) if p["rr_ratio"] > 1.5 else None

            rr = tp2_pct / stop_pct if stop_pct > 0 else 0

            opportunities.append({
                "coin":      coin,
                "direction": direction,
                "score":     score,
                "grade":     grade,
                "rvol":      rvol,
                "rsi":       rsi,
                "price":     price,
                "sl":        sl,
                "tp1":       tp1,
                "tp2":       tp2,
                "tp_trend":  True,
                "stop_pct":  stop_pct,
                "tp1_pct":   tp1_pct,
                "tp2_pct":   tp2_pct,
                "rr":        rr,
                "priority":  p["priority"],
            })
            print(f"  {coin}: {direction} sinyali! Skor={score:.1f}")

        except Exception as e:
            print(f"  {coin}: hata — {e}")

    if not opportunities:
        print("  Aktif sinyal yok.")
        return 0

    # BTC oncelikli, sonra skora gore sirala
    opportunities.sort(key=lambda x: (x["priority"], -x["score"]))

    # Her sinyal icin mesaj gonder
    for opp in opportunities:
        msg = format_signal_message(opp)
        ok  = send_message(msg)
        if ok:
            print(f"  ✅ {opp['coin']} sinyali gonderildi!")
        else:
            print(f"  ❌ {opp['coin']} gonderilemedi!")
        time.sleep(1)

    return len(opportunities)


def send_test_message():
    """Test mesaji gonder — bot calisiyor mu kontrol et."""
    msg = """
🤖 <b>SMP V2.8.2 Bot Aktif!</b>

Sinyal tarayici basariyla baglandi.

Taranacak coinler:
• BTC/USDT (Oncelikli)
• ETH/USDT
• SOL/USDT

Her 4 saatte bir otomatik tarama yapilacak.
Sinyal geldiginde buraya bildirim alacaksiniz.

⏰ Baslangic: """ + datetime.now().strftime('%d.%m.%Y %H:%M')

    ok = send_message(msg.strip())
    if ok:
        print("✅ Test mesaji gonderildi! Telegram'i kontrol edin.")
    else:
        print("❌ Mesaj gonderilemedi. Token ve Chat ID'yi kontrol edin.")


def start_auto_scan(interval_hours: int = 4):
    """Her X saatte bir otomatik tara."""
    print(f"\n🤖 SMP V2.8.2 Telegram Botu Basladi!")
    print(f"   Her {interval_hours} saatte bir tarama yapilacak.")
    print(f"   Durdurmak icin: Ctrl+C\n")

    send_message(f"🤖 <b>SMP Bot Basladi!</b>\nHer {interval_hours} saatte bir tarama yapilacak.")

    while True:
        try:
            count = scan_and_notify()
            next_scan = datetime.now()
            print(f"  Sonraki tarama {interval_hours} saat sonra...")
            time.sleep(interval_hours * 3600)
        except KeyboardInterrupt:
            print("\n\nBot durduruldu.")
            send_message("⛔ SMP Bot durduruldu.")
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
