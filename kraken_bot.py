"""
kraken_bot.py
Project KRAKEN V16 — Telegram Sinyal Botu
Ayri bot/kanal kullanir (KRAKEN_BOT_TOKEN)
"""

import sys
import os
import requests
from datetime import datetime
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.fetcher import fetch_ohlcv
from kraken.indicators import compute_kraken_indicators
from kraken.runner import BIST100, DEFAULT_PARAMS

BOT_TOKEN = os.environ.get("KRAKEN_BOT_TOKEN", "8878521032:AAEOT33_i6BIexNe5MdXd9XBqfnRejXLxRc")
CHAT_ID   = os.environ.get("CHAT_ID", "959954532")


def send_message(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"Telegram hata: {e}")
        return False


def format_kraken_signal(opp):
    symbol  = opp["symbol"].replace(".IS", "")
    price   = opp["price"]
    tp      = opp["tp"]
    sl      = opp["sl"]
    rr      = opp["rr"]
    tp_pct  = ((tp - price) / price) * 100
    sl_pct  = ((price - sl) / price) * 100

    msg = f"""🎯 <b>KRAKEN SİNYAL</b>
━━━━━━━━━━━━━━━━━━━━━
<b>{symbol}</b>  📈 LONG

💰 <b>Giriş:</b>  {price:.2f} TL

🟢 <b>TP:</b>  {tp:.2f} TL  (+%{tp_pct:.1f})
🔴 <b>SL:</b>  {sl:.2f} TL  (-%{sl_pct:.1f})

⚖️ R/R: 1:{rr:.1f}
━━━━━━━━━━━━━━━━━━━━━
📅 {datetime.now().strftime('%d.%m.%Y %H:%M')}
⚠️ Kendi analizinizi de yapin."""

    return msg.strip()


def scan_and_notify():
    print(f"\n[{datetime.now().strftime('%H:%M')}] KRAKEN BIST100 Tarama basladi...")

    opportunities = []
    for symbol in BIST100:
        try:
            df = fetch_ohlcv(symbol, interval="1d", period="5y")
            ind = compute_kraken_indicators(
                df, st_mult=DEFAULT_PARAMS["st_mult"],
                trap_tolerance=DEFAULT_PARAMS["trap_tolerance"],
                fvg_mult=DEFAULT_PARAMS["fvg_mult"]
            )

            recent_signal = (ind["buy_signal"] & ind["is_uptrend"]).iloc[-3:].any()

            if recent_signal:
                price      = df["close"].iloc[-1]
                yz_atr     = ind["yz_atr"].iloc[-1]
                supertrend = ind["supertrend"].iloc[-1]

                if pd.isna(supertrend) or pd.isna(yz_atr):
                    continue
                if supertrend >= price:
                    continue

                risk = yz_atr * DEFAULT_PARAMS["st_mult"]
                tp   = price + risk * DEFAULT_PARAMS["rr_target"]
                sl   = supertrend
                rr   = (tp - price) / (price - sl) if (price - sl) > 0 else 0

                opportunities.append({
                    "symbol": symbol, "price": price,
                    "tp": tp, "sl": sl, "rr": rr,
                })
                print(f"  {symbol}: SINYAL! Fiyat={price:.2f} TP={tp:.2f} SL={sl:.2f}")

        except Exception:
            pass

    if not opportunities:
        print("  Aktif sinyal yok.")
        return 0

    for opp in opportunities:
        msg = format_kraken_signal(opp)
        ok  = send_message(msg)
        print(f"  {opp['symbol']} mesaji {'gonderildi' if ok else 'gonderilemedi'}")

    return len(opportunities)


def send_test():
    msg = f"""🤖 <b>KRAKEN Bot Aktif!</b>

BIST100 tarayici basariyla baglandi.
Bu kendi ozel Telegram kanalinizdir.

Hafta ici her gun 18:30'da otomatik tarama yapilacak.
Sinyal geldiginde bu mesaj formatinda bildirim alacaksiniz.

Baslangic: {datetime.now().strftime('%d.%m.%Y %H:%M')}"""

    ok = send_message(msg)
    print("Test mesaji gonderildi." if ok else "Mesaj gonderilemedi.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "scan"
    if mode == "test":
        send_test()
    elif mode == "scan":
        scan_and_notify()
    else:
        print("Kullanim: python kraken_bot.py [test|scan]")