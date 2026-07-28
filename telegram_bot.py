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
  3) [DUZELTILDI] "State/dedup dosyasina artik gerek yok" onceki notu
     YANLIŞTI. Tarama 4 saatte bir calisiyor, mum boyu da 4 saat --
     "son 2 mumda sinyal var mi" kontrolu tek basina AYNI sinyali
     ardisik taramalarda tekrar tekrar gonderiyordu. state/bist_state.json
     (zaten repo'da varmis, gercek formatini kullaniciya sordum) ve YENI
     state/crypto_state.json ile dedup geri eklendi -- bkz. asagida
     "State / Dedup" bolumu. .github/workflows/scan.yml'nin artik
     state/crypto_state.json'i da commit etmesi lazim (asagidaki
     scan.yml ornegine bak).
  4) [FIX] BIST preset/eff_score/grade_filter, TradingView tarafinda
     "Preset: Auto" + 4H grafikte cozulen gercek deger olan "Aggressive"
     (effScore=3) ve "Grade Filtresi: A+ Only" (skor>=8.0) ile eslendi.
     Onceki "Default" + eff_score=5.0 + grade_filter=All (varsayilan)
     TradingView ekraninda hic gorunmeyen B/A grade sinyalleri Telegram'a
     gonderiyordu -- KORDS ornegindeki uyumsuzlugun ana sebebi buydu.
  5) [FIX] recent_buy/recent_sell son 2 barda kontrol ediliyordu ama
     fiyat/skor/RSI/RVOL her zaman SON bardan (iloc[-1]) okunuyordu.
     Sinyal 1 bar once tetiklenmisse Telegram mesajindaki fiyat/skor
     yanlis bardan geliyordu ve yon (LONG/SHORT) bile yanlis
     etiketlenebiliyordu. Artik gercek tetik barindan okunuyor ve o
     barin zaman damgasi mesaja ekleniyor (TradingView'de hangi muma
     bakman gerektigini net gostermek icin).

  NOT (GUNCEL): Pine tarafindaki MTF Likidite Filtresi (i_useMTF, 5dk/
  15dk/1sa liquidity-sweep tabanli mtfOkBull/mtfOkBear) ARTIK uygulaniyor
  -- bkz. engine/liquidity_mtf.py + engine/signals.py. Whale Onayi da
  (confBull/confBear icine giren w_buy_recent/w_sell_recent) uygulandi.
  Hala TASINMAYANLAR: (a) Alt TF Cooldown Kilidi (ltfOkBull/ltfOkBear) --
  bu bot 4H calistigi icin (isLTF=False) zaten devre disi, atlanmasi
  guvenli; (b) gorsel VAH/VAL breakout etiketi ("Roket/Selale") -- sadece
  gorsel, ana LONG/SHORT sinyaliyle ilgisi yok.
"""

import sys
import os
import time
import json
import requests
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Turkiye kalici UTC+3 (2016'dan beri DST yok). Telegram'daki sinyal/gonderim
# saatlerini TW ekranindaki (Istanbul) mumla AYNI gostermek icin.
IST = timezone(timedelta(hours=3))

from data.fetcher import fetch_ohlcv
from data.crypto_fetcher import fetch_binance_ohlcv   # [FIX] kripto verisi artik Binance (TW ile ayni kaynak)
from engine.indicators import compute_all_indicators
from engine.signals import calc_bull_bear_score, calc_triggers, generate_signals
from engine.filters import apply_all_filters
from engine.liquidity_mtf import fetch_mtf_frame
from engine.smart_money_flow import calc_smart_money_droplet
from bist100_tickers import BIST100_YF

# [YENİ] Pine'daki "MTF Likidite Filtresi" (5dk/15dk/1sa equal-level +
# sweep) esdegeri. Her sembol icin 3 EK yfinance istegi (5m/15m/1h)
# gerektirir -- BIST'te 100 hisse x 4 istek (4h+5m+15m+1h) = ~400 istek/
# tarama demektir, bu da taramayi ciddi yavaslatir ve yfinance rate-limit
# riskini artirir. Cok yavas/hata aliyorsa BIST_USE_MTF=False yapip
# sadece kripto icin acik birakabilirsin (3 coin x 4 istek = 12, sorun
# degil).
# [FIX] MTF kapatildi: Binance'ten 5m/15m/1h sayfa cekimi run'i ~4-6dk
# yavaslatiyordu + MTF portun EN AZ SADIK parcasi (TV-sapma kaynagi). Kapali =
# hizli (~1-2dk) + TV'ye daha yakin sinyal. Tekrar acmak: CRYPTO_USE_MTF = True.
CRYPTO_USE_MTF = False
BIST_USE_MTF = True

# [YENİ] BIST RAFA KALDIRILDI (crypto-only). BIST verisi (yfinance) TV'nin BIST
# feed'iyle eslesmedigi icin simdilik KAPALI; kod (scan_bist, bist100_tickers)
# repoda duruyor -- sonra ayri repo / duzgun veri kaynagiyla ele alinacak.
# Tekrar acmak icin tek satir: SCAN_BIST = True.
SCAN_BIST = False

# ── [YENİ] optimize/run_batch.py'nin ürettiği best_params.json ──
# Bir sembol icin optimize edilmis parametre varsa AŞAĞIDAKİ COINS/BIST_*
# sabitlerinin YERINE gecer (preset dahil -- kullanicinin istegi:
# "TW ile eslesmesi degil, gecmiste en iyi performans" onemli).
# Dosya yoksa veya bir sembolde yoksa, eski sabit degerlere (COINS /
# BIST_PRESET vs.) sessizce geri duser.
_OPT_PARAMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "optimize", "best_params.json")


def load_optimized_params(path=_OPT_PARAMS_PATH):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  [optimize] {len(data)} sembol icin optimize edilmis parametre yuklendi.")
        return data
    except Exception as e:
        print(f"  [optimize] best_params.json okunamadi, sabit degerlere donuluyor: {e}")
        return {}


OPTIMIZED_PARAMS = load_optimized_params()


def resolve_symbol_config(symbol: str, default_cfg: dict) -> dict:
    """
    Optuna'dan gelen params varsa onu kullan (preset/eff_score/min_conf/
    grade_filter/adr_mult/rr_ratio), eksik alanlari default_cfg'den
    tamamla. Hic optimize edilmemis sembol icin default_cfg aynen doner.
    """
    entry = OPTIMIZED_PARAMS.get(symbol)
    if not entry or "params" not in entry:
        return default_cfg
    p = entry["params"]
    cfg = dict(default_cfg)
    cfg["preset"] = p.get("preset", cfg.get("preset"))
    cfg["eff_score"] = p.get("eff_score", cfg.get("eff_score"))
    cfg["min_conf"] = p.get("min_conf", cfg.get("min_conf"))
    cfg["grade_filter"] = p.get("grade_filter", cfg.get("grade_filter"))
    cfg["adr_mult"] = p.get("adr_mult", cfg.get("adr_mult"))
    cfg["rr_ratio"] = p.get("rr_ratio", cfg.get("rr_ratio"))
    return cfg

# ── Telegram ayarlari ──
# ⚠️ Token'i GitHub Secrets'tan okuyun, koda hardcode ETMEYIN.
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

if not BOT_TOKEN or not CHAT_ID:
    print("UYARI: BOT_TOKEN veya CHAT_ID environment variable eksik.")

# ── Kripto parametreleri (mevcut, degismedi — TradingView'deki her coin
#    grafiginin Preset/Grade Filtresi ayarini da ayrica dogrulaman lazim,
#    bu dosya sadece BIST icin dogrulanan ayarlarla guncellendi) ──
COINS = {
    # [FIX] Preset/eff_score/min_conf/grade_filter artik BIST ile ayni
    # (TW'de Auto->Aggressive, Grade Filtresi=A+ Only). "Piyasa Turu"
    # (Kripto/Hisse) Pine tarafinda SADECE ayri bir gorsel VAH/VAL
    # breakout etiketini (dyn_brk_rvol, "Roket/Selale") etkiliyor --
    # LONG/SHORT ana sinyal motoruyla ilgisi yok, Python'da zaten
    # implement edilmedi. adr_mult/rr_ratio (risk yonetimi) coin'e
    # ozel kaldi -- bunlar indikator ayari degil, senin backtest'le
    # ayarladigin pozisyon buyuklugu/RR parametreleri, dokunmadim.
    "BTC-USD": {
        "preset": "Aggressive", "eff_score": 3.0, "min_conf": 2,
        "grade_filter": "A+ Only",
        "adr_mult": 1.5, "rr_ratio": 2.0, "priority": 1,
        "symbol": "BTC/USDT",
    },
    "ETH-USD": {
        "preset": "Aggressive", "eff_score": 3.0, "min_conf": 2,
        "grade_filter": "A+ Only",
        "adr_mult": 2.8, "rr_ratio": 3.5, "priority": 2,
        "symbol": "ETH/USDT",
    },
    "SOL-USD": {
        "preset": "Aggressive", "eff_score": 3.0, "min_conf": 2,
        "grade_filter": "A+ Only",
        "adr_mult": 1.9, "rr_ratio": 1.8, "priority": 3,
        "symbol": "SOL/USDT",
    },
}


# ── BIST icin ortak parametreler ──
# TradingView ekran goruntusu: Preset=Auto, HTF Filtresi=Grafik(bos=mevcut),
# Grade Filtresi=A+ Only, C-Grade Sinyalleri Gizle=acik, Min Onay Sayisi=2.
# Pine "Auto" presetinin 4H grafikte cozdugu gercek deger "Aggressive"dir
# (tfMin=240 -> "Aggressive"), "Default" DEGIL. effScore de Aggressive icin
# sabit 3'tur -- "Min Confluence Skoru (Custom)=5" alani sadece
# Preset="Custom" secilirse kullanilir, Auto modunda devre disidir.
BIST_PRESET = "Aggressive"
BIST_EFF_SCORE = 3.0
BIST_GRADE_FILTER = "A+ Only"
BIST_MIN_CONF = 2
BIST_RR_RATIO = 2.0
BIST_REQUEST_DELAY = 1.0  # ardisik yfinance istekleri arasi bekleme
                           # (MTF modulu her hisse icin 3 ek istek daha
                           # yaptigi icin eskisinden (0.6) biraz yuksek
                           # tutuldu -- BIST_USE_MTF=False ise 0.6'ya
                           # dusurebilirsin)
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


def _binance_fetch_ohlcv(symbol, interval="1h", period="60d"):
    """fetch_mtf_frame'in bekledigi (symbol, interval, period) imzasi -> Binance days."""
    days = 60
    if isinstance(period, str) and period.endswith("d"):
        try:
            days = int(period[:-1])
        except ValueError:
            days = 60
    return fetch_binance_ohlcv(symbol, interval=interval, days=days)


def _fmt_ist(ts):
    """tz-aware zaman damgasini Turkiye saatine (UTC+3) cevirip formatla."""
    if ts is None:
        return "?"
    try:
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_convert(IST) if hasattr(ts, "tz_convert") else ts.astimezone(IST)
    except Exception:
        pass
    return ts.strftime('%d.%m.%Y %H:%M')


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
    timeframe = opp.get("timeframe", "4H")
    signal_time = opp.get("signal_time")
    signal_time_str = _fmt_ist(signal_time)

    msg = f"""🚨 <b>SMP SİNYAL</b> — {asset_tag} 🚨
━━━━━━━━━━━━━━━━━━━━━
<b>{label}</b>  {dir_emoji}  |  {timeframe}
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
🕓 Sinyal Muma:  {signal_time_str}  (TSİ — TW'de bu muma bak)
⏰ Gonderim: {datetime.now(IST).strftime('%d.%m.%Y %H:%M')}"""

    if opp.get("ghost_bar_warning"):
        msg += """
⚠️ <b>DIKKAT:</b> Son mum anormal dusuk hacimli -- hayalet/yarim
seans bari olabilir, sinyali dogrulamadan islem acmayin."""

    msg += """
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
🕓 Whale Muma: {_fmt_ist(w.get('signal_time'))}  (TSİ — TW'de bu muma bak)
⏰ Gönderim: {datetime.now(IST).strftime('%d.%m.%Y %H:%M')}
━━━━━━━━━━━━━━━━━━━━━
Bu, ana SMP sinyalinden bağımsız bir para akışı uyarısıdır.
Sadece takip listesi amaçlıdır."""


def format_droplet_message(d: dict) -> str:
    """
    [YENİ] Pine'daki 💧 "Smart Money Flow" (Öncül Para Girişi) etiketi
    icin bildirim. Sadece BULLISH (Pine'da ayı versiyonu yok, bkz.
    engine/smart_money_flow.py). Ana LONG/SHORT sinyalinden ve whale
    alert'ten TAMAMEN bagimsiz, ayri bir olay.
    """
    asset_tag = "🪙 KRİPTO" if d["asset_type"] == "crypto" else "🇹🇷 BIST"
    signal_time_str = _fmt_ist(d["signal_time"])
    return f"""💧 <b>SMART MONEY FLOW</b> — {asset_tag} 💧
━━━━━━━━━━━━━━━━━━━━━
<b>{d['label']}</b>
Fiyat düşerken CVD/MFI güçleniyor -- olası öncül para girişi.

Fiyat: {d['price']:,.4f} {d['currency']}
RVOL: {d['rvol']:.2f}x
🕓 Sinyal Muma: {signal_time_str}  (TSİ — TW'de bu muma bak)
⏰ Gönderim: {datetime.now(IST).strftime('%d.%m.%Y %H:%M')}
━━━━━━━━━━━━━━━━━━━━━
Bu, ana LONG/SHORT sinyalinden ve whale alert'ten bağımsız,
öncül bir gizli-diverjans uyarısıdır. Kendi analizinizi yapın."""


# ───────────────────────── Ortak: gercek tetik barini bul ─────────────────────────

def _resolve_signal_bar(fs, lookback=2):
    """
    fs["buy_signal"] / fs["sell_signal"] icinde son `lookback` barda
    True olan var mi bak, varsa GERCEK tetik barinin index'ini ve
    yonunu dondur. Ikisi de varsa daha yeni olani sec.

    Onceki koddaki hata: sinyal 1 bar once tetiklenmis olsa bile fiyat/
    skor hep en son bardan (iloc[-1]) okunuyordu -- bu yuzden Telegram
    mesaji TradingView'deki gercek sinyal barindan farkli bir fiyat/
    zaman gosterebiliyordu.
    """
    buy_recent = fs["buy_signal"].iloc[-lookback:]
    sell_recent = fs["sell_signal"].iloc[-lookback:]

    buy_idx = buy_recent[buy_recent].index[-1] if buy_recent.any() else None
    sell_idx = sell_recent[sell_recent].index[-1] if sell_recent.any() else None

    if buy_idx is None and sell_idx is None:
        return None, None
    if buy_idx is not None and sell_idx is not None:
        if buy_idx >= sell_idx:
            return buy_idx, "LONG"
        return sell_idx, "SHORT"
    if buy_idx is not None:
        return buy_idx, "LONG"
    return sell_idx, "SHORT"


# ───────────────────────── [YENİ] State / Dedup ─────────────────────────
# [FIX] Tarama 4 saatte bir calisiyor, mum boyu da 4 saat -- yani
# _resolve_signal_bar()'in "son 2 mum" penceresi, AYNI sinyal barinin
# 2 ardisik taramada da "recent" gorunmesi demek. State dosyasi olmadan
# ayni sinyal 2 kez (bazen "hayalet bar" filtrelemesi/tatil bosluklari
# yuzunden daha da fazla) Telegram'a gidiyordu. Eskiden burada
# "state dosyasina gerek yok" diye YANLIS bir varsayimla bu tamamen
# kaldirilmisti -- state/bist_state.json'da gordugum gercek format
# ("TICKER:YON": "YYYY-MM-DD") esas alinarak geri eklendi, kripto icin
# de ayni mantikla state/crypto_state.json eklendi (workflow'da bu da
# commit edilmeli, bkz. .github/workflows/scan.yml).
STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state")
BIST_STATE_PATH = os.path.join(STATE_DIR, "bist_state.json")
CRYPTO_STATE_PATH = os.path.join(STATE_DIR, "crypto_state.json")


def load_state(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"  [state] {path} okunamadi, bos state ile devam: {e}")
    return {}


def save_state(path: str, state: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def already_notified(state: dict, label: str, direction: str, signal_bar) -> bool:
    """
    Ayni ticker+yon icin, ayni sinyal barinin tarihi daha once
    bildirildiyse True doner (bar degismedigi surece tekrar gonderme).
    """
    key = f"{label}:{direction}"
    sig_date = signal_bar.strftime("%Y-%m-%d")
    return state.get(key) == sig_date


def mark_notified(state: dict, label: str, direction: str, signal_bar):
    key = f"{label}:{direction}"
    state[key] = signal_bar.strftime("%Y-%m-%d")


# ───────────────────────── Tarama mantigi ─────────────────────────

def scan_crypto(crypto_state: dict):
    opportunities = []
    whale_events = []
    droplet_events = []
    for coin, p in COINS.items():
        p = resolve_symbol_config(coin, p)
        try:
            # [FIX] Kripto verisi artik Binance'ten (TW ile AYNI kaynak). Binance
            # 4h'i native destekler (resample yok). 300 gun -> EMA200 iyi seedlenir.
            df = fetch_binance_ohlcv(coin, interval="4h", days=300)
            if len(df) < 60:
                print(f"  {coin}: yetersiz veri ({len(df)} bar)")
                continue

            # [FIX] adr_mult artik canlida da uygulaniyor (coin'e ozel: BTC
            # 1.5 / ETH 2.8 / SOL 1.9) -- eskiden hep 1.5'te kaliyordu.
            ind = compute_all_indicators(df, preset=p["preset"], timeframe_minutes=240,
                                         adr_mult=p.get("adr_mult"))

            mtf_frame = fetch_mtf_frame(coin, df, _binance_fetch_ohlcv) if CRYPTO_USE_MTF else None
            sc = calc_bull_bear_score(ind, mtf=mtf_frame)
            tr = calc_triggers(ind, sc)
            sg = generate_signals(ind, sc, tr, preset=p["preset"],
                                   eff_score=p["eff_score"], min_conf=p["min_conf"],
                                   grade_filter=p.get("grade_filter", "All"))
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
                    "signal_time": df.index[-1],   # [FIX] whale mumu = son bar, saati TSİ gosterilecek
                })

            # [YENİ] 💧 Smart Money Flow damlasi (ana LONG/SHORT sinyalinden
            # BAGIMSIZ, sadece son bar kontrol edilir)
            droplet = calc_smart_money_droplet(ind)
            droplet_bar = ind.index[-1]
            if droplet["smart_money_droplet"].iloc[-1] and not already_notified(
                    crypto_state, coin, "DROPLET", droplet_bar):
                droplet_events.append({
                    "asset_type": "crypto", "label": p["symbol"], "currency": "USD",
                    "price": df["close"].iloc[-1], "rvol": ind["rvol"].iloc[-1],
                    "signal_time": droplet_bar, "dedup_key": coin,
                })

            signal_bar, direction = _resolve_signal_bar(fs)
            if signal_bar is None:
                print(f"  {coin}: sinyal yok")
                continue

            if already_notified(crypto_state, coin, direction, signal_bar):
                print(f"  {coin}: {direction} zaten bildirildi (bar={signal_bar.date()}), atlaniyor")
                continue

            # Guvenlik kontrolu: son bar anormal dusuk hacimliyse (olasi
            # hayalet/yarim seans bari) sinyali uyariyla isaretle
            avg_vol_recent = df["volume"].iloc[-21:-1].mean()
            last_vol = df["volume"].iloc[-1]
            ghost_bar_warning = avg_vol_recent > 0 and last_vol < avg_vol_recent * 0.15

            # [FIX] Bu 4 tanim eskiden EKSIKTI -- price/stop_pct/score/
            # tp1_pct hic atanmadan kullaniliyor, her kripto LONG/SHORT
            # sinyalinde UnboundLocalError firlatiyor ve sinyal Telegram'a
            # HIC gitmiyordu (scan_bist'te bu satirlar vardi, scan_crypto'ya
            # kopyalanmamis). Artik BIST tarafiyla birebir ayni.
            bull_score = sg.loc[signal_bar, "total_bull_score"]
            bear_score = sg.loc[signal_bar, "total_bear_score"]
            price = df.loc[signal_bar, "close"]
            stop_pct = ind.loc[signal_bar, "safe_stop_pct"]
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
                "rvol": ind.loc[signal_bar, "rvol"], "rsi": ind.loc[signal_bar, "rsi"],
                "price": price, "sl": sl, "tp1": tp1, "tp2": tp2,
                "stop_pct": stop_pct, "tp1_pct": tp1_pct, "tp2_pct": tp2_pct,
                "rr": tp2_pct / stop_pct if stop_pct > 0 else 0,
                "priority": p["priority"], "dedup_key": coin, "timeframe": "4H",
                "ghost_bar_warning": ghost_bar_warning,
                "signal_time": signal_bar,
            })
            print(f"  {coin}: {direction} sinyali (skor={score:.1f}, bar={signal_bar})")

        except Exception as e:
            print(f"  {coin}: hata — {e}")

    return opportunities, whale_events, droplet_events


def scan_bist(bist_state: dict):
    opportunities = []
    whale_events = []
    droplet_events = []
    for ticker in BIST100_YF:
        try:
            # NOT (HIZ): "1y" denendi (ema_200 & Zone POC'un tam gecerli
            # olmasi icin) ama 90 BIST hissesi x 1 yillik 1h veri, yfinance/
            # Yahoo hiz-limitine takilip taramayi ~1 saate cikardi. Bu
            # yuzden "60d"ye geri donuldu -- bu durumda BIST'te ema_200
            # tam isinmiyor ve Zone POC (rolling 200) cogu bar NaN kaliyor
            # (POC sadece +1.0 BONUS, sert filtre degil; sinyal uretimi
            # calismaya devam eder). Daha yuksek TW sadakati istenirse
            # "180d" (~6ay, ~260 bar) araya alinabilir -- POC'u tekrar
            # aktif eder, sure ~1y'in yarisi kadardir.
            df = fetch_smart(ticker, "4h", "60d", resample_offset="2h")
            if len(df) < 60:
                print(f"  {ticker}: yetersiz veri ({len(df)} bar)")
                continue

            cfg = resolve_symbol_config(ticker, {
                "preset": BIST_PRESET, "eff_score": BIST_EFF_SCORE,
                "min_conf": BIST_MIN_CONF, "grade_filter": BIST_GRADE_FILTER,
                "adr_mult": None, "rr_ratio": BIST_RR_RATIO,
            })

            ind = compute_all_indicators(df, preset=cfg["preset"], timeframe_minutes=240,
                                         adr_mult=cfg.get("adr_mult"))

            mtf_frame = fetch_mtf_frame(ticker, df, fetch_ohlcv) if BIST_USE_MTF else None
            sc = calc_bull_bear_score(ind, mtf=mtf_frame)
            tr = calc_triggers(ind, sc)
            sg = generate_signals(ind, sc, tr, preset=cfg["preset"],
                                   eff_score=cfg["eff_score"], min_conf=cfg["min_conf"],
                                   grade_filter=cfg["grade_filter"])
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
                    "signal_time": df.index[-1],   # [FIX] whale mumu = son bar, saati TSİ gosterilecek
                })

            # [YENİ] 💧 Smart Money Flow damlasi (ana LONG/SHORT sinyalinden
            # BAGIMSIZ, sadece son bar kontrol edilir)
            droplet = calc_smart_money_droplet(ind)
            droplet_bar = ind.index[-1]
            if droplet["smart_money_droplet"].iloc[-1] and not already_notified(
                    bist_state, label, "DROPLET", droplet_bar):
                droplet_events.append({
                    "asset_type": "bist", "label": label, "currency": "TRY",
                    "price": df["close"].iloc[-1], "rvol": ind["rvol"].iloc[-1],
                    "signal_time": droplet_bar, "dedup_key": label,
                })

            signal_bar, direction = _resolve_signal_bar(fs)
            if signal_bar is None:
                continue

            if already_notified(bist_state, label, direction, signal_bar):
                print(f"  {ticker}: {direction} zaten bildirildi (bar={signal_bar.date()}), atlaniyor")
                continue

            # Guvenlik kontrolu: son bar anormal dusuk hacimliyse (olasi
            # hayalet/yarim seans bari) sinyali uyariyla isaretle
            avg_vol_recent = df["volume"].iloc[-21:-1].mean()
            last_vol = df["volume"].iloc[-1]
            ghost_bar_warning = avg_vol_recent > 0 and last_vol < avg_vol_recent * 0.15

            bull_score = sg.loc[signal_bar, "total_bull_score"]
            bear_score = sg.loc[signal_bar, "total_bear_score"]
            price = df.loc[signal_bar, "close"]
            stop_pct = ind.loc[signal_bar, "safe_stop_pct"]

            rr_ratio = cfg["rr_ratio"]
            score = bull_score if direction == "LONG" else bear_score
            tp1_pct = stop_pct * 1.0
            tp2_pct = stop_pct * rr_ratio
            if direction == "LONG":
                sl = price * (1 - stop_pct / 100)
                tp1 = price * (1 + tp1_pct / 100)
                tp2 = price * (1 + tp2_pct / 100) if rr_ratio > 1.5 else None
            else:
                sl = price * (1 + stop_pct / 100)
                tp1 = price * (1 - tp1_pct / 100)
                tp2 = price * (1 - tp2_pct / 100) if rr_ratio > 1.5 else None

            opportunities.append({
                "asset_type": "bist", "label": label, "currency": "TRY",
                "direction": direction, "score": score, "grade": calc_grade(score),
                "rvol": ind.loc[signal_bar, "rvol"], "rsi": ind.loc[signal_bar, "rsi"],
                "price": price, "sl": sl, "tp1": tp1, "tp2": tp2,
                "stop_pct": stop_pct, "tp1_pct": tp1_pct, "tp2_pct": tp2_pct,
                "rr": tp2_pct / stop_pct if stop_pct > 0 else 0,
                "priority": 9, "dedup_key": label, "timeframe": "4H",
                "ghost_bar_warning": ghost_bar_warning,
                "signal_time": signal_bar,
            })
            print(f"  {ticker}: {direction} sinyali (skor={score:.1f}, bar={signal_bar})")

        except Exception as e:
            print(f"  {ticker}: hata — {e}")

        time.sleep(BIST_REQUEST_DELAY)

    return opportunities, whale_events, droplet_events


def _dispatch(opportunities, whale_events, droplet_events, crypto_state, bist_state):
    """Toplanan sinyal/whale/damla bildirimlerini Telegram'a gonder + dedup isaretle."""
    if opportunities:
        opportunities.sort(key=lambda x: (x["priority"], -x["score"]))
        for opp in opportunities:
            ok = send_message(format_signal_message(opp))
            print(f"  {opp['label']} mesaji {'gonderildi' if ok else 'gonderilemedi'}")
            if ok:
                # Basarili gonderimden SONRA isaretle (Telegram hatasinda tekrar denensin)
                state = crypto_state if opp["asset_type"] == "crypto" else bist_state
                mark_notified(state, opp["dedup_key"], opp["direction"], opp["signal_time"])
            time.sleep(1)
    if whale_events:
        whale_events.sort(key=lambda x: -x["whale_score"])
        for w in whale_events:
            ok = send_message(format_whale_message(w))
            print(f"  {w['label']} whale alert {'gonderildi' if ok else 'gonderilemedi'}")
            time.sleep(1)
    if droplet_events:
        for d in droplet_events:
            ok = send_message(format_droplet_message(d))
            print(f"  {d['label']} damla (SMF) alert {'gonderildi' if ok else 'gonderilemedi'}")
            if ok:
                state = crypto_state if d["asset_type"] == "crypto" else bist_state
                mark_notified(state, d["dedup_key"], "DROPLET", d["signal_time"])
            time.sleep(1)


def scan_and_notify():
    scope = f"3 kripto + {len(BIST100_YF)} BIST hissesi" if SCAN_BIST else "3 kripto (crypto-only)"
    print(f"\n[{datetime.now(IST).strftime('%H:%M')}] SMP Tarama basladi ({scope})...")

    crypto_state = load_state(CRYPTO_STATE_PATH)
    bist_state = load_state(BIST_STATE_PATH)

    # [FIX] KRIPTO once taranir VE HEMEN gonderilir -- yavas BIST taramasi (90
    # hisse) kripto sinyalini artik BEKLETMESIN. Eskiden ikisi de tarandiktan
    # SONRA gonderiliyordu; bu, kripto sinyalini ~10-30 dk geciktiriyordu.
    c_opps, c_whales, c_drops = scan_crypto(crypto_state)
    _dispatch(c_opps, c_whales, c_drops, crypto_state, bist_state)
    save_state(CRYPTO_STATE_PATH, crypto_state)

    # BIST sonra (yavas) -- RAFA KALDIRILDI (SCAN_BIST=False -> crypto-only).
    # Kod duruyor; sonra ayri repo/duzgun veri kaynagiyla ele alinacak.
    b_opps, b_whales, b_drops = [], [], []
    if SCAN_BIST:
        b_opps, b_whales, b_drops = scan_bist(bist_state)
        _dispatch(b_opps, b_whales, b_drops, crypto_state, bist_state)
        save_state(BIST_STATE_PATH, bist_state)

    total = (len(c_opps) + len(c_whales) + len(c_drops) +
             len(b_opps) + len(b_whales) + len(b_drops))
    if total == 0:
        print("  Aktif sinyal/whale/damla yok.")
    return total


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
