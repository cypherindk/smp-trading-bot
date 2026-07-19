"""
engine/smart_money_flow.py
Pine V2.8.2 — "💧 Smart Money Flow" (Öncül Para Girişi) damla motoru.

Pine (just_me.txt, satir ~756-776):
  global_pl_div = ta.pivotlow(low, 6, 6)
  f_hiddenDiv(pr, volume_or_mfi, len, pl_ser):
      son `len` bar icinde confirmed bir pivot-low bulunsun; eger
      (guncel fiyat < o pivottaki fiyat) VE (guncel gosterge > o
      pivottaki gosterge) ise True.
  bullCVD = f_hiddenDiv(low, cumCVD, 30, global_pl_div)
  bullMFI = f_hiddenDiv(low, mfiVal, 30, global_pl_div)
  isSmartParaBull = (bullCVD or bullMFI) and volAbove(rvol>1.2)
                    and isBull(close>=open) and emaFast>emaMid

Yani: fiyat DAHA DUSUK bir dip yaparken CVD ya da MFI o eski dibe gore
DAHA YUKSEK kaliyorsa (fiyat dusuyor ama para akisi/momentum zayiflamiyor)
+ hacim ortalama ustu + mum yesil + kisa vadeli trend yukari -> damla.

[ONEMLI] Pine'da SADECE bu boğa (bullish) versiyonu var. Ayrı bir "ayı
damlası" (bearish) YOK -- kod yanlışlıkla eksik değil, tasarım böyle.
Bu modul de bilerek sadece bullish uretiyor.

NOT — yaklasik uygulama: Pine'daki cumCVD, 1-dakikalik
request.security_lower_tf verisiyle hesaplanan, grafigin basindan beri
biriken GERCEK kumulatif bir seri. yfinance 1 dakikalik veriyi sadece
~7-8 gun geriye sagliyor, bu yuzden bunu birebir uretmek pratik degil.
Ama divergence testi sadece "guncel deger, en fazla dw_len (30) bar
onceki bir pivot barina gore DAHA MI YUKSEK" diye bakiyor -- yani iki
nokta arasindaki FARK onemli, serinin mutlak baslangic noktasi degil.
Bu yuzden elimizdeki 4H barlardan (close>open ise +volume, close<open
ise -volume) yaklasik bir "cvd_delta" hesaplanip SADECE elimizdeki veri
penceresi icinde .cumsum() aliniyor -- pencere ici GORECELI
karsilastirmalar dogru cikar, mutlak seviye Pine'inkiyle ayni olmasa da
onemli olan bu degil.
"""

import numpy as np
import pandas as pd


def _pivot_low(low: pd.Series, left: int = 6, right: int = 6) -> pd.Series:
    """ta.pivotlow(low,6,6) esdegeri -- confirmed pivot low serisi."""
    n = len(low)
    pl = pd.Series(np.nan, index=low.index)
    l = low.values
    for i in range(left, n - right):
        window = l[i - left:i + right + 1]
        if l[i] == window.min() and np.sum(window == l[i]) == 1:
            pl.iloc[i] = l[i]
    return pl


def _hidden_bull_div(price: pd.Series, indicator: pd.Series,
                     pivot_gate: pd.Series, lookback: int = 30) -> pd.Series:
    """
    f_hiddenDiv() esdegeri.

    [DUZELTME] Pine'in gercek kodu `pr < pr[i] and vol_or_mfi > vol_or_mfi[i]`
    seklinde -- yani `pl_ser[i]` SADECE "confirmed bir pivot var mi" diye
    KAPI (gate) olarak kullaniliyor; fiyat/gosterge karsilastirmasi o
    pivotun GERCEK dip degeriyle degil, confirm barinin KENDI ham
    fiyat/gosterge okumasiyla yapiliyor (`pr[i]`, `volume_or_mfi[i]` --
    ikisi de ayni i offsetinde). Ilk versiyonda yanlislikla pivotun dip
    degeri (`pl_ser[i]`'nin degeri) fiyat karsilastirmasinda
    kullanilmisti -- duzeltildi.
    """
    n = len(price)
    result = np.zeros(n, dtype=bool)
    price_v = price.values
    ind_v = indicator.values
    gate_v = pivot_gate.values  # sadece not-na kontrolu icin kullanilir

    gate_positions = [i for i in range(n) if not np.isnan(gate_v[i])]

    for idx in range(n):
        for pp in gate_positions:
            dist = idx - pp
            if dist <= 0 or dist > lookback:
                continue
            if price_v[idx] < price_v[pp] and ind_v[idx] > ind_v[pp]:
                result[idx] = True
                break
    return pd.Series(result, index=price.index)


def calc_smart_money_droplet(ind: pd.DataFrame, dw_len: int = 30,
                             pivot_left: int = 6, pivot_right: int = 6) -> pd.DataFrame:
    """
    Donen DataFrame: "smart_money_droplet" (bool) -- Pine'daki 💧
    etiketinin cikma anina karsilik gelir (sadece bullish).
    """
    low = ind["low"]
    close = ind["close"]
    open_ = ind["open"]
    rvol = ind["rvol"]
    ema_fast = ind["ema_fast"]
    ema_mid = ind["ema_mid"]
    mfi = ind["mfi"]

    is_bull_bar = close > open_
    is_bear_bar = close < open_
    cvd_buy = ind["volume"].where(is_bull_bar, 0.0)
    cvd_sell = ind["volume"].where(is_bear_bar, 0.0)
    cvd_delta = cvd_buy - cvd_sell
    cum_cvd = cvd_delta.cumsum()

    pivot_low = _pivot_low(low, pivot_left, pivot_right)
    # [FIX] ta.pivotlow(low,6,6), pivotu ANCAK right(6) bar SONRA
    # "biliyor" -- _pivot_low ise degeri pivotun GERCEK barina koyuyor.
    # Bu shift olmadan gelecegi bilen (lookahead) bir divergence testi
    # olurdu. Confirm barina kaydiriyoruz (Pine'in gercek davranisi).
    pivot_low_confirmed = pivot_low.shift(pivot_right)

    bull_cvd_div = _hidden_bull_div(low, cum_cvd, pivot_low_confirmed, dw_len)
    bull_mfi_div = _hidden_bull_div(low, mfi, pivot_low_confirmed, dw_len)

    vol_above = rvol > 1.2
    is_bull_candle = close >= open_
    trend_ok = ema_fast > ema_mid

    droplet = (bull_cvd_div | bull_mfi_div) & vol_above & is_bull_candle & trend_ok

    return pd.DataFrame({
        "smart_money_droplet": droplet,
        "cum_cvd": cum_cvd,
    }, index=ind.index)
