"""
engine/signals.py
Pine Script SMP V2.8.2 — Sinyal Motoru (Katman 2)

buySignal ve sellSignal üretir.
Pine Script'teki bullScore/bearScore + triggerBull/triggerBear mantığını uygular.
"""

import pandas as pd
import numpy as np


def calc_bull_bear_score(ind: pd.DataFrame,
                         htf_bias: pd.Series = None,
                         use_bb_filter: bool = False) -> pd.DataFrame:
    """
    Pine Script bullScore / bearScore hesabı — birebir port.
    
    ind: compute_all_indicators() çıktısı
    htf_bias: üst TF'den gelen +1/0/-1 serisi (opsiyonel)
    """
    
    # EMA karşılaştırmaları
    ema_bull_cross = ind["ema_fast"] > ind["ema_mid"]
    ema_bear_cross = ind["ema_fast"] < ind["ema_mid"]
    
    close_above_slow = ind["close"] > ind["ema_slow"]
    close_below_slow = ind["close"] < ind["ema_slow"]
    
    # RSI bölgeleri
    rsi_bull_zone = (ind["rsi"] > 50) & (ind["rsi"] < 75)
    rsi_bear_zone = (ind["rsi"] < 50) & (ind["rsi"] > 25)
    
    # MACD
    macd_bull = (ind["macd_hist"] > 0) & (ind["macd_line"] > ind["macd_signal"])
    macd_bear = (ind["macd_hist"] < 0) & (ind["macd_line"] < ind["macd_signal"])
    
    # VWAP
    above_vwap = ind["close"] > ind["vwap"]
    below_vwap = ind["close"] < ind["vwap"]
    
    # Hacim
    vol_above = ind["rvol"] > 1.2
    
    # ADX/DMI
    adx_strong = ind["adx"] > 20
    bull_dmi = adx_strong & (ind["di_plus"] > ind["di_minus"])
    bear_dmi = adx_strong & (ind["di_minus"] > ind["di_plus"])
    
    # Whale onayı
    w_bull_recent = ind["is_whale_buy"].rolling(21).max().astype(bool)
    w_sell_recent = ind["is_whale_sell"].rolling(21).max().astype(bool)
    
    # HTF bias (Pine: htfBias == 1 → +1.5 puan)
    if htf_bias is None:
        htf_bull = pd.Series(False, index=ind.index)
        htf_bear = pd.Series(False, index=ind.index)
    else:
        htf_bull = htf_bias == 1
        htf_bear = htf_bias == -1
    
    # ── BULL SCORE ──
    bull = (
        ema_bull_cross.astype(float) * 1.0 +
        close_above_slow.astype(float) * 1.0 +
        rsi_bull_zone.astype(float) * 1.0 +
        macd_bull.astype(float) * 1.0 +
        above_vwap.astype(float) * 1.0 +
        vol_above.astype(float) * 1.0 +
        bull_dmi.astype(float) * 1.0 +
        htf_bull.astype(float) * 1.5 +
        w_bull_recent.astype(float) * 1.0 +
        (ind["rvol"] > 3.0).astype(float) * 1.5
    )
    
    # BB filtresi (opsiyonel)
    if use_bb_filter:
        bull += (ind["close"] > ind["bb_mid"]).astype(float) * 0.5
    
    # ── BEAR SCORE ──
    bear = (
        ema_bear_cross.astype(float) * 1.0 +
        close_below_slow.astype(float) * 1.0 +
        rsi_bear_zone.astype(float) * 1.0 +
        macd_bear.astype(float) * 1.0 +
        below_vwap.astype(float) * 1.0 +
        vol_above.astype(float) * 1.0 +
        bear_dmi.astype(float) * 1.0 +
        htf_bear.astype(float) * 1.5 +
        w_sell_recent.astype(float) * 1.0 +
        (ind["rvol"] > 3.0).astype(float) * 1.5
    )
    
    if use_bb_filter:
        bear += (ind["close"] < ind["bb_mid"]).astype(float) * 0.5
    
    return pd.DataFrame({
        "bull_score": bull,
        "bear_score": bear,
        "bull_dmi": bull_dmi,
        "bear_dmi": bear_dmi,
        "w_bull_recent": w_bull_recent,
        "w_sell_recent": w_sell_recent,
        "macd_bull": macd_bull,
        "macd_bear": macd_bear,
    })


def calc_triggers(ind: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """
    Pine Script'teki triggerBull / triggerBear mantığı.
    
    Pine:
      triggerBull = (emaBullX and macdBull) or (crossover(close, emaFast) and recentDemandBounce)
      triggerBear = (emaBearX and macdBear) or (crossunder(close, emaFast) and recentSupplyBounce)
    """
    # EMA crossover (vectorize edilmiş)
    ema_bull_x = (ind["ema_fast"] > ind["ema_mid"]) & (ind["ema_fast"].shift(1) <= ind["ema_mid"].shift(1))
    ema_bear_x = (ind["ema_fast"] < ind["ema_mid"]) & (ind["ema_fast"].shift(1) >= ind["ema_mid"].shift(1))
    
    # close vs emaFast crossover
    close_cross_up = (ind["close"] > ind["ema_fast"]) & (ind["close"].shift(1) <= ind["ema_fast"].shift(1))
    close_cross_dn = (ind["close"] < ind["ema_fast"]) & (ind["close"].shift(1) >= ind["ema_fast"].shift(1))
    
    # "recent demand/supply bounce" — son 15 barda whale veya MTF sweep
    w_bull_15 = scores["w_bull_recent"].rolling(15).max().astype(bool)
    w_sell_15 = scores["w_sell_recent"].rolling(15).max().astype(bool)
    
    trigger_bull = (ema_bull_x & scores["macd_bull"]) | (close_cross_up & w_bull_15)
    trigger_bear = (ema_bear_x & scores["macd_bear"]) | (close_cross_dn & w_sell_15)
    
    return pd.DataFrame({
        "trigger_bull": trigger_bull,
        "trigger_bear": trigger_bear,
    })


def generate_signals(ind: pd.DataFrame,
                     scores: pd.DataFrame,
                     triggers: pd.DataFrame,
                     preset: str = "Default",
                     eff_score: float = 5.0,
                     grade_filter: str = "All",
                     use_vsa: bool = True,
                     is_scalp_mode: bool = False,
                     min_conf: int = 2) -> pd.DataFrame:
    """
    Ham sinyalleri (rawBuy / rawSell) üret, ardından son filtrelerden geçir.
    
    Pine Script:
      rawBuy = triggerBull and isAnaTrendBull and isStrongCandle and adxLockBull
               and not isWickRejectUp and rsi<80 and score>=effScore
               and fPassGrade and confOkBull and cvdOkBull and mtfOkBull
               and ltfOkBull and vsaOkBull
    """
    
    # Ana trend (Pine: close > ema200)
    ana_trend_bull = ind["close"] > ind["ema_200"]
    ana_trend_bear = ind["close"] < ind["ema_200"]
    
    # Güçlü mum (body > avg_body * 0.7)
    body = (ind["close"] - ind["open"]).abs()
    avg_body = body.rolling(14).mean()
    is_strong_candle = body > avg_body * 0.7
    
    # Wick rejection (sahte kırılım filtresi)
    wick_up = ind["high"] - pd.concat([ind["close"], ind["open"]], axis=1).max(axis=1)
    wick_dn = pd.concat([ind["close"], ind["open"]], axis=1).min(axis=1) - ind["low"]
    wick_reject_up = wick_up > body * 1.5
    wick_reject_dn = wick_dn > body * 1.5
    
    # ADX lock (Pine: adxLockBull = not bearDMI)
    adx_lock_bull = ~scores["bear_dmi"]
    adx_lock_bear = ~scores["bull_dmi"]
    
    # Confirmation sayacı (Pine: confBull, confBear)
    conf_bull = (
        (ind["rsi"] > 50).astype(int) +
        scores["macd_bull"].astype(int) +
        scores["bull_dmi"].astype(int) +
        (ind["close"] > ind["vwap"]).astype(int) +
        scores["w_bull_recent"].astype(int)
    )
    conf_bear = (
        (ind["rsi"] < 50).astype(int) +
        scores["macd_bear"].astype(int) +
        scores["bear_dmi"].astype(int) +
        (ind["close"] < ind["vwap"]).astype(int) +
        scores["w_sell_recent"].astype(int)
    )
    
    conf_ok_bull = conf_bull >= min_conf
    conf_ok_bear = conf_bear >= min_conf
    
    # Grade filtresi
    def pass_grade(score):
        if grade_filter == "A+ Only":
            return score >= 8.0
        elif grade_filter == "A+ and A":
            return score >= 6.5
        else:
            return score >= 5.0  # C grade'i gizle
    
    grade_ok_bull = scores["bull_score"].apply(pass_grade)
    grade_ok_bear = scores["bear_score"].apply(pass_grade)
    
    # Score eşiği
    score_ok_bull = scores["bull_score"] >= eff_score
    score_ok_bear = scores["bear_score"] >= eff_score
    
    # VSA zırhı (sadece scalp modunda aktif)
    if use_vsa and is_scalp_mode:
        vsa_ok_bull = ~(ind["vsa_bc"] | ind["vsa_ut"])
        vsa_ok_bear = ~(ind["vsa_sc"] | ind["vsa_spr"] | ind["vsa_dt"])
    else:
        vsa_ok_bull = pd.Series(True, index=ind.index)
        vsa_ok_bear = pd.Series(True, index=ind.index)
    
    # ── RAW SİNYALLER ──
    raw_buy = (
        triggers["trigger_bull"] &
        ana_trend_bull &
        is_strong_candle &
        adx_lock_bull &
        ~wick_reject_up &
        (ind["rsi"] < 80) &
        score_ok_bull &
        grade_ok_bull &
        conf_ok_bull &
        vsa_ok_bull
    )
    
    raw_sell = (
        triggers["trigger_bear"] &
        ana_trend_bear &
        is_strong_candle &
        adx_lock_bear &
        ~wick_reject_dn &
        (ind["rsi"] > 20) &
        score_ok_bear &
        grade_ok_bear &
        conf_ok_bear &
        vsa_ok_bear
    )
    
    # Pine: buySignal = rawBuy and lastDir != 1 (aynı yönde tekrar girmeme)
    # Bunu backtest runner'da hallederiz, burada rawBuy/rawSell döndür
    return pd.DataFrame({
        "raw_buy": raw_buy,
        "raw_sell": raw_sell,
        "conf_bull": conf_bull,
        "conf_bear": conf_bear,
    })


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from data.fetcher import fetch_ohlcv
    from engine.indicators import compute_all_indicators
    
    df = fetch_ohlcv("BTC-USD", interval="4h", period="1y")
    ind = compute_all_indicators(df, preset="Default")
    scores = calc_bull_bear_score(ind)
    triggers = calc_triggers(ind, scores)
    signals = generate_signals(ind, scores, triggers, preset="Default")
    
    total_buy  = signals["raw_buy"].sum()
    total_sell = signals["raw_sell"].sum()
    print(f"Toplam BUY sinyali: {total_buy}")
    print(f"Toplam SELL sinyali: {total_sell}")
    print(f"Son 5 bar:\n{signals[['raw_buy','raw_sell','conf_bull','conf_bear']].tail()}")
