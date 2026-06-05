"""
engine/signals.py
SMP V3.0 — Sinyal Motoru
TST bonus puan, VPA filtre kaldirildi
"""

import pandas as pd
import numpy as np


def calc_bull_bear_score(ind, htf_bias=None, use_bb_filter=False):
    ema_bull_cross   = ind["ema_fast"] > ind["ema_mid"]
    ema_bear_cross   = ind["ema_fast"] < ind["ema_mid"]
    close_above_slow = ind["close"] > ind["ema_slow"]
    close_below_slow = ind["close"] < ind["ema_slow"]
    rsi_bull_zone    = (ind["rsi"] > 50) & (ind["rsi"] < 75)
    rsi_bear_zone    = (ind["rsi"] < 50) & (ind["rsi"] > 25)
    macd_bull = (ind["macd_hist"] > 0) & (ind["macd_line"] > ind["macd_signal"])
    macd_bear = (ind["macd_hist"] < 0) & (ind["macd_line"] < ind["macd_signal"])
    above_vwap    = ind["close"] > ind["vwap"]
    below_vwap    = ind["close"] < ind["vwap"]
    vol_above     = ind["rvol"] > 1.2
    adx_strong    = ind["adx"] > 20
    bull_dmi      = adx_strong & (ind["di_plus"] > ind["di_minus"])
    bear_dmi      = adx_strong & (ind["di_minus"] > ind["di_plus"])
    w_bull_recent = ind["is_whale_buy"].rolling(21).max().astype(bool)
    w_sell_recent = ind["is_whale_sell"].rolling(21).max().astype(bool)

    if htf_bias is None:
        htf_bull = pd.Series(False, index=ind.index)
        htf_bear = pd.Series(False, index=ind.index)
    else:
        htf_bull = htf_bias == 1
        htf_bear = htf_bias == -1

    tst_bull = ind["tst_bull"] if "tst_bull" in ind.columns else pd.Series(False, index=ind.index)
    tst_bear = ind["tst_bear"] if "tst_bear" in ind.columns else pd.Series(False, index=ind.index)
    adf_ok   = ind["adf_ok"]   if "adf_ok"   in ind.columns else pd.Series(True,  index=ind.index)

    bull = (
        ema_bull_cross.astype(float)      * 1.0 +
        close_above_slow.astype(float)    * 1.0 +
        rsi_bull_zone.astype(float)       * 1.0 +
        macd_bull.astype(float)           * 1.0 +
        above_vwap.astype(float)          * 1.0 +
        vol_above.astype(float)           * 1.0 +
        bull_dmi.astype(float)            * 1.0 +
        htf_bull.astype(float)            * 1.5 +
        w_bull_recent.astype(float)       * 1.0 +
        (ind["rvol"] > 3.0).astype(float) * 1.5 +
        tst_bull.astype(float)            * 2.0
    )

    bear = (
        ema_bear_cross.astype(float)      * 1.0 +
        close_below_slow.astype(float)    * 1.0 +
        rsi_bear_zone.astype(float)       * 1.0 +
        macd_bear.astype(float)           * 1.0 +
        below_vwap.astype(float)          * 1.0 +
        vol_above.astype(float)           * 1.0 +
        bear_dmi.astype(float)            * 1.0 +
        htf_bear.astype(float)            * 1.5 +
        w_sell_recent.astype(float)       * 1.0 +
        (ind["rvol"] > 3.0).astype(float) * 1.5 +
        tst_bear.astype(float)            * 2.0
    )

    if use_bb_filter:
        bull += (ind["close"] > ind["bb_mid"]).astype(float) * 0.5
        bear += (ind["close"] < ind["bb_mid"]).astype(float) * 0.5

    return pd.DataFrame({
        "bull_score":    bull,
        "bear_score":    bear,
        "bull_dmi":      bull_dmi,
        "bear_dmi":      bear_dmi,
        "w_bull_recent": w_bull_recent,
        "w_sell_recent": w_sell_recent,
        "macd_bull":     macd_bull,
        "macd_bear":     macd_bear,
        "tst_bull":      tst_bull,
        "tst_bear":      tst_bear,
        "adf_ok":        adf_ok,
    })


def calc_triggers(ind, scores):
    ema_bull_x = (ind["ema_fast"] > ind["ema_mid"]) & (ind["ema_fast"].shift(1) <= ind["ema_mid"].shift(1))
    ema_bear_x = (ind["ema_fast"] < ind["ema_mid"]) & (ind["ema_fast"].shift(1) >= ind["ema_mid"].shift(1))
    close_cross_up = (ind["close"] > ind["ema_fast"]) & (ind["close"].shift(1) <= ind["ema_fast"].shift(1))
    close_cross_dn = (ind["close"] < ind["ema_fast"]) & (ind["close"].shift(1) >= ind["ema_fast"].shift(1))
    w_bull_15 = scores["w_bull_recent"].rolling(15).max().astype(bool)
    w_sell_15 = scores["w_sell_recent"].rolling(15).max().astype(bool)

    trigger_bull = (ema_bull_x & scores["macd_bull"]) | (close_cross_up & w_bull_15)
    trigger_bear = (ema_bear_x & scores["macd_bear"]) | (close_cross_dn & w_sell_15)

    return pd.DataFrame({
        "trigger_bull": trigger_bull,
        "trigger_bear": trigger_bear,
    })


def generate_signals(ind, scores, triggers, preset="Default",
                     eff_score=5.0, grade_filter="All",
                     use_vsa=True, is_scalp_mode=False, min_conf=2):

    ana_trend_bull   = ind["close"] > ind["ema_200"]
    ana_trend_bear   = ind["close"] < ind["ema_200"]
    body             = (ind["close"] - ind["open"]).abs()
    avg_body         = body.rolling(14).mean()
    is_strong_candle = body > avg_body * 0.7
    wick_up = ind["high"] - pd.concat([ind["close"], ind["open"]], axis=1).max(axis=1)
    wick_dn = pd.concat([ind["close"], ind["open"]], axis=1).min(axis=1) - ind["low"]
    wick_reject_up = wick_up > body * 1.5
    wick_reject_dn = wick_dn > body * 1.5
    adx_lock_bull  = ~scores["bear_dmi"]
    adx_lock_bear  = ~scores["bull_dmi"]

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

    def pass_grade(score):
        if grade_filter == "A+ Only":
            return score >= 8.0
        elif grade_filter == "A+ and A":
            return score >= 6.5
        return score >= 5.0

    grade_ok_bull = scores["bull_score"].apply(pass_grade)
    grade_ok_bear = scores["bear_score"].apply(pass_grade)
    score_ok_bull = scores["bull_score"] >= eff_score
    score_ok_bear = scores["bear_score"] >= eff_score

    if use_vsa and is_scalp_mode:
        vsa_ok_bull = ~(ind["vsa_bc"] | ind["vsa_ut"])
        vsa_ok_bear = ~(ind["vsa_sc"] | ind["vsa_spr"] | ind["vsa_dt"])
    else:
        vsa_ok_bull = pd.Series(True, index=ind.index)
        vsa_ok_bear = pd.Series(True, index=ind.index)

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

    return pd.DataFrame({
        "raw_buy":   raw_buy,
        "raw_sell":  raw_sell,
        "conf_bull": conf_bull,
        "conf_bear": conf_bear,
    })