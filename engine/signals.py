"""
engine/signals.py
SMP V3.1 — V2.8.2 + VPA Climax filtresi + Zone POC bonus puan
         + [FIX] HTF bias (self-lag, Pine "HTF Filtresi: Grafik" esdegeri)
         + [FIX] MTF Likidite Matrix entegrasyonu (mtf_ok_bull/bear,
           recent_demand/supply_bounce, mtf_score_bull/bear)
         + [FIX] isLTF=False (4H'de calisildigi icin) sabit +0.5
           muratConfirm bonusu
"""

import pandas as pd
import numpy as np


def _self_htf_bias(ind):
    """
    Pine: htfFast = request.security(tickerid, i_htfTF, ema(close,effFast)[1])
          htfSlow = request.security(tickerid, i_htfTF, ema(close,effMid)[1])
          htfBias = htfFast>htfSlow ? 1 : htfFast<htfSlow ? -1 : 0

    TradingView ekran goruntusunde "HTF Filtresi (bos=mevcut)" = "Grafik"
    -- yani i_htfTF bos string, bu da Pine'da request.security'nin
    GRAFIGIN KENDI zaman dilimini kullanmasi anlamina gelir. Yani "HTF"
    aslinda ayni 4H seri, sadece 1 bar geriden (repaint engelleme icin
    [1] kaydirmasi) bakiliyor. Gercek bir ust zaman dilimi DEGIL.
    """
    ema_fast_prev = ind["ema_fast"].shift(1)
    ema_mid_prev = ind["ema_mid"].shift(1)
    bias = pd.Series(0, index=ind.index)
    bias = bias.mask(ema_fast_prev > ema_mid_prev, 1)
    bias = bias.mask(ema_fast_prev < ema_mid_prev, -1)
    return bias


def calc_bull_bear_score(ind, htf_bias="auto", use_bb_filter=False,
                         mtf=None, is_ltf=False):
    """
    htf_bias:
      "auto" (varsayilan) -> Pine'daki "HTF Filtresi: Grafik (bos)"
             ayarinin esdegeri: ema_fast/ema_mid'in 1 bar gecikmeli
             kendi karsilastirmasi (bkz. _self_htf_bias).
      None   -> HTF bonusu tamamen kapali (eski davranis).
      pd.Series(1/-1/0) -> gercek bir ust zaman dilimi bias'i verilirse
             onu kullan.

    mtf: engine.liquidity_mtf.build_mtf_confirmation() ciktisi (veya
         None). None ise MTF Likidite Matrix bayraklari devre disi
         kalir: recent_demand/supply_bounce=False, mtf_score_bull/
         bear=0 -- yani Pine'daki i_useMTF=false davranisinin esdegeri.

    is_ltf: Pine'daki isLTF = tfMin < 60. Bu bot 4H calistigi icin
         varsayilan False -> muratConfirmBull/Bear sabit True -> her
         iki skora da +0.5 sabit bonus eklenir (Pine'da tam olarak
         boyle davraniyor cunku isLTF False oldugunda kosul dogrudan
         True'ya esitleniyor).
    """
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

    # [FIX] Pine'daki wBuyRecent/wSellRecent = barsSinceW<=21 and
    # close karsi tarafi kirmadi. ind icinde engine/indicators.py bu
    # kolonlari (w_buy_recent/w_sell_recent) uretiyorsa onlari kullan;
    # uretmiyorsa (indicators.py henuz guncellenmediyse) eski kaba
    # rolling(21) yaklasimina geri dus -- kod calismaya devam eder,
    # sadece bu tek nokta biraz daha az hassas olur.
    if "w_buy_recent" in ind.columns:
        w_bull_recent = ind["w_buy_recent"].astype(bool)
    else:
        w_bull_recent = ind["is_whale_buy"].rolling(21).max().astype(bool)
    if "w_sell_recent" in ind.columns:
        w_sell_recent = ind["w_sell_recent"].astype(bool)
    else:
        w_sell_recent = ind["is_whale_sell"].rolling(21).max().astype(bool)

    if htf_bias is None:
        htf_bull = pd.Series(False, index=ind.index)
        htf_bear = pd.Series(False, index=ind.index)
    elif isinstance(htf_bias, str) and htf_bias == "auto":
        bias = _self_htf_bias(ind)
        htf_bull = bias == 1
        htf_bear = bias == -1
    else:
        htf_bull = htf_bias == 1
        htf_bear = htf_bias == -1

    # Zone POC bonus — filtre degil, puan
    poc_bull = ind["above_poc"] if "above_poc" in ind.columns else pd.Series(False, index=ind.index)
    poc_bear = ind["below_poc"] if "below_poc" in ind.columns else pd.Series(False, index=ind.index)

    # [FIX] MTF Likidite Matrix — recent_demand/supply_bounce (+1.0 puan)
    if mtf is not None:
        recent_demand_bounce = mtf["recent_demand_bounce"].reindex(ind.index).fillna(False).astype(bool)
        recent_supply_bounce = mtf["recent_supply_bounce"].reindex(ind.index).fillna(False).astype(bool)
        mtf_score_bull = mtf["mtf_score_bull"].reindex(ind.index).fillna(0.0)
        mtf_score_bear = mtf["mtf_score_bear"].reindex(ind.index).fillna(0.0)
        mtf_ok_bull = mtf["mtf_ok_bull"].reindex(ind.index).fillna(True).astype(bool)
        mtf_ok_bear = mtf["mtf_ok_bear"].reindex(ind.index).fillna(True).astype(bool)
    else:
        recent_demand_bounce = pd.Series(False, index=ind.index)
        recent_supply_bounce = pd.Series(False, index=ind.index)
        mtf_score_bull = pd.Series(0.0, index=ind.index)
        mtf_score_bear = pd.Series(0.0, index=ind.index)
        mtf_ok_bull = pd.Series(True, index=ind.index)
        mtf_ok_bear = pd.Series(True, index=ind.index)

    # [FIX] isLTF=False (4H) -> muratConfirmBull/Bear Pine'da sabit True
    if is_ltf:
        murat_confirm_bull = ind["ema_mid"] > ind["ema_slow"]
        murat_confirm_bear = ind["ema_mid"] < ind["ema_slow"]
    else:
        murat_confirm_bull = pd.Series(True, index=ind.index)
        murat_confirm_bear = pd.Series(True, index=ind.index)

    bull = (
        ema_bull_cross.astype(float)          * 1.0 +
        close_above_slow.astype(float)        * 1.0 +
        rsi_bull_zone.astype(float)           * 1.0 +
        macd_bull.astype(float)               * 1.0 +
        above_vwap.astype(float)              * 1.0 +
        vol_above.astype(float)               * 1.0 +
        bull_dmi.astype(float)                * 1.0 +
        htf_bull.astype(float)                * 1.5 +
        w_bull_recent.astype(float)           * 1.0 +
        (ind["rvol"] > 3.0).astype(float)     * 1.5 +
        poc_bull.astype(float)                * 1.0 +
        recent_demand_bounce.astype(float)    * 1.0 +
        murat_confirm_bull.astype(float)      * 0.5
    )

    bear = (
        ema_bear_cross.astype(float)          * 1.0 +
        close_below_slow.astype(float)        * 1.0 +
        rsi_bear_zone.astype(float)           * 1.0 +
        macd_bear.astype(float)               * 1.0 +
        below_vwap.astype(float)              * 1.0 +
        vol_above.astype(float)               * 1.0 +
        bear_dmi.astype(float)                * 1.0 +
        htf_bear.astype(float)                * 1.5 +
        w_sell_recent.astype(float)           * 1.0 +
        (ind["rvol"] > 3.0).astype(float)     * 1.5 +
        poc_bear.astype(float)                * 1.0 +
        recent_supply_bounce.astype(float)    * 1.0 +
        murat_confirm_bear.astype(float)      * 0.5
    )

    if use_bb_filter:
        bull += (ind["close"] > ind["bb_mid"]).astype(float) * 0.5
        bear += (ind["close"] < ind["bb_mid"]).astype(float) * 0.5

    return pd.DataFrame({
        "bull_score":            bull,
        "bear_score":            bear,
        "bull_dmi":              bull_dmi,
        "bear_dmi":              bear_dmi,
        "w_bull_recent":         w_bull_recent,
        "w_sell_recent":         w_sell_recent,
        "macd_bull":             macd_bull,
        "macd_bear":             macd_bear,
        "recent_demand_bounce":  recent_demand_bounce,
        "recent_supply_bounce":  recent_supply_bounce,
        "mtf_score_bull":        mtf_score_bull,
        "mtf_score_bear":        mtf_score_bear,
        "mtf_ok_bull":           mtf_ok_bull,
        "mtf_ok_bear":           mtf_ok_bear,
    })


def calc_triggers(ind, scores):
    """
    [FIX] Pine: triggerBull = (emaBullX and macdBull) or
                              (crossover(close,emaFast) and recentDemandBounce)
    Eskiden burada "recentDemandBounce" yerine yanlislikla whale-bazli
    "w_bull_15" (son 15 barda whale alimi) kullaniliyordu -- bu, MTF
    likidite sweep'iyle ilgisi olmayan, tamamen farkli bir kosuldu.
    Artik scores["recent_demand_bounce"] (zaten "son 15 barda" durumunu
    iceriyor) dogrudan kullaniliyor.
    """
    ema_bull_x = (ind["ema_fast"] > ind["ema_mid"]) & (ind["ema_fast"].shift(1) <= ind["ema_mid"].shift(1))
    ema_bear_x = (ind["ema_fast"] < ind["ema_mid"]) & (ind["ema_fast"].shift(1) >= ind["ema_mid"].shift(1))
    close_cross_up = (ind["close"] > ind["ema_fast"]) & (ind["close"].shift(1) <= ind["ema_fast"].shift(1))
    close_cross_dn = (ind["close"] < ind["ema_fast"]) & (ind["close"].shift(1) >= ind["ema_fast"].shift(1))

    trigger_bull = (ema_bull_x & scores["macd_bull"]) | (close_cross_up & scores["recent_demand_bounce"])
    trigger_bear = (ema_bear_x & scores["macd_bear"]) | (close_cross_dn & scores["recent_supply_bounce"])

    return pd.DataFrame({
        "trigger_bull": trigger_bull,
        "trigger_bear": trigger_bear,
    })


def generate_signals(ind, scores, triggers, preset="Default",
                     eff_score=5.0, grade_filter="All",
                     use_vsa=True, is_scalp_mode=False, min_conf=2):

    # [FIX] Pine: isAnaTrendBull = (close>ema200) or (recentDemandBounce and wBuyRecent)
    if "w_buy_recent" in ind.columns:
        w_buy_recent_col = ind["w_buy_recent"].astype(bool)
    else:
        w_buy_recent_col = scores["w_bull_recent"]
    if "w_sell_recent" in ind.columns:
        w_sell_recent_col = ind["w_sell_recent"].astype(bool)
    else:
        w_sell_recent_col = scores["w_sell_recent"]

    ana_trend_bull = (ind["close"] > ind["ema_200"]) | (scores["recent_demand_bounce"] & w_buy_recent_col)
    ana_trend_bear = (ind["close"] < ind["ema_200"]) | (scores["recent_supply_bounce"] & w_sell_recent_col)

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
        w_buy_recent_col.astype(int)
    )
    conf_bear = (
        (ind["rsi"] < 50).astype(int) +
        scores["macd_bear"].astype(int) +
        scores["bear_dmi"].astype(int) +
        (ind["close"] < ind["vwap"]).astype(int) +
        w_sell_recent_col.astype(int)
    )

    conf_ok_bull = conf_bull >= min_conf
    conf_ok_bear = conf_bear >= min_conf

    def pass_grade(score):
        if grade_filter == "A+ Only":
            return score >= 8.0
        elif grade_filter == "A+ and A":
            return score >= 6.5
        return score >= 5.0

    # [FIX] Pine: (finalBullScore + mtfScoreBull) >= effScore ve
    # fPassGrade(finalBullScore + mtfScoreBull) -- yani MTF confluence
    # bonusu esik/grade kontrolunden ONCE skora ekleniyor.
    total_bull_score = scores["bull_score"] + scores["mtf_score_bull"]
    total_bear_score = scores["bear_score"] + scores["mtf_score_bear"]

    grade_ok_bull = total_bull_score.apply(pass_grade)
    grade_ok_bear = total_bear_score.apply(pass_grade)
    score_ok_bull = total_bull_score >= eff_score
    score_ok_bear = total_bear_score >= eff_score

    if use_vsa and is_scalp_mode:
        vsa_ok_bull = ~(ind["vsa_bc"] | ind["vsa_ut"])
        vsa_ok_bear = ~(ind["vsa_sc"] | ind["vsa_spr"] | ind["vsa_dt"])
    else:
        vsa_ok_bull = pd.Series(True, index=ind.index)
        vsa_ok_bear = pd.Series(True, index=ind.index)

    # VPA Climax filtresi
    vpa_no_buy  = ~ind["buying_climax"]  if "buying_climax"  in ind.columns else pd.Series(True, index=ind.index)
    vpa_no_sell = ~ind["selling_climax"] if "selling_climax" in ind.columns else pd.Series(True, index=ind.index)

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
        vsa_ok_bull &
        vpa_no_buy &
        scores["mtf_ok_bull"]
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
        vsa_ok_bear &
        vpa_no_sell &
        scores["mtf_ok_bear"]
    )

    return pd.DataFrame({
        "raw_buy":   raw_buy,
        "raw_sell":  raw_sell,
        "conf_bull": conf_bull,
        "conf_bear": conf_bear,
        "total_bull_score": total_bull_score,
        "total_bear_score": total_bear_score,
    })