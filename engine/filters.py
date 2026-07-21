"""
engine/filters.py
Pine Script SMP V2.8.2 — Katman 3: Onay Filtreleri

CVD Leviathan Filter (isBullTrap / isBearTrap)

NOT: "Zombi İşlem Kesici" bu canli filtre hattinda kullanilmiyor -- exit
mantigi olarak sadece backtest/runner.py icinde uygulaniyor. Eskiden
burada da kullanilmayan bir apply_zombie_killer() kopyasi vardi,
temizlendi.
"""

import pandas as pd


def calc_cvd_filter(open_: pd.Series, close: pd.Series, volume: pd.Series,
                    z_window: int = 50) -> pd.DataFrame:
    """
    CVD (Cumulative Volume Delta) Leviathan Filtresi.

    Pine Script'te lower_tf request ile hesaplanır, biz yaklaşık değer kullanıyoruz:
      cvd_delta ≈ (close > open) ? volume : -(close < open) ? volume : 0

    Pine:
      isBullTrap = (cvd_delta < 0) and (z_sell > 1.5)
      isBearTrap = (cvd_delta > 0) and (z_buy > 1.5)
    """
    is_bull_bar = close > open_
    is_bear_bar = close < open_

    cvd_buy_vol  = volume.where(is_bull_bar, 0.0)
    cvd_sell_vol = volume.where(is_bear_bar, 0.0)
    cvd_delta    = cvd_buy_vol - cvd_sell_vol

    # Z-score hesabı
    def zscore(series, window):
        mean = series.rolling(window).mean()
        std  = series.rolling(window).std()
        return (series - mean) / (std + 1e-6)

    z_sell = zscore(cvd_sell_vol, z_window)
    z_buy  = zscore(cvd_buy_vol, z_window)

    is_bull_trap = (cvd_delta < 0) & (z_sell > 1.5)
    is_bear_trap = (cvd_delta > 0) & (z_buy > 1.5)

    # CVD onay filtreleri (Pine: cvdOkBull = not isBullTrap)
    cvd_ok_bull = ~is_bull_trap
    cvd_ok_bear = ~is_bear_trap

    return pd.DataFrame({
        "cvd_delta": cvd_delta,
        "cvd_ok_bull": cvd_ok_bull,
        "cvd_ok_bear": cvd_ok_bear,
        "is_bull_trap": is_bull_trap,
        "is_bear_trap": is_bear_trap,
    })


def apply_all_filters(ind: pd.DataFrame, signals: pd.DataFrame,
                      use_cvd: bool = True) -> pd.DataFrame:
    """
    Tüm filtreleri uygula ve temizlenmiş sinyalleri döndür.
    """
    # CVD filtresi
    cvd = calc_cvd_filter(ind["open"], ind["close"], ind["volume"])

    if use_cvd:
        clean_buy  = signals["raw_buy"]  & cvd["cvd_ok_bull"]
        clean_sell = signals["raw_sell"] & cvd["cvd_ok_bear"]
    else:
        clean_buy  = signals["raw_buy"]
        clean_sell = signals["raw_sell"]

    # Son filtre: aynı yönde çift sinyal engelle (Pine: lastDir != 1)
    buy_final  = pd.Series(False, index=ind.index)
    sell_final = pd.Series(False, index=ind.index)
    last_dir = 0

    for i in range(len(ind)):
        if clean_buy.iloc[i] and last_dir != 1:
            buy_final.iloc[i] = True
            last_dir = 1
        elif clean_sell.iloc[i] and last_dir != -1:
            sell_final.iloc[i] = True
            last_dir = -1

    result = pd.DataFrame({
        "buy_signal": buy_final,
        "sell_signal": sell_final,
        "cvd_ok_bull": cvd["cvd_ok_bull"],
        "cvd_ok_bear": cvd["cvd_ok_bear"],
        "is_bull_trap": cvd["is_bull_trap"],
        "is_bear_trap": cvd["is_bear_trap"],
    })

    return result


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from data.fetcher import fetch_ohlcv
    from engine.indicators import compute_all_indicators
    from engine.signals import calc_bull_bear_score, calc_triggers, generate_signals

    df  = fetch_ohlcv("BTC-USD", interval="4h", period="1y")
    ind = compute_all_indicators(df, preset="Default")
    sc  = calc_bull_bear_score(ind)
    tr  = calc_triggers(ind, sc)
    sg  = generate_signals(ind, sc, tr, preset="Default")

    final = apply_all_filters(ind, sg)

    print(f"Temiz BUY sinyali: {final['buy_signal'].sum()}")
    print(f"Temiz SELL sinyali: {final['sell_signal'].sum()}")

    buy_bars = df[final["buy_signal"]]
    print(f"\nBUY sinyal tarihleri:\n{buy_bars.index.tolist()[-5:]}")
