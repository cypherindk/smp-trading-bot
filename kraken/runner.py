"""
kraken/runner.py
Project KRAKEN V16 — BIST100 FINAL V4
SL=nan sorunu duzeltildi (scan_today artik 5y kullaniyor)
"""

import sys
import os
import pandas as pd
import numpy as np
import vectorbt as vbt
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetcher import fetch_ohlcv
from kraken.indicators import compute_kraken_indicators

BIST100 = [
    "AEFES.IS", "AGHOL.IS", "AKBNK.IS", "AKFGY.IS", "AKSA.IS",
    "AKSEN.IS", "ALARK.IS", "ALFAS.IS", "ARCLK.IS", "ASELS.IS",
    "ASTOR.IS", "BERA.IS", "BIMAS.IS", "BRSAN.IS", "BRYAT.IS",
    "BUCIM.IS", "CCOLA.IS", "CIMSA.IS", "DOAS.IS", "DOHOL.IS",
    "ECILC.IS", "EGEEN.IS", "EKGYO.IS", "ENJSA.IS", "ENKAI.IS",
    "EREGL.IS", "EUPWR.IS", "FROTO.IS", "GARAN.IS", "GESAN.IS",
    "GUBRF.IS", "HALKB.IS", "HEKTS.IS", "ISCTR.IS", "ISGYO.IS",
    "ISMEN.IS", "IZMDC.IS", "KARSN.IS", "KAYSE.IS", "KCHOL.IS",
    "KLSER.IS", "KONTR.IS", "KORDS.IS", "KRDMD.IS",
    "MAVI.IS", "MGROS.IS", "MIATK.IS", "ODAS.IS", "OTKAR.IS",
    "OYAKC.IS", "PASEU.IS", "PETKM.IS", "PGSUS.IS", "QUAGR.IS",
    "REEDR.IS", "SAHOL.IS", "SASA.IS", "SAYAS.IS", "SISE.IS",
    "SKBNK.IS", "SMRTG.IS", "SOKM.IS", "TABGD.IS", "TAVHL.IS",
    "TCELL.IS", "THYAO.IS", "TOASO.IS", "TSKB.IS", "TTKOM.IS",
    "TTRAK.IS", "TUKAS.IS", "TUPRS.IS", "TURSG.IS", "ULKER.IS",
    "VAKBN.IS", "VESBE.IS", "VESTL.IS", "YEOTK.IS", "YKBNK.IS",
    "YYLGD.IS", "ZOREN.IS",
]

DEFAULT_PARAMS = {
    "yz_len": 20, "piv_len": 5, "cmf_len": 21,
    "train_data": 100, "st_mult": 1.5,
    "rr_target": 2.5, "trap_tolerance": 15, "fvg_mult": 0.3,
}


def run_single_backtest(symbol, period="5y", interval="1d",
                        position_pct=10,
                        initial_capital=100000,
                        commission_pct=0.04,
                        **params):
    p = {**DEFAULT_PARAMS, **params}
    try:
        df = fetch_ohlcv(symbol, interval=interval, period=period)
        if len(df) < p["train_data"] + 50:
            return None

        ind = compute_kraken_indicators(
            df, yz_len=p["yz_len"], piv_len=p["piv_len"], cmf_len=p["cmf_len"],
            train_data=p["train_data"], st_mult=p["st_mult"],
            trap_tolerance=p["trap_tolerance"], fvg_mult=p["fvg_mult"]
        )

        entries = ind["buy_signal"] & ind["is_uptrend"]

        if entries.sum() == 0:
            return {
                "symbol": symbol, "trades": 0, "return_pct": 0.0,
                "win_rate": 0.0, "pf": 0.0, "max_dd": 0.0, "sharpe": 0.0,
            }

        risk = ind["yz_atr"] * p["st_mult"]
        tp_pct = (risk * p["rr_target"]) / df["close"]
        sl_pct = ((df["close"] - ind["supertrend"]).abs() / df["close"]).clip(lower=0.005, upper=0.5)

        pf = vbt.Portfolio.from_signals(
            close=df["close"],
            entries=entries,
            exits=pd.Series(False, index=df.index),
            sl_stop=sl_pct,
            tp_stop=tp_pct,
            init_cash=initial_capital,
            fees=commission_pct / 100,
            freq=interval,
            size=position_pct / 100,
            size_type="percent",
        )

        stats = pf.stats()

        trades_df = pf.trades.records_readable
        if len(trades_df) > 0:
            wins = (trades_df["PnL"] > 0).sum()
            win_rate_manual = (wins / len(trades_df)) * 100
        else:
            win_rate_manual = 0.0

        return {
            "symbol":     symbol,
            "trades":     int(stats.get("Total Trades", 0)),
            "return_pct": float(stats.get("Total Return [%]", 0)),
            "win_rate":   win_rate_manual,
            "pf":         float(stats.get("Profit Factor", 0)) if not pd.isna(stats.get("Profit Factor", np.nan)) else 0,
            "max_dd":     float(stats.get("Max Drawdown [%]", 0)),
            "sharpe":     float(stats.get("Sharpe Ratio", 0)) if not pd.isna(stats.get("Sharpe Ratio", np.nan)) else 0,
        }
    except Exception as e:
        return None


def run_all_bist100(period="5y", interval="1d", position_pct=10, **params):
    print(f"\n{'='*75}")
    print(f"  PROJECT KRAKEN V16 — BIST100 [{interval.upper()}, {period}]")
    print(f"  Pozisyon: %{position_pct} | ST: {DEFAULT_PARAMS['st_mult']} | RR: {DEFAULT_PARAMS['rr_target']}")
    print(f"  Toplam {len(BIST100)} hisse taranacak...")
    print(f"{'='*75}\n")

    results = []
    errors = 0

    for i, symbol in enumerate(BIST100):
        r = run_single_backtest(symbol, period=period, interval=interval,
                                position_pct=position_pct, **params)
        if r:
            results.append(r)
            if r["trades"] > 0:
                print(f"[{i+1}/{len(BIST100)}] {symbol:<10} {r['return_pct']:>7.1f}%"
                      f" Win:{r['win_rate']:>5.1f}% Islem:{r['trades']:>3}"
                      f" MaxDD:{r['max_dd']:>6.1f}%")
        else:
            errors += 1

    print(f"\n{'='*75}")
    print(f"  TARAMA TAMAMLANDI: {len(results)} hisse basarili, {errors} hata/veri yok")
    print(f"{'='*75}")

    traded_results = [r for r in results if r["trades"] > 0]

    if traded_results:
        avg_return = np.mean([r["return_pct"] for r in traded_results])
        avg_winrate = np.mean([r["win_rate"] for r in traded_results])
        total_trades = sum(r["trades"] for r in traded_results)
        positive_count = sum(1 for r in traded_results if r["return_pct"] > 0)

        print(f"\nSinyal Veren Hisse Sayisi: {len(traded_results)}/{len(results)}")
        print(f"ORTALAMA Getiri: %{avg_return:.2f}")
        print(f"ORTALAMA Win Rate: %{avg_winrate:.1f}")
        print(f"TOPLAM Islem: {total_trades}")
        print(f"HAFTALIK ortalama sinyal: {total_trades / (5*52):.2f}")
        print(f"Pozitif Getirili Hisse: {positive_count}/{len(traded_results)}")

        sorted_results = sorted(traded_results, key=lambda x: x["return_pct"], reverse=True)
        print(f"\nEN IYI 10 HISSE:")
        for r in sorted_results[:10]:
            print(f"  {r['symbol']:<12} %{r['return_pct']:.1f}  (Win: %{r['win_rate']:.1f}, Islem: {r['trades']})")

        print(f"\nEN KOTU 5 HISSE:")
        for r in sorted_results[-5:]:
            print(f"  {r['symbol']:<12} %{r['return_pct']:.1f}  (Win: %{r['win_rate']:.1f}, Islem: {r['trades']})")

    return results


def scan_today():
    """Bugun BIST100'de hangi hissede sinyal var? DUZELTILMIS: 5y veri kullanir"""
    print(f"\n{'='*60}")
    print(f"  KRAKEN GUNLUK TARAMA — BIST100")
    print(f"{'='*60}")

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
                price = df["close"].iloc[-1]
                yz_atr = ind["yz_atr"].iloc[-1]
                supertrend = ind["supertrend"].iloc[-1]

                if pd.isna(supertrend) or pd.isna(yz_atr):
                    continue

                risk = yz_atr * DEFAULT_PARAMS["st_mult"]
                tp = price + risk * DEFAULT_PARAMS["rr_target"]
                sl = supertrend

                if sl >= price:
                    continue

                rr = (tp - price) / (price - sl) if (price - sl) > 0 else 0

                opportunities.append({
                    "symbol": symbol, "price": price,
                    "tp": tp, "sl": sl, "rr": rr,
                })
                print(f"  {symbol}: SINYAL! Fiyat={price:.2f} TP={tp:.2f} SL={sl:.2f} R/R=1:{rr:.1f}")
        except Exception:
            pass

    if not opportunities:
        print("  Bugun aktif/gecerli sinyal yok.")
    return opportunities


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode == "all":
        period = sys.argv[2] if len(sys.argv) > 2 else "5y"
        run_all_bist100(period=period)
    elif mode == "scan":
        scan_today()
    elif mode == "single":
        symbol = sys.argv[2] if len(sys.argv) > 2 else "EREGL.IS"
        period = sys.argv[3] if len(sys.argv) > 3 else "5y"
        r = run_single_backtest(symbol, period=period)
        if r:
            print(f"\n{symbol}: %{r['return_pct']:.2f} getiri, {r['trades']} islem, %{r['win_rate']:.1f} win")
    else:
        print("Kullanim: python kraken/runner.py [all|scan|single]")