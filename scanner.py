import sys
import os
import time
import requests
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.fetcher import fetch_ohlcv
from engine.indicators import compute_all_indicators
from engine.signals import calc_bull_bear_score, calc_triggers, generate_signals
from engine.filters import apply_all_filters
from backtest.runner import run_backtest, print_results

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


def scan_now():
    print("\n" + "="*55)
    print(f"  SINYAL TARAYICI — {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    print("="*55)

    opportunities = []

    for coin, p in COINS.items():
        try:
            df  = fetch_ohlcv(coin, interval="4h", period="3mo")
            ind = compute_all_indicators(df, preset=p["preset"])
            sc  = calc_bull_bear_score(ind)
            tr  = calc_triggers(ind, sc)
            sg  = generate_signals(ind, sc, tr, preset=p["preset"],
                                   eff_score=p["eff_score"], min_conf=p["min_conf"])
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

            opportunities.append({
                "coin": coin, "direction": direction, "score": score,
                "rvol": rvol, "rsi": rsi, "price": price,
                "sl": sl, "tp1": tp1, "tp2": tp2,
                "stop_pct": stop_pct, "tp1_pct": tp1_pct, "tp2_pct": tp2_pct,
                "rr": tp2_pct / stop_pct if stop_pct > 0 else 0,
                "priority": p["priority"],
            })
            print(f"  {coin}: {direction} sinyali (skor={score:.1f})")

        except Exception as e:
            print(f"  {coin}: hata — {e}")

    if not opportunities:
        print("\n  Su an aktif sinyal yok.")
        return

    opportunities.sort(key=lambda x: (x["priority"], -x["score"]))

    print(f"\n  {len(opportunities)} firsat bulundu:\n")
    for i, opp in enumerate(opportunities, 1):
        rr = opp["rr"]
        print(f"  {'─'*50}")
        print(f"  #{i}  {opp['coin']}  →  {opp['direction']}")
        print(f"      Fiyat:  ${opp['price']:,.4f}")
        print(f"      Skor:   {opp['score']:.1f}  |  RVOL: {opp['rvol']:.2f}x  |  RSI: {opp['rsi']:.0f}")
        print(f"      SL:     ${opp['sl']:,.4f}  (-%{opp['stop_pct']:.2f})")
        print(f"      TP1:    ${opp['tp1']:,.4f}  (+%{opp['tp1_pct']:.2f})")
        if opp["tp2"]:
            print(f"      TP2:    ${opp['tp2']:,.4f}  (+%{opp['tp2_pct']:.2f})")
        print(f"      R/R:    1:{rr:.1f}")
    print("="*55 + "\n")


def run_sequential_backtest():
    print("\n" + "="*55)
    print("  SIRALI BACKTEST — V3.1 Optimize Parametreler")
    print("="*55)

    all_data = {}
    for coin, p in COINS.items():
        print(f"\n  {coin} hazirlaniyor...")
        df  = fetch_ohlcv(coin, interval="4h", period="2y")
        ind = compute_all_indicators(df, preset=p["preset"])
        sc  = calc_bull_bear_score(ind)
        tr  = calc_triggers(ind, sc)
        sg  = generate_signals(ind, sc, tr, preset=p["preset"],
                               eff_score=p["eff_score"], min_conf=p["min_conf"])
        fs  = apply_all_filters(ind, sg, use_cvd=True)
        all_data[coin] = {"df": df, "ind": ind, "sc": sc, "fs": fs, "p": p}

    common_start = max(d["df"].index[0] for d in all_data.values())
    common_end   = min(d["df"].index[-1] for d in all_data.values())
    print(f"\n  Ortak donem: {common_start.date()} → {common_end.date()}")

    initial_capital = 10000.0
    capital = initial_capital
    in_position = False
    current_coin = None
    entry_price = 0.0
    entry_dir = None
    sl_price = 0.0
    tp_price = 0.0
    entry_bar = 0
    trades = []

    btc_df = all_data["BTC-USD"]["df"]
    all_bars = btc_df.index[(btc_df.index >= common_start) & (btc_df.index <= common_end)]

    for bar_i, bar_time in enumerate(all_bars):
        if in_position and current_coin:
            coin_df = all_data[current_coin]["df"]
            if bar_time in coin_df.index:
                cur_high  = coin_df.loc[bar_time, "high"]
                cur_low   = coin_df.loc[bar_time, "low"]
                cur_close = coin_df.loc[bar_time, "close"]
                exit_price = None
                exit_reason = None

                if entry_dir == "LONG":
                    if cur_low <= sl_price:
                        exit_price = sl_price
                        exit_reason = "SL"
                    elif cur_high >= tp_price:
                        exit_price = tp_price
                        exit_reason = "TP"
                else:
                    if cur_high >= sl_price:
                        exit_price = sl_price
                        exit_reason = "SL"
                    elif cur_low <= tp_price:
                        exit_price = tp_price
                        exit_reason = "TP"

                p = all_data[current_coin]["p"]
                if (bar_i - entry_bar) >= p.get("zombie_bars", 30):
                    exit_price = cur_close
                    exit_reason = "ZOMBIE"

                if exit_price:
                    if entry_dir == "LONG":
                        pnl_pct = (exit_price - entry_price) / entry_price * 100
                    else:
                        pnl_pct = (entry_price - exit_price) / entry_price * 100

                    risk_amt  = capital * 0.02
                    stop_dist = abs(entry_price - sl_price) / entry_price
                    trade_qty = risk_amt / (entry_price * stop_dist) if stop_dist > 0 else 0
                    pnl_usd   = trade_qty * entry_price * (pnl_pct / 100)
                    capital  += pnl_usd
                    in_position = False

                    trades.append({
                        "coin": current_coin, "direction": entry_dir,
                        "entry": entry_price, "exit": exit_price,
                        "reason": exit_reason, "pnl_pct": pnl_pct,
                        "pnl_usd": pnl_usd, "capital": capital, "bar": bar_time,
                    })
                    current_coin = None

        if not in_position:
            best_score = -1
            best_opp = None

            for coin, data in all_data.items():
                if bar_time not in data["df"].index:
                    continue
                if bar_time not in data["fs"].index:
                    continue

                buy_sig  = data["fs"].loc[bar_time, "buy_signal"]
                sell_sig = data["fs"].loc[bar_time, "sell_signal"]
                if not buy_sig and not sell_sig:
                    continue

                bull_sc = data["sc"].loc[bar_time, "bull_score"]
                bear_sc = data["sc"].loc[bar_time, "bear_score"]
                direction = "LONG" if (buy_sig and bull_sc >= bear_sc) else "SHORT"
                score     = bull_sc if direction == "LONG" else bear_sc
                adj_score = score + (4 - data["p"]["priority"]) * 0.5

                if adj_score > best_score:
                    best_score = adj_score
                    best_opp = {"coin": coin, "direction": direction, "data": data}

            if best_opp:
                coin    = best_opp["coin"]
                data    = best_opp["data"]
                p       = data["p"]
                df_coin = data["df"]
                ind_coin = data["ind"]

                if bar_time in df_coin.index and bar_time in ind_coin.index:
                    entry_price = df_coin.loc[bar_time, "close"]
                    stop_pct    = ind_coin.loc[bar_time, "safe_stop_pct"]
                    tp_pct      = stop_pct * p["rr_ratio"]

                    if best_opp["direction"] == "LONG":
                        sl_price = entry_price * (1 - stop_pct/100)
                        tp_price = entry_price * (1 + tp_pct/100)
                    else:
                        sl_price = entry_price * (1 + stop_pct/100)
                        tp_price = entry_price * (1 - tp_pct/100)

                    in_position  = True
                    current_coin = coin
                    entry_dir    = best_opp["direction"]
                    entry_bar    = bar_i

    if not trades:
        print("\n  Hic islem yapilmadi.")
        return

    trades_df = pd.DataFrame(trades)
    total_return = (capital - initial_capital) / initial_capital * 100
    win_trades   = trades_df[trades_df["pnl_usd"] > 0]
    win_rate     = len(win_trades) / len(trades_df) * 100
    pnl_series   = trades_df["pnl_usd"]
    profit_factor_val = (pnl_series[pnl_series > 0].sum() /
                         abs(pnl_series[pnl_series < 0].sum())
                         if abs(pnl_series[pnl_series < 0].sum()) > 0 else float("inf"))
    capital_series = trades_df["capital"]
    peak = capital_series.cummax()
    dd_series = (capital_series - peak) / peak * 100
    max_dd = abs(dd_series.min())

    print(f"\n  {'='*50}")
    print(f"  SIRALI SISTEM SONUCLARI (V3.1)")
    print(f"  {'='*50}")
    print(f"  Baslangic Kapital:  ${initial_capital:,.0f}")
    print(f"  Bitis Kapital:      ${capital:,.0f}")
    print(f"  Toplam Getiri:      %{total_return:.2f}")
    print(f"  Win Rate:           %{win_rate:.1f}")
    print(f"  Profit Factor:      {profit_factor_val:.2f}")
    print(f"  Max Drawdown:       %{max_dd:.2f}")
    print(f"  Toplam Islem:       {len(trades_df)}")
    print(f"\n  Coin Dagilimi:")
    for coin in COINS:
        coin_trades = trades_df[trades_df["coin"] == coin]
        if len(coin_trades) > 0:
            coin_win = len(coin_trades[coin_trades["pnl_usd"] > 0])
            coin_pnl = coin_trades["pnl_usd"].sum()
            print(f"    {coin:<12} {len(coin_trades):>3} islem  "
                  f"Win: {coin_win}/{len(coin_trades)}  "
                  f"PnL: ${coin_pnl:+,.0f}")
    print(f"\n  Son 5 Islem:")
    for _, t in trades_df.tail(5).iterrows():
        emoji = "✅" if t["pnl_usd"] > 0 else "❌"
        print(f"    {emoji} {t['coin']:<12} {t['direction']:<6} "
              f"{t['reason']:<8} %{t['pnl_pct']:+.2f}  ${t['pnl_usd']:+.0f}")
    print(f"  {'='*50}\n")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "scan"
    if mode == "scan":
        scan_now()
    elif mode == "backtest":
        run_sequential_backtest()
    else:
        print("Kullanim: python scanner.py [scan|backtest]")