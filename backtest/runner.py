"""
backtest/runner.py
VectorBT ile tam backtest motoru.

Pine Script'teki strategy.entry() / strategy.exit() mantığını uygular.
ADR tabanlı stop-loss, FVG bazlı TP, zombi kesici dahil.
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from datetime import datetime


def run_backtest(df: pd.DataFrame,
                 ind: pd.DataFrame,
                 final_signals: pd.DataFrame,
                 initial_capital: float = 10000,
                 risk_pct: float = 2.0,
                 adr_mult: float = 1.5,
                 rr_ratio: float = 2.0,
                 is_scalp_mode: bool = False,
                 zombie_bars: int = 20,
                 use_zombie: bool = True,
                 commission_pct: float = 0.04) -> dict:
    """
    Tam backtest çalıştır.
    
    Pine:
      strategy.entry("LONG", strategy.long, qty=calc_trade_qty)
      strategy.exit("EXIT_L", "LONG", limit=tp1_L, stop=sl_long)
    
    Returns: dict with stats + portfolio object
    """
    
    entries_long  = final_signals["buy_signal"]
    entries_short = final_signals["sell_signal"]
    
    # ── STOP LOSS hesapla (ADR bazlı, Pine ile aynı) ──
    stop_pct = ind["safe_stop_pct"] / 100
    
    sl_long_pct  = stop_pct          # long: fiyatın altında
    sl_short_pct = stop_pct          # short: fiyatın üstünde
    
    # ── TAKE PROFIT hesapla (R:R oranına göre) ──
    # [FIX] Telegram mesaji kullaniciya "TP1'de yarisini kapat + kalanla
    # TP2'ye git" diyor, ama burada TEK bir TP (eskiden dogrudan TP2
    # mesafesi, stop_pct*rr_ratio) simule ediliyordu -- yani Optuna'nin
    # optimize ettigi backtest, gercekte onerilen iki-asamali cikisi hic
    # test etmiyordu. Gercek partial-exit (50% TP1 + 50% TP2) VectorBT'de
    # ayri bir emir seti gerektirir (bu surumde yok); onun yerine TP1 ve
    # TP2 mesafesinin ORTALAMASI kullanilarak iki-asamali cikisin
    # BEKLENEN (blended) sonucuna daha yakin, daha durust bir yaklasik
    # deger hesaplaniyor. NOT: gercek partial-exit'ten yine de farkli
    # olabilir -- birebir degil, sadece eskisinden daha az yanlis.
    tp1_pct = stop_pct * 1.0
    tp2_pct = stop_pct * rr_ratio
    tp_long_pct  = (tp1_pct + tp2_pct) / 2.0
    tp_short_pct = (tp1_pct + tp2_pct) / 2.0
    
    # ── ZOMBI KESİCİ için exit sinyalleri ──
    # Giriş barından zombie_bars sonra kapat
    exit_long  = pd.Series(False, index=df.index)
    exit_short = pd.Series(False, index=df.index)
    
    if is_scalp_mode and use_zombie:
        long_entry_bars  = entries_long[entries_long].index.tolist()
        short_entry_bars = entries_short[entries_short].index.tolist()
        
        for entry_date in long_entry_bars:
            entry_pos = df.index.get_loc(entry_date)
            zombie_pos = entry_pos + zombie_bars
            if zombie_pos < len(df):
                exit_long.iloc[zombie_pos] = True
        
        for entry_date in short_entry_bars:
            entry_pos = df.index.get_loc(entry_date)
            zombie_pos = entry_pos + zombie_bars
            if zombie_pos < len(df):
                exit_short.iloc[zombie_pos] = True
    
    # ── POSITION SIZE hesapla ──
    # Pine: calc_trade_qty = risk_amount / (close * safe_stop_dist_pct/100)
    risk_amount   = initial_capital * (risk_pct / 100)
    qty_per_trade = risk_amount / (df["close"] * stop_pct.clip(lower=0.001))
    qty_per_trade = qty_per_trade.clip(lower=0.0001)

    # [FIX] Kimi K3'un Pine'da buldugu ayni sorun burada da vardi: ust
    # sinir yoktu. Stop mesafesi cok dar geldiginde (dusuk volatilite
    # bari), qty asiri buyuyup kaldiracsiz sermayenin kat kat uzerinde
    # nominal pozisyon acilmis gibi backtest ediliyordu. Kaldiracsiz
    # tavan: sermayenin tamami (initial_capital) ile alinabilecek max
    # miktar. NOT: bu, gercek/degisen equity'yi degil SABIT
    # initial_capital'i baz alan basit/muhafazakar bir tavandir --
    # VectorBT'nin vektorize from_signals API'si equity'yi trade-by-trade
    # izleyip dinamik boyutlandirmayi (Pine'daki strategy.equity gibi)
    # kolayca desteklemiyor, bu yuzden en azindan "tek islemde sermayenin
    # katlari kadar pozisyon" riskini kapatan bu basit tavan eklendi.
    max_qty_no_leverage = initial_capital / df["close"]
    qty_per_trade = qty_per_trade.clip(upper=max_qty_no_leverage)
    
    # VectorBT Portfolio
    try:
        portfolio = vbt.Portfolio.from_signals(
            close=df["close"],
            entries=entries_long,
            exits=exit_long,
            short_entries=entries_short,
            short_exits=exit_short,
            sl_stop=sl_long_pct,
            tp_stop=tp_long_pct,
            init_cash=initial_capital,
            fees=commission_pct / 100,
            freq="4h",  # [FIX] eskiden "1h" idi -- bu bot hep 4H calisiyor.
                        # Yanlis freq, VectorBT'nin yillik periyot sayisini
                        # (annualization factor) 4x fazla varsaymasina, bu
                        # da Sharpe Ratio'nun (~2x, sqrt(4)) yapay sekilde
                        # sisirilmis cikmasina neden oluyordu -- Optuna'nin
                        # optimize ettigi ana metrik tam da bu deger.
            size=qty_per_trade,
            size_type="amount",
            upon_opposite_entry="Reverse",
        )
        
        stats = portfolio.stats()
        
        return {
            "portfolio": portfolio,
            "stats": stats,
            "total_return_pct": stats.get("Total Return [%]", 0),
            "win_rate_pct": stats.get("Win Rate [%]", 0),
            "profit_factor": stats.get("Profit Factor", 0),
            "max_drawdown_pct": stats.get("Max Drawdown [%]", 0),
            "total_trades": stats.get("Total Trades", 0),
            "sharpe_ratio": stats.get("Sharpe Ratio", 0),
        }
    except Exception as e:
        print(f"VectorBT hatası: {e}")
        return None


def print_results(results: dict, label: str = "Backtest Sonuçları"):
    """Sonuçları güzel formatta yazdır."""
    if results is None:
        print("Backtest başarısız oldu.")
        return
    
    print(f"\n{'═'*55}")
    print(f"🏆 {label}")
    print(f"{'═'*55}")
    
    s = results["stats"]
    print(f"  Başlangıç:       {s.get('Start', 'N/A')}")
    print(f"  Bitiş:           {s.get('End', 'N/A')}")
    print(f"  Toplam Getiri:   %{results['total_return_pct']:.2f}")
    print(f"  Win Rate:        %{results['win_rate_pct']:.2f}")
    print(f"  Profit Factor:   {results['profit_factor']:.3f}")
    print(f"  Max Drawdown:    %{results['max_drawdown_pct']:.2f}")
    print(f"  Toplam İşlem:    {results['total_trades']}")
    print(f"  Sharpe Ratio:    {results['sharpe_ratio']:.3f}")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    # [FIX] yfinance "4h" interval'ini desteklemez -- fetch_smart kullan
    from telegram_bot import fetch_smart
    from engine.indicators import compute_all_indicators
    from engine.signals import calc_bull_bear_score, calc_triggers, generate_signals
    from engine.filters import apply_all_filters

    print("📡 BTC-USD verisi çekiliyor (4H)...")
    df = fetch_smart("BTC-USD", "4h", "2y")
    
    print("🧮 İndikatörler hesaplanıyor...")
    ind = compute_all_indicators(df, preset="Default")
    
    print("🎯 Sinyaller üretiliyor...")
    sc  = calc_bull_bear_score(ind)
    tr  = calc_triggers(ind, sc)
    sg  = generate_signals(ind, sc, tr, preset="Default")
    fs  = apply_all_filters(ind, sg)
    
    print(f"   BUY: {fs['buy_signal'].sum()} | SELL: {fs['sell_signal'].sum()}")
    
    print("⚡ Backtest çalışıyor...")
    results = run_backtest(df, ind, fs, adr_mult=1.5, rr_ratio=2.0)
    print_results(results, "SMP V2.8.2 — Tam Entegrasyon")
