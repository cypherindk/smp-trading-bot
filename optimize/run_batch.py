"""
optimize/run_batch.py
Her sembol (3 kripto + BIST100, ya da --symbols ile secilenler) icin
AYRI AYRI Optuna optimizasyonu calistirir ve en iyi parametreleri
optimize/best_params.json dosyasina kaydeder.

[ONEMLI] Bu, kullanicinin acik istegi uzerine preset dahil HER SEYI
optimize eder -- yani her sembol icin TradingView'deki "Auto (4H'de
Aggressive)" ayarindan FARKLI bir preset bulunabilir. Bu durumda o
sembol icin Telegram sinyalleri TW ekranindaki oklarla artik BIREBIR
ESLESMEZ -- bu, gecmiste en iyi performansi tercih etmenin bilinen ve
kabul edilmis bir sonucu.

Bu AGIR bir islemdir -- BIST100 + kripto icin saatler surebilir. Canli
tarama (telegram_bot.py) icinde CALISTIRILMAZ; ayri, zamanlanmis bir
GitHub Actions job'inda (orn. haftada/ayda bir) calistirilmasi onerilir
(bkz. .github/workflows/optimize.yml).

Kullanim:
  python optimize/run_batch.py                        # hepsi, 60 deneme/sembol
  python optimize/run_batch.py --n-trials 100
  python optimize/run_batch.py --symbols BTC-USD ETH-USD
  python optimize/run_batch.py --skip-bist             # sadece kripto
  python optimize/run_batch.py --only-bist             # sadece BIST100
  python optimize/run_batch.py --symbols AKBNK.IS --n-trials 150
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from telegram_bot import fetch_smart
from optimize.optimizer import run_optimization, validate_best_params
from bist100_tickers import BIST100_YF

CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD"]
DEFAULT_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_params.json")
BIST_REQUEST_DELAY = 1.0  # BIST icin ardisik yfinance istekleri arasi bekleme
MIN_BARS = 300            # bundan az bar varsa optimizasyon anlamsiz


def optimize_symbol(symbol: str, is_bist: bool, n_trials: int, period: str = "2y") -> dict:
    resample_offset = "2h" if is_bist else None
    df = fetch_smart(symbol, "4h", period, resample_offset=resample_offset)
    if len(df) < MIN_BARS:
        raise ValueError(f"yetersiz veri ({len(df)} bar, minimum {MIN_BARS})")

    opt_results = run_optimization(df, n_trials=n_trials, use_walk_forward=True, n_splits=3)
    val_results = validate_best_params(df, opt_results["best_params"])

    validation = None
    if val_results:
        validation = {
            "sharpe_ratio": val_results.get("sharpe_ratio"),
            "max_drawdown_pct": val_results.get("max_drawdown_pct"),
            "total_return_pct": val_results.get("total_return_pct"),
            "win_rate_pct": val_results.get("win_rate_pct"),
            "profit_factor": val_results.get("profit_factor"),
            "total_trades": val_results.get("total_trades"),
        }

    return {
        "params": opt_results["best_params"],
        "train_value": opt_results["best_value"],
        "validation": validation,
        "bar_count": len(df),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def load_existing(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def run_batch(symbols: list, is_bist_map: dict, n_trials: int, output_path: str):
    results = load_existing(output_path)
    total = len(symbols)
    ok_count = 0
    failed = []

    for i, symbol in enumerate(symbols, 1):
        is_bist = is_bist_map[symbol]
        print(f"\n[{i}/{total}] {symbol} ({'BIST' if is_bist else 'Kripto'}) optimize ediliyor...")
        try:
            entry = optimize_symbol(symbol, is_bist, n_trials)
            results[symbol] = entry
            ok_count += 1
            val_sharpe = entry["validation"]["sharpe_ratio"] if entry["validation"] else None
            msg = f"  ✅ {symbol}: train_sharpe={entry['train_value']:.3f}"
            if val_sharpe is not None:
                msg += f"  val_sharpe={val_sharpe:.3f}"
            msg += f"  preset={entry['params'].get('preset')}  grade={entry['params'].get('grade_filter')}"
            print(msg)

            # Her sembolden sonra diske yaz -- kesinti olsa bile ilerleme kaybolmaz
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        except Exception as e:
            failed.append((symbol, str(e)))
            print(f"  ❌ {symbol}: {e}")

        if is_bist:
            time.sleep(BIST_REQUEST_DELAY)

    print(f"\n{'='*55}")
    print(f"Bitti: {ok_count}/{total} basarili, {len(failed)} hata")
    if failed:
        print("Hatali semboller:")
        for sym, err in failed:
            print(f"  - {sym}: {err}")
    print(f"Sonuclar: {output_path}")
    print(f"{'='*55}")


def _build_symbol_list(args):
    is_bist_map = {}
    symbols = []
    if args.symbols:
        for s in args.symbols:
            symbols.append(s)
            is_bist_map[s] = s.upper().endswith(".IS")
    else:
        if not args.only_bist:
            for s in CRYPTO_SYMBOLS:
                symbols.append(s)
                is_bist_map[s] = False
        if not args.skip_bist:
            for s in BIST100_YF:
                symbols.append(s)
                is_bist_map[s] = True
    return symbols, is_bist_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=60,
                       help="Sembol basina Optuna deneme sayisi (varsayilan 60 -- "
                            "orijinal optimizer.py tek sembolde 150 kullaniyordu ama "
                            "103 sembol icin bu saatlerce surer, once 60 ile dene, "
                            "sonuclara gore artir)")
    parser.add_argument("--symbols", nargs="*", default=None,
                       help="Sadece belirtilen sembolleri optimize et (orn. BTC-USD AKBNK.IS)")
    parser.add_argument("--skip-bist", action="store_true", help="Sadece kripto (hizli)")
    parser.add_argument("--only-bist", action="store_true", help="Sadece BIST100")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    symbols, is_bist_map = _build_symbol_list(args)
    print(f"Toplam {len(symbols)} sembol, sembol basina {args.n_trials} deneme.")
    print(f"Tahmini sure cok degisken -- backtest.runner'in hizina bagli, "
          f"BIST100 + kripto icin saatler surebilir.\n")

    run_batch(symbols, is_bist_map, args.n_trials, args.output)
