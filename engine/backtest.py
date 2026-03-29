"""
engine/backtest.py
------------------
Top-level orchestrator.

Pipeline:
  fetch_universe → AlphaEngine.evaluate → Portfolio.simulate → compute_metrics

Returns a structured dict ready for JSON serialisation.

═══════════════════════════════════════════════════════
NEW PARAMETERS (vs V2)
═══════════════════════════════════════════════════════
  trailing_stop   : float|None  — position-level trailing stop-loss threshold
                                  (e.g. 0.05 = 5%).  None disables.
  neutralisation  : str         — "none"|"market"|"sector"|"industry"|"subindustry"
  universe        : str         — "TOPSP500"|"TOP500"|"TOP200"|"TOP1000"|"TOP3000"
                                  Ignored if `tickers` is explicitly provided.
"""

import re
import traceback
import warnings

import pandas as pd

from engine.data_fetcher import fetch_universe, get_constituents_df, get_market_cap_constituents_df
from engine.alpha        import AlphaEngine, LOOKAHEAD_ALPHAS
from engine.portfolio    import Portfolio
from engine.metrics      import compute_metrics


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _err(msg: str, detail: str = "") -> dict:
    out = {"error": msg}
    if detail:
        out["details"] = detail
    return out


def _has_lookahead(expr: str) -> bool:
    """
    True if expression uses delay() with a negative argument.
    e.g. delay(returns, -1)  →  True
         delay(close, 5)     →  False
    """
    return bool(re.search(r"delay\s*\([^,]+,\s*-\s*\d", expr))


_VALID_NEUTRALISATIONS = {"none", "market", "sector", "industry", "subindustry"}
_VALID_UNIVERSES       = {"TOPSP500", "TOP500", "TOP200", "TOP1000", "TOP3000"}


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    expression:    str,
    tickers:       list | None  = None,
    start:         str          = "2006-01-01",
    end:           str          = "2018-12-31",
    tcost_bps:     float        = 10.0,
    max_weight:    float        = 0.10,
    # trailing_stop: float | None = None,
    neutralisation: str         = "none",
    universe:      str          = "TOPSP500",
) -> dict:
    """
    Run a full backtest and return a result dict.

    On success  → {"summary": {...}, "charts": {...}, "meta": {...}}
    On failure  → {"error": "...", "details": "..."}
    """
    warnings.filterwarnings("ignore", category=FutureWarning)

    # ── 0. Validate inputs ─────────────────────────────────────────────────
    expression = (expression or "").strip()
    if not expression:
        return _err("Alpha expression is empty.")

    try:
        start_ts = pd.Timestamp(start)
        end_ts   = pd.Timestamp(end)
    except Exception as e:
        return _err(f"Invalid date: {e}")

    if end_ts <= start_ts:
        return _err("end date must be strictly after start date.")

    if not (0 < max_weight <= 1.0):
        return _err("max_weight must be between 0 (exclusive) and 1 (inclusive).")

    if tcost_bps < 0:
        return _err("tcost_bps must be ≥ 0.")

    # if trailing_stop is not None and trailing_stop < 0:
    #     return _err("trailing_stop must be ≥ 0 (or null to disable).")

    neutralisation = (neutralisation or "none").lower().strip()
    if neutralisation not in _VALID_NEUTRALISATIONS:
        return _err(f"neutralisation must be one of: {sorted(_VALID_NEUTRALISATIONS)}")

    universe = (universe or "TOPSP500").upper().strip()
    if universe not in _VALID_UNIVERSES:
        return _err(f"universe must be one of: {sorted(_VALID_UNIVERSES)}")

    # ── 1. Fetch market data ────────────────────────────────────────────────
    try:
        data = fetch_universe(tickers, start, end, universe=universe)
    except RuntimeError as e:
        return _err(str(e))
    except Exception:
        return _err("Failed to fetch market data.", traceback.format_exc())

    if "close" not in data or data["close"] is None or data["close"].empty:
        return _err(
            "No close price data available.",
            f"Requested {start} → {end}. "
            "Check internet connection or try a different date range."
        )

    close = data["close"]
    actual_start = str(close.index[0].date())
    actual_end   = str(close.index[-1].date())
    n_days_data  = len(close)
    n_tickers    = len(close.columns)

    print(f"[backtest] Data: {n_tickers} tickers × {n_days_data} days "
          f"({actual_start} → {actual_end})")


    # ---- ADD THIS BLOCK ----
    if n_tickers <= 20:   # only for small universes, to avoid flooding the console
        print("\n" + "="*80)
        print("FULL CLOSE PRICES (all rows, all columns):")
        print("="*80)
        print(close.to_string())
        print("="*80 + "\n")

    # ── 2. Evaluate alpha ───────────────────────────────────────────────────
    try:
        engine  = AlphaEngine(data)
        weights = engine.evaluate(expression)
    except (ValueError, TypeError) as e:
        return _err(f"Alpha expression error: {e}")
    except Exception:
        return _err("Unexpected error evaluating alpha.", traceback.format_exc())

    if weights is None or weights.empty:
        return _err("Alpha produced no valid weights.")

    n_weight_rows = weights.dropna(how="all").shape[0]

    print(f"[backtest] Weights: {weights.shape[0]} rows × {weights.shape[1]} tickers "
          f"({n_weight_rows} non-NaN rows)")

    # ── 3. Compute daily returns ────────────────────────────────────────────
    try:
        returns = close.pct_change()
        
        if n_tickers <= 20:
            print("\n" + "="*80)
            print("FULL DAILY RETURNS (all rows, all columns):")
            print("="*80)
            print(returns.to_string())
            print("="*80 + "\n")

    except Exception:
        return _err("Failed to compute returns from close prices.", traceback.format_exc())

    # ── 4. Load constituents for neutralisation ─────────────────────────────
    constituents_df = None
    if neutralisation in ("sector", "industry", "subindustry"):
        # Decide which file to use based on universe
        if universe in ("TOPSP500"):
            # SP500 universe → use sp500.csv
            try:
                constituents_df = get_constituents_df()
            except RuntimeError as e:
                print(f"[backtest] WARNING: {e} — falling back to market neutralisation.")
                neutralisation = "market"
        else:
            # TOP200 / TOP500 / TOP1000 / TOP3000 → use market_cap_ranked.csv
            try:
                constituents_df = get_market_cap_constituents_df()
            except RuntimeError as e:
                print(f"[backtest] WARNING: {e} — falling back to market neutralisation.")
                neutralisation = "market"

    # ── 5. Simulate portfolio ───────────────────────────────────────────────
    try:
        port = Portfolio(
            weights          = weights,
            returns          = returns,
            tcost_bps        = tcost_bps,
            max_weight       = max_weight,
            # trailing_stop    = trailing_stop,
            neutralisation   = neutralisation,
            constituents_df  = constituents_df,
        )
        sim = port.simulate()
    except ValueError as e:
        return _err(f"Portfolio error: {e}")
    except Exception:
        return _err("Unexpected error in portfolio simulation.", traceback.format_exc())

    if sim is None or sim.empty:
        return _err(
            "Portfolio simulation produced no output. "
            "Weights and returns may not overlap in dates or tickers."
        )

    print(f"[backtest] Simulation: {len(sim)} rows | "
          f"gross_cum={sim['cum_gross'].iloc[-1]:.4f} | "
          f"net_cum={sim['cum_net'].iloc[-1]:.4f}")

    # ── 6. Compute metrics ──────────────────────────────────────────────────
    try:
        result = compute_metrics(sim)
    except Exception:
        return _err("Unexpected error computing metrics.", traceback.format_exc())

    if "error" in result:
        return result

    # ── 7. Attach meta ──────────────────────────────────────────────────────
    lookahead = expression in LOOKAHEAD_ALPHAS or _has_lookahead(expression)

    result["meta"] = {
        "expression":    expression,
        "n_tickers":     weights.shape[1],
        "start":         actual_start,
        "end":           actual_end,
        "tcost_bps":     tcost_bps,
        "max_weight":    max_weight,
        # "trailing_stop": trailing_stop,
        "neutralisation": neutralisation,
        "universe":      universe,
        "has_lookahead": lookahead,
    }

    if lookahead:
        result["warning"] = (
            "⚠ LOOKAHEAD BIAS: this alpha uses future price data "
            "(delay with negative argument). Results are NOT achievable in live trading. "
            "Use this ONLY to verify the pipeline produces positive returns."
        )

    return result
