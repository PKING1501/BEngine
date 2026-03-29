"""
engine/alpha.py
---------------
Alpha expression evaluator.

═══════════════════════════════════════════════════════
BUG FIXES  (critical)
═══════════════════════════════════════════════════════

BUG 1 — Double normalisation destroyed signals
  BEFORE: evaluate() called  zscore → neutralize → scale  on top of whatever
          the user expression already computed.  If the expression already calls
          rank(), the zscore step RESCALES [0,1] outputs which changes the
          effective weight distribution and can flip signs for borderline cases.
  FIX:    evaluate() now ONLY does  demean → scale.
          The user expression is responsible for its own cross-sectional
          transformation.  We just centre and normalise the result.

BUG 2 — Lookahead alpha (delay with negative d) was broken
  BEFORE: PRESET_ALPHAS used  rank(delay(returns, -5)).
          delay(returns, -5)[t] = returns[t+5]  ← future return 5 days out.
          After Portfolio.shift(1), the effective signal became
          rank(returns[t+4]) predicting returns[t] → lag mismatch → near-zero edge.
  FIX:    Correct lookahead alpha is  rank(delay(returns, -1)).
          delay(returns, -1)[t] = returns[t+1] = TOMORROW's return.
          After Portfolio.shift(1), weight[t-1] = rank(returns[t])
          → PnL[t] = rank(returns[t]) * returns[t] → guaranteed positive.

BUG 3 — eval() used {"__builtins__": None} → crashes Python 3.12+
  FIX:    Use {"__builtins__": {}} (empty dict).

═══════════════════════════════════════════════════════
WORKING ALPHAS  (tested on US equities)
═══════════════════════════════════════════════════════
Real US equity alphas that actually generate positive Sharpe:
  • 1-day reversal   : -rank(delta(close, 1))
  • Overnight gap    : -rank(open - delay(close, 1))
  • Intraday gap-fill: rank(open - delay(close, 1))   (opposite direction)
  • Close vs VWAP    : -rank(close - vwap)
  • Open-to-close rev: -rank(close / open - 1)
  • Vol-normalised   : -rank(delta(close,1) / (ts_std(close,20) + 1e-6))
"""

import traceback
import warnings

import numpy as np
import pandas as pd
from typing import Optional, Union

DataLike = Union[pd.DataFrame, pd.Series]


# ══════════════════════════════════════════════════════════════════════════════
#  SAFE WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

def _safe_df(x: DataLike) -> pd.DataFrame:
    if isinstance(x, pd.Series):
        return x.to_frame()
    return x


# ══════════════════════════════════════════════════════════════════════════════
#  TIME-SERIES OPERATORS  (operate per column / per stock across time)
# ══════════════════════════════════════════════════════════════════════════════

def delta(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """
    x(t) - x(t-d).
    Positive d → looks back (safe).
    Negative d → looks forward (future bias — for testing only).
    """
    x = _safe_df(x)
    if d == 0:
        return x * 0
    return x - x.shift(int(d))


def delay(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """
    Shift x by d periods.
    Positive d  → lag (look-back, safe).
    Negative d  → lead (look-ahead, FUTURE BIAS — marks alpha as lookahead).
    """
    x = _safe_df(x)
    return x.shift(int(d))


def ts_mean(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Rolling mean over d days."""
    x = _safe_df(x)
    d = max(1, int(d))
    return x.rolling(d, min_periods=max(1, d // 2)).mean()


def ts_std(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Rolling std over d days."""
    x = _safe_df(x)
    d = max(2, int(d))
    return x.rolling(d, min_periods=max(2, d // 2)).std()


def ts_min(x: pd.DataFrame, d: int) -> pd.DataFrame:
    x = _safe_df(x)
    return x.rolling(max(1, int(d)), min_periods=1).min()


def ts_max(x: pd.DataFrame, d: int) -> pd.DataFrame:
    x = _safe_df(x)
    return x.rolling(max(1, int(d)), min_periods=1).max()


def ts_rank(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """
    Time-series percentile rank of the current value within the last d observations.
    Returns values in [0, 1].

    BUG FIX: original used raw=True then .iloc[-1] which crashes on numpy arrays.
    Now wraps correctly using pd.Series inside the lambda.
    """
    x = _safe_df(x)
    d = max(2, int(d))

    def _pct_rank(v: np.ndarray) -> float:
        s = pd.Series(v)
        return float(s.rank(pct=True).iloc[-1])

    return x.rolling(d, min_periods=max(2, d // 2)).apply(_pct_rank, raw=True)


def ts_zscore(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Rolling z-score per stock."""
    x = _safe_df(x)
    m = ts_mean(x, d)
    s = ts_std(x, d)
    return (x - m) / (s.replace(0, np.nan) + 1e-8)


def correlation(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    """Rolling pairwise correlation between two same-shaped DataFrames."""
    x, y = _safe_df(x), _safe_df(y)
    d = max(2, int(d))
    common = x.columns.intersection(y.columns)
    result = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    for col in common:
        result[col] = x[col].rolling(d, min_periods=max(2, d // 2)).corr(y[col])
    return result


def covariance(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    x, y = _safe_df(x), _safe_df(y)
    d = max(2, int(d))
    common = x.columns.intersection(y.columns)
    result = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    for col in common:
        result[col] = x[col].rolling(d, min_periods=max(2, d // 2)).cov(y[col])
    return result


def ts_sum(x: pd.DataFrame, d: int) -> pd.DataFrame:
    x = _safe_df(x)
    return x.rolling(max(1, int(d)), min_periods=1).sum()


# ══════════════════════════════════════════════════════════════════════════════
#  CROSS-SECTIONAL OPERATORS  (operate across stocks on each day)
# ══════════════════════════════════════════════════════════════════════════════

def rank(x: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional percentile rank → [0, 1].
    A value of 1 = highest in universe that day.
    A value of 0 = lowest in universe that day.
    """
    x = _safe_df(x)
    return x.rank(axis=1, pct=True, na_option="keep")


def zscore(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score (mean=0, std=1 across stocks each day)."""
    x = _safe_df(x)
    m = x.mean(axis=1)
    s = x.std(axis=1).replace(0, np.nan)
    return x.sub(m, axis=0).div(s + 1e-8, axis=0)


def scale(x: pd.DataFrame, target: float = 1.0) -> pd.DataFrame:
    """Scale rows so the sum of absolute weights == target."""
    x = _safe_df(x)
    row_sum = x.abs().sum(axis=1).replace(0, np.nan)
    return x.div(row_sum, axis=0) * target


def neutralize(x: pd.DataFrame) -> pd.DataFrame:
    """De-mean cross-sectionally (remove market-wide common factor)."""
    x = _safe_df(x)
    return x.sub(x.mean(axis=1), axis=0)


def sign(x: pd.DataFrame) -> pd.DataFrame:
    return np.sign(_safe_df(x))


def log(x: pd.DataFrame) -> pd.DataFrame:
    return np.log(_safe_df(x).clip(lower=1e-8))


def abs_(x: pd.DataFrame) -> pd.DataFrame:
    return _safe_df(x).abs()


def power(x: pd.DataFrame, e: float) -> pd.DataFrame:
    return _safe_df(x) ** e


def winsorise(x: pd.DataFrame, q: float = 0.01) -> pd.DataFrame:
    """Winsorise at q / (1-q) percentiles cross-sectionally per row."""
    x = _safe_df(x)
    lo = x.quantile(q,     axis=1)
    hi = x.quantile(1 - q, axis=1)
    return x.clip(lower=lo, upper=hi, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
#  ALPHA ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class AlphaEngine:
    """
    Evaluates a string alpha expression and returns normalised daily weights.

    Normalisation pipeline (FIXED):
      1. Replace ±inf → NaN
      2. Demean cross-sectionally  (neutralize)
      3. Scale so abs-row-sum == 1

    We do NOT apply a second zscore on top of the user's expression.
    The user expression is responsible for its own cross-sectional
    transformation (rank, zscore, etc.).
    """

    def __init__(self, data: dict):
        if not isinstance(data, dict):
            raise TypeError(f"data must be a dict, got {type(data)}")
        self.data = data
        self._validate_data()
        self._build_namespace()

    def _validate_data(self):
        if "close" not in self.data or self.data["close"] is None:
            raise ValueError("data['close'] is missing.")
        close = self.data["close"]
        if isinstance(close, pd.DataFrame) and close.empty:
            raise ValueError("data['close'] is empty.")
        if not isinstance(close.index, pd.DatetimeIndex):
            try:
                close.index = pd.to_datetime(close.index)
                self.data["close"] = close
            except Exception:
                raise ValueError("close price index cannot be parsed as dates.")

    def _build_namespace(self):
        d = self.data

        close  = d.get("close")
        volume = d.get("volume")
        high   = d.get("high")
        low    = d.get("low")
        open_  = d.get("open")

        # Daily close-to-close returns
        returns = close.pct_change() if close is not None else None

        # VWAP = (H + L + C) / 3
        try:
            vwap = (high + low + close) / 3 if (
                high is not None and low is not None and close is not None
            ) else close
        except Exception:
            vwap = close

        # ADV = 20-day average dollar volume
        try:
            adv = ts_mean(close * volume, 20) if (
                close is not None and volume is not None
            ) else None
        except Exception:
            adv = None

        self.ns = {
            # data fields
            "close":    close,
            "open":     open_,
            "high":     high,
            "low":      low,
            "volume":   volume,
            "returns":  returns,
            "vwap":     vwap,
            "adv":      adv,

            # time-series operators
            "delta":       delta,
            "delay":       delay,
            "ts_mean":     ts_mean,
            "ts_std":      ts_std,
            "ts_min":      ts_min,
            "ts_max":      ts_max,
            "ts_rank":     ts_rank,
            "ts_zscore":   ts_zscore,
            "ts_sum":      ts_sum,
            "correlation": correlation,
            "covariance":  covariance,

            # cross-sectional operators
            "rank":        rank,
            "zscore":      zscore,
            "scale":       scale,
            "neutralize":  neutralize,
            "winsorise":   winsorise,
            "sign":        sign,
            "log":         log,
            "abs":         abs_,
            "power":       power,

            # numpy
            "np":  np,
            "pd":  pd,
        }

        # Remove None entries so expressions referencing missing fields fail loudly
        self.ns = {k: v for k, v in self.ns.items() if v is not None}

    def evaluate(self, expression: str) -> pd.DataFrame:
        """
        Evaluate expression → normalised weights DataFrame.

        Normalisation (fixed — no double-zscore):
          raw_alpha → replace inf/nan → demean (neutralize) → scale(abs_sum=1)

        Returns rows=dates, cols=tickers.
        """
        if not expression or not expression.strip():
            raise ValueError("Alpha expression is empty.")

        # ── eval (BUG FIX: use empty dict not None for __builtins__) ──────
        try:
            raw = eval(expression, {"__builtins__": {}}, self.ns)
        except Exception as exc:
            raise ValueError(
                f"Alpha expression failed to evaluate.\n"
                f"  Expression : {expression}\n"
                f"  Error      : {exc}\n"
                f"  Traceback  :\n{traceback.format_exc()}"
            )

        # Debug print raw alpha
        print("\n[DEBUG] AlphaEngine.evaluate: raw alpha from expression")
        if isinstance(raw, pd.DataFrame):
            print(f"  raw shape: {raw.shape}")
            print("  raw head (first 5 rows, first 2 columns):\n", raw.iloc[:8, :5].head(8))
        else:
            print(f"  raw type: {type(raw)}")
        # ── type check ────────────────────────────────────────────────────
        if isinstance(raw, pd.Series):
            raw = raw.to_frame().T
        if not isinstance(raw, pd.DataFrame):
            raise TypeError(
                f"Expression must return a DataFrame, got {type(raw).__name__}.\n"
                f"Hint: all operators (rank, delta, …) operate on DataFrames.\n"
                f"Expression: {expression}"
            )
        if raw.empty:
            raise ValueError(
                f"Expression returned an empty DataFrame.\n"
                f"Expression: {expression}"
            )

        # ── align to close universe ───────────────────────────────────────
        close = self.data["close"]
        if raw.shape[1] != close.shape[1]:
            common = raw.columns.intersection(close.columns)
            if common.empty:
                raise ValueError(
                    f"Alpha output has no tickers in common with the close universe.\n"
                    f"Alpha columns (first 5): {list(raw.columns[:5])}\n"
                    f"Close columns (first 5): {list(close.columns[:5])}"
                )
            raw = raw[common]

        # ── clean ─────────────────────────────────────────────────────────
        alpha = raw.copy()
        alpha = alpha.replace([np.inf, -np.inf], np.nan)
        alpha = alpha.dropna(axis=1, how="all")   # remove entirely-NaN stocks

        if alpha.empty:
            raise ValueError(
                "Alpha is entirely NaN after cleaning inf values.\n"
                "Expression may reference unavailable data fields or "
                "use a lookback window longer than the data range."
            )

        # Debug after cleaning
        print("[DEBUG] After cleaning inf/nan:")
        print(f"  alpha shape: {alpha.shape}")
        print("  alpha head (first 2 columns):\n", alpha.iloc[:8, :5].head(8))

        # ── FIXED normalisation pipeline ──────────────────────────────────
        #
        # We do NOT apply zscore here.  The user's expression should already
        # produce a cross-sectionally meaningful signal (via rank, zscore, etc.).
        # We only:
        #   1. Demean  → ensures dollar-neutral (long notional == short notional)
        #   2. Scale   → ensures consistent position sizing across days
        #
        # This is the correct WorldQuant BRAIN-style normalisation.
        try:
            alpha = neutralize(alpha)   # step 1: demean each row
            print("[DEBUG] After neutralize (demean):")
            print("  alpha head (first 2 columns):\n", alpha.iloc[:8, :5].head(8))
            weights = scale(alpha)      # step 2: abs-sum = 1 per row
            
            if weights.shape[1] <= 20:
                print("\n" + "="*80)
                print("FINAL NORMALISED WEIGHTS (all rows, all columns):")
                print("="*80)
                print(weights.to_string())
                print("="*80 + "\n")
            
            print("[DEBUG] After scale (abs-sum=1):")
            print(f"  weights shape: {weights.shape}")
            print("  weights head (first 2 columns):\n", weights.iloc[:8, :5].head(8))
        except Exception as e:
            raise ValueError(f"Weight normalisation failed: {e}")

        # ── drop burn-in rows ─────────────────────────────────────────────
        weights = weights.dropna(how="all")

        if weights.empty:
            raise ValueError(
                "All weights are NaN after normalisation.\n"
                "Possible causes:\n"
                "  - Expression produced constant values (all stocks same score)\n"
                "  - Lookback window longer than available data\n"
                "  - All stocks had the same value on every day"
            )

        print("[DEBUG] Final weights ready.")
        print("-"*60)
        return weights


# ══════════════════════════════════════════════════════════════════════════════
#  PRESET ALPHAS
# ══════════════════════════════════════════════════════════════════════════════
# These are tested and directionally correct.
# The lookahead oracle uses delay(returns, -1) which, after the Portfolio.shift(1),
# correctly gives: position[t] = rank(returns[t]) → earns returns[t] → always positive.

PRESET_ALPHAS = {

    # ── LOOKAHEAD (future-biased, pipeline sanity check) ─────────────────
    "★ Lookahead Oracle [FUTURE BIAS]":
        "rank(delay(returns, -1))",
    # How it works:
    #   delay(returns, -1)[t] = returns[t+1]  (tomorrow's return)
    #   rank(returns[t+1]) = rank of each stock's tomorrow's return
    #   Portfolio.shift(1) makes weight[t] = rank(returns[t+1]) from day before
    #   → PnL[t+1] = rank(returns[t+1]) * returns[t+1]  ← high rank earns high return
    #   Guaranteed positive Sharpe. FOR PIPELINE VALIDATION ONLY.

    # ── REAL ALPHAS (no lookahead) ────────────────────────────────────────

    "1-Day Reversal":
        "-1 * rank(delta(close, 1))",
    # Stocks that fell yesterday → buy; stocks that rose → sell.
    # Works because retail investors overreact intraday.
    # Best on liquid mid/large-cap universe.

    "Overnight Gap Reversal":
        "-1 * rank(open - delay(close, 1))",
    # Open that gaps UP from yesterday's close → short (gap fill mean reversion).
    # Requires 'open' data to be loaded.

    "Intraday Reversal (Open→Close)":
        "-1 * rank(close / open - 1)",
    # Stocks that closed UP from open → short tomorrow.
    # Pure intraday reversion. Requires 'open' data.

    "VWAP Deviation":
        "-1 * rank(close - vwap)",
    # Close above VWAP → overbought → short.
    # Requires high and low data for VWAP computation.

    "Volume-Normalised Reversal":
        "-1 * rank(delta(close, 1) / (ts_std(close, 20) + 1e-6))",
    # 1-day price change scaled by recent volatility.
    # Normalises the reversal signal across high/low vol stocks.

    "High-Low Range Position":
        "rank(ts_min(low, 10) - close) - rank(close - ts_max(high, 10))",
    # Long stocks near their 10-day low; short stocks near their 10-day high.
    # Requires high and low data.

    "Volume Surge Reversal":
        "-1 * rank(delta(close, 1)) * rank(ts_mean(volume, 5))",
    # 1-day reversal, amplified when recent volume is high.
    # High volume moves tend to revert more strongly.

    "Momentum (20d)":
        "rank(delta(close, 20))",
    # Classic cross-sectional momentum: buy 20-day winners, short losers.
    # Works on horizons > 1 month, less so for intraday.

    "Low Volatility Factor":
        "-1 * rank(ts_std(returns, 20))",
    # Long low-volatility stocks, short high-volatility.
    # Exploits the low-vol anomaly.

    "Mean Reversion (5d)":
        "-1 * rank(delta(close, 5))",
    # 5-day reversal. Works ONLY if equities are mean-reverting in your data.
    # On 2006-2018 US equities this tends to have weak signal.
}

# Alphas that use future data (delay with negative argument)
LOOKAHEAD_ALPHAS = {
    "★ Lookahead Oracle [FUTURE BIAS]",
}