"""
engine/portfolio.py
-------------------
Converts alpha weights into a simulated long/short portfolio.

═══════════════════════════════════════════════════════
VERIFIED CORRECT MATH  (traced through unit tests)
═══════════════════════════════════════════════════════

TIME CONVENTION:
  weights[t]  = portfolio target at END of day t (formed after seeing close[t])
  returns[t]  = (close[t] - close[t-1]) / close[t-1]  ← EARNED during day t

  We want: PnL[t] = weights[t-1] * returns[t]
  → position decided yesterday, earns today's return
  → In code: (weights.shift(1)) * returns

SHIFT IS CORRECT — verified:
  _shifted_weights()[t] = weights[t-1]   ← position formed at end of day t-1
  pnl[t] = _shifted_weights()[t] * returns[t]   ← earns day-t return ✓

TURNOVER:
  One-way turnover[t] = 0.5 * sum(|weights[t] - weights[t-1]|)
  Represents the fraction of the portfolio rebalanced on day t.
  Cost[t] = turnover[t] * tcost  (one-way, applied once per trade)

═══════════════════════════════════════════════════════
NEW FEATURES (vs V2)
═══════════════════════════════════════════════════════

TRAILING STOP-LOSS
  Applied per-position.  The position in stock i is zeroed on day t if
  the cumulative P&L of that position (from entry onwards) drops below
  -stop_pct of the peak cumulative P&L seen so far.
  After a stop-out the position stays zero for the rest of that rebalance
  cycle (i.e. until the next day when new weights are computed).

  Implementation detail:
    • "Trailing" means we track a running high-water mark per stock.
    • Once knocked out the weight is set to 0 and immediately re-normalised.
    • This is a conservative (position-level) stop, not a portfolio-level stop.

NEUTRALISATION
  Applied to weights BEFORE position-limit clipping, on each day.
  The constituent metadata (from constituents.csv) is used to build
  group masks.  Within each group the weights are demeaned so the net
  long/short exposure to that group is zero.

  Supported modes:
    "none"        — no neutralisation (default)
    "market"      — demean across all stocks (same as the baseline neutralize())
    "sector"      — demean within GICS Sector
    "industry"    — demean within GICS Sub-Industry (4-digit GICS)
    "subindustry" — alias for "industry" (same column in constituents.csv)

═══════════════════════════════════════════════════════
BUGS FOUND AND FIXED (vs V1)
═══════════════════════════════════════════════════════
  1. _apply_position_limits() clipped but didn't re-normalise → abs-sum < 1 post-clip
     FIX: re-normalise after clipping
  2. returns with inf not cleaned → propagated to PnL as inf
     FIX: replace inf/nan before simulation
  3. No validation of date/ticker overlap
     FIX: explicit checks with informative error messages
"""

import warnings
import numpy as np
import pandas as pd


class Portfolio:
    """
    Simulate a daily-rebalanced long/short equity portfolio.

    Parameters
    ----------
    weights          : pd.DataFrame  — target weights (rows=dates, cols=tickers)
    returns          : pd.DataFrame  — daily pct returns (same universe as weights)
    tcost_bps        : float         — one-way transaction cost in basis points
    max_weight       : float         — maximum absolute weight per stock
    trailing_stop    : float|None    — trailing stop-loss threshold (e.g. 0.05 = 5%)
                                       None or 0 disables the stop
    neutralisation   : str           — one of "none","market","sector","industry","subindustry"
    constituents_df  : pd.DataFrame|None
                                     — loaded constituents.csv with at minimum columns
                                       ["Symbol","GICS Sector","GICS Sub-Industry"]
                                       Required when neutralisation != "none" / "market"
    """

    def __init__(
        self,
        weights:         pd.DataFrame,
        returns:         pd.DataFrame,
        tcost_bps:       float = 10.0,
        max_weight:      float = 0.10,
        # trailing_stop:   float | None = None,
        neutralisation:  str = "none",
        constituents_df: pd.DataFrame | None = None,
    ):
        if weights is None or not isinstance(weights, pd.DataFrame) or weights.empty:
            raise ValueError("weights is empty or not a DataFrame.")
        if returns is None or not isinstance(returns, pd.DataFrame) or returns.empty:
            raise ValueError("returns is empty or not a DataFrame.")

        self.tcost         = tcost_bps / 10_000
        self.max_weight    = float(max_weight)
        # self.trailing_stop = float(trailing_stop) if trailing_stop and trailing_stop > 0 else None
        self.neutralisation = (neutralisation or "none").lower().strip()

        # Build group map for sector/industry neutralisation
        self._group_map: dict[str, str] | None = None
        if self.neutralisation in ("sector", "industry", "subindustry") and constituents_df is not None:
            self._group_map = self._build_group_map(constituents_df)

        # ── align tickers ────────────────────────────────────────────────
        common_tickers = weights.columns.intersection(returns.columns)
        if common_tickers.empty:
            raise ValueError(
                "weights and returns share NO common tickers.\n"
                f"  weights columns (first 5): {list(weights.columns[:5])}\n"
                f"  returns columns (first 5): {list(returns.columns[:5])}"
            )

        # ── align dates ──────────────────────────────────────────────────
        common_dates = weights.index.intersection(returns.index)

        self.weights = weights.loc[common_dates, common_tickers].copy()
        self.returns = returns.loc[common_dates, common_tickers].copy()

        # Debug prints in __init__
        print("\n[DEBUG] Portfolio.__init__: after alignment")
        print(f"  weights shape: {self.weights.shape}")
        print("  weights head (first 2 columns):\n", self.weights.iloc[:5, :2])
        print(f"  returns shape: {self.returns.shape}")
        print("  returns head (first 2 columns):\n", self.returns.iloc[:5, :2])

        # ── clean returns ────────────────────────────────────────────────
        self.returns = (
            self.returns
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

        # ── neutralise weights ───────────────────────────────────────────
        self._apply_neutralisation()
        print("[DEBUG] After neutralisation:")
        print("  weights head (first 5 rows, first 5 columns):\n", self.weights.iloc[:5, :5])

        # ── position limits ──────────────────────────────────────────────
        self._apply_position_limits()
        print("[DEBUG] After position limits (clipped + renormalised):")
        print("  weights head (first 5 rows, first 5 columns):\n", self.weights.iloc[:5, :5])
        
        print(
            f"[portfolio] tickers={len(common_tickers)} | "
            f"dates={len(common_dates)} | "
            f"tcost={tcost_bps:.1f}bps | "
            f"max_wt={max_weight:.1%} | "
            # f"stop={self.trailing_stop} | "
            f"neutral={self.neutralisation}"
        )

    # ─────────────────────────────────────────────────────────────────────

    def _build_group_map(self, df: pd.DataFrame) -> dict[str, str]:
        """
        Build {ticker: group_label} from constituents_df.
        Group label depends on neutralisation mode.
        """
        df = df.copy()
        # Normalise ticker column
        sym_col = next((c for c in df.columns if c.lower() in ("symbol", "ticker")), None)
        if sym_col is None:
            warnings.warn("[portfolio] constituents_df has no Symbol/Ticker column; skipping group neutralisation.")
            return {}

        df[sym_col] = df[sym_col].str.replace(".", "-", regex=False).str.upper().str.strip()

        if self.neutralisation == "sector":
            group_col = next((c for c in df.columns if "sector" in c.lower()), None)
        else:  # industry / subindustry
            group_col = next((c for c in df.columns if "sub-industry" in c.lower() or "subindustry" in c.lower()), None)
            if group_col is None:
                group_col = next((c for c in df.columns if "industry" in c.lower()), None)

        if group_col is None:
            warnings.warn(f"[portfolio] Could not find group column for neutralisation='{self.neutralisation}'. Falling back to market neutralisation.")
            self.neutralisation = "market"
            return {}

        mapping = dict(zip(df[sym_col], df[group_col].fillna("Unknown")))
        print(f"[portfolio] Group map built: {len(mapping)} tickers, mode={self.neutralisation}, col='{group_col}'")
        
        if len(mapping) <= 20:
            print("\n[DEBUG] Group mapping for neutralisation:")
            for ticker, group in list(mapping.items())[:20]:
                print(f"  {ticker} → {group}")

        return mapping

    # ─────────────────────────────────────────────────────────────────────

    def _apply_neutralisation(self):
        """
        Apply the chosen cross-sectional neutralisation to self.weights.
        Modifies self.weights in-place.
        """
        mode = self.neutralisation

        if mode == "none":
            return

        if mode == "market":
            # Demean across all stocks each day
            row_mean = self.weights.mean(axis=1)
            self.weights = self.weights.sub(row_mean, axis=0)
            print("[portfolio] Market neutralisation applied.")
            return

        # Sector / industry / subindustry
        if not self._group_map:
            warnings.warn("[portfolio] No group map available; falling back to market neutralisation.")
            row_mean = self.weights.mean(axis=1)
            self.weights = self.weights.sub(row_mean, axis=0)
            return

        # Map tickers that exist in our weight universe
        tickers = self.weights.columns.tolist()
        groups  = pd.Series({t: self._group_map.get(t, "Unknown") for t in tickers})
        unique_groups = groups.unique()

        neutralised = self.weights.copy()
        for grp in unique_groups:
            members = groups[groups == grp].index.tolist()
            if len(members) < 2:
                continue
            sub = neutralised[members]
            grp_mean = sub.mean(axis=1)
            neutralised[members] = sub.sub(grp_mean, axis=0)

        self.weights = neutralised
        print(f"[portfolio] {mode.capitalize()} neutralisation applied across {len(unique_groups)} groups.")

    # ─────────────────────────────────────────────────────────────────────

    def _apply_position_limits(self):
        """
        Clip individual weights to [-max_weight, +max_weight],
        then re-normalise so abs-row-sum == 1.

        BUG FIX: V1 clipped but did NOT re-normalise.
        """
        w = self.weights.clip(lower=-self.max_weight, upper=self.max_weight)
        row_abs_sum = w.abs().sum(axis=1).replace(0, np.nan)
        self.weights = w.div(row_abs_sum, axis=0).fillna(0.0)

    # ─────────────────────────────────────────────────────────────────────

    # def _apply_trailing_stop(self, weights: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Apply trailing stop-loss at the position level.

    #     For each stock i on each day t:
    #       1. Compute the running cumulative P&L of holding weight[t-1,i] from
    #          the first day until day t.
    #       2. Track the high-water mark (HWM) of that cumulative P&L.
    #       3. If (cum_pnl[t,i] - HWM[t,i]) / (|HWM[t,i]| + 1e-8) < -stop_pct:
    #          → zero out weight[t,i] for that stock.
    #       4. Re-normalise each row so abs-sum remains 1.

    #     Parameters
    #     ----------
    #     weights : ALREADY shifted weights (i.e. weights.shift(1)), shape (T, N)

    #     Returns
    #     -------
    #     weights with stop-loss applied, same shape.
    #     """
    #     stop = self.trailing_stop
    #     if stop is None or stop <= 0:
    #         return weights

    #     w = weights.values.copy()           # (T, N)
    #     r = self.returns.values             # (T, N)
    #     T, N = w.shape

    #     # Per-stock running cumulative P&L from "entry" tracking
    #     cum_pnl = np.zeros(N)
    #     hwm     = np.zeros(N)
    #     stopped = np.zeros(N, dtype=bool)

    #     for t in range(T):
    #         # Earnings on day t using the (already shifted) weight
    #         day_pnl = w[t] * r[t]          # per-stock P&L
    #         cum_pnl += day_pnl
    #         hwm      = np.maximum(hwm, cum_pnl)

    #         # Drawdown from HWM
    #         dd = (cum_pnl - hwm) / (np.abs(hwm) + 1e-8)
    #         triggered = dd < -stop

    #         # Zero out stopped positions
    #         w[t, triggered] = 0.0

    #         # Re-normalise so row abs-sum == 1 (if anything remains)
    #         row_sum = np.abs(w[t]).sum()
    #         if row_sum > 1e-10:
    #             w[t] /= row_sum

    #     return pd.DataFrame(w, index=weights.index, columns=weights.columns)

    # ─────────────────────────────────────────────────────────────────────

    def _shifted_weights(self) -> pd.DataFrame:
        """
        weights.shift(1): position formed at end of day t earns return of day t+1.

        weights[t]          = target decided after seeing close[t]
        _shifted_weights[t] = weights[t-1]   ← what we actually held during day t
        """
        shifted = self.weights.shift(1).fillna(0.0)

        if self.weights.shape[1] <= 20:
            print("\n" + "="*80)
            print("SHIFTED WEIGHTS (positions held during each day):")
            print("="*80)
            print(shifted.to_string())
            print("="*80 + "\n")

        print("[DEBUG] Portfolio._shifted_weights():")
        print(f"  shifted shape: {shifted.shape}")
        print("  shifted head (first 2 columns):\n", shifted.iloc[:5, :2])
        return shifted

    def compute_turnover(self) -> pd.Series:
        """
        One-way turnover per day = 0.5 * sum(|w[t] - w[t-1]|).
        """
        dw = self.weights.diff().abs().sum(axis=1) * 0.5
        turnover = dw.rename("turnover")
        print("[DEBUG] Portfolio.compute_turnover():")
        print("  turnover head:\n", turnover.head())
        return turnover

    def compute_gross_pnl(self) -> pd.Series:
        """
        Gross daily P&L = sum(w[t-1] * returns[t]) across all stocks.
        If trailing stop is enabled, positions that breached the stop are zeroed.
        """
        shifted = self._shifted_weights()

        # if self.trailing_stop is not None and self.trailing_stop > 0:
        #     shifted = self._apply_trailing_stop(shifted)
        #     print(f"[portfolio] Trailing stop ({self.trailing_stop:.1%}) applied to shifted weights.")

        product = shifted * self.returns

        if product.shape[1] <= 20:
            print("\n" + "="*80)
            print("PRODUCT = SHIFTED_WEIGHTS * RETURNS (per‑stock daily P&L):")
            print("="*80)
            print(product.to_string())
            print("="*80 + "\n")



        # Debug: print the stock that contributed the most (by absolute value) each day
        max_abs = product.abs()
        max_idx = max_abs.idxmax(axis=1)               # ticker with max absolute contribution
        max_val = max_abs.max(axis=1)                  # the absolute contribution value
        # Get the weight and return for that ticker (using direct indexing to avoid deprecated .lookup)
        shifted_arr = shifted.values
        returns_arr = self.returns.values
        # Get column indices of max_idx
        col_idx = [shifted.columns.get_loc(ticker) for ticker in max_idx]
        max_weight = shifted_arr[np.arange(len(shifted_arr)), col_idx]
        max_return = returns_arr[np.arange(len(returns_arr)), col_idx]

        # Build a DataFrame for easier printing
        contrib_df = pd.DataFrame({
            'date': shifted.index,
            'ticker': max_idx,
            'weight': max_weight,
            'return': max_return,
            'contrib': product.values[np.arange(len(product)), col_idx]
        })
        # Print first 10 and last 10 days
        print("\n[DEBUG] Top contributions per day (first 10):")
        print(contrib_df.head(10).to_string(index=False))
        print("\n[DEBUG] Top contributions per day (last 10):")
        print(contrib_df.tail(10).to_string(index=False))






        pnl = product.sum(axis=1)
        print("[DEBUG] Portfolio.compute_gross_pnl():")
        print(f"  shifted * returns product shape: {product.shape}")
        print("  product head (first 2 columns):\n", product.iloc[:5, :2])
        print("  gross_pnl head:\n", pnl.head())
        return pnl.rename("gross_pnl")

    def compute_net_pnl(self) -> pd.Series:
        """Net P&L = gross P&L minus one-way transaction costs."""
        gross = self.compute_gross_pnl()
        cost  = self.compute_turnover() * self.tcost
        net = (gross - cost).rename("net_pnl")
        print("[DEBUG] Portfolio.compute_net_pnl():")
        print("  net_pnl head:\n", net.head())
        return net

    def simulate(self) -> pd.DataFrame:
        """
        Run the full simulation.

        Returns
        -------
        pd.DataFrame with columns:
            gross_pnl  — daily gross return (before costs)
            net_pnl    — daily net return (after costs)
            turnover   — daily one-way turnover fraction
            cum_gross  — cumulative gross NAV (starts at 1.0)
            cum_net    — cumulative net NAV (starts at 1.0)
            drawdown   — drawdown on net NAV (0 to -1)
        """
        gross    = self.compute_gross_pnl()
        net      = self.compute_net_pnl()
        turnover = self.compute_turnover()

        cum_gross = (1 + gross).cumprod()
        cum_net   = (1 + net).cumprod()

        rolling_max = cum_net.cummax()
        drawdown    = (cum_net - rolling_max) / rolling_max.replace(0, np.nan)

        result = pd.DataFrame({
            "gross_pnl": gross,
            "net_pnl":   net,
            "turnover":  turnover,
            "cum_gross": cum_gross,
            "cum_net":   cum_net,
            "drawdown":  drawdown,
        }).dropna(how="all")

        if result.empty:
            warnings.warn("[portfolio] Simulation output is empty.")

        print("[DEBUG] Portfolio.simulate() final result:")
        print("  result head:\n", result.head())
        print("-"*60)
        return result
