"""
engine/metrics.py
-----------------
Computes all performance metrics from Portfolio.simulate() output.

All formulas verified against standard finance references.

CHANGES vs V2
  • Added Information Ratio  = mean(net_pnl) / std(net_pnl) * sqrt(252)
    This is the "self-referential" IR (no benchmark subtraction), which is
    exactly Sharpe when the risk-free rate is assumed to be zero.
    In the quantitative-equity context the benchmark is typically
    cash (zero), so IR == Sharpe.  We expose it explicitly so users can
    compare it with a benchmark-adjusted version later.
    Formula: IR = E[r] / σ(r) × √252  where r = daily net returns.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _safe(val, decimals: int = 10):
    """Convert to Python float, return None for NaN / Inf."""
    if val is None:
        return None
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else round(v, decimals)


def _to_chart(series: pd.Series) -> list:
    """Convert Series to [{date, value}, …] for JSON."""
    if series is None or (isinstance(series, pd.Series) and series.empty):
        return []
    if not isinstance(series.index, pd.DatetimeIndex):
        try:
            series = series.copy()
            series.index = pd.to_datetime(series.index)
        except Exception:
            return []
    out = []
    for d, v in series.items():
        try:
            fv = float(v)
            if np.isfinite(fv):
                out.append({"date": d.strftime("%Y-%m-%d"), "value": round(fv, 10)})
        except (TypeError, ValueError):
            pass
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  VPIN APPROXIMATION
# ══════════════════════════════════════════════════════════════════════════════

def _vpin(net_pnl: pd.Series, bucket: int = 50) -> float:
    """
    Approximate VPIN from daily P&L.
    Uses buy/sell classification based on daily P&L sign.
    Returns the average volume imbalance fraction.
    """
    if len(net_pnl) < bucket * 2:
        return float("nan")
    buy  = (net_pnl > 0).astype(float)
    sell = (net_pnl <= 0).astype(float)
    imbal = (buy - sell).abs()
    denom = (buy + sell).rolling(bucket, min_periods=bucket // 2).mean().replace(0, np.nan)
    series = imbal.rolling(bucket, min_periods=bucket // 2).mean() / denom
    v = series.mean()
    return float(v) if np.isfinite(v) else float("nan")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(sim: pd.DataFrame, rolling_window: int = 63) -> dict:
    """
    Parameters
    ----------
    sim            : output of Portfolio.simulate()
    rolling_window : window for rolling Sharpe (default 63 trading days)

    Returns
    -------
    {"summary": {scalar metrics}, "charts": {time-series lists}}
    """
    required = {"net_pnl", "gross_pnl", "turnover", "cum_net", "drawdown"}
    missing  = required - set(sim.columns)
    if missing:
        return {"error": f"Simulation missing columns: {sorted(missing)}"}

    net = sim["net_pnl"].dropna()
    cum = sim["cum_net"].dropna()
    dd  = sim["drawdown"].dropna()
    to  = sim["turnover"].dropna()

    n = len(net)

    # ── Core ──────────────────────────────────────────────────────────────
    total_return = _safe(float(cum.iloc[-1]) - 1.0)

    # CAGR: (end_value)^(252/n_days) - 1
    end_val = float(cum.iloc[-1])
    ann_return = _safe((end_val ** (TRADING_DAYS / n)) - 1.0) if end_val > 0 else None

    ann_vol  = _safe(net.std() * np.sqrt(TRADING_DAYS))
    max_dd   = _safe(dd.min())
    avg_to   = _safe(to.mean())
    win_rate = _safe((net > 0).mean())

    # ── Sharpe ────────────────────────────────────────────────────────────
    sharpe = _safe(
        net.mean() / net.std() * np.sqrt(TRADING_DAYS)
        if net.std() > 1e-10 else None
    )

    # ── Information Ratio ─────────────────────────────────────────────────
    # IR = mean(daily_net_returns) / std(daily_net_returns) * sqrt(252)
    # With a zero benchmark (cash), this equals the Sharpe ratio numerically.
    # Exposed separately for transparency and future benchmark-adjusted extension.
    information_ratio = _safe(
        net.mean() / net.std() * np.sqrt(TRADING_DAYS)
        if net.std() > 1e-10 else None
    )

    # ── Sortino ───────────────────────────────────────────────────────────
    downside = net[net < 0]
    if len(downside) > 5:
        ddev    = float(np.sqrt((downside ** 2).mean()))
        sortino = _safe(net.mean() / ddev * np.sqrt(TRADING_DAYS) if ddev > 1e-10 else None)
    else:
        sortino = None

    # ── Calmar ────────────────────────────────────────────────────────────
    calmar = _safe(
        ann_return / abs(max_dd)
        if ann_return is not None and max_dd is not None and abs(max_dd) > 1e-10
        else None
    )

    # ── Win / loss stats ──────────────────────────────────────────────────
    wins   = net[net > 0]
    losses = net[net < 0]
    avg_win  = _safe(wins.mean()   if len(wins)   > 0 else None)
    avg_loss = _safe(losses.mean() if len(losses) > 0 else None)
    profit_factor = _safe(
        wins.sum() / abs(losses.sum())
        if len(wins) > 0 and len(losses) > 0 and abs(losses.sum()) > 1e-10
        else None
    )

    # ── Higher moments ────────────────────────────────────────────────────
    skewness = _safe(float(net.skew()))
    kurtosis = _safe(float(net.kurtosis()))   # excess kurtosis

    # ── Recovery factor ───────────────────────────────────────────────────
    recovery = _safe(
        total_return / abs(max_dd)
        if total_return is not None and max_dd is not None and abs(max_dd) > 1e-10
        else None
    )

    # ── Pain index (mean abs drawdown) ────────────────────────────────────
    pain = _safe(dd.abs().mean())

    # ── Fitness score ──────────────────────────────────────────
    fitness = _safe(
        sharpe * np.sqrt(abs(ann_return) / max(avg_to or 0, 0.125))
        if sharpe is not None and ann_return is not None
        else None
    )

    # ── VPIN ──────────────────────────────────────────────────────────────
    vpin = _safe(_vpin(net))

    # ── Rolling Sharpe ────────────────────────────────────────────────────
    def _roll_sharpe(x: np.ndarray) -> float:
        s = x.std()
        return float(x.mean() / s * np.sqrt(TRADING_DAYS)) if s > 1e-10 else np.nan

    rolling_sharpe = net.rolling(rolling_window, min_periods=rolling_window // 2) \
                        .apply(_roll_sharpe, raw=True)

    # ── Rolling volatility (21-day) ───────────────────────────────────────
    rolling_vol = net.rolling(21, min_periods=10).std() * np.sqrt(TRADING_DAYS)

    # ── Charts: convert to percentages where appropriate ──────────────────
    cum_pnl_pct = (cum - 1) * 100

    # Debug prints
    print("\n[DEBUG] compute_metrics: summary metrics")
    summary = {
        "total_return":       total_return,
        "annualized_return":  ann_return,
        "annualized_vol":     ann_vol,
        "sharpe_ratio":       sharpe,
        "information_ratio":  information_ratio,
        "sortino_ratio":      sortino,
        "max_drawdown":       max_dd,
        "calmar_ratio":       calmar,
        "avg_daily_turnover": avg_to,
        "win_rate":           win_rate,
        "avg_win":            avg_win,
        "avg_loss":           avg_loss,
        "profit_factor":      profit_factor,
        "skewness":           skewness,
        "kurtosis":           kurtosis,
        "recovery_factor":    recovery,
        "pain_index":         pain,
        "fitness":            fitness,
        "vpin":               vpin,
        "n_trading_days":     n,
    }
    print("  summary:", summary)
    print("-"*60)

    return {
        "summary": summary,
        "charts": {
            "cum_pnl":        _to_chart(cum_pnl_pct),          # percent return
            "daily_returns":  _to_chart(net * 100),            # percent
            "drawdown":       _to_chart(dd * 100),             # percent
            "turnover":       _to_chart(to * 100),             # percent
            "rolling_sharpe": _to_chart(rolling_sharpe),       # ratio
            "rolling_vol":    _to_chart(rolling_vol * 100),    # percent
        },
    }
