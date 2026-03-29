"""
engine/data_fetcher.py
----------------------
Fetches daily OHLCV data from Yahoo Finance via yfinance.
Stores per-ticker CSV files under  cache/<TICKER>/  with one CSV per field.

Uses DISK cache only — no RAM cache to avoid stale data across different date ranges.

═══════════════════════════════════════════════════════
NEW FEATURES (vs V2)
═══════════════════════════════════════════════════════

UNIVERSE SELECTION
  get_universe_tickers(universe) returns the ticker list for a named universe:

    "TOPSP500"  — all tickers in sp500.csv  (~500 stocks)
    "TOP200"    — top 200 by market cap proxy (first 200 from SP500)
    "TOP500"    — alias for TOPSP500
    "TOP1000"   — SP500 + extended mid-cap supplement list (~1000 stocks)
    "TOP3000"   — broad US equity universe (~3000 stocks, Russell 3000 proxy)

  NOTE ON TOP1000 / TOP3000:
    These require an extended ticker list.  We supply a curated supplement
    list of mid-cap and small-cap US tickers that are commonly traded.
    For a production system you would source these from a data vendor
    (e.g. CRSP, Compustat, Bloomberg) or a free index provider.
    See the HOW TO GET UNIVERSE DATA section below for guidance.

═══════════════════════════════════════════════════════
HOW TO GET UNIVERSE DATA
═══════════════════════════════════════════════════════

  SP500      — Wikipedia / sp500.csv (already provided)
               https://en.wikipedia.org/wiki/List_of_S%26P_500_companies

  Russell 1000/3000 —
    • FTSE Russell publishes membership files quarterly:
      https://www.ftserussell.com/products/indices/russell-us

    • iShares ETF CSV download (IWB = Russell 1000, IWV = Russell 3000):
      On the iShares fund page → "Download" → "Holdings" → CSV
      Gives you ticker, name, weight, sector, market cap tier.

    • Quandl / Nasdaq Data Link  SHARADAR/TICKERS  dataset:
      Includes exchange, sector, industry, market-cap tier.
      Free tier available.  One-time Python download, store locally.

    • yfinance + Wikipedia:
      The Russell page lists index constituents; scrape once and cache.

  CRSP (comprehensive) —
    • University access required.  Covers ALL listed US stocks with
      historical constituent flags, SIC codes, exchanges.
"""

import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── optional curl_cffi for better rate-limit avoidance ───────────────────────
try:
    from curl_cffi import requests as curl_requests
    _CURL = True
except ImportError:
    _CURL = False

import yfinance as yf

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).resolve().parent.parent   # project root
CACHE_DIR  = _HERE / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_SKIP_FILE = CACHE_DIR / "_skip_list.json"            # tickers that yield no data
_RANKED_CSV = _HERE / "market_cap_ranked.csv"

# ── constants ─────────────────────────────────────────────────────────────────
MASTER_START   = "2000-01-01"
DOWNLOAD_DELAY = 0.5          # seconds between tickers
MAX_RETRIES    = 3
_FIELDS        = ["open", "high", "low", "close", "volume"]

# ══════════════════════════════════════════════════════════════════════════════
#  SKIP LIST  — tickers that returned no data; avoid repeated 404 hits
# ══════════════════════════════════════════════════════════════════════════════

def _load_skip_list() -> set:
    try:
        if _SKIP_FILE.exists():
            return set(json.loads(_SKIP_FILE.read_text()))
    except Exception:
        pass
    return set()


def _add_to_skip_list(ticker: str):
    sl = _load_skip_list()
    sl.add(ticker)
    try:
        _SKIP_FILE.write_text(json.dumps(sorted(sl)))
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  UNIVERSE SELECTION
# ══════════════════════════════════════════════════════════════════════════════
def get_sp500_tickers() -> list:
    """Return SP500 ticker list. Delegates to get_constituents_df()."""
    df = get_constituents_df()
    tickers = df["Symbol"].dropna().tolist()
    tickers = [t for t in tickers if t and t != "NAN"]
    print(f"[fetcher] {len(tickers)} SP500 tickers from sp500.csv")
    return tickers

def get_constituents_df() -> pd.DataFrame:
    """
    Load sp500.csv as a DataFrame.
    Always present — raises RuntimeError if missing or malformed.
    Normalises the Symbol column to uppercase with dot→dash.
    """
    csv_path = _HERE / "sp500.csv"
    try:
        df = pd.read_csv(csv_path)
        sym_col = next((c for c in df.columns if c.lower() in ("symbol", "ticker")), None)
        if sym_col is None:
            raise ValueError("No Symbol/Ticker column found in sp500.csv")
        df = df.rename(columns={sym_col: "Symbol"})
        df["Symbol"] = (
            df["Symbol"]
            .astype(str)
            .str.strip()
            .str.upper()
            .str.replace(".", "-", regex=False)
        )
        return df
    except Exception as e:
        raise RuntimeError(f"[fetcher] Failed to load sp500.csv: {e}")

def get_universe_tickers(universe: str = "TOPSP500") -> list:
    universe = (universe or "TOPSP500").upper().strip()

    # TOPSP500 always comes from sp500.csv (has sector/industry metadata)
    if universe in ("TOPSP500", "TOP500"):
        return get_sp500_tickers()

    if _RANKED_CSV.exists():
        try:
            df = pd.read_csv(_RANKED_CSV)

            # Find the symbol column (case-insensitive)
            sym_col = next(
                (c for c in df.columns if c.strip().lower() in ("symbol", "ticker")),
                None
            )
            if sym_col is None:
                warnings.warn("[fetcher] market_cap_ranked.csv has no Symbol/Ticker column.")
                return get_sp500_tickers()

            # Clean: strip whitespace, uppercase, drop blanks
            tickers = (
                df[sym_col]
                .astype(str)
                .str.strip()
                .str.upper()
                .str.replace(".", "-", regex=False)   # BRK.B → BRK-B
                .dropna()
                .tolist()
            )
            tickers = [t for t in tickers if t and t != "NAN"]

            limits = {"TOP200": 200, "TOP1000": 1000, "TOP3000": 3000}
            if universe in limits:
                tickers = tickers[:limits[universe]]
                print(f"[fetcher] Universe {universe}: {len(tickers)} tickers from ranked CSV")
                return tickers

        except Exception as e:
            warnings.warn(f"[fetcher] Failed to read market_cap_ranked.csv: {e}")

    # Fallback if file missing
    warnings.warn(
        f"[fetcher] market_cap_ranked.csv not found. "
        f"Place the market-cap ranked CSV at: {_HERE / 'market_cap_ranked.csv'}"
    )
    return get_sp500_tickers()


# ══════════════════════════════════════════════════════════════════════════════
#  PER-TICKER DISK HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _ticker_dir(ticker: str) -> Path:
    return CACHE_DIR / ticker.replace("/", "_")


def _ticker_has_data(ticker: str) -> bool:
    td = _ticker_dir(ticker)
    return td.exists() and any(td.glob("*.csv"))


def _latest_cached_date(ticker: str) -> Optional[pd.Timestamp]:
    """Return the last date stored in close.csv, or None."""
    cp = _ticker_dir(ticker) / "close.csv"
    if not cp.exists():
        return None
    try:
        df = pd.read_csv(cp, index_col=0, parse_dates=True)
        if not df.empty:
            return df.index.max()
    except Exception:
        pass
    return None


def _save_ticker_data(ticker: str, df: pd.DataFrame):
    """
    Save OHLCV columns from df into per-field CSV files.
    """
    if df is None or df.empty:
        return

    td = _ticker_dir(ticker)
    td.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(x) for x in c).strip() for c in df.columns]

    df = df.sort_index()

    for field in _FIELDS:
        if field in df.columns:
            s = df[field].dropna()
            if not s.empty:
                try:
                    s.to_csv(td / f"{field}.csv", header=True)
                except Exception as e:
                    warnings.warn(f"[fetcher] save {ticker}/{field}: {e}")


def _load_ticker_data(ticker: str, start: str, end: str) -> Optional[dict]:
    """
    Load per-field CSVs, slice to [start, end], return dict or None.
    """
    td = _ticker_dir(ticker)
    if not td.exists():
        return None

    data = {}
    for field in _FIELDS:
        fp = td / f"{field}.csv"
        if not fp.exists():
            continue
        try:
            df = pd.read_csv(fp, index_col=0, parse_dates=True)
            if df.empty:
                continue
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]
            sliced = df.loc[start:end]
            if sliced.empty:
                continue
            sliced.columns = [ticker]
            data[field] = sliced
        except Exception as e:
            warnings.warn(f"[fetcher] load {ticker}/{field}: {e}")

    return data if data else None


# ══════════════════════════════════════════════════════════════════════════════
#  DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

def _download_ticker(ticker: str) -> pd.DataFrame:
    """
    Download full history for one ticker with retries (exponential backoff).
    """
    for attempt in range(MAX_RETRIES):
        try:
            if _CURL:
                session = curl_requests.Session(impersonate="safari15_5")
                t = yf.Ticker(ticker, session=session)
            else:
                t = yf.Ticker(ticker)

            df = t.history(
                start=MASTER_START,
                end=datetime.now().strftime("%Y-%m-%d"),
                auto_adjust=True,
                actions=False,
            )

            if df is None or df.empty:
                return pd.DataFrame()

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

            keep = [c for c in df.columns if c.lower() in _FIELDS]
            df = df[keep].copy()
            df.columns = [c.lower() for c in df.columns]

            if "close" not in df.columns or df["close"].dropna().empty:
                return pd.DataFrame()

            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.sort_index()
            return df

        except Exception as e:
            wait = min(2 ** attempt, 60)
            if attempt < MAX_RETRIES - 1:
                print(f"  [{ticker}] attempt {attempt+1} failed ({e}); retry in {wait}s")
                time.sleep(wait)
            else:
                print(f"  [{ticker}] all retries exhausted: {e}")

    return pd.DataFrame()


def _ensure_ticker_downloaded(ticker: str, force: bool = False, required_end: Optional[str] = None):
    """
    Download ticker if not on disk yet or if stale.
    """
    skip = _load_skip_list()
    if ticker in skip and not force:
        return

    needs_download = force or not _ticker_has_data(ticker)

    if not needs_download and required_end:
        latest = _latest_cached_date(ticker)
        if latest is None or latest < pd.Timestamp(required_end):
            needs_download = True

    if not needs_download:
        return

    print(f"[fetcher] ↓ {ticker} ...", end=" ", flush=True)
    df = _download_ticker(ticker)

    if df.empty:
        print("no data — skipped")
        _add_to_skip_list(ticker)
        return

    _save_ticker_data(ticker, df)
    print(f"ok  ({len(df)} rows)")
    time.sleep(DOWNLOAD_DELAY)


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def fetch_universe(
    tickers:        Optional[list] = None,
    start:          str = "2006-01-01",
    end:            str = "2018-12-31",
    force_refresh:  bool = False,
    universe:       str = "TOPSP500",
) -> dict:
    """
    Returns { field: DataFrame(index=DatetimeIndex, columns=tickers) }
    Fields: open, high, low, close, volume.

    Parameters
    ----------
    tickers       : explicit list of tickers (overrides `universe`)
    universe      : named universe if `tickers` is None
                    ("TOPSP500","TOP500","TOP200","TOP1000","TOP3000")
    """
    if tickers is None:
        tickers = get_universe_tickers(universe)
    if not tickers:
        raise ValueError("Empty ticker list.")

    tickers = [str(t).upper().strip() for t in tickers if t]

    for tkr in tickers:
        try:
            _ensure_ticker_downloaded(tkr, force=force_refresh, required_end=end)
        except Exception as e:
            warnings.warn(f"[fetcher] Unexpected error downloading {tkr}: {e}")

    field_frames: dict[str, list] = {f: [] for f in _FIELDS}
    loaded, skipped = 0, 0

    for tkr in tickers:
        td = _load_ticker_data(tkr, start, end)
        if td:
            for fld in _FIELDS:
                if fld in td:
                    field_frames[fld].append(td[fld])
            loaded += 1
        else:
            skipped += 1

    print(f"[fetcher] Loaded {loaded} tickers, skipped {skipped}")

    if loaded == 0:
        raise RuntimeError(
            "No data loaded for any ticker in the universe. "
            "Check your internet connection or date range."
        )

    combined: dict[str, pd.DataFrame] = {}
    for fld, frames in field_frames.items():
        if not frames:
            continue
        df = pd.concat(frames, axis=1, sort=True)
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()
        df = df.ffill(limit=5)
        df = df.dropna(axis=1, how="all")
        combined[fld] = df

    if "close" not in combined or combined["close"].empty:
        raise RuntimeError("close price data is empty after loading universe.")

    print("\n[DEBUG] fetch_universe: loaded data (sliced to requested range)")
    for fld, df in combined.items():
        print(f"  {fld}: shape {df.shape}, columns: {list(df.columns)[:5]}...")
    print("[DEBUG] close head:\n", combined['close'].head(5))
    print("-"*60)

    return combined


# ── Convenience helpers ───────────────────────────────────────────────────────

def get_close(tickers=None, start="2006-01-01", end="2018-12-31") -> pd.DataFrame:
    return fetch_universe(tickers, start, end)["close"]


def get_returns(tickers=None, start="2006-01-01", end="2018-12-31") -> pd.DataFrame:
    return get_close(tickers, start, end).pct_change().iloc[1:]


def cache_info() -> dict:
    dirs = [p.name for p in CACHE_DIR.iterdir() if p.is_dir()]
    return {
        "cached_tickers": dirs,
        "count": len(dirs),
        "cache_dir": str(CACHE_DIR),
        "skip_list": sorted(_load_skip_list()),
    }

def get_market_cap_constituents_df() -> pd.DataFrame:
    """
    Load market_cap_ranked.csv and return a DataFrame with columns
    Symbol, GICS Sector, GICS Sub-Industry (normalised).
    """
    csv_path = _HERE / "market_cap_ranked.csv"
    try:
        df = pd.read_csv(csv_path)
        # Find the symbol column
        sym_col = next((c for c in df.columns if c.strip().lower() in ("symbol", "ticker")), None)
        if sym_col is None:
            raise ValueError("No Symbol/Ticker column found in market_cap_ranked.csv")
        # Find sector and industry columns
        sector_col = next((c for c in df.columns if c.strip().lower() == "sector"), None)
        industry_col = next((c for c in df.columns if c.strip().lower() == "industry"), None)
        if sector_col is None:
            raise ValueError("No Sector column found in market_cap_ranked.csv")
        if industry_col is None:
            raise ValueError("No Industry column found in market_cap_ranked.csv")
        # Rename and select columns
        df = df.rename(columns={
            sym_col: "Symbol",
            sector_col: "GICS Sector",
            industry_col: "GICS Sub-Industry"
        })
        # Normalise Symbol (same as sp500.csv)
        df["Symbol"] = (
            df["Symbol"]
            .astype(str)
            .str.strip()
            .str.upper()
            .str.replace(".", "-", regex=False)
        )
        # Drop any rows where Symbol is missing
        df = df.dropna(subset=["Symbol"])
        # Keep only needed columns
        df = df[["Symbol", "GICS Sector", "GICS Sub-Industry"]].copy()
        return df
    except Exception as e:
        raise RuntimeError(f"[fetcher] Failed to load market_cap_ranked.csv: {e}")