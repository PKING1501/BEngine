# Quant Forge v3 — Alpha Backtesting Engine

Quant Forge is a professional-grade alpha backtesting platform. It allows you to write, test, and analyse quantitative trading strategies using a powerful expression language and realistic transaction costs. The system includes a Flask API, a web frontend, and optional AI‑powered feedback from LLMs (OpenAI, Anthropic, DeepSeek, Gemini, or local Ollama).

---

## Features

- **Alpha Expression Engine** – Evaluate custom formulas using time‑series and cross‑sectional operators (`rank`, `zscore`, `delta`, `ts_mean`, `correlation`, …)
- **Realistic Backtesting** – Daily long/short portfolio simulation with transaction costs, position limits, and neutralisation (market, sector, industry).
- **Multiple Universes** – SP500, TOP200, TOP500, TOP1000, TOP3000 (via `market_cap_ranked.csv`).
- **Performance Metrics** – Sharpe, Sortino, Calmar, drawdown, turnover, profit factor, VPIN, and more.
- **LLM Feedback** – Automatically generate strategy critique and improvement suggestions using your favourite LLM provider.
- **Web Interface** – Interactive frontend to run backtests and view results.
- **Disk Cache** – Stores historical OHLCV data locally to avoid repeated downloads.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/quant-forge.git
cd quant-forge
```

### 2. Set up a Python environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r req.txt
```

If you encounter issues with `curl_cffi`, you may need to install it separately:

```bash
pip install curl_cffi --upgrade
```

### 4. Environment variables (`.env` file)

Create a `.env` file in the project root and add API keys for your chosen LLM provider (optional – the backtest works without LLM feedback).

```ini
# Choose one of: openai, anthropic, deepseek, gemini, local
LLM_PROVIDER=gemini

# Keys (only needed for the selected provider)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
DEEPSEEK_API_KEY=...
GEMINI_API_KEY=...

# Optional: override Gemini model (default gemini-2.5-flash)
GEMINI_MODEL=gemini-2.5-pro
```

The LLM provider is configured inside `llm_feedback.py` (variable `PROVIDER`). By default it is set to `"gemini"`. Change it there if you want a different provider.

### 5. Universe data

The repository already includes:
- `sp500.csv` – SP500 constituents with sectors and sub‑industries (used for sector/industry neutralisation).
- `market_cap_ranked.csv` – Market‑cap ranked universe for TOP200/1000/3000.

If you want to update or customise the universes, replace these CSV files with your own data. The column names must include `Symbol`, `Sector`, and `Sub-Industry` (or `Industry`).

---

## Running the Application

Start the Flask server:

```bash
python app.py
```

The server will start at `http://localhost:5000`. Open this URL in your browser to access the frontend.

The API will automatically download missing price data into the `cache/` directory. **First runs may take a while** (downloading hundreds of tickers). Subsequent runs are much faster.

---

## API Endpoints

| Method | Endpoint          | Description                               |
|--------|-------------------|-------------------------------------------|
| GET    | `/`               | Serve the frontend (`frontend/index.html`)|
| GET    | `/api/health`     | Liveness probe                            |
| GET    | `/api/presets`    | List preset alphas, available universes, neutralisation modes |
| GET    | `/api/cache`      | Show cached tickers and skip list         |
| POST   | `/api/run`        | Run a backtest (see payload below)        |

### `POST /api/run` – Example Payload

```json
{
  "expression": "-1 * rank(delta(close, 1))",
  "tickers": null,
  "start": "2010-01-01",
  "end": "2020-12-31",
  "tcost_bps": 10.0,
  "max_weight": 0.10,
  "neutralisation": "market",
  "universe": "TOPSP500"
}
```

| Field               | Type          | Description                                                                 |
|---------------------|---------------|-----------------------------------------------------------------------------|
| `expression`        | string        | Alpha expression (required).                                               |
| `tickers`           | list or null  | Override universe with a specific list of tickers.                         |
| `start`, `end`      | string (YYYY-MM-DD) | Backtest period.                                                           |
| `tcost_bps`         | float         | One‑way transaction cost in basis points (default 10).                     |
| `max_weight`        | float         | Maximum absolute weight per stock (default 0.10).                          |
| `neutralisation`    | string        | `"none"`, `"market"`, `"sector"`, `"industry"`, `"subindustry"`.           |
| `universe`          | string        | `"TOPSP500"`, `"TOP500"`, `"TOP200"`, `"TOP1000"`, `"TOP3000"`.            |

The response contains `summary` (performance metrics), `charts` (time series data), `meta` (backtest parameters), and optional `llm_feedback`.

---

## Example Alpha Expressions

| Name                          | Expression                                                                 |
|-------------------------------|----------------------------------------------------------------------------|
| 1‑Day Reversal                | `-1 * rank(delta(close, 1))`                                               |
| Overnight Gap Reversal        | `-1 * rank(open - delay(close, 1))`                                        |
| VWAP Deviation                | `-1 * rank(close - vwap)`                                                  |
| Volume‑Normalised Reversal    | `-1 * rank(delta(close, 1) / (ts_std(close, 20) + 1e-6))`                  |
| Momentum (20d)                | `rank(delta(close, 20))`                                                   |
| Low Volatility Factor         | `-1 * rank(ts_std(returns, 20))`                                           |

> **Note:** The engine **does not** have a `market_cap` variable. Use `adv` (20‑day average dollar volume) as a size proxy.

---

## Changing the Version String in the Frontend

The frontend is located at `frontend/index.html`. The version number (e.g., `Quant Forge v3`) is hardcoded in that file.

To change it:

1. Open `frontend/index.html` in any text editor.
2. Search for the string `v3` or `Quant Forge v3` (case‑insensitive).
3. Replace it with your desired version, e.g., `Quant Forge v4` or `Alpha Lab v1`.
4. Save the file and refresh your browser.

If you want the version to be dynamically served from the backend, you would need to modify `app.py` to pass a version variable to the template and use a templating engine (like Jinja2). The current implementation is a static HTML file, so a direct edit is the simplest way.

---

## Customisation & Advanced Configuration

### LLM Provider

Edit `llm_feedback.py` and change the `PROVIDER` variable at the top:

```python
PROVIDER = "openai"   # or "anthropic", "deepseek", "gemini", "local"
```

For local Ollama, ensure Ollama is running with a model like `llama3.2:3b` and the API endpoint `http://localhost:11434`.

### Transaction Costs & Position Limits

Default values can be changed in the `/api/run` payload. To change global defaults, modify the corresponding parameters in `app.py` (e.g., `tcost_bps = 10.0`).

### Trailing Stop‑Loss (commented out)

The code includes a trailing stop‑loss implementation that is currently **commented out** in `portfolio.py` and `app.py`. To enable it, uncomment the relevant sections and pass `trailing_stop` in the API request.

### Adding New Data Fields

The `AlphaEngine` namespace includes `close`, `open`, `high`, `low`, `volume`, `returns`, `vwap`, and `adv`. If you need additional fields (e.g., fundamental data), you must extend `data_fetcher.py` and `alpha.py`.

---

## Project Structure

```
quant-forge/
├── app.py                     # Flask API entry point
├── llm_feedback.py            # LLM feedback generation
├── engine/
│   ├── alpha.py               # Alpha expression evaluator
│   ├── backtest.py            # Orchestrator
│   ├── data_fetcher.py        # Yahoo Finance download & caching
│   ├── metrics.py             # Performance metrics
│   └── portfolio.py           # Portfolio simulation
├── frontend/
│   └── index.html             # Web UI (edit here to change version)
├── cache/                     # OHLCV cache (auto‑created)
├── sp500.csv                  # SP500 constituents
├── market_cap_ranked.csv      # Extended universe data
├── req.txt                    # Python dependencies
└── .env                       # API keys (not committed)
```

---

## Known Limitations

- **Lookahead alphas** (e.g., `delay(returns, -1)`) are marked with a warning – they use future data and are **not** suitable for live trading.
- The default date range for data download starts from `2000-01-01`. To change it, edit `MASTER_START` in `data_fetcher.py`.
- `yfinance` can be rate‑limited. The code includes retries and a delay between downloads.
- `TOP1000` and `TOP3000` universes rely on `market_cap_ranked.csv`. The provided file includes 5516 rows; if you need a different composition, replace the CSV.

---

## License

This project is provided for educational and research purposes. Use at your own risk.

---

## Acknowledgements

- Built with [yfinance](https://github.com/ranaroussi/yfinance), [Flask](https://flask.palletsprojects.com/), [pandas](https://pandas.pydata.org/), and [NumPy](https://numpy.org/).