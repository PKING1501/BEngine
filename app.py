"""
app.py
------
Thin Flask API layer.

Endpoints
---------
  GET  /                   — serve frontend/index.html
  GET  /api/health         — liveness probe
  GET  /api/presets        — preset alphas + SP500 universe
  GET  /api/cache          — cache stats
  POST /api/run            — run a backtest

NEW PARAMETERS (vs V2)
  trailing_stop   : float|null  — trailing stop-loss threshold (0.05 = 5%)
  neutralisation  : str         — "none"|"market"|"sector"|"industry"|"subindustry"
  universe        : str         — "TOPSP500"|"TOP500"|"TOP200"|"TOP1000"|"TOP3000"
"""

import json
import sys
import traceback
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, Response, jsonify, request, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.backtest     import run_backtest
from engine.alpha        import PRESET_ALPHAS, LOOKAHEAD_ALPHAS
from engine.data_fetcher import get_sp500_tickers, cache_info
from llm_feedback import get_llm_feedback

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")

app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024   # 1 MB request cap

_VALID_NEUTRALISATIONS = {"none", "market", "sector", "industry", "subindustry"}
_VALID_UNIVERSES       = {"TOPSP500", "TOP500", "TOP200", "TOP1000", "TOP3000"}


# ── CORS ─────────────────────────────────────────────────────────────────────
@app.after_request
def _cors(response: Response) -> Response:
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/", methods=["OPTIONS"])
@app.route("/api/<path:_>", methods=["OPTIONS"])
def _options(_=None):
    return Response(status=204)


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    if not FRONTEND_DIR.exists():
        return (
            "<h2>Frontend directory not found.</h2>"
            f"<p>Expected: <code>{FRONTEND_DIR}</code></p>"
            "<p>Create <code>frontend/index.html</code> or run from the project root.</p>",
            404,
        )
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "version": "v3"})


@app.route("/api/presets")
def presets():
    return jsonify({
        "presets":            list(PRESET_ALPHAS.keys()),
        "expressions":        PRESET_ALPHAS,
        "lookahead_alphas":   list(LOOKAHEAD_ALPHAS),
        "default_universe":   get_sp500_tickers(),
        "universes":          sorted(_VALID_UNIVERSES),
        "neutralisations":    sorted(_VALID_NEUTRALISATIONS),
    })


@app.route("/api/cache")
def cache_status():
    try:
        return jsonify(cache_info())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/run", methods=["POST"])
def run():
    body = {}
    try:
        body = request.get_json(force=True, silent=True) or {}
    except Exception:
        pass

    # ── required ──────────────────────────────────────────────────────────
    expression = (body.get("expression") or "").strip()
    if not expression:
        return jsonify({"error": "No alpha expression provided."}), 400

    # ── tickers ───────────────────────────────────────────────────────────
    tickers = body.get("tickers") or None
    if tickers is not None:
        if not isinstance(tickers, list):
            return jsonify({"error": "'tickers' must be a JSON array."}), 400
        tickers = [str(t).strip().upper() for t in tickers if t]
        if not tickers:
            tickers = None

    # ── dates ─────────────────────────────────────────────────────────────
    try:
        start = str(body.get("start", "2006-01-01"))
        end   = str(body.get("end",   "2018-12-31"))
    except Exception:
        return jsonify({"error": "Invalid start/end date."}), 400

    # ── tcost / max_weight ────────────────────────────────────────────────
    try:
        tcost_bps = float(body.get("tcost_bps", 10.0))
    except (TypeError, ValueError):
        return jsonify({"error": "'tcost_bps' must be a number."}), 400

    try:
        max_weight = float(body.get("max_weight", 0.10))
    except (TypeError, ValueError):
        return jsonify({"error": "'max_weight' must be a number."}), 400

    # ── trailing stop ─────────────────────────────────────────────────────
    # raw_stop = body.get("trailing_stop", None)
    # trailing_stop = None
    # if raw_stop is not None and raw_stop != "" and raw_stop is not False:
    #     try:
    #         trailing_stop = float(raw_stop)
    #         if trailing_stop <= 0:
    #             trailing_stop = None
    #     except (TypeError, ValueError):
    #         return jsonify({"error": "'trailing_stop' must be a positive number or null."}), 400

    # ── neutralisation ────────────────────────────────────────────────────
    neutralisation = str(body.get("neutralisation", "none")).lower().strip()
    if neutralisation not in _VALID_NEUTRALISATIONS:
        return jsonify({"error": f"'neutralisation' must be one of: {sorted(_VALID_NEUTRALISATIONS)}"}), 400

    # ── universe ──────────────────────────────────────────────────────────
    universe = str(body.get("universe", "TOPSP500")).upper().strip()
    if universe not in _VALID_UNIVERSES:
        return jsonify({"error": f"'universe' must be one of: {sorted(_VALID_UNIVERSES)}"}), 400

    # ── run ───────────────────────────────────────────────────────────────
    print(f"\n[api/run] expr={expression!r}  start={start}  end={end}  "
          f"tcost={tcost_bps}bps  max_wt={max_weight}  "
          f"neutral={neutralisation}  universe={universe}")

    try:
        result = run_backtest(
            expression     = expression,
            tickers        = tickers,
            start          = start,
            end            = end,
            tcost_bps      = tcost_bps,
            max_weight     = max_weight,
            # trailing_stop  = trailing_stop,
            neutralisation = neutralisation,
            universe       = universe,
        )
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[api/run] UNEXPECTED ERROR:\n{tb}")
        return jsonify({
            "error": f"Internal server error: {exc}",
            "details": tb,
        }), 500

    if "error" in result and "summary" not in result:
        return jsonify(result), 422

    if "error" not in result:
        try:
            feedback = get_llm_feedback(result)
            result["llm_feedback"] = feedback
        except Exception as e:
            result["llm_feedback"] = f"Could not get AI feedback: {e}"
    
    return jsonify(result), 200


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  QUANT FORGE  v3  —  Alpha Backtesting Engine")
    print("=" * 60)
    print(f"  Frontend : {FRONTEND_DIR}")
    print(f"  URL      : http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
