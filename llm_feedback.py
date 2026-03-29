"""
llm_feedback.py
---------------
Backtest feedback generator using external LLM APIs.
Supports OpenAI, Anthropic, DeepSeek, Google Gemini, and local Ollama.

═══════════════════════════════════════════════════════════════
TEMPERATURE — WHAT IT MEANS
═══════════════════════════════════════════════════════════════

Temperature controls how "random" the model's token choices are.

At each step the model computes a probability distribution over all
possible next tokens.  Temperature scales that distribution:

  temperature = 0.0  →  always pick the single highest-probability token
                         Fully deterministic. Same input always gives
                         identical output. Very focused and repetitive.

  temperature = 0.3  →  slightly sharpen the distribution.
                         Near-deterministic but allows small variations.
                         Best for structured analytical tasks like this one.
                         You get consistent formatting with natural phrasing.

  temperature = 0.7  →  flatten the distribution noticeably.
                         More creative, more varied vocabulary, less
                         predictable. Good for brainstorming or writing.
                         BAD for structured output — the model may
                         spontaneously decide to add bullet points or
                         reorder sections.

  temperature = 1.0  →  use the raw unscaled distribution.
                         High variance. Sometimes brilliant, often wrong.

  temperature > 1.0  →  flatten so aggressively that low-probability
                         tokens get selected frequently. Output becomes
                         incoherent or hallucination-prone.

For THIS use case (strict plain-text, exact section labels, valid
alpha expressions) we use temperature = 0.2.  We want the model to
follow the format precisely and produce syntactically valid expressions,
not explore creative alternatives.

═══════════════════════════════════════════════════════════════
TOKEN BUDGET
═══════════════════════════════════════════════════════════════

max_tokens in all APIs = OUTPUT tokens only (prompt not counted).

Why 1024 cut responses mid-sentence:
  Without a length instruction, the model planned a ~2000-token
  markdown essay and got hard-cut at 1024.

Two-layer fix:
  1. Prompt says "keep response under 520 tokens"
     → model plans a concise reply from token 1
  2. max_tokens = 600  → hard safety ceiling (520 + 15% buffer)

Per-section budget:
  OVERALL ASSESSMENT  (3 lines)         ~  60 tokens
  RETURNS SNAPSHOT    (3 phrases)        ~  25 tokens
  KEY ISSUES          (4 lines)          ~ 100 tokens
  NEXT ALPHA IDEAS    (3 expr + intuit)  ~ 150 tokens
  Labels + spacing                       ~  20 tokens
  ─────────────────────────────────────────────────
  Total                                  ~ 355 tokens
  With 1.4× safety buffer               ~ 500 tokens  ← prompt instruction
  API max_tokens                           600         ← hard ceiling

═══════════════════════════════════════════════════════════════
NOTE ON market_cap
═══════════════════════════════════════════════════════════════
market_cap is NOT available in the engine.  The closest proxy is:
  adv  =  ts_mean(close * volume, 20)   (20-day avg dollar volume)
The prompt uses adv and explicitly forbids market_cap to prevent
the LLM from suggesting expressions that would crash at eval time.
"""

import os
import requests
from dotenv import load_dotenv
load_dotenv()

# ── Provider ──────────────────────────────────────────────────────────────────
# Options: "openai" | "anthropic" | "deepseek" | "gemini" | "local"
PROVIDER = "gemini"

# ── API keys ──────────────────────────────────────────────────────────────────
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
DEEPSEEK_API_KEY  = os.environ.get("DEEPSEEK_API_KEY")
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY")

# ── Token budget ──────────────────────────────────────────────────────────────
PROMPT_TOKEN_LIMIT = 4096   # ceiling stated inside the prompt text
MAX_OUTPUT_TOKENS  = 5000   # hard API ceiling  (PROMPT_TOKEN_LIMIT × 1.2)

# ── Temperature ───────────────────────────────────────────────────────────────
# 0.2 = near-deterministic, strict format compliance, valid expression syntax.
# Do NOT raise above 0.4 for this task — higher values cause the model to
# spontaneously add markdown, reorder sections, or hallucinate operator names.
TEMPERATURE = 0.2


# ══════════════════════════════════════════════════════════════════════════════
#  STATIC REFERENCE BLOCKS  (sourced verbatim from alpha.py)
# ══════════════════════════════════════════════════════════════════════════════

# Every operator that exists in AlphaEngine._build_namespace()
_ALLOWED_OPERATORS = (
    "rank, zscore, delta, ts_mean, ts_std, ts_min, ts_max, ts_rank, ts_zscore, "
    "ts_sum, delay, correlation, covariance, scale, neutralize, "
    "sign, log, abs, power, winsorise"
)

# Every data variable that exists in AlphaEngine._build_namespace()
# NOTE: market_cap is NOT available. adv (20-day avg dollar volume) is the proxy.
_ALLOWED_DATA = (
    "close, open, high, low, volume, returns, vwap, adv"
)

# Operator signatures — so the LLM writes syntactically correct calls
_OPERATOR_SIGNATURES = """\
delta(x, d)           x(t) - x(t-d)
delay(x, d)           shift x by d bars
ts_mean(x, d)         rolling mean, d days
ts_std(x, d)          rolling std, d days
ts_min(x, d)          rolling min, d days
ts_max(x, d)          rolling max, d days
ts_rank(x, d)         time-series pct rank [0,1], d days
ts_zscore(x, d)       rolling z-score, d days
ts_sum(x, d)          rolling sum, d days
correlation(x, y, d)  rolling correlation, d days
covariance(x, y, d)   rolling covariance, d days
rank(x)               cross-sectional pct rank [0,1]
zscore(x)             cross-sectional z-score
neutralize(x)         cross-sectional demean
scale(x)              scale so abs-row-sum == 1
sign(x)               element-wise sign
log(x)                natural log
abs(x)                absolute value
power(x, e)           x ** e
winsorise(x, q)       clip at q / (1-q) percentile"""


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_metrics(summary: dict) -> str:
    lines = []
    for k, v in summary.items():
        if v is not None:
            lines.append(
                f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}"
            )
    return "\n".join(lines)


def _build_prompt(result: dict) -> str:
    summary = result.get("summary", {})
    meta    = result.get("meta",    {})
    warning = result.get("warning", "")

    lookahead_note = (
        "\nCRITICAL WARNING: This alpha uses lookahead bias (future price data). "
        "All metrics below are artificially inflated and unachievable in live trading. "
        "Treat the performance figures as meaningless. Focus feedback on structural issues "
        "and suggest genuinely forward-looking alphas.\n"
    ) if meta.get("has_lookahead") else ""

    metrics_block = _fmt_metrics(summary)

    # ── THE PROMPT ────────────────────────────────────────────────────────
    return f"""You are a senior quantitative analyst at a top systematic hedge fund.
Your task is to evaluate the backtest results of an alpha strategy and provide highly actionable, concise, and implementation-ready feedback.
{lookahead_note}
BACKTEST DETAILS
  Alpha expression : {meta.get('expression', 'N/A')}
  Universe         : {meta.get('n_tickers', 'N/A')} stocks, S&P 500 constituents
  Period           : {meta.get('start', 'N/A')} to {meta.get('end', 'N/A')}
  Trading days     : {summary.get('n_trading_days', 'N/A')}
  Transaction cost : {meta.get('tcost_bps', 'N/A')} bps one-way per trade
  Max position     : {meta.get('max_weight', 'N/A')} of portfolio per stock
  Portfolio type   : Long/Short, daily-rebalanced, dollar-neutral

PERFORMANCE METRICS
{metrics_block}

METRIC DEFINITIONS
  total_return         cumulative net return  (0.08 = +8%)
  annualized_return    geometric CAGR
  annualized_vol       annualised std of daily net PnL
  sharpe_ratio         mean(pnl) / std(pnl) * sqrt(252),  no risk-free deduction
  sortino_ratio        mean(pnl) / downside_deviation * sqrt(252)
  max_drawdown         worst peak-to-trough  (negative decimal)
  calmar_ratio         annualized_return / abs(max_drawdown)
  avg_daily_turnover   avg fraction of portfolio rebalanced per day
  win_rate             fraction of days with positive net PnL
  avg_win / avg_loss   mean PnL on winning / losing days
  profit_factor        abs(sum of wins) / abs(sum of losses)
  skewness             negative = left tail risk
  kurtosis             excess kurtosis, positive = fat tails
  recovery_factor      total_return / abs(max_drawdown)
  pain_index           mean absolute drawdown across full period
  fitness              Sharpe * sqrt(abs(ann_return) / max(turnover, 0.125))
  vpin                 Volume-synchronised Probability of Informed Trading [0,1]

ALLOWED OPERATORS — use ONLY these, nothing else:
{_ALLOWED_OPERATORS}

OPERATOR SIGNATURES (for correct syntax):
{_OPERATOR_SIGNATURES}

ALLOWED DATA VARIABLES — use ONLY these, nothing else:
{_ALLOWED_DATA}
Note: market_cap does not exist. Use adv (20-day avg dollar volume) as size proxy.

STRICT OUTPUT FORMAT — PLAIN TEXT ONLY
No markdown. No asterisks. No hash symbols. No bullet points. No dashes. No bold.
Keep total response under {PROMPT_TOKEN_LIMIT} tokens.
Output EXACTLY 4 sections in this order, each starting with its label on its own line:

OVERALL ASSESSMENT
Write exactly 3 lines. Give a blunt, professional evaluation of strategy quality. State whether it is profitable, cite the most damning metric by its actual value, and give a one-line verdict on viability.

RETURNS SNAPSHOT
Write exactly 3 short phrases, each 5 to 6 words, separated by semicolons on a single line. Describe return behaviour — for example: unstable cumulative PnL; high loss day frequency; severe prolonged drawdown.

KEY ISSUES
Write exactly 4 lines. Identify the most critical problems affecting performance. Be specific — reference actual metric values. Cover alpha signal quality, turnover cost drag, risk exposure, or distribution properties as relevant.

NEXT ALPHA IDEAS
Suggest exactly 3 new alpha expressions. For each, write the expression on one line, then immediately below it write one short line of intuition (max 10 words). No labels, no numbering, no extra punctuation. Format strictly as:
expression
intuition in max ten words

IMPORTANT RULES
Do not use markdown formatting of any kind.
Do not explain basics or define terms.
Do not be generic or use placeholder language.
Be concise, specific, and critical.
Every alpha expression must be valid Python using only the allowed operators and data variables listed above.
Alpha expressions must be copy-pasteable directly into the system without modification.
""".strip()


# ══════════════════════════════════════════════════════════════════════════════
#  API CALLERS
# ══════════════════════════════════════════════════════════════════════════════

def get_llm_feedback(backtest_result: dict) -> str:
    """Call the configured LLM provider and return plain-text feedback."""

    key_map = {
        "openai":    (OPENAI_API_KEY,    "OPENAI_API_KEY"),
        "anthropic": (ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY"),
        "deepseek":  (DEEPSEEK_API_KEY,  "DEEPSEEK_API_KEY"),
        "gemini":    (GEMINI_API_KEY,    "GEMINI_API_KEY"),
    }
    if PROVIDER in key_map:
        key, name = key_map[PROVIDER]
        if not key:
            return f"{name} is not set. Add it to your .env file."

    prompt = _build_prompt(backtest_result)
    print(f"\n[llm_feedback] provider={PROVIDER} | prompt={len(prompt)} chars "
          f"| max_tokens={MAX_OUTPUT_TOKENS} | temperature={TEMPERATURE}")

    # ── OpenAI ────────────────────────────────────────────────────────────
    if PROVIDER == "openai":
        try:
            from openai import OpenAI
            resp = OpenAI(api_key=OPENAI_API_KEY).chat.completions.create(
                model       = "gpt-4o-mini",
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = MAX_OUTPUT_TOKENS,
                temperature = TEMPERATURE,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"OpenAI error: {e}"

    # ── DeepSeek ──────────────────────────────────────────────────────────
    elif PROVIDER == "deepseek":
        try:
            from openai import OpenAI
            resp = OpenAI(
                api_key  = DEEPSEEK_API_KEY,
                base_url = "https://api.deepseek.com/v1",
            ).chat.completions.create(
                model       = "deepseek-chat",
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = MAX_OUTPUT_TOKENS,
                temperature = TEMPERATURE,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"DeepSeek error: {e}"

    # ── Anthropic ─────────────────────────────────────────────────────────
    elif PROVIDER == "anthropic":
        headers = {
            "x-api-key":         ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type":      "application/json",
        }
        payload = {
            "model":      "claude-haiku-4-5-20251001",
            "max_tokens": MAX_OUTPUT_TOKENS,
            "messages":   [{"role": "user", "content": prompt}],
        }
        try:
            r = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers, json=payload, timeout=30,
            )
            print(f"[llm_feedback] status={r.status_code}")
            r.raise_for_status()
            return r.json()["content"][0]["text"].strip()
        except requests.exceptions.HTTPError as e:
            return f"Anthropic error: {e} — {r.text[:300]}"
        except Exception as e:
            return f"Anthropic error: {e}"

    # ── Google Gemini ─────────────────────────────────────────────────────
    elif PROVIDER == "gemini":
        model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        url   = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={GEMINI_API_KEY}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature":     TEMPERATURE,
                "maxOutputTokens": MAX_OUTPUT_TOKENS,
            },
        }
        try:
            r = requests.post(url, json=payload, timeout=30)
            print(f"[llm_feedback] status={r.status_code}")
            r.raise_for_status()
            return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except requests.exceptions.HTTPError as e:
            return f"Gemini error: {e} — {r.text[:300]}"
        except Exception as e:
            return f"Gemini error: {e}"

    # ── Local Ollama ──────────────────────────────────────────────────────
    elif PROVIDER == "local":
        try:
            r = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model":    "llama3.2:3b",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream":   False,
                    "options":  {"temperature": TEMPERATURE},
                },
                timeout=60,
            )
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except Exception as e:
            return f"Ollama error: {e}"

    else:
        return (
            f"Unknown provider '{PROVIDER}'. "
            "Choose from: openai, anthropic, deepseek, gemini, local."
        )