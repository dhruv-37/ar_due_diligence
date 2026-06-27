"""
agents/red_flag_agent.py
========================
Analyses financial ratios (computed from Excel + live FMP data) and
produces a structured list of red flags with severity ratings.

Red flag categories
-------------------
1. Profitability   — PAT margin, revenue growth vs FMP net profit margin
2. Quality         — Operating CF / PAT (earnings quality)
3. Leverage        — Debt/Equity, Interest Coverage
4. OCI             — OCI / PAT ratio (aggressive OCI usage signal)
5. Exceptional     — Exceptional items / PBT (one-time noise signal)
6. Valuation       — P/E, ROE, ROA from FMP

Severity levels
---------------
HIGH   — materially outside acceptable range; needs immediate attention
MEDIUM — warrants monitoring; may be sector-specific
LOW    — minor deviation; informational only
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# ── Project root on sys.path ──────────────────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tools.ratio_tool import ratio_tool
from tools.excel_reader_tool import excel_reader_tool


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

_THRESHOLDS = {
    # Profitability
    "pat_margin_pct":           {"low": 5,   "high": 20},   # below low = red flag
    "revenue_growth_pct":       {"low": 0,   "high": 30},   # negative = red flag
    # Quality
    "operating_cf_to_pat":      {"low": 0.7, "high": 2.0},  # below 0.7 = poor quality
    # OCI / Exceptional noise
    "oci_to_pat_pct":           {"warn": 20},                # above 20% = flag
    "exceptional_to_pbt_pct":   {"warn": 10},                # above 10% = flag
    # FMP ratios
    "debt_to_equity":           {"warn": 2.0},               # above 2 = high leverage
    "current_ratio":            {"low": 1.0},                # below 1 = liquidity risk
    "roe":                      {"low": 0.10},               # below 10% = weak returns
}


# ─────────────────────────────────────────────────────────────────────────────
# RULE ENGINE  (pure Python, no LLM needed for flag detection)
# ─────────────────────────────────────────────────────────────────────────────

def _flag(metric: str, value: float, message: str, severity: str) -> dict:
    return {
        "metric":   metric,
        "value":    value,
        "message":  message,
        "severity": severity,
    }


def detect_red_flags(computed: dict, fmp: dict) -> list[dict]:
    flags: list[dict] = []
    t = _THRESHOLDS

    # ── PAT Margin ────────────────────────────────────────────────────────────
    v = computed.get("pat_margin_pct")
    if v is not None:
        if v < t["pat_margin_pct"]["low"]:
            flags.append(_flag("pat_margin_pct", v,
                f"PAT margin {v:.1f}% is below 5% — weak profitability.", "HIGH"))
        elif v < 10:
            flags.append(_flag("pat_margin_pct", v,
                f"PAT margin {v:.1f}% is below 10% — moderate concern.", "MEDIUM"))

    # ── Revenue Growth ────────────────────────────────────────────────────────
    v = computed.get("revenue_growth_pct")
    if v is not None:
        if v < t["revenue_growth_pct"]["low"]:
            flags.append(_flag("revenue_growth_pct", v,
                f"Revenue declined {v:.1f}% YoY — contraction signal.", "HIGH"))
        elif v > t["revenue_growth_pct"]["high"]:
            flags.append(_flag("revenue_growth_pct", v,
                f"Revenue grew {v:.1f}% — unusually high; verify organic vs inorganic.", "LOW"))

    # ── Earnings Quality ──────────────────────────────────────────────────────
    v = computed.get("operating_cf_to_pat")
    if v is not None:
        if v < t["operating_cf_to_pat"]["low"]:
            flags.append(_flag("operating_cf_to_pat", v,
                f"Operating CF / PAT = {v:.2f} — earnings not backed by cash flow.", "HIGH"))
        elif v > t["operating_cf_to_pat"]["high"]:
            flags.append(_flag("operating_cf_to_pat", v,
                f"Operating CF / PAT = {v:.2f} — unusually high; check working capital.", "LOW"))

    # ── OCI Noise ─────────────────────────────────────────────────────────────
    v = computed.get("oci_to_pat_pct")
    if v is not None and abs(v) > t["oci_to_pat_pct"]["warn"]:
        flags.append(_flag("oci_to_pat_pct", v,
            f"OCI is {v:.1f}% of PAT — aggressive OCI usage may mask true earnings.", "MEDIUM"))

    # ── Exceptional Items ─────────────────────────────────────────────────────
    v = computed.get("exceptional_to_pbt_pct")
    if v is not None and abs(v) > t["exceptional_to_pbt_pct"]["warn"]:
        flags.append(_flag("exceptional_to_pbt_pct", v,
            f"Exceptional items = {v:.1f}% of PBT — one-time items distorting earnings.", "MEDIUM"))

    # ── FMP: Debt / Equity ────────────────────────────────────────────────────
    v = fmp.get("debt_to_equity")
    if v is not None and v > t["debt_to_equity"]["warn"]:
        flags.append(_flag("debt_to_equity", v,
            f"Debt/Equity = {v:.2f} — high leverage; monitor interest coverage.", "HIGH"))

    # ── FMP: Current Ratio ────────────────────────────────────────────────────
    v = fmp.get("current_ratio")
    if v is not None and v < t["current_ratio"]["low"]:
        flags.append(_flag("current_ratio", v,
            f"Current ratio = {v:.2f} — below 1; short-term liquidity risk.", "HIGH"))

    # ── FMP: ROE ──────────────────────────────────────────────────────────────
    v = fmp.get("roe")
    if v is not None and v < t["roe"]["low"]:
        flags.append(_flag("roe", v,
            f"ROE = {v*100:.1f}% — below 10%; weak return on shareholder equity.", "MEDIUM"))

    return flags


# ─────────────────────────────────────────────────────────────────────────────
# AGENT
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are a senior financial analyst specialising in Indian listed companies.

You have access to two tools:
- ratio_tool      : fetches live FMP ratios + computes ratios from Excel
- excel_reader_tool: reads the structured Excel file into sheet data

Your job:
1. Call ratio_tool with the ticker and xlsx_path provided by the user.
2. Parse the JSON response.
3. Return ONLY a valid JSON object with this exact structure:
{{
  "ticker": "<ticker>",
  "segment": "<Standalone|Consolidated>",
  "red_flags": [ {{ "metric": "", "value": 0, "message": "", "severity": "HIGH|MEDIUM|LOW" }} ],
  "ratios_used": {{ "computed": {{}}, "fmp": {{}} }}
}}
Do not add any explanation outside the JSON.
"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])


def run_red_flag_agent(ticker: str, xlsx_path: str) -> dict:
    """
    Runs the Red Flag Agent and returns a structured dict of flags.

    Args:
        ticker:    NSE ticker without suffix, e.g. "RELIANCE"
        xlsx_path: Path to the Step2 Excel output

    Returns:
        dict with keys: ticker, segment, red_flags, ratios_used
    """
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        raise EnvironmentError("GEMINI_API_KEY not set.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=gemini_key,
        temperature=0,
    )

    tools = [ratio_tool, excel_reader_tool]
    agent = create_tool_calling_agent(llm, tools, _PROMPT)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    user_input = (
        f"Analyse financial red flags for ticker={ticker}, "
        f"xlsx_path={xlsx_path}. "
        f"Call ratio_tool first, then return the structured JSON."
    )

    raw = executor.invoke({"input": user_input})
    output = raw.get("output", "{}")

    # ── Parse LLM JSON output ─────────────────────────────────────────────────
    try:
        result = json.loads(output)
    except json.JSONDecodeError:
        # LLM sometimes wraps in ```json ... ```
        import re
        match = re.search(r"\{.*\}", output, re.DOTALL)
        result = json.loads(match.group()) if match else {"error": output}

    # ── Run deterministic rule engine on top of LLM result ───────────────────
    # This ensures flags are never missed even if LLM hallucinates
    ratios_used = result.get("ratios_used", {})
    computed    = ratios_used.get("computed", {})
    fmp         = ratios_used.get("fmp", {})

    deterministic_flags = detect_red_flags(computed, fmp)

    # Merge: deduplicate by metric, prefer deterministic over LLM
    existing_metrics = {f["metric"] for f in result.get("red_flags", [])}
    for flag in deterministic_flags:
        if flag["metric"] not in existing_metrics:
            result.setdefault("red_flags", []).append(flag)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("ticker",    help="NSE ticker, e.g. RELIANCE")
    parser.add_argument("xlsx_path", help="Path to Step2 Excel output")
    args = parser.parse_args()

    result = run_red_flag_agent(args.ticker, args.xlsx_path)
    print(json.dumps(result, indent=2))