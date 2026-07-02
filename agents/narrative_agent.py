"""
agents/narrative_agent.py
=========================
Cross-checks management claims in the MD&A section against the actual
numbers extracted by Step2.

Pipeline
--------
1.  mda_extractor_tool  → raw MD&A text
2.  Chunk + embed into Chroma vector store (Google text-embedding-004)
3.  excel_reader_tool   → structured financial data
4.  LLM agent queries the vector store for specific claims, then
    compares each claim against the extracted numbers.
5.  Returns a structured list of claim verdicts.

Verdict types
-------------
CONFIRMED   — claim matches extracted number within 2% tolerance
OVERSTATED  — management claim is higher than actual number
UNDERSTATED — management claim is lower than actual number
UNVERIFIABLE— claim found in MD&A but no matching taxonomy node exists
"""

import json
import os
import re
import sys
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# ── Project root on sys.path ──────────────────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tools.mda_extractor_tool import mda_extractor_tool
from tools.excel_reader_tool import excel_reader_tool
from tools.llm_cache import cached_invoke


# ─────────────────────────────────────────────────────────────────────────────
# CLAIM EXTRACTOR  (LLM — extracts structured claims from MD&A)
# ─────────────────────────────────────────────────────────────────────────────

_CLAIM_EXTRACTION_PROMPT = """
You are a financial analyst. Below is the MD&A section of an Indian Annual Report.

Extract every quantitative claim management makes about financial performance.
Focus on: revenue, profit, margins, growth rates, cash flow, debt, EPS.

Return ONLY a JSON array. Each element must have:
{{
  "claim_text": "<exact or close paraphrase of management statement>",
  "metric":     "<what is being claimed, e.g. revenue_growth_pct>",
  "value":      <numeric value as float, e.g. 18.5>,
  "unit":       "<%, ₹ Cr, x, etc.>"
}}

MD&A Text:
{mda_text}
"""


def _extract_claims(mda_text: str, llm: ChatGoogleGenerativeAI) -> list[dict]:
    """Uses LLM to extract quantitative claims from MD&A text."""
    # Use only first 12000 chars to stay within context limits
    truncated = mda_text[:12000]
    prompt    = _CLAIM_EXTRACTION_PROMPT.format(mda_text=truncated)

    response = llm.invoke(prompt)
    raw      = response.content if hasattr(response, "content") else str(response)

    try:
        # Strip markdown fences if present
        clean = re.sub(r"```json|```", "", raw).strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        return json.loads(match.group()) if match else []


# ─────────────────────────────────────────────────────────────────────────────
# CLAIM VERIFIER  (LLM-driven — LLM reads raw sheet snippets and the claim,
# extracts the number itself, computes, and returns the verdict. No Python-side
# metric/alias matching — that was silently mis-extracting values like
# consolidated revenue and producing wrong verdicts.)
# ─────────────────────────────────────────────────────────────────────────────

# Keyword prefilter: claim text/metric → which statement(s) to send.
# Keeps the prompt small by only sending sheets relevant to the claim,
# instead of dumping the full workbook for every claim.
_STMT_KEYWORDS = {
    "pnl": [
        "revenue", "sales", "income", "profit", "pat", "pbt", "ebitda",
        "margin", "expense", "cost", "eps", "earnings", "tax",
    ],
    "balance_sheet": [
        "asset", "liabilit", "debt", "borrowing", "equity", "networth",
        "net worth", "reserve", "capital employed",
    ],
    "cash_flow": [
        "cash flow", "cash generated", "cash from operating",
        "investing activities", "financing activities", "free cash flow",
    ],
    "equity": [
        "shareholding", "share capital", "dividend", "buyback",
    ],
}


def _prefilter_sheets(claim: dict, sheets: dict) -> dict:
    """Returns only the sheet snippets relevant to this claim's keywords.
    Falls back to all sheets if nothing matches (avoids starving the LLM
    of data on unmapped keywords)."""
    text = f"{claim.get('claim_text', '')} {claim.get('metric', '')}".lower()

    scope = None
    if "consolidated" in text:
        scope = "consolidated"
    elif "standalone" in text:
        scope = "standalone"
    prefixes = [scope] if scope else ["standalone", "consolidated"]

    stmt_keys = [k for k, kws in _STMT_KEYWORDS.items() if any(kw in text for kw in kws)]
    if not stmt_keys:
        stmt_keys = list(_STMT_KEYWORDS.keys())

    snippet = {}
    for prefix in prefixes:
        for stmt_key in stmt_keys:
            sheet_name = f"{prefix}_{stmt_key}"
            records = sheets.get(sheet_name)
            if records:
                snippet[sheet_name] = records

    return snippet if snippet else sheets


_COMPARE_PROMPT = """You are a financial analyst verifying a management claim from an Indian
Annual Report's MD&A against the company's actual reported financial statements.

Claim:
{claim_json}

Relevant extracted Excel sheet data (raw rows: line_item, current_year, previous_year):
{sheets_json}

Instructions:
- Find the correct line item(s) yourself from the sheet data — do not assume any
  Python-side mapping is correct. Watch for standalone vs consolidated scope.
- Do the calculation yourself (growth %, margin, etc. as needed).
- Compare your computed actual value against the claimed value (2% tolerance = CONFIRMED).
- Also weigh in qualitatively where relevant — e.g. if the number is technically
  overstated/understated but explainable by market conditions, competitive
  dynamics, business model shifts, one-offs, etc., reflect that nuance in your note
  rather than a flat pass/fail. Don't force every verdict into the same tone.

Return ONLY JSON, no markdown fences:
{{
  "verdict": "CONFIRMED" | "OVERSTATED" | "UNDERSTATED" | "UNVERIFIABLE",
  "actual_value": <float or null>,
  "delta_pct": <float or null>,
  "note": "<your reasoning, including any qualitative/contextual color>"
}}
"""


def _llm_compare(claim: dict, sheets_snippet: dict, llm: ChatGoogleGenerativeAI, model_name: str) -> dict:
    """Sends claim + raw sheet snippets to the LLM and lets it extract, compute,
    and verdict in one shot. Cached via cached_invoke (key = model + full prompt,
    i.e. keyed on the LLM comparison result, not raw sheet extraction)."""
    prompt = _COMPARE_PROMPT.format(
        claim_json=json.dumps(claim, ensure_ascii=False),
        sheets_json=json.dumps(sheets_snippet, ensure_ascii=False)[:20000],
    )
    response = cached_invoke(llm, prompt, model_name)
    raw = response.content if hasattr(response, "content") else str(response)

    try:
        clean = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        result = json.loads(match.group()) if match else {
            "verdict": "UNVERIFIABLE", "actual_value": None,
            "delta_pct": None, "note": "LLM response could not be parsed",
        }

    return {**claim, **result}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AGENT RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_narrative_agent(pdf_path: str, xlsx_path: str) -> dict:
    """
    Runs the Narrative Agent end-to-end.

    Args:
        pdf_path:  Path to the original (untrimmed) AR PDF.
        xlsx_path: Path to the Step2 Excel output.

    Returns:
        dict with keys:
            status   — "success" | "error"
            verdicts — list of claim verdict dicts
            summary  — counts by verdict type
            error    — only on failure
    """
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        return {"status": "error", "error": "GEMINI_API_KEY not set."}

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # gemini-2.0-flash retired ~June 2026 — was hardcoded to a dead model
        google_api_key=gemini_key,
        temperature=0,
    )

    # ── Step 1: Extract MD&A text ─────────────────────────────────────────────
    print("\n── Narrative Agent: Extracting MD&A text...")
    mda_result = json.loads(mda_extractor_tool.invoke({"pdf_path": pdf_path}))
    if mda_result.get("status") == "error":
        return {"status": "error", "error": f"MD&A extraction failed: {mda_result.get('error')}"}

    mda_text   = mda_result["text"]
    pdf_stem   = Path(pdf_path).stem
    print(f"  ✅ MD&A extracted — pages {mda_result['start_page']}–{mda_result['end_page']}")

    # ── Step 2: Extract claims from MD&A via LLM ─────────────────────────────
    print("\n── Narrative Agent: Extracting quantitative claims from MD&A...")
    claims = _extract_claims(mda_text, llm)
    print(f"  ✅ {len(claims)} claims extracted.")

    # ── Step 4: Load Excel sheets ─────────────────────────────────────────────
    print("\n── Narrative Agent: Loading Excel data...")
    excel_result = json.loads(excel_reader_tool.invoke({"xlsx_path": xlsx_path}))
    if excel_result.get("status") == "error":
        return {"status": "error", "error": f"Excel read failed: {excel_result.get('error')}"}

    sheets = excel_result["sheets"]

    # ── Step 5: Verify each claim via LLM ─────────────────────────────────────
    print("\n── Narrative Agent: Verifying claims against extracted numbers (LLM)...")
    model_name = "gemini-2.5-flash"
    verdicts = [
        _llm_compare(claim, _prefilter_sheets(claim, sheets), llm, model_name)
        for claim in claims
    ]

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        "total":        len(verdicts),
        "CONFIRMED":    sum(1 for v in verdicts if v["verdict"] == "CONFIRMED"),
        "OVERSTATED":   sum(1 for v in verdicts if v["verdict"] == "OVERSTATED"),
        "UNDERSTATED":  sum(1 for v in verdicts if v["verdict"] == "UNDERSTATED"),
        "UNVERIFIABLE": sum(1 for v in verdicts if v["verdict"] == "UNVERIFIABLE"),
    }

    print(f"\n  ✅ Verification complete: {summary}")

    return {
        "status":   "success",
        "verdicts": verdicts,
        "summary":  summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path",  help="Path to original AR PDF")
    parser.add_argument("xlsx_path", help="Path to Step2 Excel output")
    args = parser.parse_args()

    result = run_narrative_agent(args.pdf_path, args.xlsx_path)
    print(json.dumps(result, indent=2))