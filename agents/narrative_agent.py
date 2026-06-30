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

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# ── Project root on sys.path ──────────────────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tools.mda_extractor_tool import mda_extractor_tool
from tools.excel_reader_tool import excel_reader_tool

_VECTORSTORE_DIR = str(Path(_PROJECT_ROOT) / "data" / "vectorstore")


# ─────────────────────────────────────────────────────────────────────────────
# VECTOR STORE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_vectorstore(mda_text: str, pdf_stem: str, gemini_key: str) -> Chroma:
    """
    Chunks MD&A text and embeds into a persistent Chroma vector store.
    Re-uses existing store if already built for this PDF stem.
    """
    persist_dir = str(Path(_VECTORSTORE_DIR) / pdf_stem)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=gemini_key,
    )

    # Re-use existing store
    if Path(persist_dir).exists():
        print(f"  ✅ Loading existing vectorstore for {pdf_stem}")
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )

    print(f"  🔨 Building vectorstore for {pdf_stem}...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_text(mda_text)

    store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    print(f"  ✅ Vectorstore built — {len(chunks)} chunks indexed.")
    return store


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
# CLAIM VERIFIER  (deterministic — compares claims against Excel numbers)
# ─────────────────────────────────────────────────────────────────────────────

# Maps metric names LLM might use → taxonomy nodes in Excel
_METRIC_TO_NODE = {
    "revenue":                  "REVENUE_FROM_OPERATIONS",
    "revenue_growth":           "REVENUE_FROM_OPERATIONS",
    "revenue_growth_pct":       "REVENUE_FROM_OPERATIONS",
    "profit":                   "PROFIT_FOR_THE_YEAR",
    "pat":                      "PROFIT_FOR_THE_YEAR",
    "net_profit":               "PROFIT_FOR_THE_YEAR",
    "profit_growth_pct":        "PROFIT_FOR_THE_YEAR",
    "pbt":                      "PROFIT_BEFORE_TAX",
    "profit_before_tax":        "PROFIT_BEFORE_TAX",
    "total_income":             "TOTAL_INCOME",
    "ebitda":                   "PROFIT_BEFORE_EXCEPTIONAL",
    "cash_flow":                "NET_CASH_FROM_OPERATING",
    "operating_cash_flow":      "NET_CASH_FROM_OPERATING",
    "eps":                      "EARNINGS_PER_SHARE",
}

_TOLERANCE = 0.02  # 2% tolerance for CONFIRMED verdict


def _find_actual(node: str, sheets: dict) -> tuple[float | None, float | None]:
    """Returns (current_year, previous_year) for a taxonomy node from sheets."""
    for prefix in ["Standalone", "Consolidated"]:
        for stmt_suffix in ["P&L", "Balance Sheet", "Cash Flow", "Changes in Equity"]:
            sheet_name = f"{prefix} - {stmt_suffix}"
            records = sheets.get(sheet_name, [])
            for r in records:
                if str(r.get("taxonomy_node", "")).upper() == node:
                    cur  = r.get("current_year")
                    prev = r.get("previous_year")
                    try:
                        return float(cur), float(prev) if prev is not None else None
                    except (TypeError, ValueError):
                        continue
    return None, None


def _verify_claim(claim: dict, sheets: dict) -> dict:
    """
    Compares a single management claim against extracted numbers.
    Returns a verdict dict.
    """
    metric   = str(claim.get("metric", "")).lower()
    cl_value = claim.get("value")
    unit     = claim.get("unit", "")

    node = _METRIC_TO_NODE.get(metric)
    if not node:
        return {**claim, "verdict": "UNVERIFIABLE", "actual_value": None,
                "delta_pct": None, "note": f"No taxonomy mapping for metric '{metric}'"}

    cur, prev = _find_actual(node, sheets)

    if cur is None:
        return {**claim, "verdict": "UNVERIFIABLE", "actual_value": None,
                "delta_pct": None, "note": f"Node {node} not found in extracted data"}

    # For growth % claims, compute actual YoY growth
    if "growth" in metric and prev and prev != 0:
        actual = (cur - prev) / abs(prev) * 100
    else:
        actual = cur

    if cl_value is None:
        return {**claim, "verdict": "UNVERIFIABLE", "actual_value": actual,
                "delta_pct": None, "note": "No numeric value in claim"}

    try:
        cl_float = float(cl_value)
        if cl_float == 0:
            delta_pct = 0.0
        else:
            delta_pct = (actual - cl_float) / abs(cl_float)
    except (TypeError, ValueError):
        return {**claim, "verdict": "UNVERIFIABLE", "actual_value": actual,
                "delta_pct": None, "note": "Could not parse claim value as float"}

    if abs(delta_pct) <= _TOLERANCE:
        verdict = "CONFIRMED"
    elif delta_pct < 0:
        verdict = "OVERSTATED"   # actual < claimed
    else:
        verdict = "UNDERSTATED"  # actual > claimed

    return {
        **claim,
        "verdict":      verdict,
        "actual_value": round(actual, 2),
        "delta_pct":    round(delta_pct * 100, 2),
        "note":         f"Actual={actual:.2f} vs Claimed={cl_float:.2f} ({unit})",
    }


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

    # ── Step 2: Build / load vector store ────────────────────────────────────
    print("\n── Narrative Agent: Building vector store...")
    _build_vectorstore(mda_text, pdf_stem, gemini_key)

    # ── Step 3: Extract claims from MD&A via LLM ─────────────────────────────
    print("\n── Narrative Agent: Extracting quantitative claims from MD&A...")
    claims = _extract_claims(mda_text, llm)
    print(f"  ✅ {len(claims)} claims extracted.")

    # ── Step 4: Load Excel sheets ─────────────────────────────────────────────
    print("\n── Narrative Agent: Loading Excel data...")
    excel_result = json.loads(excel_reader_tool.invoke({"xlsx_path": xlsx_path}))
    if excel_result.get("status") == "error":
        return {"status": "error", "error": f"Excel read failed: {excel_result.get('error')}"}

    sheets = excel_result["sheets"]

    # ── Step 5: Verify each claim ─────────────────────────────────────────────
    print("\n── Narrative Agent: Verifying claims against extracted numbers...")
    verdicts = [_verify_claim(claim, sheets) for claim in claims]

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