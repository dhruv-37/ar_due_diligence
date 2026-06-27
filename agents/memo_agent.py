"""
agents/memo_agent.py
====================
LangGraph orchestrator that runs the full due diligence pipeline:

    Extractor → Red Flag Agent → Narrative Agent → Memo Writer

Graph nodes
-----------
extract       : runs Step1 + Step2 via extractor_tool
red_flags     : runs red_flag_agent
narrative     : runs narrative_agent
write_memo    : LLM writes the final markdown memo
END

Conditional edge
----------------
After red_flags: if any HIGH severity flag exists → skip narrative,
go straight to write_memo with an "unreliable AR" warning.
Otherwise → run narrative agent first.

Output
------
Writes <pdf_stem>_memo.md to the output/ directory.
Returns the memo text.
"""

import json
import os
import sys
from pathlib import Path
from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

# ── Project root on sys.path ──────────────────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tools.extractor_tool import extract_financials_tool
from agents.red_flag_agent import run_red_flag_agent
from agents.narrative_agent import run_narrative_agent

_OUTPUT_DIR = Path(_PROJECT_ROOT) / "output"


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH STATE
# ─────────────────────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    # Inputs
    pdf_path   : str
    ticker     : str
    output_xlsx: str

    # Intermediate results
    extraction : dict          # extractor_tool result
    red_flags  : dict          # red_flag_agent result
    narrative  : dict          # narrative_agent result

    # Output
    memo_text  : str
    memo_path  : str
    errors     : Annotated[list[str], operator.add]


# ─────────────────────────────────────────────────────────────────────────────
# NODE: EXTRACT
# ─────────────────────────────────────────────────────────────────────────────

def node_extract(state: PipelineState) -> dict:
    print("\n══ Node: EXTRACT ══════════════════════════════════════════")
    result = json.loads(
        extract_financials_tool.invoke({
            "pdf_path":    state["pdf_path"],
            "output_xlsx": state["output_xlsx"],
        })
    )
    if result.get("status") == "error":
        return {"extraction": result, "errors": [f"Extraction failed: {result.get('error')}"]}

    print(f"  ✅ Extraction complete → {result.get('output_xlsx')}")
    return {"extraction": result}


# ─────────────────────────────────────────────────────────────────────────────
# NODE: RED FLAGS
# ─────────────────────────────────────────────────────────────────────────────

def node_red_flags(state: PipelineState) -> dict:
    print("\n══ Node: RED FLAGS ═════════════════════════════════════════")
    extraction = state.get("extraction", {})

    if extraction.get("status") == "error":
        return {"red_flags": {"error": "Skipped — extraction failed.", "red_flags": []}}

    xlsx_path = extraction.get("output_xlsx") or state["output_xlsx"]

    try:
        result = run_red_flag_agent(state["ticker"], xlsx_path)
        high_count = sum(1 for f in result.get("red_flags", []) if f.get("severity") == "HIGH")
        print(f"  ✅ Red flags: {len(result.get('red_flags', []))} total, {high_count} HIGH")
        return {"red_flags": result}
    except Exception as exc:
        return {"red_flags": {"error": str(exc), "red_flags": []},
                "errors": [f"Red flag agent error: {exc}"]}


# ─────────────────────────────────────────────────────────────────────────────
# CONDITIONAL EDGE: should we run narrative?
# ─────────────────────────────────────────────────────────────────────────────

def should_run_narrative(state: PipelineState) -> str:
    """
    Skip narrative if extraction failed or too many HIGH flags
    (AR deemed unreliable — no point cross-checking narrative).
    """
    if state.get("extraction", {}).get("status") == "error":
        return "write_memo"

    flags     = state.get("red_flags", {}).get("red_flags", [])
    high_count = sum(1 for f in flags if f.get("severity") == "HIGH")

    if high_count >= 3:
        print(f"\n  ⚠️  {high_count} HIGH severity flags — skipping narrative, flagging AR as unreliable.")
        return "write_memo"

    return "narrative"


# ─────────────────────────────────────────────────────────────────────────────
# NODE: NARRATIVE
# ─────────────────────────────────────────────────────────────────────────────

def node_narrative(state: PipelineState) -> dict:
    print("\n══ Node: NARRATIVE ═════════════════════════════════════════")
    xlsx_path = state.get("extraction", {}).get("output_xlsx") or state["output_xlsx"]

    try:
        result = run_narrative_agent(state["pdf_path"], xlsx_path)
        summary = result.get("summary", {})
        print(f"  ✅ Narrative: {summary}")
        return {"narrative": result}
    except Exception as exc:
        return {"narrative": {"status": "error", "error": str(exc), "verdicts": []},
                "errors": [f"Narrative agent error: {exc}"]}


# ─────────────────────────────────────────────────────────────────────────────
# NODE: WRITE MEMO
# ─────────────────────────────────────────────────────────────────────────────

_MEMO_PROMPT = """
You are a senior CA and equity analyst. Write a professional due diligence memo
in markdown for the following Indian company annual report analysis.

Use this structure exactly:
# Due Diligence Memo — {ticker}

## 1. Executive Summary
2-3 sentences summarising the overall assessment.

## 2. Financial Extraction
State whether extraction succeeded and how many pages were processed.

## 3. Red Flags
List every red flag with its severity. If none, say "No red flags identified."
Format each as:
- **[SEVERITY]** `metric`: message

## 4. Narrative vs Numbers
List claim verdicts. Group by verdict type (CONFIRMED / OVERSTATED / UNDERSTATED / UNVERIFIABLE).
If narrative analysis was skipped, explain why.

## 5. Overall Assessment
**Reliability Rating**: RELIABLE / CAUTION / UNRELIABLE
Justify the rating in 2-3 sentences based on the above.

## 6. Recommended Next Steps
3 bullet points of actionable follow-up items.

---
DATA:
Extraction: {extraction}
Red Flags: {red_flags}
Narrative: {narrative}
"""


def node_write_memo(state: PipelineState) -> dict:
    print("\n══ Node: WRITE MEMO ════════════════════════════════════════")
    gemini_key = os.environ.get("GEMINI_API_KEY")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # gemini-2.0-flash retired ~June 2026 — was hardcoded to a dead model
        google_api_key=gemini_key,
        temperature=0.2,
    )

    # Summarise data to keep prompt concise
    extraction_summary = {
        "status":      state.get("extraction", {}).get("status"),
        "total_pages": state.get("extraction", {}).get("total_pages"),
        "output_xlsx": state.get("extraction", {}).get("output_xlsx"),
    }

    red_flags_summary = {
        "flags": state.get("red_flags", {}).get("red_flags", []),
        "error": state.get("red_flags", {}).get("error"),
    }

    narrative_summary = {
        "summary":  state.get("narrative", {}).get("summary"),
        "verdicts": state.get("narrative", {}).get("verdicts", [])[:20],  # cap at 20
        "error":    state.get("narrative", {}).get("error"),
    }

    prompt = _MEMO_PROMPT.format(
        ticker     = state["ticker"],
        extraction = json.dumps(extraction_summary, indent=2),
        red_flags  = json.dumps(red_flags_summary,  indent=2),
        narrative  = json.dumps(narrative_summary,  indent=2),
    )

    response  = llm.invoke(prompt)
    memo_text = response.content if hasattr(response, "content") else str(response)

    # ── Write memo to output/ ─────────────────────────────────────────────────
    pdf_stem  = Path(state["pdf_path"]).stem
    memo_path = str(_OUTPUT_DIR / f"{pdf_stem}_memo.md")
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(memo_path, "w", encoding="utf-8") as f:
        f.write(memo_text)

    print(f"  ✅ Memo written → {memo_path}")
    return {"memo_text": memo_text, "memo_path": memo_path}


# ─────────────────────────────────────────────────────────────────────────────
# BUILD GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(PipelineState)

    g.add_node("extract",    node_extract)
    g.add_node("red_flags",  node_red_flags)
    g.add_node("narrative",  node_narrative)
    g.add_node("write_memo", node_write_memo)

    g.set_entry_point("extract")
    g.add_edge("extract", "red_flags")

    g.add_conditional_edges(
        "red_flags",
        should_run_narrative,
        {"narrative": "narrative", "write_memo": "write_memo"},
    )

    g.add_edge("narrative",  "write_memo")
    g.add_edge("write_memo", END)

    return g.compile()


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_due_diligence(pdf_path: str, ticker: str, output_xlsx: str = "") -> dict:
    """
    Runs the full due diligence pipeline.

    Args:
        pdf_path:     Path to the original AR PDF.
        ticker:       NSE ticker without suffix, e.g. "RELIANCE"
        output_xlsx:  Optional output Excel path. Auto-derived if not given.

    Returns:
        Final PipelineState dict containing memo_text and memo_path.
    """
    pdf_stem    = Path(pdf_path).stem
    output_xlsx = output_xlsx or str(_OUTPUT_DIR / f"{pdf_stem}.xlsx")

    graph = build_graph()

    initial_state: PipelineState = {
        "pdf_path":    pdf_path,
        "ticker":      ticker,
        "output_xlsx": output_xlsx,
        "extraction":  {},
        "red_flags":   {},
        "narrative":   {},
        "memo_text":   "",
        "memo_path":   "",
        "errors":      [],
    }

    final_state = graph.invoke(initial_state)

    if final_state.get("errors"):
        print("\n⚠️  Pipeline completed with errors:")
        for err in final_state["errors"]:
            print(f"  - {err}")

    return final_state


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="AR Due Diligence Agent")
    parser.add_argument("pdf_path", help="Path to the Annual Report PDF")
    parser.add_argument("ticker",   help="NSE ticker, e.g. RELIANCE")
    parser.add_argument("--output-xlsx", default="", help="Optional Excel output path")
    args = parser.parse_args()

    result = run_due_diligence(args.pdf_path, args.ticker, args.output_xlsx)

    print(f"\n✅  Due diligence complete.")
    print(f"    Memo  → {result.get('memo_path')}")
    print(f"    Excel → {result.get('extraction', {}).get('output_xlsx')}")