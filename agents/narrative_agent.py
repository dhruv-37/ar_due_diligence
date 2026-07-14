"""
agents/narrative_agent.py
=========================
Cross-checks management claims in the MD&A section against the actual
numbers extracted by Step2.

Pipeline
--------
1.  mda_extractor_tool  → raw MD&A text
2.  _extract_claims     → LLM, schema-driven: each claim is tagged with a
                          taxonomy_node/scope/metric_type at extraction time
                          (same schema-driven-extraction shift as Step2).
3.  excel_reader_tool   → structured financial data (line_item, taxonomy_node,
                          scope, excel_row per row)
4.  _verify_claims      → deterministic Python lookup + variance calculation.
                          No LLM involved in the math verification step.
5.  Returns a structured list of claim verdicts.

Verdict types
-------------
CONFIRMED    — claim within 2% tolerance of the extracted number
OVERSTATED   — management claim is higher than actual number (beyond tolerance)
UNDERSTATED  — management claim is lower than actual number (beyond tolerance)
UNVERIFIABLE — taxonomy_node is UNMAPPED, scope is UNKNOWN, or no matching
               Excel row was found for the node/scope pair
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

# ── Project root on sys.path ──────────────────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tools.mda_extractor_tool import mda_extractor_tool
from tools.excel_reader_tool import excel_reader_tool
from tools.llm_cache import cached_invoke

try:
    from pipeline.taxonomy import get_taxonomy_enums
except ImportError:
    def get_taxonomy_enums() -> list:
        return ["UNMAPPED"]

VARIANCE_TOLERANCE = 0.02  # 2%


# ─────────────────────────────────────────────────────────────────────────────
# CLAIM EXTRACTOR  (LLM — extracts structured claims from MD&A)
# ─────────────────────────────────────────────────────────────────────────────

_CLAIM_EXTRACTION_PROMPT = """
You are a financial analyst. Below is the MD&A section of an Indian Annual Report.

Extract every quantitative claim management makes about financial performance.
Focus on: revenue, profit, margins, growth rates, cash flow, debt, EPS.

For EACH claim, output:
- claim_text     : exact or close paraphrase of the management statement.
- taxonomy_node  : the Internal Taxonomy Node this claim is about, chosen from the
                    allowed enum list. If nothing fits, use "UNMAPPED" — do NOT guess.
                    Node disambiguation for revenue claims:
                      * Use "REVENUE_FROM_OPERATIONS" ONLY when the claim explicitly
                        says "revenue from operations" (or "net/total revenue from
                        operations").
                      * Any other generic revenue claim (e.g. "Value of Sales and
                        Services (Revenue)", "revenue", "turnover", "gross revenue")
                        → use "REVENUE_GROSS".
- scope          : "STANDALONE", "CONSOLIDATED", or "UNKNOWN" if the claim doesn't
                    specify (or you can't tell) which segment it refers to.
- claimed_value  : the numeric value management is claiming, as a float.
- metric_type    : "ABSOLUTE" (a plain rupee figure), "GROWTH_PCT" (a YoY growth %),
                    or "MARGIN_PCT" (a margin/ratio %).
- period         : "CURRENT_YEAR" or "PREVIOUS_YEAR" — which year's figure this claim
                    refers to. Default to "CURRENT_YEAR" if not specified.

MD&A Text:
{mda_text}
"""


def _build_claim_extraction_schema():
    """
    Gemini response_schema for schema-driven claim extraction — taxonomy_node
    and scope are assigned by the LLM at extraction time (mirrors Step2's
    schema-driven extraction), instead of being inferred later by a second
    LLM comparison pass.
    """
    enums = get_taxonomy_enums()
    return {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "claim_text":    {"type": "STRING"},
                "taxonomy_node": {"type": "STRING", "enum": enums},
                "scope":         {"type": "STRING", "enum": ["STANDALONE", "CONSOLIDATED", "UNKNOWN"]},
                "claimed_value": {"type": "NUMBER"},
                "metric_type":   {"type": "STRING", "enum": ["ABSOLUTE", "GROWTH_PCT", "MARGIN_PCT"]},
                "period":        {"type": "STRING", "enum": ["CURRENT_YEAR", "PREVIOUS_YEAR"]},
            },
            "required": ["claim_text", "taxonomy_node", "scope", "claimed_value", "metric_type", "period"],
        },
    }


def _extract_claims(mda_text: str, llm: ChatGoogleGenerativeAI) -> list[dict]:
    """Uses LLM to extract quantitative claims from MD&A text, tagging each
    with taxonomy_node/scope/metric_type directly via response_schema."""
    # Use only first 12000 chars to stay within context limits
    truncated = mda_text[:12000]
    prompt    = _CLAIM_EXTRACTION_PROMPT.format(mda_text=truncated)

    # Schema is bound onto the llm itself (not passed through cached_invoke,
    # whose signature/caching logic is untouched) — .bind() returns a Runnable
    # with the same .invoke(prompt) interface cached_invoke already expects.
    schema_llm = llm.bind(
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": _build_claim_extraction_schema(),
        }
    )

    # Routed through cached_invoke: retries transient 503 overload errors
    # with backoff, and caches so a re-run on the same PDF/prompt is free.
    response = cached_invoke(schema_llm, prompt, "gemini-2.5-flash-claim-extract-schema")
    raw = response.content if hasattr(response, "content") else str(response)

    try:
        # Strip markdown fences if present
        clean = re.sub(r"```json|```", "", raw).strip()
        claims = json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        claims = json.loads(match.group()) if match else []

    # Defensive normalisation — if the LLM ever returns a node/scope outside
    # the allowed enums (e.g. cached_invoke fell back to a non-schema call),
    # treat it as UNMAPPED/UNKNOWN so the verifier marks it UNVERIFIABLE
    # instead of raising a KeyError downstream.
    allowed_nodes = set(get_taxonomy_enums())
    for c in claims:
        if c.get("taxonomy_node") not in allowed_nodes:
            c["taxonomy_node"] = "UNMAPPED"
        if c.get("scope") not in ("STANDALONE", "CONSOLIDATED", "UNKNOWN"):
            c["scope"] = "UNKNOWN"
        if c.get("period") not in ("CURRENT_YEAR", "PREVIOUS_YEAR"):
            c["period"] = "CURRENT_YEAR"
    return claims


# ─────────────────────────────────────────────────────────────────────────────
# CLAIM VERIFIER  (deterministic Python — no LLM in the math verification
# step). taxonomy_node/scope were assigned at extraction time (schema-driven),
# so verification is now a straight dict lookup + variance calculation.
# ─────────────────────────────────────────────────────────────────────────────

# Maps a canonical excel_reader_tool sheet key's statement suffix → which
# statement group a taxonomy node's `TAXONOMY[node].statement` value falls
# under, so we know which sheet(s) to look the node up in.
_STATEMENT_TO_SHEET_SUFFIX = {
    "Profit and Loss":    "pnl",
    "Balance Sheet":      "balance_sheet",
    "Cash Flow":          "cash_flow",
    "Changes in Equity":  "equity",
}


def _candidate_sheet_keys(taxonomy_node: str, scope: str, sheets: dict) -> list[str]:
    """
    Returns the canonical sheet key(s) (as produced by excel_reader_tool,
    e.g. 'standalone_pnl') to search for a given taxonomy_node + scope.
    If scope is known, only that segment's sheet is searched; the
    statement suffix is derived from the node's own registry metadata,
    with a fallback to searching every sheet for that scope if the node
    isn't found in the taxonomy (defensive — should only happen if
    taxonomy.py and the schema enum ever drift out of sync).
    """
    try:
        from pipeline.taxonomy import TAXONOMY
        node = TAXONOMY.get(taxonomy_node)
    except ImportError:
        node = None

    prefixes = [scope.lower()] if scope in ("STANDALONE", "CONSOLIDATED") else ["standalone", "consolidated"]

    if node is not None:
        suffix = _STATEMENT_TO_SHEET_SUFFIX.get(node.statement.value)
        if suffix:
            return [f"{p}_{suffix}" for p in prefixes if f"{p}_{suffix}" in sheets]

    # Fallback: search every sheet for the relevant scope prefix(es).
    return [k for k in sheets.keys() if any(k.startswith(f"{p}_") for p in prefixes)]


def _find_excel_row(taxonomy_node: str, scope: str, sheets: dict) -> Optional[dict]:
    """
    Looks up the exact Excel row for a given taxonomy_node + scope by
    scanning the candidate sheet(s)' records for a matching taxonomy_node
    value. Returns the row dict (line_item, current_year, previous_year,
    taxonomy_node, excel_row, sheet_name, scope) or None if not found.

    If the taxonomy_node appears on multiple Excel rows (e.g. a headline
    total and a sub-component both mapped to the same node), the row with
    the largest-magnitude current_year value is returned, since the
    headline total is virtually always the larger figure.
    """
    matches = []
    for sheet_key in _candidate_sheet_keys(taxonomy_node, scope, sheets):
        for row in sheets.get(sheet_key, []):
            if row.get("taxonomy_node") == taxonomy_node:
                matches.append(row)

    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    return max(matches, key=lambda r: abs(r.get("current_year") or 0))


_EBITDA_DEBT_KEYWORDS = ("ebitda", "debt", "borrowing")


def _verify_claims(claims: list[dict], sheets: dict, llm=None) -> list[dict]:
    """
    Deterministic verification loop — no LLM call, except for the
    EBITDA/debt UNMAPPED fallback batch (see _llm_compute_verify).

    For each claim:
      - taxonomy_node == "UNMAPPED" and claim text mentions
        ebitda/debt/borrowing  → batched to _llm_compute_verify
      - taxonomy_node == "UNMAPPED" (other) or scope == "UNKNOWN"  → UNVERIFIABLE
      - otherwise, look up the exact value (current_year or previous_year,
        based on the claim's period) in the Excel JSON via taxonomy_node +
        scope, compute variance = (claimed - actual) / |actual|,
        and apply a 2% tolerance:
          within tolerance  → CONFIRMED
          claimed > actual  → OVERSTATED
          claimed < actual  → UNDERSTATED
    """
    verdicts = []
    compute_batch = []  # claims routed to _llm_compute_verify

    for claim in claims:
        taxonomy_node = claim.get("taxonomy_node", "UNMAPPED")
        scope         = claim.get("scope", "UNKNOWN")
        claimed_value = claim.get("claimed_value")
        period        = claim.get("period", "CURRENT_YEAR")

        if taxonomy_node == "UNMAPPED":
            claim_text_lower = str(claim.get("claim_text", "")).lower()
            if llm is not None and any(kw in claim_text_lower for kw in _EBITDA_DEBT_KEYWORDS):
                compute_batch.append(claim)
                continue
            verdicts.append({
                **claim,
                "verdict": "UNVERIFIABLE",
                "actual_value": None,
                "variance_pct": None,
                "note": f"Unverifiable: taxonomy_node={taxonomy_node}, scope={scope}.",
            })
            continue

        if scope == "UNKNOWN":
            verdicts.append({
                **claim,
                "verdict": "UNVERIFIABLE",
                "actual_value": None,
                "variance_pct": None,
                "note": f"Unverifiable: taxonomy_node={taxonomy_node}, scope={scope}.",
            })
            continue

        row = _find_excel_row(taxonomy_node, scope, sheets)
        if row is None:
            verdicts.append({
                **claim,
                "verdict": "UNVERIFIABLE",
                "actual_value": None,
                "variance_pct": None,
                "note": f"No matching Excel row found for {taxonomy_node} ({scope}).",
            })
            continue

        value_col    = "previous_year" if period == "PREVIOUS_YEAR" else "current_year"
        actual_value = row.get(value_col)
        if actual_value in (None, 0) or claimed_value is None:
            verdicts.append({
                **claim,
                "verdict": "UNVERIFIABLE",
                "actual_value": actual_value,
                "variance_pct": None,
                "note": f"Excel value for {taxonomy_node} ({scope}, {period}) is missing or zero — "
                        f"cannot compute variance.",
            })
            continue

        variance = (claimed_value - actual_value) / abs(actual_value)
        sheet_label = str(row.get("sheet_name", taxonomy_node))
        excel_row   = row.get("excel_row")
        location    = f"Found in {sheet_label}, Row {excel_row}" if excel_row else f"Found in {sheet_label}"

        if abs(variance) <= VARIANCE_TOLERANCE:
            verdict = "CONFIRMED"
        elif variance > 0:
            verdict = "OVERSTATED"
        else:
            verdict = "UNDERSTATED"

        variance_pct = round(variance * 100, 2)
        note = f"{verdict.capitalize()}. Variance: {variance_pct}%. {location}."

        verdicts.append({
            **claim,
            "verdict": verdict,
            "actual_value": actual_value,
            "variance_pct": variance_pct,
            "note": note,
        })

    if compute_batch:
        verdicts.extend(_llm_compute_verify(compute_batch, sheets, llm))

    return verdicts


def _llm_compute_verify(claims: list[dict], sheets: dict, llm) -> list[dict]:
    """
    For UNMAPPED claims whose text mentions ebitda/debt/borrowing, has the
    LLM derive the figure from raw P&L/BS line items (standard EBITDA /
    gross-debt / net-debt formulas) instead of returning a blanket
    UNVERIFIABLE. Flags a mismatch only if outside a 7% tolerance.
    """
    TOLERANCE = 0.07
    raw_lines = []
    for sheet_key, rows in sheets.items():
        for row in rows:
            li = row.get("line_item")
            cy = row.get("current_year")
            py = row.get("previous_year")
            if li is not None:
                raw_lines.append(f"[{sheet_key}] {li}: CY={cy}, PY={py}")
    raw_lines_text = "\n".join(raw_lines)

    prompt = f"""
You are a financial analyst. Below are raw P&L / Balance Sheet line items
extracted from an Indian Annual Report, followed by a list of management
claims about EBITDA / debt / borrowings that could not be mapped to a
taxonomy node.

For EACH claim, derive the actual figure using standard formulas
(EBITDA = Profit before tax + Depreciation + Finance costs - Other income
where applicable; Gross Debt = total borrowings (current + non-current);
Net Debt = Gross Debt - Cash & cash equivalents), using the period
(CURRENT_YEAR or PREVIOUS_YEAR) the claim refers to.

Return ONLY a JSON array, one object per claim in the same order, each with:
- "claim_text": copy of the claim text
- "computed_value": the derived actual figure (float), or null if it cannot
  be derived from the line items given.
- "basis": short note on which line items/formula were used.

Raw line items:
{raw_lines_text}

Claims:
{json.dumps([{"claim_text": c.get("claim_text"), "claimed_value": c.get("claimed_value"), "period": c.get("period", "CURRENT_YEAR")} for c in claims])}
"""

    response = cached_invoke(llm, prompt, "gemini-2.5-flash-compute-verify")
    raw = response.content if hasattr(response, "content") else str(response)

    try:
        clean = re.sub(r"```json|```", "", raw).strip()
        computed = json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        computed = json.loads(match.group()) if match else []

    verdicts = []
    for i, claim in enumerate(claims):
        comp = computed[i] if i < len(computed) else {}
        actual_value = comp.get("computed_value")
        claimed_value = claim.get("claimed_value")

        if actual_value in (None, 0) or claimed_value is None:
            verdicts.append({
                **claim,
                "verdict": "UNVERIFIABLE",
                "actual_value": actual_value,
                "variance_pct": None,
                "note": f"LLM could not derive figure from raw line items. {comp.get('basis', '')}".strip(),
            })
            continue

        variance = (claimed_value - actual_value) / abs(actual_value)
        if abs(variance) <= TOLERANCE:
            verdict = "CONFIRMED"
        elif variance > 0:
            verdict = "OVERSTATED"
        else:
            verdict = "UNDERSTATED"

        variance_pct = round(variance * 100, 2)
        verdicts.append({
            **claim,
            "verdict": verdict,
            "actual_value": actual_value,
            "variance_pct": variance_pct,
            "note": f"{verdict.capitalize()} (LLM-derived). Variance: {variance_pct}%. {comp.get('basis', '')}".strip(),
        })

    return verdicts


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
        max_retries=1,  # was defaulting to 2 — each 429 near quota ceiling was costing 3x billed attempts
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

    # Cap to the most material claims (by absolute value) — fewer claims means
    # fewer batches on top of the batching itself, and 56 raw claims is more
    # than we need to cross-check for a due diligence read.
    MAX_CLAIMS = 20
    if len(claims) > MAX_CLAIMS:
        original_count = len(claims)
        claims = sorted(claims, key=lambda c: abs(c.get("claimed_value") or 0), reverse=True)[:MAX_CLAIMS]
        print(f"  ⚠️  Capped {original_count} claims → top {MAX_CLAIMS} by materiality.")

    # ── Step 4: Load Excel sheets ─────────────────────────────────────────────
    print("\n── Narrative Agent: Loading Excel data...")
    excel_result = json.loads(excel_reader_tool.invoke({"xlsx_path": xlsx_path}))
    if excel_result.get("status") == "error":
        return {"status": "error", "error": f"Excel read failed: {excel_result.get('error')}"}

    sheets = excel_result["sheets"]

    # ── Step 5: Verify each claim deterministically (no LLM) ──────────────────
    print("\n── Narrative Agent: Verifying claims against extracted numbers (deterministic)...")
    verdicts = _verify_claims(claims, sheets, llm)
    print(f"  ✅ Verified {len(claims)} claims.")

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