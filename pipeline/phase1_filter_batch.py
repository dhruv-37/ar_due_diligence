"""
Phase 1 — Statutory Auditor Signature Block Extractor
=======================================================
FULL REPLACEMENT of the old TOC-based / header-scan FS-page filter.
All previous logic (heuristic density-gradient filter, TOC prefilter,
header-scan filter, FSType/FilteredPages/TokenBatch dataclasses, etc.)
has been removed entirely and replaced with auditor-signature-block
detection, ported from extract_auditor_signature.py and wrapped as a
LangChain tool so a LangGraph agent node can call it directly.

NOTE — BREAKING CHANGE:
    pipeline/phase2_llm_classify.py and any agent code that imports
    run_phase1 / Phase1Result / TokenBatch / FSType / FilteredPages
    from this module WILL break. Those call sites need to be updated
    separately to consume auditor_signature_tool's output instead.

Three-stage pipeline (unchanged from the standalone script)
-------------------------------------------------------------
Stage 1 — Signature detection
    Score each page against auditor-signature signals (firm name, FRN,
    Chartered Accountants, Partner, Membership No., Date, Place, UDIN).
    Only pages that meet min_score proceed.

Stage 2 — Header drop filter (applied to Stage-1 survivors)
    If the first 300 characters of the page contain "auditor's report"
    or "annexure" (case-insensitive), discard the page. These are intro
    / header pages that carry the phrase but are not the signing page.

Stage 3 — Number-density filter (applied to Stage-2 survivors)
    digit_density = digits / total_non_whitespace_chars
    KEEP the page only if digit_density > num_density_pct.
    Pages dominated by prose (the auditor's narrative opinion) are dropped.

    Look-back on Stage-3 pass: for every page that passes Stage 3, the
    two pages immediately before it are each tested against Stage 3
    alone. Any preceding page that passes is also included in the
    output. All page numbers are deduplicated across direct passes and
    look-backs.

Public API
----------
    extract_auditor_signatures(pdf_path, min_score, num_density_pct,
                                output_pdf, no_pdf) -> list[dict]
        Direct Python call — same behaviour as the original CLI script.

    auditor_signature_tool(pdf_path, min_score=7, num_density_pct=10.0,
                            output_pdf="", no_pdf=False) -> str
        @tool-wrapped entry point (langchain_core.tools.tool) for use
        inside a LangGraph / AgentExecutor pipeline. Returns a JSON
        string (see its docstring for schema) instead of printing to
        stdout and instead of raising/exiting on bad input.

Dependencies
------------
    pip install pdfplumber pypdf rapidfuzz langchain-core
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from langchain_core.tools import tool

try:
    import pdfplumber
except ImportError as e:
    raise ImportError(
        "pdfplumber not found. Run: pip install pdfplumber"
    ) from e

try:
    from pypdf import PdfReader, PdfWriter
except ImportError as e:
    raise ImportError(
        "pypdf not found. Run: pip install pypdf"
    ) from e

try:
    from rapidfuzz import fuzz
    _FUZZY_OK = True
except ImportError:
    _FUZZY_OK = False

log = logging.getLogger(__name__)
if not _FUZZY_OK:
    log.warning("rapidfuzz not found - fuzzy matching disabled. Run: pip install rapidfuzz")


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN LIBRARY
# ═══════════════════════════════════════════════════════════════════════════════

_RE_FIRM     = re.compile(r"For\s+[A-Z][A-Za-z&,\.\s]{3,60?}(?:LLP|& Co\.?|Associates)?", re.I)
_RE_CA       = re.compile(r"Chartered\s+Accountants?", re.I)
_RE_FRN      = re.compile(
    r"(?:Firm\s+Reg(?:istration)?\.?\s*(?:No\.?|Number)?|FRN)[:\s]*"
    r"([A-Z0-9]{5,10}(?:[/\-][A-Z0-9]{5,10})?)",
    re.I,
)
_RE_FRN_BARE = re.compile(r"\b([A-Z]\d{5,6}(?:[/\-][A-Z]\d{5,6})?)\b")
_RE_PARTNER  = re.compile(r"\bPartner\b", re.I)
_RE_MEMNO    = re.compile(r"(?:Membership\s+No\.?|M\.?\s*No\.?)[:\s]*(\d{5,6})", re.I)
_RE_DATE     = re.compile(
    r"Date\s*[:\-]?\s*"
    r"(?:\d{1,2}[thstndrd]*\s+\w+\s+\d{4}"
    r"|\w+\s+\d{1,2},?\s+\d{4}"
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
    re.I,
)
_RE_PLACE    = re.compile(r"Place\s*[:\-]\s*[A-Za-z\s,]+", re.I)
_RE_UDIN     = re.compile(r"UDIN\s*[:\-]?\s*[A-Z0-9]{18,20}", re.I)

# Drop-filter: checked against the first 300 chars of the full page text.
# Matches "auditor's report", "auditors report", "auditor report", "annexure".
_RE_DROP_HEADER = re.compile(r"auditors?'?\s*report|annexure", re.I)

_FUZZY_ANCHORS = ["Chartered Accountants", "Firm Registration No", "Membership No", "Partner"]

_SIGNAL_WEIGHTS = {
    "firm_name":      2,
    "chartered_acct": 2,
    "frn":            3,
    "partner":        2,
    "membership_no":  3,
    "date":           1,
    "place":          1,
    "udin":           1,
}
MAX_SCORE = sum(_SIGNAL_WEIGHTS.values()) + len(_FUZZY_ANCHORS)  # 15 + 4 = 19

_SECTION_PATTERNS = [
    (re.compile(r"Independent\s+Auditor['']?s?\s+Report", re.I), "Independent Auditor's Report"),
    (re.compile(r"Audit(?:or)?['']?s?\s+Report", re.I),          "Auditor's Report"),
    (re.compile(r"Balance\s+Sheet", re.I),                        "Balance Sheet"),
    (re.compile(r"Standalone\s+Financial\s+Statements?", re.I),   "Standalone Financial Statements"),
    (re.compile(r"Consolidated\s+Financial\s+Statements?", re.I), "Consolidated Financial Statements"),
    (re.compile(r"Statement\s+of\s+(?:Profit|Loss)", re.I),       "Statement of Profit & Loss"),
    (re.compile(r"Cash\s+Flow\s+Statement", re.I),                "Cash Flow Statement"),
    (re.compile(r"Notes?\s+to\s+(?:the\s+)?(?:Financial\s+)?Accounts?", re.I), "Notes to Accounts"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — SIGNATURE SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def score_block(text: str) -> dict:
    signals = {
        "firm_name":      bool(_RE_FIRM.search(text)),
        "chartered_acct": bool(_RE_CA.search(text)),
        "frn":            bool(_RE_FRN.search(text) or _RE_FRN_BARE.search(text)),
        "partner":        bool(_RE_PARTNER.search(text)),
        "membership_no":  bool(_RE_MEMNO.search(text)),
        "date":           bool(_RE_DATE.search(text)),
        "place":          bool(_RE_PLACE.search(text)),
        "udin":           bool(_RE_UDIN.search(text)),
    }
    score = sum(_SIGNAL_WEIGHTS[k] for k, v in signals.items() if v)
    if _FUZZY_OK:
        for anchor in _FUZZY_ANCHORS:
            if fuzz.partial_ratio(anchor.lower(), text.lower()) >= 80:
                score += 1
    signals["total_score"] = score
    return signals


def _extract_sub_blocks(text: str) -> list:
    lines  = [ln.rstrip() for ln in text.splitlines()]
    cutoff = max(0, int(len(lines) * 0.55))
    lower  = "\n".join(lines[cutoff:])
    windows = [lower]
    start = max(0, int(len(lines) * 0.40))
    for i in range(start, len(lines) - 5):
        windows.append("\n".join(lines[i: i + 20]))
    return windows


def find_auditor_blocks(text: str, min_score: int) -> list:
    seen, found = set(), []
    for window in _extract_sub_blocks(text):
        if not window.strip():
            continue
        sig = score_block(window)
        if sig["total_score"] < min_score:
            continue
        norm = " ".join(window.split())[:200]
        if norm in seen:
            continue
        seen.add(norm)
        found.append({"text": window.strip(), "signals": sig})

    if not found:
        return []
    found.sort(key=lambda x: x["signals"]["total_score"], reverse=True)
    # Keep both blocks for joint-auditor (side-by-side) pages.
    if (len(found) > 1 and
            found[0]["signals"]["total_score"] - found[1]["signals"]["total_score"] <= 2):
        return found[:2]
    return found[:1]


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — HEADER DROP FILTER
# ═══════════════════════════════════════════════════════════════════════════════

def should_drop(full_page_text: str) -> tuple[bool, str]:
    """
    Return (True, reason) if the page should be excluded despite matching
    the signature pattern.

    Rule: if the first 300 characters of the page text contain
    "auditor's report" (or variant) or "annexure", drop it. These are
    intro/header pages, not the actual signing page.
    """
    header = full_page_text[:300]
    m = _RE_DROP_HEADER.search(header)
    if m:
        return True, f"'{m.group(0)}' found in first 300 chars"
    return False, ""


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — NUMBER-DENSITY FILTER
# ═══════════════════════════════════════════════════════════════════════════════

def number_density(full_page_text: str) -> float:
    """
    Return the fraction of non-whitespace characters on the page that are
    decimal digits (0-9).

        density = digit_count / non_whitespace_char_count

    Returns 0.0 if the page has no non-whitespace content.
    """
    non_ws = [ch for ch in full_page_text if not ch.isspace()]
    if not non_ws:
        return 0.0
    digits = sum(1 for ch in non_ws if ch.isdigit())
    return digits / len(non_ws)


def passes_number_density(full_page_text: str, threshold_pct: float = 10.0) -> tuple[bool, float]:
    """
    Return (True, density_pct) when the page's digit density exceeds
    threshold_pct (default 10%), meaning the page contains significant
    numeric/financial content alongside the signature.

    Return (False, density_pct) for prose-heavy pages that should be dropped.
    """
    density = number_density(full_page_text)
    density_pct = density * 100
    return density_pct > threshold_pct, density_pct


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def infer_section(page_text: str, prev: str) -> str:
    for pat, label in _SECTION_PATTERNS:
        if pat.search(page_text):
            return label
    return prev or "Unknown Section"


# ═══════════════════════════════════════════════════════════════════════════════
# CONSOLE / LOG OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def _log_result(r: dict, idx: int) -> None:
    source_tag = f" [{r.get('inclusion_source', '')}]" if r.get("inclusion_source") else ""
    log.info("MATCH #%d%s | page=%d | section=%s", idx + 1, source_tag, r["page_number"], r["section"])
    if r["score"] is not None:
        log.info(
            "  confidence=%d/%d signals=%s",
            r["score"], MAX_SCORE, ",".join(r["signals_hit"]),
        )
    else:
        log.info("  confidence=n/a (included via look-back, no signature block on this page)")
    log.info("  digit_density=%.2f%%", r.get("digit_density_pct", 0))
    if r.get("frn"):
        log.info("  FRN=%s", r["frn"])
    if r.get("membership_no"):
        log.info("  membership_no=%s", r["membership_no"])
    if r.get("date"):
        log.info("  date=%s", r["date"])


def _log_dropped(page_num: int, reason: str) -> None:
    log.info("Page %d matched signature but was DROPPED - %s", page_num, reason)


# ═══════════════════════════════════════════════════════════════════════════════
# PDF ASSEMBLY (pure extraction, no drawing)
# ═══════════════════════════════════════════════════════════════════════════════

def build_output_pdf(source_pdf: str, results: list, out_path: str) -> None:
    """
    Copy only the unique matched pages verbatim from source_pdf into out_path.
    No overlays, no annotations, no modifications whatsoever.
    """
    reader = PdfReader(source_pdf)
    writer = PdfWriter()

    seen_pages: set[int] = set()
    ordered = []
    for r in results:
        pn = r["page_number"]
        if pn not in seen_pages:
            seen_pages.add(pn)
            ordered.append(r)

    log.info("Extracting %d unique page(s) into output PDF...", len(ordered))

    for r in ordered:
        page_idx = r["page_number"] - 1  # 0-based
        writer.add_page(reader.pages[page_idx])

    with open(out_path, "wb") as fh:
        writer.write(fh)

    log.info("Output PDF saved -> %s", out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXTRACTION LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def extract_auditor_signatures(
    pdf_path: str,
    min_score: int = 7,
    num_density_pct: float = 10.0,
    output_pdf: str | None = None,
    no_pdf: bool = False,
) -> list[dict]:
    """
    Run the full 3-stage auditor-signature extraction pipeline against a
    single PDF and (optionally) write a clean extracted PDF of the
    matched pages.

    Raises:
        FileNotFoundError: pdf_path does not exist.
        ValueError: pdf_path is not a .pdf file.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF: {pdf_path}")

    log.info("Processing: %s", path.name)
    log.info("  Stage 1 - Min signature score: %d/%d", min_score, MAX_SCORE)
    log.info("  Stage 2 - Header drop filter: \"auditor's report\" or \"annexure\" in first 300 chars")
    log.info("  Stage 3 - Number density: keep pages with digit density > %.1f%%", num_density_pct)
    log.info("    Look-back: each Stage-3 pass also pulls in the 2 preceding pages if they pass Stage 3")

    # ── Pass 1: Stages 1 & 2 — collect all candidate pages ───────────────────
    # Cache every page's text so Pass 2 look-backs are free (no re-open).
    page_texts: dict[int, str] = {}
    stage12_survivors: list[dict] = []
    dropped: list[dict] = []
    current_section = "Unknown Section"

    with pdfplumber.open(str(path)) as pdf:
        total = len(pdf.pages)
        log.info("  Total pages: %d", total)

        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            page_texts[page_num] = text
            if not text.strip():
                continue

            current_section = infer_section(text, current_section)

            # Stage 1 — signature detection
            blocks = find_auditor_blocks(text, min_score)
            if not blocks:
                continue

            # Stage 2 — header drop filter
            drop, reason = should_drop(text)
            if drop:
                dropped.append({"page_number": page_num, "stage": 2, "reason": reason})
                _log_dropped(page_num, f"[Stage 2] {reason}")
                continue

            stage12_survivors.append({
                "page_number": page_num,
                "section":     current_section,
                "blocks":      blocks,
                "text":        text,
            })

    # ── Pass 2: Stage 3 — number-density filter + look-back ──────────────────
    #
    # For every Stage-1/2 survivor:
    #   - Run Stage 3 on the candidate page itself.
    #   - If it PASSES -> include it directly.
    #   - If it FAILS  -> look at the two pages immediately before it
    #                     (page_num-1 and page_num-2, clamped to page 1).
    #                     For each of those preceding pages, run Stage 3 only.
    #                     Any preceding page that passes is included.
    #   - Deduplicate by page number across all inclusions.
    #
    # Preceding pages are included as plain page-number entries (no signature
    # block metadata) because they passed on density, not on signature signals.

    results: list[dict] = []
    accepted_pages: set[int] = set()

    def _add_page(page_num: int, section: str, blocks: list | None,
                   text: str, density_pct: float, source: str) -> None:
        """Append a result record; silently skip if already included."""
        if page_num in accepted_pages:
            return
        accepted_pages.add(page_num)

        if blocks:
            blk   = blocks[0]
            sig   = blk["signals"]
            btxt  = blk["text"]
            frn_m = _RE_FRN.search(btxt) or _RE_FRN_BARE.search(btxt)
            mem_m = _RE_MEMNO.search(btxt)
            dat_m = _RE_DATE.search(btxt)
            results.append({
                "page_number":       page_num,
                "section":           section,
                "score":             sig["total_score"],
                "signals_hit":       [k for k, v in sig.items() if v and k != "total_score"],
                "extracted_block":   btxt,
                "frn":               frn_m.group(0).strip() if frn_m else None,
                "membership_no":     mem_m.group(1).strip() if mem_m else None,
                "date":              dat_m.group(0).strip() if dat_m else None,
                "digit_density_pct": round(density_pct, 2),
                "inclusion_source":  source,
            })
        else:
            results.append({
                "page_number":       page_num,
                "section":           section,
                "score":             None,
                "signals_hit":       [],
                "extracted_block":   "",
                "frn":               None,
                "membership_no":     None,
                "date":              None,
                "digit_density_pct": round(density_pct, 2),
                "inclusion_source":  source,
            })

    for candidate in stage12_survivors:
        pn      = candidate["page_number"]
        text    = candidate["text"]
        blocks  = candidate["blocks"]
        section = candidate["section"]

        passes, density_pct = passes_number_density(text, num_density_pct)

        if not passes:
            reason = (f"digit density {density_pct:.1f}% <= {num_density_pct:.1f}% threshold "
                      f"- prose-heavy page")
            dropped.append({"page_number": pn, "stage": 3, "reason": reason})
            _log_dropped(pn, f"[Stage 3] {reason}")
        else:
            log.info("Page %d digit density %.1f%% -> KEPT [Stage 3 pass]", pn, density_pct)
            _add_page(pn, section, blocks, text, density_pct, "direct")

            for offset in (1, 2):
                lb_pn = pn - offset
                if lb_pn < 1:
                    continue
                lb_text = page_texts.get(lb_pn, "")
                if not lb_text.strip():
                    continue
                lb_passes, lb_density = passes_number_density(lb_text, num_density_pct)
                if lb_passes:
                    lb_section = infer_section(lb_text, section)
                    log.info(
                        "  look-back page %d digit density %.1f%% -> KEPT [look-back from page %d]",
                        lb_pn, lb_density, pn,
                    )
                    _add_page(lb_pn, lb_section, None, lb_text, lb_density,
                              f"look-back from page {pn}")
                else:
                    log.info(
                        "  look-back page %d digit density %.1f%% -> skip (below threshold)",
                        lb_pn, lb_density,
                    )

    # Sort final results by page number so the output PDF is in document order.
    results.sort(key=lambda r: r["page_number"])

    s2_drops = sum(1 for d in dropped if d["stage"] == 2)
    s3_drops = sum(1 for d in dropped if d["stage"] == 3)
    if dropped:
        log.info("Pages dropped - Stage 2 (header filter): %d | Stage 3 (below density threshold): %d",
                  s2_drops, s3_drops)

    if not results:
        log.warning("No auditor signature pages survived all filters.")
        log.warning("Suggestions: lower min_score (currently %d) or lower num_density_pct (currently %s)",
                     min_score, num_density_pct)
        return results

    log.info("%d unique page(s) retained after all stages:", len(results))
    for i, r in enumerate(results):
        _log_result(r, i)

    if not no_pdf:
        dest = output_pdf or str(path.parent / (path.stem + "_auditor_pages.pdf"))
        build_output_pdf(str(path), results, dest)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# LANGCHAIN TOOL WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def auditor_signature_tool(
    pdf_path: str,
    min_score: int = 7,
    num_density_pct: float = 10.0,
    output_pdf: str = "",
    no_pdf: bool = False,
) -> str:
    """
    Detects statutory auditor signature blocks in an Indian annual report
    PDF (firm name, FRN, Chartered Accountants, Partner, Membership No.,
    Date, Place, UDIN) and returns the matched pages, optionally writing
    a clean extracted PDF of just those pages.

    Runs a 3-stage pipeline: Stage 1 scores each page against signature
    signals; Stage 2 drops intro/header pages that mention "auditor's
    report" or "annexure" near the top; Stage 3 keeps only pages with a
    high enough digit density (the signature page sits inside numeric
    financial content, not pure prose), and additionally pulls in the
    1-2 immediately preceding pages if they too pass the density check.

    Args:
        pdf_path:         Path to the annual report PDF.
        min_score:        Stage 1 minimum signature confidence score,
                           0-19 (default: 7).
        num_density_pct:  Stage 3 minimum digit-density percentage to
                           keep a page (default: 10.0).
        output_pdf:       Output PDF path. Empty string uses
                           "<stem>_auditor_pages.pdf" next to the source
                           PDF (default: "").
        no_pdf:           If True, skip writing the extracted PDF and
                           only return the JSON match data (default: False).

    Returns:
        JSON string with keys:
            status        — "success" | "error"
            pdf_path      — the input path that was processed
            output_pdf    — path the extracted PDF was written to, or
                             null if no_pdf was True or there were no matches
            match_count   — number of unique matched pages
            matches       — list of per-page records, each with:
                             page_number, section, score (0-19 or null
                             if included only via look-back), signals_hit,
                             extracted_block, frn, membership_no, date,
                             digit_density_pct, inclusion_source
                             ("direct" or "look-back from page N")
            error         — error message (only on failure)
    """
    try:
        results = extract_auditor_signatures(
            pdf_path=pdf_path,
            min_score=min_score,
            num_density_pct=num_density_pct,
            output_pdf=output_pdf or None,
            no_pdf=no_pdf,
        )
    except (FileNotFoundError, ValueError) as e:
        return json.dumps({"status": "error", "error": str(e)})

    written_path = None
    if not no_pdf and results:
        path = Path(pdf_path)
        written_path = output_pdf or str(path.parent / (path.stem + "_auditor_pages.pdf"))

    return json.dumps({
        "status":      "success",
        "pdf_path":    pdf_path,
        "output_pdf":  written_path,
        "match_count": len(results),
        "matches":     results,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "Extract statutory auditor signature pages from a corporate "
            "annual report PDF into a clean output PDF."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline
  Stage 1  Signature score  >= --min-score
  Stage 2  Drop if first 300 chars contain "auditor's report" or "annexure"
  Stage 3  Keep only if digit density > --num-density-pct

Examples:
  python phase1_filter_batch.py annual_report.pdf
  python phase1_filter_batch.py annual_report.pdf --min-score 5
  python phase1_filter_batch.py annual_report.pdf --num-density-pct 8
  python phase1_filter_batch.py annual_report.pdf --output signed_pages.pdf
  python phase1_filter_batch.py annual_report.pdf --no-pdf
        """,
    )
    parser.add_argument("pdf", metavar="PDF_PATH",
                         help="Path to the corporate annual report PDF.")
    parser.add_argument("--min-score", type=int, default=7, metavar="N",
                         help="Stage 1: minimum signature confidence score 0-19 (default: 7).")
    parser.add_argument("--num-density-pct", type=float, default=10.0, metavar="PCT",
                         help="Stage 3: minimum digit-density %% to keep a page (default: 10.0).")
    parser.add_argument("--output", metavar="PDF_PATH",
                         help="Output PDF path (default: <stem>_auditor_pages.pdf).")
    parser.add_argument("--no-pdf", action="store_true",
                         help="Skip PDF generation; log report only.")
    args = parser.parse_args()

    try:
        extract_auditor_signatures(
            args.pdf,
            args.min_score,
            args.num_density_pct,
            args.output,
            args.no_pdf,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()