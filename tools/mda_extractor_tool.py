"""
tools/mda_extractor_tool.py
===========================
Extracts the Management Discussion & Analysis (MD&A) section from the
original (untrimmed) Indian Annual Report PDF using PyMuPDF.

Strategy
--------
1.  Scan every page for MD&A header keywords.
2.  Once the section start is found, collect text until a known
    terminating section header appears (e.g. Corporate Governance,
    Directors' Report, Financial Statements).
3.  Return the raw text so the Narrative Agent can chunk + embed it.

No Gemini call is made here — pure heuristic text extraction.
"""

import json
import re
from pathlib import Path

import fitz  # PyMuPDF
from langchain.tools import tool

# ── Keywords that signal MD&A section start ───────────────────────────────────
_MDA_START_PATTERNS = [
    r"management\s+discussion\s+and\s+analysis",
    r"management['']s\s+discussion\s+and\s+analysis",
    r"md\s*&\s*a",
    r"management\s+review",
]

# ── Keywords that signal MD&A section end ────────────────────────────────────
_MDA_END_PATTERNS = [
    r"corporate\s+governance",
    r"directors['']?\s+report",
    r"board\s+of\s+directors",
    r"auditors['']?\s+report",
    r"independent\s+auditor",
    r"financial\s+statements",
    r"notes\s+to\s+(the\s+)?financial",
    r"annexure",
    r"business\s+responsibility",
]

_START_RE = re.compile("|".join(_MDA_START_PATTERNS), re.IGNORECASE)
_END_RE   = re.compile("|".join(_MDA_END_PATTERNS),   re.IGNORECASE)


def _is_section_header(text: str, pattern: re.Pattern) -> bool:
    """
    Returns True only if the pattern matches within the first 300 chars
    of the page — avoids false positives from body text references.
    """
    return bool(pattern.search(text[:300]))


@tool
def mda_extractor_tool(pdf_path: str) -> str:
    """
    Extracts the MD&A section text from an Indian Annual Report PDF.

    Args:
        pdf_path: Path to the original (untrimmed) AR PDF.

    Returns:
        JSON string with keys:
            status      — "success" | "error"
            text        — extracted MD&A text (may be multi-page)
            start_page  — 1-indexed page where MD&A was found
            end_page    — 1-indexed page where extraction stopped
            error       — error message (only on failure)
    """
    pdf_path = str(Path(pdf_path).resolve())

    if not Path(pdf_path).exists():
        return json.dumps({"status": "error", "error": f"PDF not found: {pdf_path}"})

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        return json.dumps({"status": "error", "error": f"Could not open PDF: {exc}"})

    mda_pages: list[str] = []
    in_mda     = False
    start_page = None
    end_page   = None

    try:
        for page_num, page in enumerate(doc):
            text = page.get_text("text")

            if not in_mda:
                if _is_section_header(text, _START_RE):
                    in_mda     = True
                    start_page = page_num + 1  # 1-indexed
                    mda_pages.append(text)
            else:
                # Check for terminating section — but only after at least
                # 2 pages of MD&A content to avoid premature cutoff
                if len(mda_pages) >= 2 and _is_section_header(text, _END_RE):
                    end_page = page_num + 1
                    break
                mda_pages.append(text)

    finally:
        doc.close()

    if not mda_pages:
        return json.dumps({
            "status": "error",
            "error":  "MD&A section not found. The section header may use "
                      "non-standard wording — check the PDF manually.",
        })

    full_text = "\n\n".join(mda_pages)
    end_page  = end_page or (start_page + len(mda_pages) - 1)

    return json.dumps({
        "status":     "success",
        "text":       full_text,
        "start_page": start_page,
        "end_page":   end_page,
        "page_count": len(mda_pages),
    })