"""
Phase 3 — Assembly & Output Generation
=======================================
Consumes the ``page_mapping`` dict produced by Phase 2's
``ClassifiedPages.to_dict()`` and writes a new, compacted PDF that contains
only the four core financial-statement pages, in their original document order.

Public surface
--------------
    generate_output_pdf(source_pdf_path, output_pdf_path, page_mapping)
        → OutputResult

    print_execution_summary(result)
        → None  (side-effect: pretty-printed JSON to stdout)

    extract_core_financial_statements(pdf_path, output_path, api_key)
        → OutputResult  (end-to-end master orchestrator)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

# ── Import Phase 1 & Phase 2 public APIs ─────────────────────────────────────
# These modules must be importable from the same package / working directory.
# Adjust the import paths to match your project layout if needed.
try:
    from pipeline.phase1_filter_batch import run_phase1          # type: ignore[import]
    from pipeline.phase2_llm_classify import classify_page_batches  # type: ignore[import]
except ModuleNotFoundError as _e:  # pragma: no cover
    raise ImportError(
        "Could not import Phase 1 / Phase 2 modules.  "
        "Ensure 'phase1_filter_batch.py' and 'phase2_llm_classify.py' are on "
        "sys.path before importing this module."
    ) from _e

log = logging.getLogger(__name__)

# ── Category metadata (display order matters for the summary) ─────────────────
_CATEGORY_META: dict[str, dict[str, str]] = {
    "balance_sheet_pages": {
        "label": "Balance Sheet",
        "code":  "BS",
    },
    "profit_loss_pages": {
        "label": "Profit & Loss / Income Statement",
        "code":  "PL",
    },
    "cash_flow_pages": {
        "label": "Cash Flow Statement",
        "code":  "CF",
    },
    "equity_pages": {
        "label": "Statement of Changes in Equity",
        "code":  "EQ",
    },
}


# ─── DATA STRUCTURES ──────────────────────────────────────────────────────────


@dataclass
class OutputResult:
    """
    Everything Phase 3 knows about the assembled PDF.

    Attributes
    ----------
    output_pdf_path : str
        Absolute path of the PDF that was written to disk.
    page_mapping : dict[str, list[int]]
        The exact ``{category: [0-indexed page nums]}`` dict that was used.
    assembled_pages : list[int]
        Sorted, de-duplicated list of all 0-indexed page numbers that were
        inserted into the output PDF (in document order).
    total_pages : int
        Number of pages in the generated PDF (== ``len(assembled_pages)``).
    elapsed_seconds : float
        Wall-clock time taken by Phase 3 alone (PDF I/O, not the LLM calls).
    """
    output_pdf_path: str
    page_mapping:    dict[str, list[int]]
    assembled_pages: list[int]
    total_pages:     int
    elapsed_seconds: float = 0.0
    # Any pages the LLM classified but that fell outside the source PDF's
    # valid range — kept for diagnostics, not inserted.
    skipped_out_of_range: list[int] = field(default_factory=list)


# ─── PHASE 3 STEP 1: PDF COMPILATION ─────────────────────────────────────────


def generate_output_pdf(
    source_pdf_path: str,
    output_pdf_path: str,
    page_mapping: dict[str, list[int]],
) -> OutputResult:
    """
    Assemble a new PDF from the confirmed financial-statement pages.

    Algorithm
    ---------
    1.  Collect all unique page numbers across the four category lists.
    2.  Clamp any out-of-range page numbers and log a warning (don't crash).
    3.  Sort the valid page numbers so pages appear in original document order.
    4.  Open the source PDF, create a blank destination document, and insert
        (``insert_pdf``) only the selected pages.
    5.  Save the destination PDF with garbage collection + deflation, then
        close **both** document handles in a ``finally`` block regardless of
        whether an exception occurs.

    Parameters
    ----------
    source_pdf_path : str
        Path to the original, full annual-report PDF.
    output_pdf_path : str
        Destination path for the compacted financial-statements PDF.
        Parent directories are created automatically.
    page_mapping : dict[str, list[int]]
        ``{"balance_sheet_pages": [...], "profit_loss_pages": [...], ...}``
        with **0-indexed** page numbers as produced by Phase 2.

    Returns
    -------
    OutputResult
        Populated result object; ``total_pages == 0`` if nothing was inserted.

    Raises
    ------
    FileNotFoundError
        If ``source_pdf_path`` does not exist.
    ValueError
        If ``page_mapping`` contains none of the four expected keys.
    fitz.FileDataError
        If the source PDF is corrupt / unreadable by PyMuPDF.
    """
    t0 = time.perf_counter()

    source_path = Path(source_pdf_path).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source PDF not found: {source_path}")

    # ── Validate page_mapping keys ────────────────────────────────────────────
    expected_keys = set(_CATEGORY_META.keys())
    present_keys  = expected_keys & set(page_mapping)
    if not present_keys:
        raise ValueError(
            f"page_mapping must contain at least one of {sorted(expected_keys)}; "
            f"got keys: {sorted(page_mapping)}"
        )

    # ── Collect & deduplicate all candidate page numbers ──────────────────────
    all_pages: set[int] = set()
    for key in expected_keys:
        all_pages.update(page_mapping.get(key, []))

    log.info("Phase 3 ─ Assembly & Output Generation")
    log.info("  Source PDF      : %s", source_path)
    log.info("  Destination PDF : %s", output_pdf_path)
    log.info("  Raw candidate pages (0-indexed, deduplicated): %s", sorted(all_pages))

    # ── Open source document ──────────────────────────────────────────────────
    src_doc: fitz.Document | None = None
    dst_doc: fitz.Document | None = None

    try:
        src_doc = fitz.open(str(source_path))
        max_valid_page = len(src_doc) - 1  # 0-indexed upper bound

        # ── Range-check every candidate page ─────────────────────────────────
        valid_pages:   list[int] = []
        out_of_range:  list[int] = []

        for pg in sorted(all_pages):
            if 0 <= pg <= max_valid_page:
                valid_pages.append(pg)
            else:
                out_of_range.append(pg)
                log.warning(
                    "Page %d (0-indexed) is out of range for a %d-page PDF — "
                    "skipping.",
                    pg, len(src_doc),
                )

        assembled_pages = sorted(valid_pages)   # document order

        if not assembled_pages:
            log.error(
                "No valid pages remain after range-checking — refusing to "
                "write an empty PDF. This almost always means Phase 2 "
                "classification produced zero results (check upstream logs "
                "for LLM API errors, e.g. an invalid/retired model name), "
                "not that the source document genuinely has no matching "
                "pages."
            )
            raise RuntimeError(
                "generate_output_pdf: no pages to write — page_mapping "
                "resolved to an empty page list. Check Phase 2 classification "
                "output before retrying."
            )

        # ── Build destination document ────────────────────────────────────────
        dst_doc = fitz.open()   # new, empty in-memory document

        # insert_pdf accepts a list of page numbers via `from_page` / `to_page`
        # but for a non-contiguous selection we insert page-by-page so we
        # preserve exact ordering and avoid pulling in unwanted pages.
        for pg_num in assembled_pages:
            dst_doc.insert_pdf(
                src_doc,
                from_page=pg_num,
                to_page=pg_num,
            )
            log.debug("  Inserted page %d (0-indexed) → output page %d",
                      pg_num, dst_doc.page_count)

        # ── Write to disk ─────────────────────────────────────────────────────
        out_path = Path(output_pdf_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        dst_doc.save(
            str(out_path),
            garbage=4,      # maximum cross-reference cleanup
            deflate=True,   # compress streams
            clean=True,     # sanitise content streams
        )

        elapsed = time.perf_counter() - t0
        total   = len(assembled_pages)

        log.info(
            "Phase 3 complete: %d page(s) written to '%s' in %.2fs",
            total, out_path, elapsed,
        )

        return OutputResult(
            output_pdf_path      = str(out_path),
            page_mapping         = page_mapping,
            assembled_pages      = assembled_pages,
            total_pages          = total,
            elapsed_seconds      = elapsed,
            skipped_out_of_range = out_of_range,
        )

    finally:
        # Always close both handles — even if an exception was raised above.
        if dst_doc is not None:
            dst_doc.close()
        if src_doc is not None:
            src_doc.close()


# ─── PHASE 3 STEP 2: EXECUTION SUMMARY ───────────────────────────────────────


def print_execution_summary(result: OutputResult) -> None:
    """
    Print a human-readable, JSON-formatted execution summary to *stdout*.

    The summary includes:
    * Per-category breakdown of which pages were allocated (1-indexed for
      readability) and how many that is.
    * A complete, ordered list of all pages in the output PDF.
    * Total output page count.
    * Wall-clock time taken by Phase 3.
    * Any pages that were skipped due to being out of the source PDF's range.

    Parameters
    ----------
    result : OutputResult
        The object returned by ``generate_output_pdf()``.
    """
    # Build an inverse map: 0-indexed page num → list of category codes
    # (a single page might appear in two categories, e.g. a combined BS+PL page)
    page_to_codes: dict[int, list[str]] = {}
    for key, meta in _CATEGORY_META.items():
        for pg in result.page_mapping.get(key, []):
            page_to_codes.setdefault(pg, []).append(meta["code"])

    # ── Per-category section ──────────────────────────────────────────────────
    categories_summary: list[dict[str, Any]] = []
    for key, meta in _CATEGORY_META.items():
        pages_0idx = sorted(result.page_mapping.get(key, []))
        # Convert to 1-indexed for human display
        pages_1idx = [p + 1 for p in pages_0idx]
        categories_summary.append({
            "statement_type":  meta["label"],
            "code":            meta["code"],
            "page_count":      len(pages_1idx),
            "pages_in_source": pages_1idx,   # 1-indexed (human-readable)
        })

    # ── Overall section ───────────────────────────────────────────────────────
    summary: dict[str, Any] = {
        "execution_summary": {
            "output_pdf_path":      result.output_pdf_path,
            "total_pages_in_output": result.total_pages,
            "output_pages_in_order": [p + 1 for p in result.assembled_pages],  # 1-indexed
            "phase3_elapsed_seconds": round(result.elapsed_seconds, 3),
        },
        "financial_statement_breakdown": categories_summary,
    }

    if result.skipped_out_of_range:
        summary["warnings"] = {
            "skipped_out_of_range_pages": [p + 1 for p in result.skipped_out_of_range],
            "reason": "These page numbers were returned by the LLM but exceed "
                      "the source PDF's page count and were not inserted.",
        }

    print(json.dumps(summary, indent=2))


# ─── MASTER ORCHESTRATOR ─────────────────────────────────────────────────────


def extract_core_financial_statements(
    pdf_path:    str,
    output_path: str,
    api_key:     str,
) -> OutputResult:
    """
    End-to-end pipeline: Phase 1 → Phase 2 → Phase 3.

    Phase 1 — Heuristic Filtering & Token Batching
        Opens the source PDF, extracts text from every page, runs a keyword
        heuristic to shortlist candidate pages, then packs them into
        token-bounded batches safe to send to the LLM.

    Phase 2 — Structured LLM Classification
        Sends each batch to the Gemini API and aggregates the ``[page, code]``
        pairs into a ``ClassifiedPages`` object.  Pages the LLM marks ``XX``
        (non-financial) are silently discarded here.

    Phase 3 — Assembly & Output Generation
        Collects the confirmed page numbers, sorts them, and writes a new PDF
        containing only the four core financial statements.  Prints a JSON
        execution summary on success.

    Parameters
    ----------
    pdf_path : str
        Path to the source annual-report PDF.
    output_path : str
        Desired path for the generated financial-statements PDF.
    api_key : str
        Gemini API key.

    Returns
    -------
    OutputResult
        Contains the output path, assembled pages, total page count, and
        timing information for Phase 3.

    Raises
    ------
    FileNotFoundError
        If ``pdf_path`` does not exist.
    ImportError
        If Phase 1 or Phase 2 modules cannot be imported.
    Any exception propagated from PyMuPDF or the Gemini HTTP client.
    """
    pipeline_start = time.perf_counter()

    log.info("=" * 65)
    log.info("  Core Financial Statement Extraction Pipeline — START")
    log.info("=" * 65)
    log.info("  Source PDF  : %s", pdf_path)
    log.info("  Output PDF  : %s", output_path)

    # ── Phase 0: PDF Text Extraction ─────────────────────────────────────────
    try:
        source_doc = fitz.open(pdf_path)
        pages: dict[int, str] = {
            i: page.get_text("text") for i, page in enumerate(source_doc)
        }
        source_doc.close()
    except Exception as e:
        log.error("Failed to open or read PDF: %s", pdf_path)
        raise e

    log.info("  Extracted %d pages.", len(pages))

    # ── Phase 1: Heuristic filter (active) & build token batches ─────────────
    log.info("-" * 65)
    log.info("  PHASE 1 ─ Heuristic Filtering & Token Batching")
    log.info("-" * 65)

    phase1_result = run_phase1(pages)
    batches: list[dict[int, str]] = [b.pages for b in phase1_result.batches]

    log.info(
        "  Phase 1 produced %d batch(es) covering %d candidate page(s).",
        len(batches),
        sum(len(b) for b in batches),
    )

    if not batches:
        log.warning(
            "Phase 1 returned zero batches — the PDF may contain no "
            "financial-statement candidate pages.  Aborting pipeline."
        )
        # Return an empty result rather than crashing.
        return OutputResult(
            output_pdf_path = output_path,
            page_mapping    = {k: [] for k in _CATEGORY_META},
            assembled_pages = [],
            total_pages     = 0,
        )

    # ── Phase 2: LLM classification ───────────────────────────────────────────
    log.info("-" * 65)
    log.info("  PHASE 2 ─ Structured LLM Classification")
    log.info("-" * 65)

    classified = classify_page_batches(batches, api_key=api_key)
    page_mapping: dict[str, list[int]] = classified.to_dict()

    log.info("  Phase 2 page mapping:")
    for category, pages in page_mapping.items():
        log.info("    %-28s → %s", category, pages)

    # ── Phase 3: PDF assembly & output ────────────────────────────────────────
    log.info("-" * 65)
    log.info("  PHASE 3 ─ Assembly & Output Generation")
    log.info("-" * 65)

    result = generate_output_pdf(
        source_pdf_path = pdf_path,
        output_pdf_path = output_path,
        page_mapping    = page_mapping,
    )

    total_elapsed = time.perf_counter() - pipeline_start

    log.info("=" * 65)
    log.info(
        "  Pipeline complete in %.2fs — output: %s  (%d pages)",
        total_elapsed,
        result.output_pdf_path,
        result.total_pages,
    )
    log.info("=" * 65)

    # ── Print human-readable JSON summary to stdout ───────────────────────────
    print_execution_summary(result)

    return result


# ─── CLI ENTRY POINT ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Minimal CLI wrapper so the full pipeline can be invoked as:

        python phase3_assembly_output.py <source.pdf> <output.pdf>

    The Gemini API key is read from the ``GEMINI_API_KEY`` environment variable.
    """
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        stream = sys.stderr,    # keep logs on stderr; JSON summary goes to stdout
    )

    if len(sys.argv) != 3:
        print(
            f"Usage: python {Path(__file__).name} <source_pdf> <output_pdf>",
            file=sys.stderr,
        )
        sys.exit(1)

    _pdf_path    = sys.argv[1]
    _output_path = sys.argv[2]
    _api_key     = os.environ.get("GEMINI_API_KEY", "")

    if not _api_key:
        print(
            "ERROR: GEMINI_API_KEY environment variable is not set.",
            file=sys.stderr,
        )
        sys.exit(1)

    extract_core_financial_statements(
        pdf_path    = _pdf_path,
        output_path = _output_path,
        api_key     = _api_key,
    )