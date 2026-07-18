"""
Phase 3 — Assembly & Output Generation
=======================================
Consumes the ``page_mapping`` dict produced by Phase 2's
``ClassifiedPages.to_dict()`` and writes a new, compacted PDF that contains
only the four core financial-statement pages, in their original document order.

Also hosts the master orchestrator ``extract_core_financial_statements()``,
which wires together the full pipeline:

    Phase 1 (auditor_signature_tool) → Phase 2 (classify_page_batches) → Phase 3 (this module)

Phase 1 no longer does heuristic/TOC-based filtering — it locates the
statutory auditor's signature page(s) via ``pipeline.phase1_filter_batch.
auditor_signature_tool`` and leans on that tool's look-back to pull in the
1-2 preceding pages, which is where the actual financial-statement numbers
sit in Indian annual reports. That narrowed page set (re-read from the
original source PDF, 0-indexed) is what gets handed to Phase 2.

Phase 1's intermediate batch result is cached in ``gemini_cachelite``
(SQLite, ``gemini_cachelite.sqlite3`` next to this module), keyed on the
PDF's content hash plus the Phase 1 scan parameters — separate from Phase
2's own Gemini-response cache (``gemini_cache.sqlite3``).

Public surface
--------------
    generate_output_pdf(source_pdf_path, output_pdf_path, page_mapping)
        → OutputResult

    print_execution_summary(result)
        → None  (side-effect: pretty-printed JSON to stdout)

    extract_core_financial_statements(pdf_path, output_path, api_key, ...)
        → OutputResult  (end-to-end master orchestrator)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF


def _fill_page_gaps(pages: list[int]) -> list[int]:
    """
    Given a sorted set of page numbers, fill in the pages inside every
    consecutive gap EXCEPT the single largest gap, which is left alone
    (that gap is assumed to be a genuine section boundary — e.g. between
    the Auditor's Report signature block and the financial statements —
    rather than a page Phase 2 simply missed).

    Example
    -------
    Input  (sorted): 24, 25, 28, 29, 30, 60, 61, 62, 64
    Gaps:            1,  3,  1,  1, 30,  1,  1,  2
    Largest gap is 30 - 29 = 30 (between page 30 and page 60) → left as-is.
    All other gaps get filled:
        25 → 28  (gap 3) → fill 26, 27
        62 → 64  (gap 2) → fill 63
    Output: 24, 25, 26, 27, 28, 29, 30, 60, 61, 62, 63, 64

    If there are multiple gaps tied for the largest size, all of them are
    left unfilled (only gaps strictly smaller than the max get filled).

    Parameters
    ----------
    pages : list[int]
        Candidate page numbers (any indexing convention — 0-indexed or
        1-indexed, doesn't matter, this function is agnostic to it).

    Returns
    -------
    list[int]
        Sorted, de-duplicated page numbers with small gaps filled in.
    """
    unique_sorted = sorted(set(pages))
    if len(unique_sorted) < 2:
        return unique_sorted

    gaps = [unique_sorted[i + 1] - unique_sorted[i] for i in range(len(unique_sorted) - 1)]
    max_gap = max(gaps)

    filled: list[int] = [unique_sorted[0]]
    for i in range(1, len(unique_sorted)):
        prev_page = unique_sorted[i - 1]
        curr_page = unique_sorted[i]
        gap = gaps[i - 1]

        if gap > 1 and gap != max_gap:
            # Small/medium gap — fill in every page in between.
            filled.extend(range(prev_page + 1, curr_page))

        filled.append(curr_page)

    return sorted(set(filled))

# ── Import Phase 1 & Phase 2 public APIs ─────────────────────────────────────
# These modules must be importable from the same package / working directory.
# Adjust the import paths to match your project layout if needed.
#
# Phase 1 is now the auditor-signature LangChain tool (pipeline/phase1_filter_batch.py).
# It locates each statutory auditor's signature page in the source PDF and,
# via its Stage-3 look-back, pulls in the 1-2 pages immediately preceding it
# — which is where the actual financial-statement numbers live in Indian ARs.
# That narrowed set of (page_number, text) pairs is what gets batched and
# handed to Phase 2 for BS/PL/CF/EQ classification.
try:
    from pipeline.phase1_filter_batch import auditor_signature_tool        # type: ignore[import]
    from pipeline.phase2_llm_classify import classify_page_batches          # type: ignore[import]
except ModuleNotFoundError as _e:  # pragma: no cover
    raise ImportError(
        "Could not import Phase 1 / Phase 2 modules.  "
        "Ensure 'phase1_filter_batch.py' and 'phase2_llm_classify.py' are on "
        "sys.path before importing this module."
    ) from _e

log = logging.getLogger(__name__)

# ── gemini_cachelite — Step1's own lightweight cache ──────────────────────────
# Step1 does not call the Gemini API directly (that's Phase 2's job, cached
# separately in gemini_cache.sqlite3). What Step1 DOES own is the expensive,
# repeatable work of re-running the Phase 1 auditor-signature scan and
# re-reading every page of the source PDF on every retry. gemini_cachelite
# caches that intermediate result — the batches that get handed to Phase 2 —
# keyed on the PDF's content hash plus the scan parameters, so re-running the
# same PDF with the same Phase 1 settings skips straight to Phase 2.
_DEFAULT_CACHELITE_PATH: Path = Path(__file__).resolve().parent / "gemini_cachelite.sqlite3"

_CACHELITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS gemini_cachelite (
    cache_key    TEXT PRIMARY KEY,
    pdf_path     TEXT NOT NULL,
    min_score    INTEGER NOT NULL,
    num_density  REAL NOT NULL,
    batches_json TEXT NOT NULL,   -- the cached list[dict[int, str]] payload
    created_at   REAL NOT NULL
)
"""


def _get_cachelite_conn(cache_path: Path) -> sqlite3.Connection:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(cache_path))
    conn.execute(_CACHELITE_SCHEMA)
    conn.commit()
    return conn


def _hash_pdf_content(pdf_path: str) -> str:
    """Content hash of the PDF bytes — invalidates automatically if the file changes."""
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _compute_cachelite_key(pdf_content_hash: str, min_score: int, num_density_pct: float) -> str:
    """
    Cache key covers the PDF's own bytes plus every Phase 1 scan parameter
    that can change which pages get selected. Any change to min_score or
    num_density_pct automatically invalidates the cached batch set.
    """
    raw = f"{pdf_content_hash}:{min_score}:{num_density_pct}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cachelite_get(conn: sqlite3.Connection, cache_key: str) -> list[dict[int, str]] | None:
    row = conn.execute(
        "SELECT batches_json FROM gemini_cachelite WHERE cache_key = ?",
        (cache_key,),
    ).fetchone()
    if row is None:
        return None
    # JSON object keys are always strings — convert page numbers back to int.
    raw_batches: list[dict[str, str]] = json.loads(row[0])
    return [{int(pg): text for pg, text in batch.items()} for batch in raw_batches]


def _cachelite_set(
    conn: sqlite3.Connection,
    cache_key: str,
    pdf_path: str,
    min_score: int,
    num_density_pct: float,
    batches: list[dict[int, str]],
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO gemini_cachelite "
        "(cache_key, pdf_path, min_score, num_density, batches_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            cache_key,
            pdf_path,
            min_score,
            num_density_pct,
            json.dumps(batches),
            time.time(),
        ),
    )
    conn.commit()

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

    # ── Fill small gaps between candidate pages ───────────────────────────────
    # Phase 2's per-category classification can leave small holes in an
    # otherwise-contiguous run of financial-statement pages (e.g. a page that
    # is 90% table but got classified as "narrative" and dropped). We treat
    # the single largest gap in the sorted candidate list as a genuine section
    # boundary (e.g. Auditor's Report → Financial Statements) and leave it
    # alone, but fill in every smaller gap so we don't lose pages in the
    # middle of a statement (this is what was silently truncating the
    # Income Statement / P&L pages out of some trimmed PDFs).
    pre_fill_pages = sorted(all_pages)
    all_pages = set(_fill_page_gaps(pre_fill_pages))
    filled_in = sorted(all_pages - set(pre_fill_pages))
    if filled_in:
        log.info("  Gap-fill added %d page(s) inside small gaps: %s", len(filled_in), filled_in)

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
    pdf_path:          str,
    output_path:       str,
    api_key:           str,
    min_score:         int   = 7,
    num_density_pct:   float = 10.0,
    use_cachelite:     bool  = True,
    cachelite_path:    Path | str = _DEFAULT_CACHELITE_PATH,
) -> OutputResult:
    """
    End-to-end pipeline: Phase 1 → Phase 2 → Phase 3.

    Phase 1 — Auditor-Signature-Anchored Page Narrowing
        Runs ``auditor_signature_tool`` against the source PDF to locate
        the statutory auditor's signature page(s). Its built-in look-back
        also pulls in the 1-2 pages immediately preceding each signature
        page — in Indian ARs that's where the actual financial-statement
        numbers (Balance Sheet / P&L / Cash Flow / Equity) live, since the
        auditor signs directly below or after the statements. The matched
        page numbers are then re-read from the ORIGINAL source PDF (not
        the tool's repaginated side-output) so page numbering stays
        consistent all the way through Phase 3, and packed into a single
        token batch for Phase 2. The whole step is cached in
        ``gemini_cachelite`` (see module docstring), keyed on the PDF's
        content hash plus these scan parameters.

    Phase 2 — Structured LLM Classification
        Sends the batch to the Gemini API and aggregates the
        ``[page, code]`` pairs into a ``ClassifiedPages`` object. Pages
        the LLM marks ``XX`` (non-financial) are silently discarded here.

    Phase 3 — Assembly & Output Generation
        Collects the confirmed page numbers, sorts them, and writes a new
        PDF containing only the four core financial statements. Prints a
        JSON execution summary on success.

    Parameters
    ----------
    pdf_path : str
        Path to the source annual-report PDF.
    output_path : str
        Desired path for the generated financial-statements PDF.
    api_key : str
        Gemini API key (passed through to Phase 2).
    min_score : int
        Phase 1 ``auditor_signature_tool`` Stage 1 minimum signature
        confidence score, 0-19 (default: 7).
    num_density_pct : float
        Phase 1 ``auditor_signature_tool`` Stage 3 minimum digit-density
        percentage to keep a page (default: 10.0).
    use_cachelite : bool
        If True (default), check ``gemini_cachelite`` for a cached Phase 1
        batch before re-running the auditor-signature scan. Re-running the
        same PDF with the same scan parameters becomes instant.
    cachelite_path : Path | str
        Path to the ``gemini_cachelite`` SQLite cache file. Defaults to
        ``gemini_cachelite.sqlite3`` next to this module.

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
    Any exception propagated from PyMuPDF, the auditor-signature tool, or
    the Gemini HTTP client.
    """
    pipeline_start = time.perf_counter()

    log.info("=" * 65)
    log.info("  Core Financial Statement Extraction Pipeline — START")
    log.info("=" * 65)
    log.info("  Source PDF  : %s", pdf_path)
    log.info("  Output PDF  : %s", output_path)

    # ── Phase 1: Auditor-signature-anchored page narrowing ────────────────────
    log.info("-" * 65)
    log.info("  PHASE 1 ─ Auditor-Signature-Anchored Page Narrowing")
    log.info("-" * 65)

    cachelite_conn: sqlite3.Connection | None = None
    cache_key: str | None = None
    batches: list[dict[int, str]] = []

    if use_cachelite:
        cachelite_conn = _get_cachelite_conn(Path(cachelite_path))
        pdf_hash = _hash_pdf_content(pdf_path)
        cache_key = _compute_cachelite_key(pdf_hash, min_score, num_density_pct)
        cached_batches = _cachelite_get(cachelite_conn, cache_key)
        if cached_batches is not None:
            log.info("  gemini_cachelite hit — skipping Phase 1 re-scan.")
            batches = cached_batches

    if not batches:
        tool_result_json = auditor_signature_tool.invoke({
            "pdf_path":         pdf_path,
            "min_score":        min_score,
            "num_density_pct":  num_density_pct,
            "no_pdf":           True,   # Step1 only needs the match data, not a side PDF
        })
        tool_result = json.loads(tool_result_json)

        if tool_result["status"] != "success":
            log.error("Phase 1 auditor-signature scan failed: %s", tool_result.get("error"))
            raise RuntimeError(
                f"auditor_signature_tool failed: {tool_result.get('error')}"
            )

        matched_pages = sorted(m["page_number"] for m in tool_result["matches"])
        log.info(
            "  auditor_signature_tool matched %d page(s) (1-indexed): %s",
            len(matched_pages), matched_pages,
        )

        if not matched_pages:
            log.warning(
                "Phase 1 found no auditor-signature pages — the PDF may not "
                "contain a recognisable statutory auditor signature block. "
                "Aborting pipeline."
            )
            return OutputResult(
                output_pdf_path = output_path,
                page_mapping    = {k: [] for k in _CATEGORY_META},
                assembled_pages = [],
                total_pages     = 0,
            )

        # ── Re-read the matched pages from the ORIGINAL source PDF ────────────
        # auditor_signature_tool's page_number is 1-indexed into pdf_path.
        # Phase 2 / Phase 3 expect 0-indexed page numbers into that same
        # original PDF, so convert here and read text directly — we
        # deliberately do NOT read from the tool's repaginated side-output.
        try:
            source_doc = fitz.open(pdf_path)
            page_texts: dict[int, str] = {
                pg - 1: source_doc[pg - 1].get_text("text")
                for pg in matched_pages
                if 0 <= pg - 1 < len(source_doc)
            }
            source_doc.close()
        except Exception as e:
            log.error("Failed to re-read matched pages from source PDF: %s", pdf_path)
            raise e

        # Single batch — Phase 1's narrowing already keeps this small relative
        # to the full document, and Phase 2 takes list[dict[int, str]].
        batches = [page_texts]

        if use_cachelite and cachelite_conn is not None and cache_key is not None:
            _cachelite_set(
                cachelite_conn, cache_key, pdf_path, min_score, num_density_pct, batches,
            )

    if cachelite_conn is not None:
        cachelite_conn.close()

    log.info(
        "  Phase 1 produced %d batch(es) covering %d candidate page(s).",
        len(batches),
        sum(len(b) for b in batches),
    )

    if not batches or not any(batches):
        log.warning(
            "Phase 1 returned zero usable pages — aborting pipeline."
        )
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