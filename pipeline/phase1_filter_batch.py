"""
Phase 1 — Heuristic Filtering & Token Batching
===============================================
Sits between raw PDF page extraction (Step1_phase2.py Stage 1) and the
future Phase 2 LLM API call.

Public surface
--------------
    run_phase1(pages, scored_pages) -> Phase1Result

Or call the three building-blocks individually:
    heuristic_filter(pages, scored_pages)   -> FilteredPages
    compress_text_for_llm(text)             -> str
    chunk_pages_by_tokens(filtered, max_t)  -> list[TokenBatch]
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Avoid a hard import dependency; Step1_phase2 is the caller, not the callee.
    from step1_phase2 import PageScore

log = logging.getLogger(__name__)

# ── Mirror the thresholds that live in Step1_phase2 ──────────────────────────
# Import them when available; fall back to their documented defaults so this
# module is also usable in isolation (e.g. unit tests).
try:
    from step1_phase2 import (  # type: ignore[import]
        KEYWORD_SCORE_THRESHOLD,
        TABLE_DENSITY_THRESHOLD,
    )
except ImportError:
    KEYWORD_SCORE_THRESHOLD: int = 8
    TABLE_DENSITY_THRESHOLD: float = 0.02

# ── Regex constants (compiled once at import time) ────────────────────────────

# 3 or more consecutive newlines → single newline
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

# Multiple spaces / tabs between non-newline chars → single space
_MULTI_SPACE_RE = re.compile(r"[^\S\n]{2,}")

# Isolated page numbers: a line that contains *only* digits (1–4), optional whitespace
_STANDALONE_PAGE_NUM_RE = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)

# Common repeating headers / footers found in Indian annual reports
_HEADER_FOOTER_RE = re.compile(
    r"(?im)"
    r"("
    r"annual\s+report\s+\d{4}"           # "Annual Report 2023"
    r"|cin\s*[:\s]"                       # "CIN: L12345..."
    r"|^page\s+\d+"                       # "Page 12"
    r"|^\s*\d+\s*\|\s*"                  # "12 | " (pipe-style page numbers)
    r"|©\s*\d{4}"                         # copyright lines
    r"|all\s+rights\s+reserved"          # boilerplate footer
    # --- Expanded pattern for "Notes to..." headers ---
    # This is critical because the LLM prompt strictly instructs the model
    # to discard pages with these headers. By removing a wider variety of
    # them here (e.g., "Note on...", "Schedules forming part of..."), we
    # allow the LLM to classify the page based on its actual table content.
    r"|(note|notes|schedules)\s+(to|on|forming\s+part\s+of)\s+(the\s+)?financial\s+statements"
    r")",
)

# Junk strings: 40+ consecutive alphanumeric chars with no spaces
# (base64 blobs, hex dump fragments, garbled PDF artefacts)
_JUNK_STRING_RE = re.compile(r"(?<!\w)[A-Za-z0-9+/]{40,}(?!\w)")

# ─── DATA STRUCTURES ─────────────────────────────────────────────────────────


@dataclass
class FilteredPages:
    """
    Output of heuristic_filter().
    Maps original (0-indexed) page number → raw (uncompressed) text for
    every page that survived the filter.
    Carries a summary so callers can log without re-iterating the dict.
    """
    pages: dict[int, str] = field(default_factory=dict)
    total_input_pages: int = 0
    pages_removed: int = 0

    @property
    def pages_kept(self) -> int:
        return len(self.pages)


@dataclass
class TokenBatch:
    """
    One batch destined for a single LLM API call.
    ``pages`` maps original page number → *compressed* text.
    ``estimated_tokens`` is the fast-approximation token count for the batch.
    """
    batch_index: int
    pages: dict[int, str] = field(default_factory=dict)
    estimated_tokens: int = 0

    def page_numbers(self) -> list[int]:
        """Sorted list of page numbers (0-indexed) in this batch."""
        return sorted(self.pages)


@dataclass
class Phase1Result:
    """
    Aggregated output of run_phase1(); hand this directly to Phase 2.
    """
    batches: list[TokenBatch] = field(default_factory=list)
    filtered: FilteredPages = field(default_factory=FilteredPages)

    # Convenience metrics
    @property
    def total_batches(self) -> int:
        return len(self.batches)

    @property
    def total_pages_in_batches(self) -> int:
        return sum(len(b.pages) for b in self.batches)

    @property
    def total_estimated_tokens(self) -> int:
        return sum(b.estimated_tokens for b in self.batches)


# ─── STEP 1: HEURISTIC PRE-FILTER ────────────────────────────────────────────


def heuristic_filter(
    pages: dict[int, str],
    scored_pages: list["PageScore"],
) -> FilteredPages:
    """
    Retain only pages that pass at least one of the existing quality gates:

        keyword_score  > KEYWORD_SCORE_THRESHOLD   (Stage 4 metric)
        table_density  > TABLE_DENSITY_THRESHOLD    (Stage 5 metric)

    Parameters
    ----------
    pages:
        Full dict of ``{page_num: raw_text}`` from Stage 1.
    scored_pages:
        List of ``PageScore`` objects produced by Stage 5
        (``filter_by_table_density``). Each object already carries both
        ``keyword_score`` and ``table_density``.

    Returns
    -------
    FilteredPages
        A wrapper around the surviving ``{page_num: raw_text}`` subset.
    """
    surviving: dict[int, str] = {}

    for ps in scored_pages:
        passes_keyword = ps.keyword_score > KEYWORD_SCORE_THRESHOLD
        passes_density = ps.table_density > TABLE_DENSITY_THRESHOLD

        if passes_keyword or passes_density:
            raw_text = pages.get(ps.page_num, "")
            if raw_text:  # skip genuinely empty pages
                surviving[ps.page_num] = raw_text
        else:
            log.debug(
                "Page %d dropped — keyword_score=%d, table_density=%.4f",
                ps.page_num + 1,
                ps.keyword_score,
                ps.table_density,
            )

    result = FilteredPages(
        pages=surviving,
        total_input_pages=len(scored_pages),
        pages_removed=len(scored_pages) - len(surviving),
    )
    log.info(
        "Heuristic filter: %d/%d pages kept (%d removed)",
        result.pages_kept,
        result.total_input_pages,
        result.pages_removed,
    )
    return result


# ─── STEP 2: TEXT COMPRESSION ────────────────────────────────────────────────


def compress_text_for_llm(text: str) -> str:
    """
    Aggressively clean a page's raw text to minimise LLM token consumption
    while preserving every financially meaningful token.

    Transformations (applied in order)
    ------------------------------------
    1. Strip common repeating headers / footers (annual report title,
       CIN lines, page labels, copyright notices).
    2. Remove isolated standalone page numbers (lines containing only digits).
    3. Drop long uninterrupted alphanumeric blobs — base64 image data,
       hex strings, and other PDF parse artefacts.
    4. Collapse 3+ consecutive newlines to a single ``\\n``.
    5. Collapse 2+ spaces / tabs on the same line to a single space.
    6. Strip leading / trailing whitespace.

    Parameters
    ----------
    text : str
        Raw page text as returned by PyMuPDF's ``get_text()``.

    Returns
    -------
    str
        Compressed text suitable for inclusion in an LLM prompt.
    """
    if not text:
        return ""

    # 1. Headers / footers
    text = _HEADER_FOOTER_RE.sub("", text)

    # 2. Isolated page numbers
    text = _STANDALONE_PAGE_NUM_RE.sub("", text)

    # 3. Junk blobs (base64 / hex artefacts)
    text = _JUNK_STRING_RE.sub("", text)

    # 4. Excessive newlines (3+ → 1)
    text = _MULTI_NEWLINE_RE.sub("\n", text)

    # 5. Redundant horizontal whitespace
    text = _MULTI_SPACE_RE.sub(" ", text)

    # 6. Trim
    return text.strip()


# ─── STEP 3: TOKEN-AWARE BATCHING ────────────────────────────────────────────

# Approximation constant: 1 LLM token ≈ 4 characters (GPT / Claude average)
_CHARS_PER_TOKEN: int = 4

# Overhead budget per batch: accounts for the system-prompt, page-number
# delimiters, and JSON wrapper that Phase 2 will add.
_BATCH_PROMPT_OVERHEAD_TOKENS: int = 300


def _estimate_tokens(text: str) -> int:
    """Fast token count approximation: len(text) // 4."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def chunk_pages_by_tokens(
    filtered_pages: FilteredPages,
    max_tokens: int = 15_000,
) -> list[TokenBatch]:
    """
    Compress each surviving page and pack pages into token-bounded batches.

    Algorithm
    ---------
    Pages are processed in ascending page-number order (preserving document
    order within each batch).  A new batch is started whenever adding the
    next page would push the running total past ``max_tokens``.

    A page whose *compressed* text alone exceeds the token budget is placed
    in a batch by itself and a warning is logged — the caller / Phase 2
    must handle oversized pages (e.g. by chunking at the paragraph level).

    Parameters
    ----------
    filtered_pages : FilteredPages
        Output of ``heuristic_filter()``.
    max_tokens : int
        Hard ceiling per batch (default 15 000).  Leave headroom for the
        Phase 2 system-prompt; the function subtracts
        ``_BATCH_PROMPT_OVERHEAD_TOKENS`` from the effective budget.

    Returns
    -------
    list[TokenBatch]
        Ordered list of batches, each ready to be serialised into a Phase 2
        prompt.  Every batch retains the original 0-indexed page numbers as
        keys so Phase 2 can label its output correctly.
    """
    effective_budget = max_tokens - _BATCH_PROMPT_OVERHEAD_TOKENS
    if effective_budget <= 0:
        raise ValueError(
            f"max_tokens ({max_tokens}) is too small to accommodate even the "
            f"prompt overhead ({_BATCH_PROMPT_OVERHEAD_TOKENS} tokens)."
        )

    batches: list[TokenBatch] = []
    current_batch: dict[int, str] = {}
    current_tokens: int = 0
    batch_index: int = 0

    # Sort by page number to maintain document order
    for page_num in sorted(filtered_pages.pages):
        raw_text = filtered_pages.pages[page_num]
        compressed = compress_text_for_llm(raw_text)

        # Per-page delimiter that Phase 2 will use: "--- PAGE N ---\n"
        delimiter = f"--- PAGE {page_num + 1} ---\n"
        page_token_cost = _estimate_tokens(delimiter + compressed)

        # Oversized single page — emit it in isolation with a warning
        if page_token_cost > effective_budget:
            log.warning(
                "Page %d compressed to ~%d tokens, which exceeds the effective "
                "budget of %d tokens. Emitting as a standalone batch.",
                page_num + 1,
                page_token_cost,
                effective_budget,
            )
            # Flush current batch first (if non-empty)
            if current_batch:
                batches.append(
                    TokenBatch(
                        batch_index=batch_index,
                        pages=current_batch,
                        estimated_tokens=current_tokens,
                    )
                )
                batch_index += 1
                current_batch = {}
                current_tokens = 0

            batches.append(
                TokenBatch(
                    batch_index=batch_index,
                    pages={page_num: compressed},
                    estimated_tokens=page_token_cost,
                )
            )
            batch_index += 1
            continue

        # Would adding this page overflow the current batch?
        if current_batch and (current_tokens + page_token_cost) > effective_budget:
            batches.append(
                TokenBatch(
                    batch_index=batch_index,
                    pages=current_batch,
                    estimated_tokens=current_tokens,
                )
            )
            batch_index += 1
            current_batch = {}
            current_tokens = 0

        current_batch[page_num] = compressed
        current_tokens += page_token_cost

    # Flush the final (possibly partial) batch
    if current_batch:
        batches.append(
            TokenBatch(
                batch_index=batch_index,
                pages=current_batch,
                estimated_tokens=current_tokens,
            )
        )

    log.info(
        "Token batching: %d pages → %d batch(es) "
        "(max_tokens=%d, overhead=%d, effective_budget=%d)",
        filtered_pages.pages_kept,
        len(batches),
        max_tokens,
        _BATCH_PROMPT_OVERHEAD_TOKENS,
        effective_budget,
    )
    for b in batches:
        log.debug(
            "  Batch %d: pages %s | ~%d tokens",
            b.batch_index,
            b.page_numbers(),
            b.estimated_tokens,
        )

    return batches


# ─── ORCHESTRATOR ────────────────────────────────────────────────────────────


def run_phase1(
    pages: dict[int, str],
    scored_pages: list["PageScore"],
    max_tokens_per_batch: int = 15_000,
) -> Phase1Result:
    """
    Execute the full Phase 1 pipeline and return a ``Phase1Result`` whose
    ``batches`` list is ready to be consumed by Phase 2.

    Parameters
    ----------
    pages : dict[int, str]
        Raw page text from ``extract_pages()`` (Stage 1 of Step1_phase2).
    scored_pages : list[PageScore]
        Shortlisted pages from ``filter_by_table_density()`` (Stage 5).
    max_tokens_per_batch : int
        Token ceiling per LLM call. Defaults to 15 000.

    Returns
    -------
    Phase1Result
        Contains ``batches`` (for Phase 2) and ``filtered`` (for diagnostics).

    Example
    -------
    >>> # --- inside Step1_phase2.extract_fs_pages() ---
    >>> from phase1_filter_batch import run_phase1
    >>>
    >>> pages = extract_pages(doc)                               # Stage 1
    >>> ...
    >>> shortlisted = filter_by_table_density(scored, pages)    # Stage 5
    >>>
    >>> phase1 = run_phase1(pages, shortlisted, max_tokens_per_batch=15_000)
    >>> # Hand off to Phase 2:
    >>> for batch in phase1.batches:
    ...     llm_response = call_llm_api(batch)   # Phase 2 (not implemented here)
    """
    log.info("Phase 1 ─ Heuristic Filtering & Token Batching")
    log.info("  Input scored pages : %d", len(scored_pages))

    filtered = heuristic_filter(pages, scored_pages)
    batches = chunk_pages_by_tokens(filtered, max_tokens=max_tokens_per_batch)

    result = Phase1Result(batches=batches, filtered=filtered)

    log.info(
        "Phase 1 complete: %d pages → %d batch(es) | "
        "~%d total tokens across all batches",
        result.total_pages_in_batches,
        result.total_batches,
        result.total_estimated_tokens,
    )
    return result
