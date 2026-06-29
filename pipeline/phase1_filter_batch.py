"""
Phase 1 — Standalone / Consolidated FS Page Detection & Token Batching
=======================================================================

Step 0 — TOC-Based Range Pre-Filter (NEW)
    Many Indian Annual Reports print a "Contents of this Report" page
    where every section is listed with its printed page number directly
    in front of it (see sample TOC layout). For each FS type, the TOC
    gives us two anchor rows:
        "Independent Auditor's Report on Financial Statement"  -> page X
        "Notes to the Financial Statement"                     -> page Y
    Because the actual Balance Sheet / P&L / Statement of Changes in
    Equity / Cash Flow always sit between the auditor's report and the
    Notes section, we can derive a tight candidate page range directly:
        range = [X + 1, Y - 1]   (notes page itself excluded)
    This range is resolved from *printed* folio numbers to *physical*
    0-indexed PDF page numbers (the two rarely match 1:1 because of
    unnumbered cover/TOC pages), and is computed independently for
    STANDALONE and CONSOLIDATED.

    This stage runs FIRST in the pipeline. When it succeeds, every
    downstream filter only ever sees pages inside these TOC-derived ranges.

Output
------
Phase1Result  →  passed directly into phase2_llm_classify.py.

Public API (additive — all previous functions unchanged)
--------------------------------------------------------
    run_phase1(pages) -> Phase1Result
    toc_range_prefilter(pages) -> TocPrefilterResult
    compress_text_for_llm(text) -> str
    chunk_pages_by_tokens(filtered, max_t) -> list[TokenBatch]
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from difflib import SequenceMatcher
from typing import Optional

try:
    from rapidfuzz import fuzz as _fuzz
except ImportError:
    _fuzz = None




log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════

# Chars from top of page used for TOC header detection.
_HEADER_SCAN_CHARS: int = 800

# ── Header-scan filter constants ───────────────────────────────────────────────
_FS_SCAN_CHARS: int = 300
_FS_FUZZY_THRESHOLD: int = 82

_STANDALONE_PHRASES: list[str] = [
    "standalone financial statements",
    "standalone financial statement",
]
_CONSOLIDATED_PHRASES: list[str] = [
    "consolidated financial statements",
    "consolidated financial statement",
]
_NOTES_PHRASES: list[str] = ["notes"]



# Only scan the first N pages of the PDF when hunting for the contents page.
# TOCs in Indian ARs are essentially always within the first 5 pages.
_TOC_SEARCH_PAGE_LIMIT: int = 6

# Fuzzy phrases that identify the contents/index page itself.
_TOC_HEADER_PHRASES: list[str] = [
    "contents of this report",
    "table of contents",
    "index",
]
_TOC_HEADER_THRESHOLD: int = 78

# A TOC page must also show the "page no." column header to avoid false
# positives on random text pages that just happen to say "contents".
_TOC_PAGE_NO_PHRASES: list[str] = ["page no", "page number", "page nos"]

# Matches a line that is *just* an integer — i.e. the page-number sits on
# its own line (common when PyMuPDF splits a two-column TOC layout).
_TOC_PURE_NUM_LINE_RE = re.compile(r"^\s*(\d{1,4})\s*$")

# Matches a line ending in an integer, preceded by label text on the same
# line (common single-column / tab-leader TOC layout).
_TOC_LABEL_NUM_SAMELINE_RE = re.compile(r"^(.*\S)[ \t.\u2026]{1,}(\d{1,4})\s*$")

# Section header phrase that flips the "current FS type" while walking the
# TOC rows in order.
_TOC_SECTION_RE = re.compile(
    r"(standalone|consolidated)\s+financial\s+statements?", re.IGNORECASE
)
# The auditor's report anchor row.
_TOC_AUDITOR_RE = re.compile(r"independent\s+auditor", re.IGNORECASE)
# The notes anchor row (end boundary, excluded from the final range).
_TOC_NOTES_RE = re.compile(
    r"\bnotes?\b[^\n]{0,40}financial\s+statements?", re.IGNORECASE
)

# When resolving printed folio numbers -> physical 0-indexed PDF pages, only
# trust a detected (physical_idx - printed_folio) offset if it's the winner
# by at least this many votes over the runner-up, otherwise fall back to a
# zero offset (i.e. assume the PDF has no front-matter offset at all).
_TOC_OFFSET_MIN_SAMPLE: int = 5

# ── Text compression constants (used in Step 2/3, unchanged) ──────────────────
_MULTI_NEWLINE_RE          = re.compile(r"\n{3,}")
_MULTI_SPACE_RE            = re.compile(r"[^\S\n]{2,}")
_STANDALONE_PAGE_NUM_RE    = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)
_HEADER_FOOTER_RE          = re.compile(
    r"(?im)"
    r"("
    r"annual\s+report\s+\d{4}"
    r"|cin\s*[:\s]"
    r"|^page\s+\d+"
    r"|^\s*\d+\s*\|\s*"
    r"|©\s*\d{4}"
    r"|all\s+rights\s+reserved"
    r"|(note|notes|schedules)\s+(to|on|forming\s+part\s+of)\s+(the\s+)?financial\s+statements"
    r")",
)
_JUNK_STRING_RE = re.compile(r"(?<!\w)[A-Za-z0-9+/]{40,}(?!\w)")
_CHARS_PER_TOKEN: int = 4
_BATCH_PROMPT_OVERHEAD_TOKENS: int = 300


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMERATIONS & DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class FSType(Enum):
    STANDALONE   = auto()
    CONSOLIDATED = auto()


@dataclass
class TocPrefilterResult:
    """
    Output of toc_range_prefilter() — Step 0 of the pipeline.
    """
    toc_found:        bool                              = False
    toc_page:          Optional[int]                    = None
    page_ranges:       dict[FSType, tuple[int, int]]     = field(default_factory=dict)
    candidate_pages:   set[int]                          = field(default_factory=set)
    printed_to_physical_offset: int                      = 0


@dataclass
class FilteredPages:
    """Pages selected by the TOC pre-filter."""
    pages:            dict[int, str]     = field(default_factory=dict)
    page_fs_type:     dict[int, FSType]  = field(default_factory=dict)
    total_input_pages: int = 0
    pages_removed:     int = 0

    @property
    def pages_kept(self) -> int:
        return len(self.pages)


@dataclass
class TokenBatch:
    """One batch destined for a single LLM API call (phase2_llm_classify.py)."""
    batch_index:      int
    fs_type:          Optional[FSType]   = None
    pages:            dict[int, str]     = field(default_factory=dict)
    estimated_tokens: int = 0

    def page_numbers(self) -> list[int]:
        return sorted(self.pages)


@dataclass
class Phase1Result:
    """Aggregated output of run_phase1(); passed directly to Phase 2."""
    batches:  list[TokenBatch] = field(default_factory=list)
    filtered: FilteredPages    = field(default_factory=FilteredPages)

    @property
    def total_batches(self) -> int:
        return len(self.batches)

    @property
    def total_pages_in_batches(self) -> int:
        return sum(len(b.pages) for b in self.batches)

    @property
    def total_estimated_tokens(self) -> int:
        return sum(b.estimated_tokens for b in self.batches)


# ═══════════════════════════════════════════════════════════════════════════════
# CORE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_page_dict(page_dict: dict) -> tuple[str, list[dict]]:
    """
    Flattens a PyMuPDF page dictionary into a single raw text string 
    and a list of spans with font sizes.
    """
    full_text = ""
    spans_info = []
    
    for block in page_dict.get("blocks", []):
        if block.get("type") == 0:  # Text block
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if not text:
                        continue
                        
                    start_idx = len(full_text)
                    full_text += text
                    end_idx = len(full_text)
                    
                    spans_info.append({
                        "text": text,
                        "size": round(span.get("size", 0.0), 2),
                        "start": start_idx,
                        "end": end_idx
                    })
                full_text += "\n" 
                
    return full_text, spans_info


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _fuzzy_contains(haystack: str, needle: str, threshold: int) -> bool:
    if _fuzz is not None:
        return _fuzz.partial_ratio(needle, haystack) >= threshold
    n = len(needle)
    for i in range(max(1, len(haystack) - n + 1)):
        chunk = haystack[i : i + int(n * 1.3)]
        if int(SequenceMatcher(None, chunk, needle).ratio() * 100) >= threshold:
            return True
    return False


def _get_size_at(spans: list[dict], char_idx: int) -> float:
    for s in spans:
        if s["start"] <= char_idx < s["end"]:
            return s["size"]
    return 0.0


def _fuzzy_find(text: str, phrases: list[str], threshold: int) -> Optional[int]:
    """Return the char index of the first fuzzy match of any phrase, or None."""
    norm = _normalise(text)
    for phrase in phrases:
        if _fuzz is not None:
            # slide a window to locate approximate position
            n = len(phrase)
            win = int(n * 1.3)
            for i in range(max(1, len(norm) - n + 1)):
                chunk = norm[i : i + win]
                if _fuzz.partial_ratio(phrase, chunk) >= threshold:
                    return i
        else:
            n = len(phrase)
            for i in range(max(1, len(norm) - n + 1)):
                chunk = norm[i : i + int(n * 1.3)]
                if int(SequenceMatcher(None, chunk, phrase).ratio() * 100) >= threshold:
                    return i
    return None


def header_scan_filter(pages: dict[int, dict]) -> FilteredPages:
    """
    For each page, scan the first _FS_SCAN_CHARS for a fuzzy match of
    'standalone/consolidated financial statements'. Accept the page and
    tag its FSType. Then check if 'notes' also appears in the same window
    with a larger font than the FS phrase — if so, discard the page.
    """
    surviving:    dict[int, str]    = {}
    page_fs_type: dict[int, FSType] = {}
    total = len(pages)

    for pg in sorted(pages):
        text, spans = _parse_page_dict(pages[pg])
        window = text[:_FS_SCAN_CHARS]

        # Determine FS type via fuzzy match
        fs_type: Optional[FSType] = None
        fs_char_idx: Optional[int] = None

        idx = _fuzzy_find(window, _STANDALONE_PHRASES, _FS_FUZZY_THRESHOLD)
        if idx is not None:
            fs_type, fs_char_idx = FSType.STANDALONE, idx

        if fs_type is None:
            idx = _fuzzy_find(window, _CONSOLIDATED_PHRASES, _FS_FUZZY_THRESHOLD)
            if idx is not None:
                fs_type, fs_char_idx = FSType.CONSOLIDATED, idx

        if fs_type is None:
            log.debug("Page %d: no FS header found; skipping.", pg + 1)
            continue

        # Check for 'notes' with larger font
        notes_idx = _fuzzy_find(window, _NOTES_PHRASES, _FS_FUZZY_THRESHOLD)
        if notes_idx is not None:
            size_fs    = _get_size_at(spans, fs_char_idx)
            size_notes = _get_size_at(spans, notes_idx)
            if size_notes > size_fs:
                log.info(
                    "Page %d: discarded — 'notes' (%.1fpt) > FS phrase (%.1fpt).",
                    pg + 1, size_notes, size_fs,
                )
                continue

        surviving[pg]    = text
        page_fs_type[pg] = fs_type
        log.info("Page %d: accepted as %s.", pg + 1, fs_type.name)

    return FilteredPages(
        pages=dict(sorted(surviving.items())),
        page_fs_type=page_fs_type,
        total_input_pages=total,
        pages_removed=total - len(surviving),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 0 — TOC-BASED RANGE PRE-FILTER
# ═══════════════════════════════════════════════════════════════════════════════

def _find_toc_page(pages: dict[int, dict]) -> Optional[int]:
    sorted_nums = sorted(pages)[:_TOC_SEARCH_PAGE_LIMIT]
    for pg in sorted_nums:
        text, _ = _parse_page_dict(pages[pg])
        header = _normalise(text[:_HEADER_SCAN_CHARS])

        has_toc_phrase = any(
            _fuzzy_contains(header, phrase, _TOC_HEADER_THRESHOLD)
            for phrase in _TOC_HEADER_PHRASES
        )
        if not has_toc_phrase:
            continue

        has_page_no_col = any(p in header for p in _TOC_PAGE_NO_PHRASES)
        if has_page_no_col:
            return pg

    return None


def _extract_toc_entries(text: str) -> list[tuple[str, int]]:
    entries: list[tuple[str, int]] = []
    pending_label_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m_pure = _TOC_PURE_NUM_LINE_RE.match(line)
        if m_pure:
            if pending_label_lines:
                label = " ".join(pending_label_lines).strip()
                entries.append((label, int(m_pure.group(1))))
                pending_label_lines = []
            continue

        m_same = _TOC_LABEL_NUM_SAMELINE_RE.match(line)
        if m_same and not re.fullmatch(r"[\d\s.,\-]+", m_same.group(1)):
            label_part = m_same.group(1).strip()
            full_label = " ".join(pending_label_lines + [label_part]).strip()
            entries.append((full_label, int(m_same.group(2))))
            pending_label_lines = []
            continue

        pending_label_lines.append(line)

    return entries


def _resolve_fs_anchor_rows(
    entries: list[tuple[str, int]],
) -> dict[FSType, tuple[int, int]]:
    auditor_pg: dict[FSType, int] = {}
    notes_pg:   dict[FSType, int] = {}
    current_fs: Optional[FSType] = None

    for label, num in entries:
        sec_match = _TOC_SECTION_RE.search(label)
        if sec_match:
            current_fs = (
                FSType.STANDALONE
                if sec_match.group(1).lower() == "standalone"
                else FSType.CONSOLIDATED
            )

        if current_fs and _TOC_AUDITOR_RE.search(label) and current_fs not in auditor_pg:
            auditor_pg[current_fs] = num

        if current_fs and _TOC_NOTES_RE.search(label):
            notes_pg[current_fs] = num

    return {
        fs: (auditor_pg[fs], notes_pg[fs])
        for fs in (FSType.STANDALONE, FSType.CONSOLIDATED)
        if fs in auditor_pg and fs in notes_pg
    }


def _detect_folio_on_page(text: str, edge_lines: int = 4) -> Optional[int]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None

    candidates = lines[:edge_lines] + lines[-edge_lines:]
    for ln in candidates:
        if re.fullmatch(r"\d{1,4}", ln):
            return int(ln)
    return None


def _compute_folio_offset(pages: dict[int, dict]) -> int:
    votes: Counter[int] = Counter()

    for pg in sorted(pages):
        text, _ = _parse_page_dict(pages[pg])
        folio = _detect_folio_on_page(text)
        if folio is not None and 0 < folio <= len(pages) + 50:
            votes[pg - folio] += 1

    if not votes:
        return 0

    (best_offset, best_count), *rest = votes.most_common()
    runner_up_count = rest[0][1] if rest else 0

    if best_count < _TOC_OFFSET_MIN_SAMPLE and best_count <= runner_up_count:
        return 0

    return best_offset


def toc_range_prefilter(pages: dict[int, dict]) -> TocPrefilterResult:
    log.info("Step 0: Searching for TOC / contents page...")

    toc_pg = _find_toc_page(pages)
    if toc_pg is None:
        log.info("  -> No contents page found in first %d pages; skipping TOC pre-filter.",
                  _TOC_SEARCH_PAGE_LIMIT)
        return TocPrefilterResult(toc_found=False)

    log.info("  -> Contents page located at physical page %d", toc_pg + 1)

    toc_text, _ = _parse_page_dict(pages[toc_pg])
    entries = _extract_toc_entries(toc_text)
    fs_anchor_rows = _resolve_fs_anchor_rows(entries)

    if not fs_anchor_rows:
        log.info("  -> Contents page found but no auditor's-report/notes anchor "
                  "pair could be resolved; skipping TOC pre-filter.")
        return TocPrefilterResult(toc_found=True, toc_page=toc_pg)

    offset = _compute_folio_offset(pages)
    log.info("  -> Printed-folio -> physical-index offset resolved to %+d", offset)

    page_ranges: dict[FSType, tuple[int, int]] = {}
    candidate_pages: set[int] = set()
    total_physical_pages = len(pages)

    for fs_type, (auditor_no, notes_no) in fs_anchor_rows.items():
        start_pg = auditor_no + 1 + offset
        end_pg   = notes_no - 1 + offset

        start_pg = max(0, start_pg)
        end_pg   = min(total_physical_pages - 1, end_pg)

        if start_pg > end_pg:
            log.warning(
                "  -> %s range came out inverted/empty (printed %d->%d, "
                "physical %d->%d); discarding this FS type's range.",
                fs_type.name, auditor_no, notes_no, start_pg, end_pg,
            )
            continue

        page_ranges[fs_type] = (start_pg, end_pg)
        candidate_pages.update(range(start_pg, end_pg + 1))
        log.info(
            "  -> %-12s printed pages %d-%d (auditor+1 .. notes-1) "
            "-> physical pages %d-%d",
            fs_type.name, auditor_no + 1, notes_no - 1, start_pg + 1, end_pg + 1,
        )

    if not candidate_pages:
        log.info("  -> All resolved ranges were empty/invalid; skipping TOC pre-filter.")
        return TocPrefilterResult(toc_found=True, toc_page=toc_pg,
                                   printed_to_physical_offset=offset)

    return TocPrefilterResult(
        toc_found=True,
        toc_page=toc_pg,
        page_ranges=page_ranges,
        candidate_pages=candidate_pages,
        printed_to_physical_offset=offset,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — TEXT COMPRESSION
# ═══════════════════════════════════════════════════════════════════════════════

def compress_text_for_llm(text: str) -> str:
    """
    Strip boilerplate from raw page text to minimise LLM token consumption
    while preserving every financially meaningful token.
    """
    if not text:
        return ""
    text = _HEADER_FOOTER_RE.sub("", text)
    text = _STANDALONE_PAGE_NUM_RE.sub("", text)
    text = _JUNK_STRING_RE.sub("", text)
    text = _MULTI_NEWLINE_RE.sub("\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — TOKEN-AWARE BATCHING
# ═══════════════════════════════════════════════════════════════════════════════

def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def chunk_pages_by_tokens(
    filtered_pages: FilteredPages,
    max_tokens: int = 15_000,
) -> list[TokenBatch]:
    effective_budget = max_tokens - _BATCH_PROMPT_OVERHEAD_TOKENS
    if effective_budget <= 0:
        raise ValueError(
            f"max_tokens ({max_tokens}) is too small to fit the prompt "
            f"overhead ({_BATCH_PROMPT_OVERHEAD_TOKENS} tokens)."
        )

    batches:       list[TokenBatch] = []
    current_batch: dict[int, str]   = {}
    current_tokens: int             = 0
    current_type:  Optional[FSType] = None
    batch_index:   int              = 0

    def _flush():
        nonlocal current_batch, current_tokens, current_type, batch_index
        if current_batch:
            batches.append(TokenBatch(
                batch_index=batch_index,
                fs_type=current_type,
                pages=current_batch,
                estimated_tokens=current_tokens,
            ))
            batch_index  += 1
            current_batch = {}
            current_tokens = 0
            current_type  = None

    for page_num in sorted(filtered_pages.pages):
        raw_text    = filtered_pages.pages[page_num]
        compressed  = compress_text_for_llm(raw_text)
        delimiter   = f"--- PAGE {page_num + 1} ---\n"
        page_cost   = _estimate_tokens(delimiter + compressed)
        page_type   = filtered_pages.page_fs_type.get(page_num)

        if page_cost > effective_budget:
            log.warning(
                "Page %d compressed to ~%d tokens, exceeds budget %d. "
                "Emitting as standalone batch.",
                page_num + 1, page_cost, effective_budget,
            )
            _flush()
            batches.append(TokenBatch(
                batch_index=batch_index,
                fs_type=page_type,
                pages={page_num: compressed},
                estimated_tokens=page_cost,
            ))
            batch_index += 1
            continue

        type_boundary  = (current_type is not None and page_type != current_type)
        budget_overrun = (current_batch and (current_tokens + page_cost) > effective_budget)

        if type_boundary or budget_overrun:
            reason = "type boundary" if type_boundary else "token budget"
            log.debug("Flushing batch at page %d (%s)", page_num + 1, reason)
            _flush()

        current_batch[page_num] = compressed
        current_tokens         += page_cost
        current_type            = page_type

    _flush()

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
            "  Batch %d | type=%-12s | pages=%s | ~%d tokens",
            b.batch_index,
            b.fs_type.name if b.fs_type else "UNKNOWN",
            b.page_numbers(),
            b.estimated_tokens,
        )
    return batches


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase1(
    pages: dict[int, dict],
    scored_pages=None,
    max_tokens_per_batch: int = 15_000,
) -> Phase1Result:
    log.info("=" * 60)
    log.info("Phase 1 — Standalone / Consolidated FS Detection")
    log.info("  Total PDF pages: %d", len(pages))

    toc_result = toc_range_prefilter(pages)

    if toc_result.toc_found and toc_result.page_ranges:
        # Narrow to TOC-derived ranges first, then apply header scan per range
        merged_pages:        dict[int, str]    = {}
        merged_page_fs_type: dict[int, FSType] = {}

        for fs_type, (start_pg, end_pg) in toc_result.page_ranges.items():
            range_pages = {pg: pages[pg] for pg in range(start_pg, end_pg + 1) if pg in pages}
            range_filtered = header_scan_filter(range_pages)
            for pg, text in range_filtered.pages.items():
                merged_pages[pg]        = text
                merged_page_fs_type[pg] = fs_type  # trust TOC for type assignment

        filtered = FilteredPages(
            pages=dict(sorted(merged_pages.items())),
            page_fs_type=merged_page_fs_type,
            total_input_pages=len(pages),
            pages_removed=len(pages) - len(merged_pages),
        )
        log.info("Step 0 narrowed to %d range(s); header scan kept %d page(s).",
                 len(toc_result.page_ranges), len(merged_pages))
    else:
        log.info("Step 0 found no TOC range; running header scan on all pages.")
        filtered = header_scan_filter(pages)

    batches = chunk_pages_by_tokens(filtered, max_tokens=max_tokens_per_batch)
    result  = Phase1Result(batches=batches, filtered=filtered)

    log.info(
        "Phase 1 complete: %d pages → %d batch(es) | ~%d total tokens",
        result.total_pages_in_batches,
        result.total_batches,
        result.total_estimated_tokens,
    )
    log.info("=" * 60)
    return result

if __name__ == "__main__":
    import argparse
    import fitz
    import sys

    # Set up argument parsing to take the path from the terminal
    parser = argparse.ArgumentParser(description="Run Phase 1 FS detection.")
    parser.add_argument("pdf_path", help="Path to the PDF file (e.g., ../data/pdfs/file.pdf)")
    args = parser.parse_args()

    # Open the file from the terminal-provided path
    try:
        doc = fitz.open(args.pdf_path)
        log.info(f"📄 Opened: {args.pdf_path}")
        
        # Build the page dictionary as required by run_phase1
        mock_pages = {i: page.get_text("dict") for i, page in enumerate(doc)}
        doc.close()
        
        # Execute the orchestrator
        result = run_phase1(mock_pages)
        # Add this right after run_phase1(mock_pages)
        print("\n--- List of Kept Pages (1-indexed) ---")
        for batch in result.batches:
            print(f"Batch {batch.batch_index} ({batch.fs_type.name}): {sorted([p + 1 for p in batch.pages.keys()])}")        
        print("\n--- Phase 1 Execution Summary ---")
        print(f"Total Batches: {result.total_batches}")
        print(f"Total Pages Kept: {result.total_pages_in_batches}")

    except Exception as e:
        log.error(f"❌ Failed to process: {e}")
        sys.exit(1)