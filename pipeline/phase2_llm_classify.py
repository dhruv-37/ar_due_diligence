"""
Phase 2 — Structured LLM Classification
========================================
Consumes the token-bounded batches produced by Phase 1 and classifies each
page into one of seven financial-statement categories using the Gemini 1.5
API.

Public surface
--------------
    classify_page_batches(batches, api_key, ...)  ->  ClassifiedPages

Or call the two building-blocks individually:
    call_gemini_batch(batch, api_key, model, url)  ->  list[list]
    aggregate_classifications(raw)                 ->  ClassifiedPages
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ── API constants ─────────────────────────────────────────────────────────────

_DEFAULT_MODEL: str = "gemini-2.5-flash"

# Gemini generateContent endpoint (v1beta supports response_mime_type)
_GEMINI_API_URL: str = (
    "https://generativelanguage.googleapis.com/v1beta/models"
    "/{model}:generateContent?key={api_key}"
)

# Seconds to sleep between successive batch calls — avoids 429s on free tier.
_INTER_BATCH_DELAY: float = 1.0

# Maximum retries per batch on transient errors (5xx / network timeout).
_MAX_RETRIES: int = 3
_RETRY_BACKOFF_BASE: float = 2.0   # seconds; doubles on each retry

# ── Cache constants ────────────────────────────────────────────────────────────

# Default cache location: alongside this module, not the caller's CWD, so the
# cache persists regardless of where Step1.py is invoked from.
_DEFAULT_CACHE_PATH: Path = Path(__file__).resolve().parent / "gemini_cache.sqlite3"

# ── Category code ↔ master-dict key mapping ───────────────────────────────────

_CODE_TO_KEY: dict[str, str] = {
    "BS": "balance_sheet_pages",
    "PL": "profit_loss_pages",
    "CF": "cash_flow_pages",
    "EQ": "equity_pages",
    # XX is the discard sentinel — it is intentionally absent here so the
    # aggregation loop silently skips any page the LLM marks as XX.
}

# ── System / user prompt templates ───────────────────────────────────────────

_SYSTEM_PROMPT: str = """\
You are a financial document classifier specialising in Indian corporate annual reports.

TASK
----
Classify every page in the batch below into exactly one category using the
abbreviated codes defined here:

  BS  – Balance Sheet (assets, liabilities, equity as of a date)
  PL  – Profit & Loss / Income Statement (revenue, expenses, profit)
  CF  – Cash Flow Statement (operating / investing / financing activities)
  EQ  – Statement of Changes in Equity
  XX  – Discard (use for EVERYTHING ELSE — see critical rule below)

CRITICAL DISCARD RULE
---------------------
You MUST assign XX to absolutely every page that is not one of the four core
financial statements above.  This includes — but is not limited to:

  * Notes to Financial Statements (accounting policies, disclosures, schedules)
  * Auditor's / Independent Auditor's Reports
  * Management Discussion & Analysis (MD&A)
  * Director's Reports and Board Reports
  * Corporate Governance Reports
  * Cover pages, tables of contents, blank pages
  * Graphs, infographics, or any non-tabular summary pages
  * Any other narrative, regulatory, or supplementary section

When in doubt, assign XX.  Only assign BS, PL, CF, or EQ when the page
unambiguously contains the primary financial statement table itself.

OUTPUT FORMAT — CRITICAL
------------------------
Return ONLY a JSON array of arrays.  Each inner array must be exactly two
elements: [page_number_integer, "CATEGORY_CODE_STRING"].
No keys, no objects, no markdown, no explanation.

Example (do NOT copy these page numbers — use the real ones from the input):
[[4,"BS"],[5,"BS"],[6,"PL"],[7,"CF"],[8,"XX"],[9,"XX"],[10,"XX"]]
"""

_USER_TEMPLATE: str = """\
Classify the following pages.

{page_blocks}

Return the JSON array of arrays only.
"""


# ─── CACHE LAYER ──────────────────────────────────────────────────────────────
#
# Caches the raw [[page_num, "CODE"], ...] result of a single Gemini batch
# call, keyed on a hash of everything that could change the answer: the
# model name, the system prompt (so editing the classification rules
# auto-invalidates old entries), and the exact batch content (page numbers
# + compressed text). Re-running the same PDF through the same prompt/model
# is then a SQLite lookup instead of an API call.

_SCHEMA = """
CREATE TABLE IF NOT EXISTS gemini_cache (
    cache_key   TEXT PRIMARY KEY,
    model       TEXT NOT NULL,
    page_nums   TEXT NOT NULL,   -- JSON list, for human debugging only
    result_json TEXT NOT NULL,   -- the cached [[page_num, "CODE"], ...] payload
    created_at  REAL NOT NULL
);
"""


def _get_cache_conn(cache_path: Path) -> sqlite3.Connection:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(cache_path))
    conn.execute(_SCHEMA)
    conn.commit()
    return conn


def _compute_cache_key(batch: dict[int, str], model: str) -> str:
    """
    Deterministic hash of (model, system prompt, batch content).

    Including ``_SYSTEM_PROMPT`` means any edit to the classification rules
    or output format automatically invalidates every previously cached
    entry — there is no separate "prompt version" constant to remember to
    bump by hand.
    """
    hasher = hashlib.sha256()
    hasher.update(model.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(_SYSTEM_PROMPT.encode("utf-8"))
    hasher.update(b"\x00")
    # Sort by page number so key is stable regardless of dict ordering.
    for page_num in sorted(batch):
        hasher.update(str(page_num).encode("utf-8"))
        hasher.update(b"\x01")
        hasher.update(batch[page_num].encode("utf-8"))
        hasher.update(b"\x02")
    return hasher.hexdigest()


def _cache_get(conn: sqlite3.Connection, cache_key: str) -> list[list] | None:
    row = conn.execute(
        "SELECT result_json FROM gemini_cache WHERE cache_key = ?",
        (cache_key,),
    ).fetchone()
    if row is None:
        return None
    return json.loads(row[0])


def _cache_set(
    conn: sqlite3.Connection,
    cache_key: str,
    model: str,
    batch: dict[int, str],
    result: list[list],
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO gemini_cache "
        "(cache_key, model, page_nums, result_json, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            cache_key,
            model,
            json.dumps(sorted(batch)),
            json.dumps(result),
            time.time(),
        ),
    )
    conn.commit()


# ─── DATA STRUCTURES ─────────────────────────────────────────────────────────


@dataclass
class ClassifiedPages:
    """
    Aggregated output of classify_page_batches().

    Only the four core financial-statement categories are retained.
    Pages classified as XX (Discard) by the LLM are silently dropped and
    never stored here.

    All page numbers are the original 0-indexed integers from Phase 1
    (i.e. page_num == PDF page index, NOT the human-readable page label).
    """
    balance_sheet_pages:   list[int] = field(default_factory=list)
    profit_loss_pages:     list[int] = field(default_factory=list)
    cash_flow_pages:       list[int] = field(default_factory=list)
    equity_pages:          list[int] = field(default_factory=list)

    # Pages the LLM returned an unrecognised code for — kept for diagnostics.
    unrecognised_pages:    list[tuple[int, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[int]]:
        """Return the canonical master-dict expected by downstream phases."""
        return {
            "balance_sheet_pages": sorted(self.balance_sheet_pages),
            "profit_loss_pages":   sorted(self.profit_loss_pages),
            "cash_flow_pages":     sorted(self.cash_flow_pages),
            "equity_pages":        sorted(self.equity_pages),
        }

    @property
    def total_classified(self) -> int:
        return (
            len(self.balance_sheet_pages)
            + len(self.profit_loss_pages)
            + len(self.cash_flow_pages)
            + len(self.equity_pages)
        )


# ─── STEP 1: SINGLE-BATCH API CALL ───────────────────────────────────────────


def _build_page_blocks(batch: dict[int, str]) -> str:
    """
    Serialise a batch dict into the ``--- PAGE N ---`` delimited string
    that the LLM prompt references.
    """
    parts: list[str] = []
    for page_num in sorted(batch):
        parts.append(f"--- PAGE {page_num + 1} ---\n{batch[page_num]}")
    return "\n\n".join(parts)


def _build_request_payload(batch: dict[int, str]) -> dict[str, Any]:
    """
    Construct the Gemini generateContent request body.

    Key design decisions
    --------------------
    * ``response_mime_type: "application/json"`` forces the model to emit
      valid JSON at the infrastructure level — no regex stripping required.
    * ``system_instruction`` carries the role-definition and output spec so
      the user turn stays lean.
    * ``temperature: 0`` maximises determinism for a classification task.
    * ``thinkingConfig.thinkingBudget: 0`` disables internal reasoning
      tokens. Gemini 2.5+ models reason by default and that reasoning is
      billed against the SAME max_output_tokens budget as the visible
      answer — for a classification task with a fixed, mechanical output
      format we don't need it, and leaving it on silently starves the
      actual JSON answer once batches exceed ~10 pages.
    * ``max_output_tokens`` scales with batch size: each page costs roughly
      ~12 output tokens for its "[N,"XX"]," entry, plus a fixed margin for
      JSON punctuation. A flat 512-token cap truncates any batch beyond
      roughly 15-20 pages.
    """
    page_blocks = _build_page_blocks(batch)
    user_text = _USER_TEMPLATE.format(page_blocks=page_blocks)

    # ~12 tokens per page entry (e.g. `[123,"XX"],`) + fixed JSON overhead,
    # with generous headroom and a sane floor.
    output_budget = max(512, len(batch) * 20 + 128)

    return {
        "system_instruction": {
            "parts": [{"text": _SYSTEM_PROMPT}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_text}],
            }
        ],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.0,
            "max_output_tokens": output_budget,
            "thinking_config": {"thinking_budget": 0},
        },
    }


def call_gemini_batch(
    batch: dict[int, str],
    api_key: str,
    model: str = _DEFAULT_MODEL,
    base_url: str = _GEMINI_API_URL,
    cache_conn: sqlite3.Connection | None = None,
) -> list[list]:
    """
    Send one token batch to the Gemini API and return the parsed
    ``[[page_num, "CODE"], ...]`` array.

    Parameters
    ----------
    batch:
        ``{page_num: compressed_text}`` dict for a single batch
        (as produced by Phase 1's ``TokenBatch.pages``).
    api_key:
        Gemini API key.
    model:
        Gemini model identifier.
    base_url:
        Override for testing / proxies.
    cache_conn:
        Open SQLite connection to the response cache. If provided, an
        exact match (same model + system prompt + batch content) is
        returned without touching the network. Pass ``None`` to disable
        caching for this call.

    Returns
    -------
    list[list]
        Parsed inner arrays, e.g. ``[[4, "BS"], [5, "PL"]]``.
        Returns an empty list on unrecoverable failure (error already logged).

    Raises
    ------
    Does NOT raise — all errors are caught, logged, and returned as [].
    This keeps the orchestrator loop running even when a single batch fails.
    """
    cache_key = _compute_cache_key(batch, model) if cache_conn is not None else None

    if cache_conn is not None:
        cached = _cache_get(cache_conn, cache_key)
        if cached is not None:
            log.info(
                "  Cache hit for pages %s — skipping Gemini call",
                sorted(batch),
            )
            return cached

    url = base_url.format(model=model, api_key=api_key)
    payload = _build_request_payload(batch)
    body = json.dumps(payload).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")

            with urllib.request.urlopen(req, timeout=60) as resp:
                raw_bytes = resp.read()

            response_json: dict[str, Any] = json.loads(raw_bytes)

            # Navigate the Gemini response envelope
            candidates = response_json.get("candidates", [])
            if not candidates:
                log.error("Gemini returned no candidates for batch with pages %s", sorted(batch))
                return []

            finish_reason = candidates[0].get("finishReason", "")
            if finish_reason == "MAX_TOKENS":
                log.error(
                    "Gemini truncated its response (finishReason=MAX_TOKENS) "
                    "for pages %s — output budget was exhausted before the "
                    "JSON array could be completed. This batch will be "
                    "treated as unclassified; consider raising "
                    "max_output_tokens or shrinking the batch size.",
                    sorted(batch),
                )
                return []

            content_text: str = (
                candidates[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
                .strip()
            )

            if not content_text:
                log.error(
                    "Empty text in Gemini response for pages %s "
                    "(finishReason=%s)",
                    sorted(batch), finish_reason or "<none>",
                )
                return []

            parsed = json.loads(content_text)

            # Validate basic shape: must be a list of 2-element lists
            if not isinstance(parsed, list):
                log.error(
                    "Unexpected LLM output type %s for pages %s — expected list",
                    type(parsed).__name__,
                    sorted(batch),
                )
                return []

            validated: list[list] = []
            for item in parsed:
                if (
                    isinstance(item, (list, tuple))
                    and len(item) == 2
                    and isinstance(item[0], int)
                    and isinstance(item[1], str)
                ):
                    # Normalise page_num: LLM was given 1-indexed labels;
                    # convert back to 0-indexed to match Phase 1 convention.
                    validated.append([item[0] - 1, item[1].upper()])
                else:
                    log.warning("Skipping malformed LLM item: %r", item)

            log.debug(
                "Batch pages %s → %d classifications received",
                sorted(batch),
                len(validated),
            )

            if cache_conn is not None:
                _cache_set(cache_conn, cache_key, model, batch, validated)

            return validated

        except urllib.error.HTTPError as exc:
            status = exc.code
            if status == 429 or status >= 500:
                # Transient — back off and retry
                wait = _RETRY_BACKOFF_BASE ** attempt
                log.warning(
                    "HTTP %d on attempt %d/%d for pages %s — retrying in %.1fs",
                    status, attempt, _MAX_RETRIES, sorted(batch), wait,
                )
                time.sleep(wait)
            else:
                body_text = exc.read().decode("utf-8", errors="replace")
                if status == 404 and "is not found for API version" in body_text:
                    # Model name itself is invalid/retired — every batch will
                    # fail identically, so don't burn through the rest of the
                    # run producing a misleadingly empty result. Fail loudly.
                    raise RuntimeError(
                        f"Gemini model '{model}' is invalid or has been "
                        f"retired (HTTP 404). Update _DEFAULT_MODEL / the "
                        f"model argument to a currently supported model "
                        f"(see https://ai.google.dev/gemini-api/docs/models). "
                        f"Raw response: {body_text}"
                    ) from exc
                # Other permanent 4xx (bad request, content blocked, etc.)
                # — log and bail on this batch only.
                log.error(
                    "HTTP %d (permanent) for pages %s: %s",
                    status, sorted(batch), body_text,
                )
                return []

        except urllib.error.URLError as exc:
            wait = _RETRY_BACKOFF_BASE ** attempt
            log.warning(
                "Network error on attempt %d/%d for pages %s: %s — retrying in %.1fs",
                attempt, _MAX_RETRIES, sorted(batch), exc.reason, wait,
            )
            time.sleep(wait)

        except json.JSONDecodeError as exc:
            log.error(
                "JSON decode failed for pages %s: %s", sorted(batch), exc
            )
            return []

        except Exception as exc:  # noqa: BLE001
            log.error(
                "Unexpected error on attempt %d/%d for pages %s: %r",
                attempt, _MAX_RETRIES, sorted(batch), exc,
            )
            time.sleep(_RETRY_BACKOFF_BASE ** attempt)

    log.error(
        "All %d retries exhausted for batch with pages %s — skipping.",
        _MAX_RETRIES, sorted(batch),
    )
    return []


# ─── STEP 2: AGGREGATE RAW CLASSIFICATIONS ────────────────────────────────────


def aggregate_classifications(
    raw: list[list],
) -> ClassifiedPages:
    """
    Convert the flat ``[[page_num_0indexed, "CODE"], ...]`` list produced
    by concatenating all batch results into a ``ClassifiedPages`` object.

    Parameters
    ----------
    raw:
        Combined list of ``[page_num, code]`` pairs from all batches.
        ``page_num`` must already be 0-indexed (``call_gemini_batch`` handles
        the 1→0 conversion automatically).

    Returns
    -------
    ClassifiedPages
        All page lists are de-duplicated and will be sorted in ``to_dict()``.
    """
    result = ClassifiedPages()

    # Deduplicate: if the same page appears in multiple batches (shouldn't
    # happen by design but guard anyway) the last classification wins.
    seen: dict[int, str] = {}
    for item in raw:
        page_num: int = item[0]
        code: str = item[1]
        seen[page_num] = code

    for page_num, code in seen.items():
        # XX is the discard sentinel — skip it entirely (no list, no warning).
        if code == "XX":
            log.debug("Page %d classified as XX (discard) — skipping", page_num + 1)
            continue

        target_key = _CODE_TO_KEY.get(code)
        if target_key is None:
            log.warning("Unrecognised category code %r for page %d", code, page_num + 1)
            result.unrecognised_pages.append((page_num, code))
            continue

        # Append to the appropriate list on ClassifiedPages
        getattr(result, target_key).append(page_num)

    log.info(
        "Aggregation complete: %d pages classified, %d unrecognised",
        result.total_classified,
        len(result.unrecognised_pages),
    )
    if result.unrecognised_pages:
        log.warning("Unrecognised pages: %r", result.unrecognised_pages)

    return result


# ─── ORCHESTRATOR ────────────────────────────────────────────────────────────


def classify_page_batches(
    batches: list[dict[int, str]],
    api_key: str,
    model: str = _DEFAULT_MODEL,
    inter_batch_delay: float = _INTER_BATCH_DELAY,
    use_cache: bool = True,
    cache_path: Path | str = _DEFAULT_CACHE_PATH,
) -> ClassifiedPages:
    """
    Execute the full Phase 2 pipeline: classify every batch via the Gemini
    API, then aggregate results into a master ``ClassifiedPages`` object.

    Parameters
    ----------
    batches:
        ``list[dict[int, str]]`` where each dict maps 0-indexed page numbers
        to their *compressed* text (i.e. ``[b.pages for b in phase1.batches]``).
    api_key:
        Gemini API key (e.g. from ``os.environ["GEMINI_API_KEY"]``).
    model:
        Gemini model to use (default: ``gemini-1.5-flash-latest``).
    inter_batch_delay:
        Seconds to sleep between successive API calls.  Increase to 2–5 s
        if you encounter sustained 429 rate-limit errors.
    use_cache:
        If True (default), check a local SQLite cache before calling the
        API for each batch, and store successful results there. Re-running
        the same PDF (same pages, same prompt, same model) becomes free and
        instant on subsequent runs. Set False to force fresh API calls.
    cache_path:
        Path to the SQLite cache file. Defaults to ``gemini_cache.sqlite3``
        next to this module. Only relevant when ``use_cache=True``.

    Returns
    -------
    ClassifiedPages
        Call ``.to_dict()`` for the clean master dict expected by Phase 3.

    Example
    -------
    >>> import os
    >>> from phase1_filter_batch import run_phase1
    >>> from phase2_llm_classify import classify_page_batches
    >>>
    >>> phase1 = run_phase1(pages, shortlisted)
    >>> batches = [b.pages for b in phase1.batches]
    >>>
    >>> classified = classify_page_batches(batches, api_key=os.environ["GEMINI_API_KEY"])
    >>> print(classified.to_dict())
    {
        "balance_sheet_pages": [11, 12],
        "profit_loss_pages":   [13],
        "cash_flow_pages":     [14],
        ...
    }
    """
    log.info("Phase 2 ─ Structured LLM Classification")
    log.info("  Total batches to classify : %d", len(batches))
    log.info("  Model                     : %s", model)

    cache_conn: sqlite3.Connection | None = None
    if use_cache:
        cache_conn = _get_cache_conn(Path(cache_path))
        log.info("  Cache                     : %s", cache_path)

    all_raw: list[list] = []
    cache_hits = 0

    try:
        for idx, batch in enumerate(batches):
            log.info(
                "  Calling Gemini — batch %d/%d (%d pages: %s)",
                idx + 1,
                len(batches),
                len(batch),
                sorted(batch),
            )

            was_cached = (
                cache_conn is not None
                and _cache_get(cache_conn, _compute_cache_key(batch, model)) is not None
            )

            raw = call_gemini_batch(batch, api_key=api_key, model=model, cache_conn=cache_conn)
            all_raw.extend(raw)

            if was_cached:
                cache_hits += 1

            # Rate-limit guard — skip the delay after the final batch, and
            # skip it entirely for cache hits since no request was made.
            if idx < len(batches) - 1 and not was_cached:
                time.sleep(inter_batch_delay)
    finally:
        if cache_conn is not None:
            cache_conn.close()

    if use_cache and cache_hits:
        log.info(
            "  Cache hits                : %d/%d batch(es) skipped the API",
            cache_hits, len(batches),
        )

    classified = aggregate_classifications(all_raw)

    log.info(
        "Phase 2 complete: %d total pages classified across %d batch(es)",
        classified.total_classified,
        len(batches),
    )

    if batches and classified.total_classified == 0:
        log.warning(
            "Zero pages were classified into BS/PL/CF/EQ across all %d "
            "batch(es). If the source document plausibly contains core "
            "financial statements, this usually means every batch failed "
            "or was truncated upstream (check for HTTP errors, "
            "finishReason=MAX_TOKENS, or empty-response log lines above) "
            "rather than the document genuinely lacking them.",
            len(batches),
        )

    # Emit a concise summary at INFO level for quick visual confirmation
    master = classified.to_dict()
    for category, page_list in master.items():
        if page_list:
            log.info("  %-25s → pages %s", category, page_list)

    return classified