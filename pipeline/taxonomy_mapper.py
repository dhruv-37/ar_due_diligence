"""
taxonomy_mapper.py
==================
Deterministic, non-LLM mapping of extracted line items to the predefined
items in fs_dictionary.py (scope -> statement -> item -> keywords), via
fuzzy string matching only. If no keyword scores above MATCH_THRESHOLD,
the node name falls back to the raw line item text itself.

Usage
-----
    from pipeline.taxonomy_mapper import map_line_items
    records = map_line_items(records)   # adds taxonomy_node, fs_statement, match_score
"""

from __future__ import annotations
import re

try:
    from pipeline.fs_dictionary import FS_DICTIONARY, new_fs_dictionary
except ImportError:
    from fs_dictionary import FS_DICTIONARY, new_fs_dictionary

try:
    from rapidfuzz import fuzz
    _SCORER = lambda a, b: fuzz.token_sort_ratio(a, b)
except ImportError:
    import difflib
    _SCORER = lambda a, b: difflib.SequenceMatcher(None, a, b).ratio() * 100

MATCH_THRESHOLD = 72  # 0-100; below this, node name = raw line item text


def _normalize(text: str) -> str:
    text = str(text or "").lower().strip()
    text = re.sub(r"[^a-z0-9%()/\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


# Per-item ordered keyword lists (order preserved as written in fs_dictionary.py):
# (item_name, statement, [normalized_keyword, ...])  -- item_name itself appended last as a fallback keyword.
_ITEM_KEYWORDS: list[tuple[str, str, list[str]]] = []
for _statement, _items in FS_DICTIONARY["STANDALONE"].items():
    for _item_name, _meta in _items.items():
        _ordered = [_normalize(_kw) for _kw in _meta["keywords"]] + [_normalize(_item_name)]
        _ITEM_KEYWORDS.append((_item_name, _statement, _ordered))


def _ranked_matches(raw_string: str) -> list[tuple[str, str, float]]:
    """All (item_name, statement, score) candidates for raw_string, sorted
    best-first, restricted to those meeting MATCH_THRESHOLD.

    Per item, keywords are checked in the exact order they are written in
    fs_dictionary.py. The first keyword that clears MATCH_THRESHOLD seals
    that item's score immediately — later keywords for that same item are
    not checked."""
    target = _normalize(raw_string)
    if not target:
        return []
    scored: dict[tuple[str, str], float] = {}
    for item_name, statement, keywords in _ITEM_KEYWORDS:
        for candidate in keywords:
            score = _SCORER(target, candidate)
            if score >= MATCH_THRESHOLD:
                scored[(item_name, statement)] = score
                break  # sealed on first qualifying keyword, in written order
    ranked = [(k[0], k[1], s) for k, s in scored.items()]
    ranked.sort(key=lambda t: t[2], reverse=True)
    return ranked


def match_taxonomy_node(raw_string: str) -> tuple[str, str, float]:
    """Fuzzy-match a raw line item to the closest predefined FS dictionary item.

    Returns (node_name, fs_statement, score). If no candidate meets
    MATCH_THRESHOLD, node_name falls back to the raw line item text as-is
    and fs_statement is "". (Standalone use only — does not enforce the
    one-node-per-item uniqueness that map_line_items applies across a batch.)
    """
    ranked = _ranked_matches(raw_string)
    if not ranked:
        return str(raw_string).strip(), "", 0.0
    return ranked[0]


def map_line_items(records: list) -> list:
    """Assign taxonomy_node (+ fs_statement, match_score) to every record via
    fuzzy matching only. No LLM/AI logic is involved.

    A single dictionary node can be allocated to at most one line item
    within the same scope (STANDALONE/CONSOLIDATED). Records are resolved
    in descending order of their best match score so the strongest matches
    claim their node first; any other record that would have mapped to an
    already-claimed node falls through to its next-best unclaimed candidate,
    or to its own raw line item text if none remain.
    """
    # (record_index, ranked candidate list)
    candidates_per_record = [(i, _ranked_matches(rec.get("raw_string", ""))) for i, rec in enumerate(records)]

    def _scope_of(rec: dict) -> str:
        return "CONSOLIDATED" if str(rec.get("scope", "")).upper() == "CONSOLIDATED" else "STANDALONE"

    # Process strongest overall matches first.
    order = sorted(
        range(len(records)),
        key=lambda i: (candidates_per_record[i][1][0][2] if candidates_per_record[i][1] else -1),
        reverse=True,
    )

    claimed: set[tuple[str, str, str]] = set()  # (scope, statement, item_name)

    for i in order:
        rec = records[i]
        scope = _scope_of(rec)
        ranked = candidates_per_record[i][1]

        chosen_item, chosen_statement, chosen_score = "", "", 0.0
        for item_name, statement, score in ranked:
            key = (scope, statement, item_name)
            if key not in claimed:
                chosen_item, chosen_statement, chosen_score = item_name, statement, score
                claimed.add(key)
                break

        if not chosen_item:
            chosen_item = str(rec.get("raw_string", "")).strip()
            chosen_statement = ""
            chosen_score = ranked[0][2] if ranked else 0.0

        rec["taxonomy_node"] = chosen_item
        rec["fs_statement"] = chosen_statement
        rec["match_score"] = round(chosen_score, 1)

    return records


def build_populated_dictionary(df) -> dict:
    """
    Builds a fresh scope/statement-segregated dictionary (see fs_dictionary.py)
    and fills in current_year/previous_year for every matched item, per scope,
    from the extraction dataframe `df` (columns: report_type, taxonomy_node,
    fs_statement, current_year, previous_year).
    """
    out = new_fs_dictionary()
    for _, row in df.iterrows():
        scope = "CONSOLIDATED" if str(row.get("report_type", "")).lower() == "consolidated" else "STANDALONE"
        statement = row.get("fs_statement", "")
        node = row.get("taxonomy_node", "")
        if statement and node and statement in out[scope] and node in out[scope][statement]:
            out[scope][statement][node]["current_year"] = row.get("current_year")
            out[scope][statement][node]["previous_year"] = row.get("previous_year")
    return out