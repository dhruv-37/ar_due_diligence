"""
tools/ratio_tool.py
===================
Fetches live fundamental ratios for an Indian listed company from Yahoo
Finance (via yfinance — free, no API key required), finds sector peers
from a curated NSE peer-ticker map, and returns the company's ratios
alongside a peer-median benchmark computed fresh every run from LIVE
per-peer data (only the peer *ticker membership* is curated — every
ratio value, including the peer median, is fetched live, never hardcoded).

Switched from Financial Modeling Prep (FMP) because FMP's free/low tiers
return HTTP 403 on the /ratios and /stock_peers endpoints for NSE tickers.

Ticker format: RELIANCE.NS  (NSE-listed stocks)

Computed ratios (from Excel):
    revenue_growth_pct, pat_margin_pct, ebitda_margin_pct,
    operating_cf_to_pat, oci_to_pat_pct, exceptional_to_pbt_pct

Live ratios (per company, via yfinance):
    pe_ratio, debt_to_equity, current_ratio, quick_ratio, roe, roa,
    operating_profit_margin, net_profit_margin

Peer benchmark (live, computed per run):
    For each ratio above, the median value across live peers resolved
    from _SECTOR_PEER_TICKERS (keyed by yfinance's reported sector/
    industry string). If the company's sector isn't in the map, or none
    of the curated peers return usable data, this comes back empty and
    the caller is told explicitly via peer_benchmark_meta.status.
"""

import json
import re
import statistics
from pathlib import Path

import pandas as pd
import yfinance as yf
from langchain.tools import tool
from tools.excel_reader_tool import excel_reader_tool

# Max peers to query ratios for — keeps latency sane, not a benchmark value
_MAX_PEERS = 6

# Curated NSE peer-ticker sets keyed by a lowercase substring of yfinance's
# 'sector' / 'industry' field. Only TICKER MEMBERSHIP is hardcoded here —
# every ratio for every peer is still fetched live via yfinance below, so
# this is not a static sector_medians.json equivalent.
_SECTOR_PEER_TICKERS: dict[str, list[str]] = {
    "energy":          ["ONGC.NS", "BPCL.NS", "IOC.NS", "HINDPETRO.NS", "GAIL.NS"],
    "information technology": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    "technology":      ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    "financial":       ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"],
    "bank":            ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"],
    "consumer defensive": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS"],
    "household":       ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS"],
    "auto":            ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS"],
    "pharma":          ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "LUPIN.NS"],
    "drug":            ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "LUPIN.NS"],
    "steel":           ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "SAIL.NS"],
    "metal":           ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "SAIL.NS"],
    "telecom":         ["BHARTIARTL.NS", "IDEA.NS"],
}


def _safe(val, default=None):
    """Return val if it's a finite number, else default."""
    try:
        f = float(val)
        return round(f, 4) if f == f else default  # NaN check
    except (TypeError, ValueError):
        return default


def _fetch_company_ratios(ns_ticker: str) -> dict:
    """
    Fetch a single company's live ratios via yfinance. Returns {} on
    failure (delisted ticker, network error, no fundamentals available).
    """
    try:
        info = yf.Ticker(ns_ticker).info
    except Exception:
        return {}

    if not info or info.get("trailingPE") is None and info.get("regularMarketPrice") is None:
        return {}

    # yfinance reports debtToEquity as a percentage (e.g. 41.2 == 0.412
    # ratio) — normalise to a plain ratio to match the rest of the codebase
    # (e.g. red_flag_agent's absolute D/E > 2.0 check).
    de_raw = _safe(info.get("debtToEquity"))
    debt_to_equity = round(de_raw / 100, 4) if de_raw is not None else None

    return {
        "pe_ratio":                _safe(info.get("trailingPE")),
        "debt_to_equity":          debt_to_equity,
        "current_ratio":           _safe(info.get("currentRatio")),
        "quick_ratio":             _safe(info.get("quickRatio")),
        "roe":                     _safe(info.get("returnOnEquity")),
        "roa":                     _safe(info.get("returnOnAssets")),
        "operating_profit_margin": _safe(info.get("operatingMargins")),
        "net_profit_margin":       _safe(info.get("profitMargins")),
    }


def _fetch_peer_tickers(ns_ticker: str) -> list[str]:
    """
    Resolves sector/market-cap peers via the curated _SECTOR_PEER_TICKERS
    map, keyed by yfinance's reported sector/industry for this ticker.
    Returns [] if the sector can't be resolved or isn't mapped — caller
    must treat that as "no peer data", not silently use a fixed number.
    """
    try:
        info = yf.Ticker(ns_ticker).info
    except Exception:
        return []

    haystack = f"{info.get('sector', '')} {info.get('industry', '')}".lower()
    for key, peers in _SECTOR_PEER_TICKERS.items():
        if key in haystack:
            return [p for p in peers if p != ns_ticker][:_MAX_PEERS]
    return []


def _peer_median_ratios(ns_ticker: str) -> tuple[dict, dict]:
    """
    Returns (peer_medians, meta) where peer_medians maps each ratio name
    to the live median across resolved peers, and meta describes how the
    benchmark was obtained (for transparency to the caller).
    """
    peer_tickers = _fetch_peer_tickers(ns_ticker)
    meta = {"peer_tickers": peer_tickers, "peers_with_data": 0, "status": "ok"}

    if not peer_tickers:
        meta["status"] = "no_peers_found"
        return {}, meta

    per_peer_ratios = []
    for peer in peer_tickers:
        ratios = _fetch_company_ratios(peer)
        if ratios:
            per_peer_ratios.append(ratios)

    meta["peers_with_data"] = len(per_peer_ratios)
    if not per_peer_ratios:
        meta["status"] = "peers_found_but_no_ratio_data"
        return {}, meta

    medians = {}
    for key in ["pe_ratio", "debt_to_equity", "current_ratio", "quick_ratio",
                "roe", "roa", "operating_profit_margin", "net_profit_margin"]:
        vals = [p[key] for p in per_peer_ratios if p.get(key) is not None]
        if vals:
            medians[key] = round(statistics.median(vals), 4)

    return medians, meta



# ── Line-item alias matching (replaces taxonomy_node, which Step2's Excel
#    does not produce) ────────────────────────────────────────────────────
def _normalize_line_item(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace for fuzzy matching."""
    s = str(text or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Each metric maps to a list of normalized alias phrases (longest/most
# specific first where overlap is possible).
_LINE_ITEM_ALIASES = {
    "REVENUE": [
        "revenue from operations", "value of sales", "net sales", "revenue",
    ],
    "PROFIT_BEFORE_TAX": [
        "profit before tax and exceptional items", "profit before tax", "pbt",
    ],
    "PROFIT_AFTER_TAX": [
        "profit for the year", "profit after tax", "net profit", "pat",
    ],
    "TOTAL_OCI": [
        "total other comprehensive income", "other comprehensive income", "oci",
    ],
    "EXCEPTIONAL_ITEMS": [
        "exceptional items", "exceptional item",
    ],
    "OPERATING_CASH_FLOW": [
        "net cash from operating activities",
        "net cash generated from operating activities",
        "cash flow from operating activities",
        "net cash flow from operating activities",
    ],
}


def _find_by_alias(records: list, metric: str):
    """
    Scans a sheet's records for a line_item matching an alias of `metric`.
    Returns (current_year, previous_year) as floats, or (None, None) if
    no usable match is found — never raises, so missing metrics are
    skipped gracefully.

    Matching strategy (fixes two real-data issues found in testing):
    1. Iterate ALIASES first, records second — so a more specific alias
       (e.g. "revenue from operations") is preferred over a less specific
       one (e.g. "revenue") even if the less-specific row appears earlier
       in the sheet (e.g. "Value of Sales" appearing before "Revenue from
       Operations").
    2. Skip rows whose current_year is not a usable number — Indian ARs
       often have section-header rows (e.g. "Other Comprehensive Income")
       with blank values immediately above the real total row (e.g.
       "Total Other Comprehensive Income ... (Net of Tax)"); these blank
       headers must not shadow the real total.
    """
    aliases = _LINE_ITEM_ALIASES.get(metric, [])

    # Pass 1: exact normalized match, alias-priority order, value required
    for alias in aliases:
        for r in records:
            if _normalize_line_item(r.get("line_item")) == alias:
                cur = _safe(r.get("current_year"))
                if cur is not None:
                    return cur, _safe(r.get("previous_year"))

    # Pass 2: substring match, alias-priority order, value required
    for alias in aliases:
        for r in records:
            if alias in _normalize_line_item(r.get("line_item")):
                cur = _safe(r.get("current_year"))
                if cur is not None:
                    return cur, _safe(r.get("previous_year"))

    return None, None


def _compute_ratios(sheets: dict) -> dict:
    """
    Derive key ratios from the extracted Excel sheets using normalized
    line_item alias matching (no taxonomy_node dependency).
    Tries Standalone first, falls back to Consolidated.
    """
    ratios = {}

    if "standalone_pnl" in sheets:
        pl = sheets["standalone_pnl"]
        cf = sheets.get("standalone_cash_flow", [])
        prefix = "Standalone"
    elif "consolidated_pnl" in sheets:
        pl = sheets["consolidated_pnl"]
        cf = sheets.get("consolidated_cash_flow", [])
        prefix = "Consolidated"
    else:
        return ratios

    ratios["segment"] = prefix
    if not pl:
        return ratios

    # ── Alias-based lookups (replaces taxonomy_node lookups) ────────────────
    rev_cur,  rev_prev = _find_by_alias(pl, "REVENUE")
    pat_cur,  _         = _find_by_alias(pl, "PROFIT_AFTER_TAX")
    pbt_cur,  _         = _find_by_alias(pl, "PROFIT_BEFORE_TAX")
    oci_cur,  _         = _find_by_alias(pl, "TOTAL_OCI")
    exc_cur,  _         = _find_by_alias(pl, "EXCEPTIONAL_ITEMS")
    cfo_cur,  _         = _find_by_alias(cf, "OPERATING_CASH_FLOW")

    if rev_cur and rev_prev and rev_prev != 0:
        ratios["revenue_growth_pct"] = round((rev_cur - rev_prev) / abs(rev_prev) * 100, 2)

    if pat_cur and rev_cur and rev_cur != 0:
        ratios["pat_margin_pct"] = round(pat_cur / rev_cur * 100, 2)

    if pat_cur and cfo_cur and pat_cur != 0:
        ratios["operating_cf_to_pat"] = round(cfo_cur / pat_cur, 2)

    if oci_cur and pat_cur and pat_cur != 0:
        ratios["oci_to_pat_pct"] = round(oci_cur / pat_cur * 100, 2)

    if exc_cur and pbt_cur and pbt_cur != 0:
        ratios["exceptional_to_pbt_pct"] = round(exc_cur / pbt_cur * 100, 2)

    return ratios


@tool
def ratio_tool(ticker: str, xlsx_path: str) -> str:
    """
    Fetches live ratios for an NSE-listed stock via yfinance, computes
    additional ratios from the extracted Excel file, and benchmarks the
    company's ratios against the LIVE median of its sector peers (peer
    ticker membership is curated; every ratio value is fetched live).

    Args:
        ticker:     NSE ticker symbol, e.g. "RELIANCE" (without .NS suffix).
        xlsx_path:  Path to the structured Excel file from Step2.

    Returns:
        JSON string with keys:
            status            — "success" | "error"
            ticker            — ticker used
            computed          — ratios derived from Excel data
            fmp               — company's own live ratios (sourced from
                                 yfinance; key name kept as "fmp" for
                                 backward compatibility with callers)
            peer_benchmark    — median of live peer ratios (may be {})
            peer_benchmark_meta — how the benchmark was derived: which
                                   peer tickers were used, how many had
                                   usable data, and a status flag the
                                   caller MUST check before treating
                                   peer_benchmark as reliable. Status is
                                   one of: "ok", "no_peers_found",
                                   "peers_found_but_no_ratio_data".
            error             — error message (only on failure)
    """
    xlsx_path = str(Path(xlsx_path).resolve())
    if not Path(xlsx_path).exists():
        return json.dumps({"status": "error", "error": f"Excel file not found: {xlsx_path}"})

    # ── Load Excel ────────────────────────────────────────────────────────────
    try:
        result = json.loads(excel_reader_tool.invoke({"xlsx_path": xlsx_path}))

        print("\n===== EXCEL READER RESULT =====")
        print(result)

        if result["status"] != "success":
            return json.dumps(result)

        sheets = result["sheets"]

    except Exception as exc:
        return json.dumps({"status": "error", "error": f"Excel read failed: {exc}"})
    # ── Computed ratios from Excel ────────────────────────────────────────────
    computed = _compute_ratios(sheets)

    # ── Live ratios for this company (via yfinance) ──────────────────────────
    ns_ticker = f"{ticker.upper()}.NS"
    live_ratios = _fetch_company_ratios(ns_ticker)
    if not live_ratios:
        live_ratios = {"fetch_error": f"No usable yfinance data for {ns_ticker}."}

    # ── Live peer benchmark ──────────────────────────────────────────────────
    peer_benchmark, peer_meta = _peer_median_ratios(ns_ticker)

    return json.dumps({
        "status":              "success",
        "ticker":               ns_ticker,
        "computed":             computed,
        "fmp":                  live_ratios,
        "peer_benchmark":       peer_benchmark,
        "peer_benchmark_meta":  peer_meta,
    })