"""
tools/ratio_tool.py
===================
Fetches live fundamental ratios for an Indian listed company from the
Financial Modeling Prep (FMP) API, finds live sector/market-cap peers
via FMP, and returns the company's ratios alongside a peer-median
benchmark — computed fresh every run, never from a static file.

FMP endpoints used:
    /api/v3/ratios/{ticker}?limit=1&apikey={key}
    /api/v4/stock_peers?symbol={ticker}&apikey={key}

Ticker format: RELIANCE.NS  (NSE-listed stocks)

Computed ratios (from Excel):
    revenue_growth_pct, pat_margin_pct, ebitda_margin_pct,
    operating_cf_to_pat, oci_to_pat_pct, exceptional_to_pbt_pct

FMP ratios (live, per company):
    pe_ratio, debt_to_equity, current_ratio, roe, roa,
    operating_profit_margin, net_profit_margin

Peer benchmark (live, computed per run):
    For each FMP ratio above, the median value across live peers
    returned by FMP's stock_peers endpoint. No hardcoded sector
    list, no static JSON file — if FMP can't resolve peers (plan
    tier, missing NSE coverage, etc.) this comes back empty and
    the caller is told explicitly via `peer_benchmark_status`.
"""

import json
import os
import statistics
import urllib.request
import urllib.error
from pathlib import Path

import pandas as pd
from langchain.tools import tool

FMP_BASE_V3 = "https://financialmodelingprep.com/api/v3"
FMP_BASE_V4 = "https://financialmodelingprep.com/api/v4"

# Max peers to query ratios for — keeps latency/quota sane, not a benchmark value
_MAX_PEERS = 6


def _fmp_get(url: str) -> dict | list | None:
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return {"fmp_error": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"fmp_error": str(e)}


def _safe(val, default=None):
    """Return val if it's a finite number, else default."""
    try:
        f = float(val)
        return round(f, 4) if f == f else default  # NaN check
    except (TypeError, ValueError):
        return default


def _fetch_company_ratios(ns_ticker: str, api_key: str) -> dict:
    """Fetch a single company's live FMP ratios. Returns {} on failure."""
    raw = _fmp_get(f"{FMP_BASE_V3}/ratios/{ns_ticker}?limit=1&apikey={api_key}")
    if not (isinstance(raw, list) and raw):
        return {}
    r = raw[0]
    return {
        "pe_ratio":                _safe(r.get("priceEarningsRatio")),
        "debt_to_equity":          _safe(r.get("debtEquityRatio")),
        "current_ratio":           _safe(r.get("currentRatio")),
        "roe":                     _safe(r.get("returnOnEquity")),
        "roa":                     _safe(r.get("returnOnAssets")),
        "operating_profit_margin": _safe(r.get("operatingProfitMargin")),
        "net_profit_margin":       _safe(r.get("netProfitMargin")),
    }


def _fetch_peer_tickers(ns_ticker: str, api_key: str) -> list[str]:
    """
    Calls FMP's stock_peers endpoint (same exchange, sector, market-cap band).
    Returns [] if unavailable — caller must treat that as "no peer data",
    not silently fall back to a fixed number.
    """
    raw = _fmp_get(f"{FMP_BASE_V4}/stock_peers?symbol={ns_ticker}&apikey={api_key}")
    if isinstance(raw, dict) and "fmp_error" in raw:
        return []
    if isinstance(raw, list) and raw:
        peers = raw[0].get("peersList", []) if isinstance(raw[0], dict) else []
        return [p for p in peers if isinstance(p, str)][:_MAX_PEERS]
    return []


def _peer_median_ratios(ns_ticker: str, api_key: str) -> tuple[dict, dict]:
    """
    Returns (peer_medians, meta) where peer_medians maps each FMP ratio
    name to the live median across resolved peers, and meta describes
    how the benchmark was obtained (for transparency to the caller).
    """
    peer_tickers = _fetch_peer_tickers(ns_ticker, api_key)
    meta = {"peer_tickers": peer_tickers, "peers_with_data": 0, "status": "ok"}

    if not peer_tickers:
        meta["status"] = "no_peers_found"
        return {}, meta

    per_peer_ratios = []
    for peer in peer_tickers:
        ratios = _fetch_company_ratios(peer, api_key)
        if ratios:
            per_peer_ratios.append(ratios)

    meta["peers_with_data"] = len(per_peer_ratios)
    if not per_peer_ratios:
        meta["status"] = "peers_found_but_no_ratio_data"
        return {}, meta

    medians = {}
    for key in ["pe_ratio", "debt_to_equity", "current_ratio", "roe",
                "roa", "operating_profit_margin", "net_profit_margin"]:
        vals = [p[key] for p in per_peer_ratios if p.get(key) is not None]
        if vals:
            medians[key] = round(statistics.median(vals), 4)

    return medians, meta


def _compute_ratios(sheets: dict) -> dict:
    """
    Derive key ratios from the extracted Excel sheets.
    Tries Standalone first, falls back to Consolidated.
    """
    ratios = {}

    # ── helper: find value by taxonomy_node in a sheet record list ───────────
    def find(records: list, node: str):
        for r in records:
            if str(r.get("taxonomy_node", "")).upper() == node:
                return _safe(r.get("current_year")), _safe(r.get("previous_year"))
        return None, None

    for prefix in ["Standalone", "Consolidated"]:
        pl  = sheets.get(f"{prefix} - P&L", [])
        cf  = sheets.get(f"{prefix} - Cash Flow", [])

        if not pl:
            continue

        rev_cur,  rev_prev  = find(pl, "REVENUE_FROM_OPERATIONS")
        pat_cur,  _         = find(pl, "PROFIT_FOR_THE_YEAR")
        pbt_cur,  _         = find(pl, "PROFIT_BEFORE_TAX")
        oci_cur,  _         = find(pl, "TOTAL_OCI")
        exc_cur,  _         = find(pl, "EXCEPTIONAL_ITEMS")
        cfo_cur,  _         = find(cf, "NET_CASH_FROM_OPERATING")

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

        ratios["segment"] = prefix
        break  # use first segment that has data

    return ratios


@tool
def ratio_tool(ticker: str, xlsx_path: str) -> str:
    """
    Fetches live FMP ratios for an NSE-listed stock, computes additional
    ratios from the extracted Excel file, and benchmarks the company's
    FMP ratios against the LIVE median of its FMP-resolved sector/market-cap
    peers (no static thresholds, no hardcoded peer list).

    Args:
        ticker:     NSE ticker symbol, e.g. "RELIANCE" (without .NS suffix).
        xlsx_path:  Path to the structured Excel file from Step2.

    Returns:
        JSON string with keys:
            status            — "success" | "error"
            ticker            — ticker used
            computed          — ratios derived from Excel data
            fmp               — company's own live ratios from FMP API
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
    api_key = os.environ.get("FMP_API_KEY")
    if not api_key:
        return json.dumps({"status": "error", "error": "FMP_API_KEY not set in environment."})

    xlsx_path = str(Path(xlsx_path).resolve())
    if not Path(xlsx_path).exists():
        return json.dumps({"status": "error", "error": f"Excel file not found: {xlsx_path}"})

    # ── Load Excel ────────────────────────────────────────────────────────────
    try:
        xf = pd.ExcelFile(xlsx_path)
        sheets = {}
        for sheet_name in xf.sheet_names:
            df = pd.read_excel(xf, sheet_name=sheet_name)
            df.columns = [str(c).strip().lower() for c in df.columns]
            core = ["line_item", "current_year", "previous_year", "taxonomy_node"]
            for col in core:
                if col not in df.columns:
                    df[col] = None
            df = df[core].dropna(subset=["line_item"])
            if not df.empty:
                sheets[sheet_name] = df.where(pd.notna(df), other=None).to_dict(orient="records")
    except Exception as exc:
        return json.dumps({"status": "error", "error": f"Excel read failed: {exc}"})

    # ── Computed ratios from Excel ────────────────────────────────────────────
    computed = _compute_ratios(sheets)

    # ── FMP live ratios for this company ─────────────────────────────────────
    ns_ticker = f"{ticker.upper()}.NS"
    fmp_ratios = _fetch_company_ratios(ns_ticker, api_key)
    if not fmp_ratios:
        # Could be a real API error — surface it instead of silently empty
        raw = _fmp_get(f"{FMP_BASE_V3}/ratios/{ns_ticker}?limit=1&apikey={api_key}")
        if isinstance(raw, dict) and "fmp_error" in raw:
            fmp_ratios = raw

    # ── Live peer benchmark (replaces static sector_medians.json) ───────────
    peer_benchmark, peer_meta = _peer_median_ratios(ns_ticker, api_key)

    return json.dumps({
        "status":              "success",
        "ticker":               ns_ticker,
        "computed":             computed,
        "fmp":                  fmp_ratios,
        "peer_benchmark":       peer_benchmark,
        "peer_benchmark_meta":  peer_meta,
    })