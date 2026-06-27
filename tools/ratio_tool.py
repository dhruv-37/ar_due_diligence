"""
tools/ratio_tool.py
===================
Fetches live fundamental ratios for an Indian listed company from the
Financial Modeling Prep (FMP) API and returns them alongside ratios
computed from the extracted Excel data.

FMP endpoint used:
    /api/v3/ratios/{ticker}?limit=1&apikey={key}

Ticker format: RELIANCE.NS  (NSE-listed stocks)

Computed ratios (from Excel):
    revenue_growth_pct, pat_margin_pct, ebitda_margin_pct,
    operating_cf_to_pat, oci_to_pat_pct, exceptional_to_pbt_pct

FMP ratios (live):
    pe_ratio, debt_to_equity, current_ratio, roe, roa,
    operating_profit_margin, net_profit_margin
"""

import json
import os
import urllib.request
import urllib.error
from pathlib import Path

import pandas as pd
from langchain.tools import tool

FMP_BASE = "https://financialmodelingprep.com/api/v3"


def _fmp_get(endpoint: str, api_key: str) -> dict | list | None:
    url = f"{FMP_BASE}{endpoint}&apikey={api_key}"
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
    Fetches live FMP ratios for an NSE-listed stock and computes
    additional ratios from the extracted Excel file.

    Args:
        ticker:     NSE ticker symbol, e.g. "RELIANCE" (without .NS suffix).
        xlsx_path:  Path to the structured Excel file from Step2.

    Returns:
        JSON string with keys:
            status          — "success" | "error"
            ticker          — ticker used
            computed        — ratios derived from Excel data
            fmp             — ratios from FMP API
            error           — error message (only on failure)
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

    # ── FMP live ratios ───────────────────────────────────────────────────────
    ns_ticker = f"{ticker.upper()}.NS"
    raw = _fmp_get(f"/ratios/{ns_ticker}?limit=1", api_key)

    fmp_ratios = {}
    if isinstance(raw, list) and raw:
        r = raw[0]
        fmp_ratios = {
            "pe_ratio":               _safe(r.get("priceEarningsRatio")),
            "debt_to_equity":         _safe(r.get("debtEquityRatio")),
            "current_ratio":          _safe(r.get("currentRatio")),
            "roe":                    _safe(r.get("returnOnEquity")),
            "roa":                    _safe(r.get("returnOnAssets")),
            "operating_profit_margin":_safe(r.get("operatingProfitMargin")),
            "net_profit_margin":      _safe(r.get("netProfitMargin")),
        }
    elif isinstance(raw, dict) and "fmp_error" in raw:
        fmp_ratios = raw  # pass error through, don't fail entire tool

    return json.dumps({
        "status":   "success",
        "ticker":   ns_ticker,
        "computed": computed,
        "fmp":      fmp_ratios,
    })