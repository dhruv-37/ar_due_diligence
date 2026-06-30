"""
agents/red_flag_agent.py
========================
Analyses financial ratios (computed from Excel + live FMP data) and
produces a structured list of red flags with severity ratings.

Benchmarking approach
----------------------
Wherever FMP can supply live peer data (via ratio_tool's stock_peers
lookup), flags are raised based on DEVIATION FROM THE LIVE PEER MEDIAN
— not a fixed number. This means the same ROE, say, can be flagged for
one company and not another, depending on what its actual sector peers
are doing *right now*. No static sector_medians.json, no hardcoded
sector-to-number mapping.

A small number of metrics (OCI/PAT, Exceptional items/PBT, Operating
CF/PAT) have no FMP peer equivalent — these are accounting-quality
signals about a SINGLE company's own statements, not something peers
inherently bound. For those, a fixed rule-of-thumb threshold is used
and explicitly labeled as such (see _FALLBACK_RULES below) so it's
never confused with a sector-aware benchmark.

Red flag categories
-------------------
1. Profitability   — PAT margin, revenue growth, FMP net profit margin vs peers
2. Quality         — Operating CF / PAT (earnings quality) — fixed rule-of-thumb
3. Leverage        — Debt/Equity, Interest Coverage vs peers
4. OCI             — OCI / PAT ratio (aggressive OCI usage signal) — fixed rule-of-thumb
5. Exceptional     — Exceptional items / PBT (one-time noise signal) — fixed rule-of-thumb
6. Valuation       — P/E, ROE, ROA from FMP vs peers

Severity levels
---------------
HIGH   — materially outside acceptable range; needs immediate attention
MEDIUM — warrants monitoring; may be sector-specific
LOW    — minor deviation; informational only
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

# ── Project root on sys.path ──────────────────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tools.ratio_tool import ratio_tool


def _call_ratio_tool(ticker: str, xlsx_path: str) -> dict:
    """
    Invokes ratio_tool directly — no LLM/agent layer. Supports a plain
    callable as well as a LangChain @tool-wrapped callable (which exposes
    .func / .invoke rather than being directly callable with kwargs).
    """
    if hasattr(ratio_tool, "func"):
        raw = ratio_tool.func(ticker=ticker, xlsx_path=xlsx_path)
    elif hasattr(ratio_tool, "invoke"):
        raw = ratio_tool.invoke({"ticker": ticker, "xlsx_path": xlsx_path})
    else:
        raw = ratio_tool(ticker=ticker, xlsx_path=xlsx_path)

    if isinstance(raw, str):
        raw = json.loads(raw)
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK RULES — only for metrics with no FMP peer equivalent.
# These are accounting-quality signals intrinsic to a company's own
# statements (cash backing of earnings, OCI/exceptional-item noise),
# not benchmarks that vary meaningfully by sector. Everything that
# CAN be peer-compared (margins, leverage, returns, valuation) is
# NOT in this dict — see detect_red_flags() for the peer-relative logic.
# ─────────────────────────────────────────────────────────────────────────────

_FALLBACK_RULES = {
    "operating_cf_to_pat":      {"low": 0.7, "high": 2.0},   # below 0.7 = poor quality
    "oci_to_pat_pct":           {"warn": 20},                 # above 20% = flag
    "exceptional_to_pbt_pct":   {"warn": 10},                 # above 10% = flag
}

# How far (as a fraction of the peer median) a value must deviate
# before it's worth flagging at all. This is a sensitivity knob for
# the rule engine, not a sector benchmark — it applies identically
# regardless of which sector or peer set is involved.
_PEER_DEVIATION_MEDIUM = 0.25   # 25% worse than peer median -> MEDIUM
_PEER_DEVIATION_HIGH   = 0.50   # 50% worse than peer median -> HIGH


# ─────────────────────────────────────────────────────────────────────────────
# RULE ENGINE  (pure Python, no LLM needed for flag detection)
# ─────────────────────────────────────────────────────────────────────────────

def _flag(metric: str, value: float, message: str, severity: str) -> dict:
    return {
        "metric":   metric,
        "value":    value,
        "message":  message,
        "severity": severity,
    }


def _peer_relative_flag(metric: str, value: float, peer_median: float,
                         higher_is_better: bool, label: str) -> dict | None:
    """
    Flags `value` if it deviates badly from `peer_median`, in the
    direction that matters for that metric (e.g. low ROE is bad,
    high Debt/Equity is bad).

    Returns None if within tolerance or peer_median is unusable (0).
    """
    if peer_median in (None, 0):
        return None

    deviation = (value - peer_median) / abs(peer_median)
    # Normalise so "bad" is always negative deviation for direction purposes
    bad_deviation = deviation if higher_is_better else -deviation

    if bad_deviation <= -_PEER_DEVIATION_HIGH:
        severity = "HIGH"
    elif bad_deviation <= -_PEER_DEVIATION_MEDIUM:
        severity = "MEDIUM"
    else:
        return None

    pct = abs(bad_deviation) * 100
    direction = "below" if higher_is_better else "above"
    return _flag(metric, value,
        f"{label} = {value:.2f} is {pct:.0f}% {direction} the live peer "
        f"median ({peer_median:.2f}).", severity)


def detect_red_flags(computed: dict, fmp: dict, peer_benchmark: dict,
                      peer_meta: dict) -> list[dict]:
    flags: list[dict] = []

    # ── Peer-relative flags (only meaningful if peer data resolved) ─────────
    peers_ok = peer_meta.get("status") == "ok" and peer_benchmark

    if peers_ok:
        peer_specs = [
            # (fmp key, higher_is_better, label)
            ("net_profit_margin",       True,  "Net profit margin"),
            ("operating_profit_margin", True,  "Operating profit margin"),
            ("roe",                     True,  "ROE"),
            ("roa",                     True,  "ROA"),
            ("current_ratio",           True,  "Current ratio"),
            ("quick_ratio",             True,  "Quick ratio"),
            ("debt_to_equity",          False, "Debt/Equity"),
            ("interest_coverage",       True,  "Interest coverage"),
            ("pe_ratio",                None,  "P/E ratio"),  # informational, no direction
        ]
        for key, higher_is_better, label in peer_specs:
            v = fmp.get(key)
            pm = peer_benchmark.get(key)
            if v is None or pm is None or higher_is_better is None:
                continue
            f = _peer_relative_flag(key, v, pm, higher_is_better, label)
            if f:
                flags.append(f)
    else:
        # Peer data unavailable this run — say so explicitly rather than
        # silently skipping or silently using a hardcoded number instead.
        flags.append(_flag("peer_benchmark", 0,
            f"Live peer benchmark unavailable this run (status: "
            f"{peer_meta.get('status', 'unknown')}). Leverage/return/"
            f"valuation ratios were NOT benchmarked against peers.", "LOW"))

    # ── Revenue Growth (no FMP peer equivalent for YoY growth; directional) ──
    v = computed.get("revenue_growth_pct")
    if v is not None:
        if v < 0:
            flags.append(_flag("revenue_growth_pct", v,
                f"Revenue declined {v:.1f}% YoY — contraction signal.", "HIGH"))
        elif v > 30:
            flags.append(_flag("revenue_growth_pct", v,
                f"Revenue grew {v:.1f}% — unusually high; verify organic vs inorganic.", "LOW"))

    # ── PAT Margin: prefer peer comparison via FMP net_profit_margin; the
    #    Excel-computed PAT margin itself has no live peer figure to compare
    #    against the company's *own* statements, so only flag extreme cases.
    v = computed.get("pat_margin_pct")
    if v is not None and v < 0:
        flags.append(_flag("pat_margin_pct", v,
            f"PAT margin is negative ({v:.1f}%) — the company posted a loss.", "HIGH"))

    # ── Earnings Quality (fallback rule — no FMP peer equivalent) ───────────
    v = computed.get("operating_cf_to_pat")
    fr = _FALLBACK_RULES["operating_cf_to_pat"]
    if v is not None:
        if v < fr["low"]:
            flags.append(_flag("operating_cf_to_pat", v,
                f"Operating CF / PAT = {v:.2f} — earnings not backed by cash flow "
                f"(fixed rule-of-thumb, not peer-benchmarked).", "HIGH"))
        elif v > fr["high"]:
            flags.append(_flag("operating_cf_to_pat", v,
                f"Operating CF / PAT = {v:.2f} — unusually high; check working capital "
                f"(fixed rule-of-thumb, not peer-benchmarked).", "LOW"))

    # ── OCI Noise (fallback rule — no FMP peer equivalent) ──────────────────
    v = computed.get("oci_to_pat_pct")
    fr = _FALLBACK_RULES["oci_to_pat_pct"]
    if v is not None and abs(v) > fr["warn"]:
        flags.append(_flag("oci_to_pat_pct", v,
            f"OCI is {v:.1f}% of PAT — aggressive OCI usage may mask true earnings "
            f"(fixed rule-of-thumb, not peer-benchmarked).", "MEDIUM"))

    # ── Exceptional Items (fallback rule — no FMP peer equivalent) ─────────
    v = computed.get("exceptional_to_pbt_pct")
    fr = _FALLBACK_RULES["exceptional_to_pbt_pct"]
    if v is not None and abs(v) > fr["warn"]:
        flags.append(_flag("exceptional_to_pbt_pct", v,
            f"Exceptional items = {v:.1f}% of PBT — one-time items distorting earnings "
            f"(fixed rule-of-thumb, not peer-benchmarked).", "MEDIUM"))

    # ── Liquidity (absolute, intrinsic — solvency risk regardless of peers) ──
    cr = computed.get("current_ratio")
    if cr is not None and cr < 1.0:
        flags.append(_flag("current_ratio", cr,
            f"Current ratio = {cr:.2f} (<1.0) — current liabilities exceed current "
            f"assets; short-term solvency risk.", "HIGH"))

    # ── Leverage (absolute, intrinsic — independent of sector norms) ────────
    de = computed.get("debt_to_equity")
    if de is not None and de > 2.0:
        flags.append(_flag("debt_to_equity", de,
            f"Debt/Equity = {de:.2f} (>2.0) — high absolute leverage; elevated "
            f"financial risk regardless of sector.", "HIGH"))

    ic = computed.get("interest_coverage")
    if ic is not None and ic < 1.5:
        flags.append(_flag("interest_coverage", ic,
            f"Interest coverage = {ic:.2f} (<1.5) — earnings barely cover interest "
            f"obligations; default risk signal.", "HIGH"))

    # ── Contradictory trends — internally inconsistent signals worth a flag ──
    rg  = computed.get("revenue_growth_pct")
    pm  = computed.get("pat_margin_pct")
    pat_growth = computed.get("pat_growth_pct")
    if rg is not None and pat_growth is not None and rg > 5 and pat_growth < 0:
        flags.append(_flag("revenue_vs_pat_growth", pat_growth,
            f"Revenue grew {rg:.1f}% YoY but PAT fell {abs(pat_growth):.1f}% — "
            f"margin compression or one-off costs eroding profitability.", "MEDIUM"))

    ocf_pat = computed.get("operating_cf_to_pat")
    if pm is not None and ocf_pat is not None and pm > 0 and ocf_pat < 0:
        flags.append(_flag("pat_vs_operating_cf", ocf_pat,
            f"Company reports a positive PAT margin ({pm:.1f}%) but negative "
            f"operating cash flow — possible earnings manipulation or aggressive "
            f"accrual accounting.", "HIGH"))

    roe = fmp.get("roe")
    roa = fmp.get("roa")
    if de is not None and roe is not None and roa is not None and de > 1.5 and roa < 5 and roe > 15:
        flags.append(_flag("roe_leverage_driven", roe,
            f"ROE ({roe:.1f}%) looks healthy but ROA ({roa:.1f}%) is weak alongside "
            f"high Debt/Equity ({de:.2f}) — returns are leverage-driven, not "
            f"operationally driven.", "MEDIUM"))

    return flags


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT  (deterministic — no LLM, no LangChain agent)
# ─────────────────────────────────────────────────────────────────────────────

def run_red_flag_agent(ticker: str, xlsx_path: str) -> dict:
    """
    Runs the Red Flag rule engine and returns a structured dict of flags.
    Calls ratio_tool directly (no LLM/agent layer involved) and applies
    the deterministic rule engine in detect_red_flags().

    Args:
        ticker:    NSE ticker without suffix, e.g. "RELIANCE"
        xlsx_path: Path to the Step2 Excel output

    Returns:
        dict with keys: ticker, segment, red_flags, ratios_used
    """
    tool_output = _call_ratio_tool(ticker, xlsx_path)

    computed             = tool_output.get("computed", {})
    fmp                  = tool_output.get("fmp", {})
    peer_benchmark       = tool_output.get("peer_benchmark", {})
    peer_benchmark_meta  = tool_output.get("peer_benchmark_meta", {})
    segment              = tool_output.get("segment", "Standalone")

    red_flags = detect_red_flags(computed, fmp, peer_benchmark, peer_benchmark_meta)

    return {
        "ticker": ticker,
        "segment": segment,
        "red_flags": red_flags,
        "ratios_used": {
            "computed": computed,
            "fmp": fmp,
            "peer_benchmark": peer_benchmark,
            "peer_benchmark_meta": peer_benchmark_meta,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("ticker",    help="NSE ticker, e.g. RELIANCE")
    parser.add_argument("xlsx_path", help="Path to Step2 Excel output")
    args = parser.parse_args()

    result = run_red_flag_agent(args.ticker, args.xlsx_path)
    print(json.dumps(result, indent=2))