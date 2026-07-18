"""
forensic_diagnostic_engine.py
==============================
Phase 2 — ForensicDiagnosticEngine

Ingests parsed financial JSON (line_items + fs_dictionary, STANDALONE /
CONSOLIDATED) and runs data-integrity, double-entry, and forensic
red-flag diagnostics, producing a structured JSON report + Markdown
summary.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger("forensic_diagnostic_engine")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

TOLERANCE = 1_000.0  # rounding slack for thousands/millions/crore reporting

_SCOPES = ("STANDALONE", "CONSOLIDATED")

# fs_dictionary metric-name aliases the engine looks for (best-effort — the
# dictionary is free-form, so multiple spellings are tried per concept)
_METRIC_ALIASES: dict[str, list[str]] = {
    "revenue": ["Revenue from Operations", "Revenue", "Total Revenue", "Turnover"],
    "profit_for_the_year": ["Profit for the Year", "Net Income", "Net Profit", "Profit After Tax"],
    "total_assets": ["Total Assets"],
    "total_liabilities": ["Total Liabilities"],
    "total_equity": ["Total Equity", "Shareholders' Equity", "Total Stockholders' Equity"],
    "operating_cash_flow": ["Net Cash From Operating Activities", "Net Cash Flow From Operations"],
    "investing_cash_flow": ["Net Cash From Investing Activities", "Net Cash Flow From Investing"],
    "financing_cash_flow": ["Net Cash From Financing Activities", "Net Cash Flow From Financing"],
    "cash_opening": ["Cash and Cash Equivalents at the Beginning of the Year", "Opening Cash Balance"],
    "cash_closing": ["Cash and Cash Equivalents at the End of the Year", "Closing Cash Balance"],
    "dividends_paid": ["Dividends Paid", "Dividend Paid"],
}


def _norm(s: Any) -> str:
    return " ".join(str(s).strip().lower().split())


def _f(value: Any) -> Optional[float]:
    """Best-effort numeric coercion; returns None on failure."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").replace("₹", "").replace("`", "").strip()
        neg = cleaned.startswith("(") and cleaned.endswith(")")
        cleaned = cleaned.strip("()")
        if not cleaned:
            return None
        try:
            num = float(cleaned)
            return -num if neg else num
        except ValueError:
            return None
    return None


class ForensicDiagnosticEngine:
    """
    Phase 2 diagnostic engine. Construct with the parsed financial JSON
    (line_items + fs_dictionary), then call `run_full_diagnostics()` to
    execute all three pipelines and produce the structured report.
    """

    def __init__(self, financial_data: dict[str, Any]):
        self.raw = financial_data
        self.line_items: list[dict[str, Any]] = financial_data.get("line_items", []) or []
        self.fs_dictionary: dict[str, Any] = financial_data.get("fs_dictionary", {}) or {}

        self.taxonomy_gaps: list[dict[str, Any]] = []
        self.mathematical_discrepancies: list[dict[str, Any]] = []
        self.forensic_red_flags: list[dict[str, Any]] = []

    # ─────────────────────────────────────────────────────────────────
    # METRIC LOOKUP HELPERS
    # ─────────────────────────────────────────────────────────────────

    def _scope_dict(self, scope: str) -> dict[str, Any]:
        """
        fs_dictionary can be keyed either as {scope: {metric: {...}}} or
        {metric: {scope: {...}}} — try both shapes tolerantly.
        """
        direct = self.fs_dictionary.get(scope) or self.fs_dictionary.get(scope.upper()) or {}
        if direct:
            return direct
        # inverted shape: {metric: {"STANDALONE": {...}, "CONSOLIDATED": {...}}}
        inverted: dict[str, Any] = {}
        for metric_name, scopes in self.fs_dictionary.items():
            if isinstance(scopes, dict) and scope in scopes:
                inverted[metric_name] = scopes[scope]
        return inverted

    def _get_metric(self, scope: str, canonical: str, period: str = "current_year") -> Optional[float]:
        """
        Look up a canonical metric (e.g. 'revenue') in the given scope's
        fs_dictionary slice, trying each known alias, case/whitespace
        insensitive, for the requested period ('current_year' / 'previous_year').
        """
        scope_dict = self._scope_dict(scope)
        normalized_lookup = {_norm(k): v for k, v in scope_dict.items()}
        for alias in _METRIC_ALIASES.get(canonical, [canonical]):
            key = _norm(alias)
            if key in normalized_lookup:
                entry = normalized_lookup[key]
                if isinstance(entry, dict):
                    val = _f(entry.get(period))
                else:
                    val = _f(entry)
                if val is not None:
                    return val
        return None

    # ─────────────────────────────────────────────────────────────────
    # A. DATA INTEGRITY & TAXONOMY DIAGNOSTICS
    # ─────────────────────────────────────────────────────────────────

    def validate_taxonomy(self) -> list[dict[str, Any]]:
        gaps: list[dict[str, Any]] = []

        total_assets = max(
            (self._get_metric(s, "total_assets") or 0.0) for s in _SCOPES
        ) or None
        revenue = max(
            (self._get_metric(s, "revenue") or 0.0) for s in _SCOPES
        ) or None
        denom = total_assets or revenue

        for item in self.line_items:
            match_score = _f(item.get("match_score"))
            fs_statement = item.get("fs_statement")
            label = item.get("raw_line_item", item.get("line_item", "UNKNOWN"))
            value = _f(item.get("value", item.get("current_year")))

            unmapped = (match_score == 0.0) or (fs_statement in (None, "", "UNMAPPED"))
            if not unmapped:
                continue

            entry: dict[str, Any] = {
                "line_item": label,
                "match_score": match_score,
                "fs_statement": fs_statement,
                "value": value,
                "high_value": False,
            }

            if value is not None and denom:
                pct_of_base = abs(value) / abs(denom) if denom else 0.0
                entry["pct_of_base"] = round(pct_of_base * 100, 2)
                if pct_of_base > 0.05:
                    entry["high_value"] = True
                    entry["basis"] = "Total Assets" if total_assets else "Revenue"

            gaps.append(entry)

        self.taxonomy_gaps = gaps
        return gaps

    # ─────────────────────────────────────────────────────────────────
    # B. MATHEMATICAL & DOUBLE-ENTRY VERIFICATION
    # ─────────────────────────────────────────────────────────────────

    def verify_accounting_math(self) -> list[dict[str, Any]]:
        discrepancies: list[dict[str, Any]] = []

        for scope in _SCOPES:
            for period, period_label in (("current_year", "t"), ("previous_year", "t_minus_1")):
                discrepancies.extend(self._check_balance_sheet(scope, period, period_label))
                discrepancies.extend(self._check_cash_flow_reconciliation(scope, period, period_label))

        discrepancies.extend(self._check_consolidated_vs_standalone())

        self.mathematical_discrepancies = discrepancies
        return discrepancies

    def _check_balance_sheet(self, scope: str, period: str, period_label: str) -> list[dict[str, Any]]:
        out = []
        assets = self._get_metric(scope, "total_assets", period)
        liabilities = self._get_metric(scope, "total_liabilities", period)
        equity = self._get_metric(scope, "total_equity", period)

        if assets is None or liabilities is None or equity is None:
            return out

        diff = assets - (liabilities + equity)
        if abs(diff) > TOLERANCE:
            out.append({
                "check": "balance_sheet_identity",
                "scope": scope,
                "period": period_label,
                "description": (
                    f"[{scope} {period_label}] Total Assets ({assets:,.2f}) does not equal "
                    f"Total Liabilities + Equity ({liabilities + equity:,.2f})"
                ),
                "variance": round(diff, 2),
            })
        return out

    def _check_cash_flow_reconciliation(self, scope: str, period: str, period_label: str) -> list[dict[str, Any]]:
        out = []
        ocf = self._get_metric(scope, "operating_cash_flow", period)
        icf = self._get_metric(scope, "investing_cash_flow", period)
        fcf = self._get_metric(scope, "financing_cash_flow", period)
        opening = self._get_metric(scope, "cash_opening", period)
        closing = self._get_metric(scope, "cash_closing", period)

        if None in (ocf, icf, fcf, opening, closing):
            return out

        computed_net_change = ocf + icf + fcf
        actual_net_change = closing - opening
        diff = computed_net_change - actual_net_change

        if abs(diff) > TOLERANCE:
            out.append({
                "check": "cash_flow_reconciliation",
                "scope": scope,
                "period": period_label,
                "description": (
                    f"[{scope} {period_label}] Operating+Investing+Financing cash flows "
                    f"({computed_net_change:,.2f}) do not reconcile with the opening-to-closing "
                    f"cash delta ({actual_net_change:,.2f})"
                ),
                "variance": round(diff, 2),
            })
        return out

    def _check_consolidated_vs_standalone(self) -> list[dict[str, Any]]:
        out = []
        for canonical, label in (("revenue", "Revenue"), ("profit_for_the_year", "Profit for the Year")):
            for period, period_label in (("current_year", "t"), ("previous_year", "t_minus_1")):
                standalone_val = self._get_metric("STANDALONE", canonical, period)
                consolidated_val = self._get_metric("CONSOLIDATED", canonical, period)
                if standalone_val is None or consolidated_val is None:
                    continue
                variance = standalone_val - consolidated_val
                if variance > TOLERANCE:
                    out.append({
                        "check": "consolidated_vs_standalone_delta",
                        "metric": label,
                        "period": period_label,
                        "description": (
                            f"[{period_label}] Standalone {label} ({standalone_val:,.2f}) exceeds "
                            f"Consolidated {label} ({consolidated_val:,.2f}) without apparent "
                            f"sub-entity elimination — investigate intercompany adjustments."
                        ),
                        "variance": round(variance, 2),
                    })
        return out

    # ─────────────────────────────────────────────────────────────────
    # C. FORENSIC ACCOUNTING RED FLAGS
    # ─────────────────────────────────────────────────────────────────

    def detect_red_flags(self) -> list[dict[str, Any]]:
        flags: list[dict[str, Any]] = []
        for scope in _SCOPES:
            flags.extend(self._check_dividend_sustainability(scope))
            flags.extend(self._check_cash_flow_quality(scope))
        flags.extend(self._check_asset_concentration())

        self.forensic_red_flags = flags
        return flags

    def _check_dividend_sustainability(self, scope: str) -> list[dict[str, Any]]:
        out = []
        dividends = self._get_metric(scope, "dividends_paid")
        profit = self._get_metric(scope, "profit_for_the_year")
        if dividends is None or profit is None:
            return out
        dividends = abs(dividends)
        if dividends > profit:
            out.append({
                "risk_level": "CRITICAL",
                "metric": f"{scope} Dividend Sustainability",
                "description": (
                    f"Dividends Paid ({dividends:,.2f}) exceed Profit for the Year "
                    f"({profit:,.2f}) — dividend is not covered by current-year earnings."
                ),
                "variance": round(dividends - profit, 2),
            })
        return out

    def _check_cash_flow_quality(self, scope: str) -> list[dict[str, Any]]:
        out = []
        ocf = self._get_metric(scope, "operating_cash_flow")
        profit = self._get_metric(scope, "profit_for_the_year")
        if ocf is None or profit is None:
            return out

        if profit > 0 and ocf <= 0:
            out.append({
                "risk_level": "CRITICAL",
                "metric": f"{scope} Cash Flow Quality",
                "description": (
                    f"Profit for the Year is positive ({profit:,.2f}) but Net Cash From "
                    f"Operating Activities is negative or zero ({ocf:,.2f}) — possible "
                    f"earnings-quality risk (non-cash or aggressive revenue recognition)."
                ),
                "variance": round(ocf - profit, 2),
            })
        elif profit > 0:
            ratio = ocf / profit
            if ratio < 0.2:
                out.append({
                    "risk_level": "WARNING",
                    "metric": f"{scope} Cash Flow Quality",
                    "description": (
                        f"Operating Cash Flow to Profit ratio is low ({ratio:.2f}) — OCF "
                        f"({ocf:,.2f}) covers only {ratio*100:.1f}% of Profit ({profit:,.2f})."
                    ),
                    "variance": round(ratio, 4),
                })
            else:
                out.append({
                    "risk_level": "INFO",
                    "metric": f"{scope} Cash Flow Quality",
                    "description": f"OCF/Profit ratio healthy at {ratio:.2f}.",
                    "variance": round(ratio, 4),
                })
        return out

    def _check_asset_concentration(self) -> list[dict[str, Any]]:
        out = []
        total_assets = max(
            (v for v in (self._get_metric(s, "total_assets") for s in _SCOPES) if v), default=None
        )
        if not total_assets:
            return out

        for item in self.line_items:
            fs_statement = item.get("fs_statement")
            if fs_statement not in (None, "", "UNMAPPED"):
                continue  # only unclassified items are in scope for this check
            value = _f(item.get("value", item.get("current_year")))
            label = item.get("raw_line_item", item.get("line_item", "UNKNOWN"))
            if value is None:
                continue
            pct = abs(value) / abs(total_assets)
            if pct > 0.15:
                out.append({
                    "risk_level": "WARNING",
                    "metric": "Asset Concentration",
                    "description": (
                        f"Unclassified line item '{label}' ({value:,.2f}) represents "
                        f"{pct*100:.1f}% of Total Assets ({total_assets:,.2f}) — "
                        f"concentration risk in an unmapped asset."
                    ),
                    "variance": round(pct * 100, 2),
                })
        return out

    # ─────────────────────────────────────────────────────────────────
    # SCORING & REPORT ASSEMBLY
    # ─────────────────────────────────────────────────────────────────

    def _compute_integrity_score(self) -> float:
        """
        Scale 0-100. Starts at 100 and deducts points per flagged issue:
        - CRITICAL red flag or mathematical discrepancy: -15
        - WARNING red flag / high-value taxonomy gap: -7
        - INFO red flag / ordinary taxonomy gap: -2
        Floors at 0.
        """
        score = 100.0
        score -= 15.0 * len(self.mathematical_discrepancies)

        for flag in self.forensic_red_flags:
            level = flag.get("risk_level")
            if level == "CRITICAL":
                score -= 15.0
            elif level == "WARNING":
                score -= 7.0
            else:
                score -= 2.0

        for gap in self.taxonomy_gaps:
            if gap.get("high_value"):
                score -= 7.0
            # low-value unmatched items are disclosed in the Markdown report
            # but don't penalize the score — they're completeness noise,
            # not a financial concern, unless they're material (>5% of base)

        return max(0.0, round(score, 2))

    def run_full_diagnostics(self) -> dict[str, Any]:
        """Run all three pipelines and assemble the structured report."""
        logger.info("Running taxonomy validation...")
        self.validate_taxonomy()
        logger.info("Running accounting math verification...")
        self.verify_accounting_math()
        logger.info("Running forensic red-flag detection...")
        self.detect_red_flags()

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "integrity_score": self._compute_integrity_score(),
            "taxonomy_gaps": self.taxonomy_gaps,
            "mathematical_discrepancies": self.mathematical_discrepancies,
            "forensic_red_flags": self.forensic_red_flags,
        }
        logger.info("Diagnostics complete. Integrity score: %.2f", report["integrity_score"])
        return report

    # ─────────────────────────────────────────────────────────────────
    # MARKDOWN SUMMARY
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def to_markdown(report: dict[str, Any]) -> str:
        lines = []
        lines.append(f"# Forensic Diagnostic Report")
        lines.append(f"_Generated: {report['timestamp']}_")
        lines.append(f"\n**Integrity Score: {report['integrity_score']} / 100**\n")

        lines.append(f"## Taxonomy Gaps ({len(report['taxonomy_gaps'])})")
        if report["taxonomy_gaps"]:
            high_value = [g for g in report["taxonomy_gaps"] if g.get("high_value")]
            low_value = [g for g in report["taxonomy_gaps"] if not g.get("high_value")]
            lines.append(
                f"- {len(high_value)} high-value (>5% of base) — penalize the score, "
                f"{len(low_value)} low-value — disclosed below, do not penalize the score.\n"
            )
            for g in report["taxonomy_gaps"]:
                flag = " 🔴 HIGH-VALUE" if g.get("high_value") else ""
                lines.append(f"- `{g['line_item']}` — match_score={g.get('match_score')}, "
                             f"fs_statement={g.get('fs_statement')!r}{flag}")
        else:
            lines.append("- None")

        lines.append(f"\n## Mathematical Discrepancies ({len(report['mathematical_discrepancies'])})")
        if report["mathematical_discrepancies"]:
            for d in report["mathematical_discrepancies"]:
                lines.append(f"- **{d['check']}**: {d['description']} (variance: {d['variance']})")
        else:
            lines.append("- None")

        lines.append(f"\n## Forensic Red Flags ({len(report['forensic_red_flags'])})")
        if report["forensic_red_flags"]:
            icon = {"CRITICAL": "🔴", "WARNING": "🟠", "INFO": "🔵"}
            for f in report["forensic_red_flags"]:
                lines.append(
                    f"- {icon.get(f['risk_level'], '')} **[{f['risk_level']}] {f['metric']}**: "
                    f"{f['description']} (variance: {f['variance']})"
                )
        else:
            lines.append("- None")

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
# PIPELINE ADAPTER — Step2 taxonomy JSON + normalize_financials output
# ─────────────────────────────────────────────────────────────────────────

def build_diagnostic_input_from_pipeline(
    taxonomy_json: dict[str, Any],
    normalized_standalone: dict[str, Any],
    normalized_consolidated: dict[str, Any],
) -> dict[str, Any]:
    """
    Combine Step2's taxonomy JSON (line_items, both scopes) with two
    normalize_financials() outputs (one per scope) into the
    {"line_items": [...], "fs_dictionary": {"STANDALONE": {...}, "CONSOLIDATED": {...}}}
    shape ForensicDiagnosticEngine expects.
    """
    line_items = []
    for rec in taxonomy_json.get("line_items", []):
        line_items.append({
            "raw_line_item": rec.get("line_item"),
            "value": rec.get("current_year"),
            "fs_statement": rec.get("fs_statement") or rec.get("statement"),
            "match_score": rec.get("match_score"),
            "report_type": rec.get("report_type"),
        })

    def _metric_dict(normalized: dict[str, Any]) -> dict[str, Any]:
        inc = normalized.get("income_statement", {})
        bs = normalized.get("balance_sheet", {})
        cf = normalized.get("cash_flow_statement", {})
        return {
            "Revenue from Operations": inc.get("revenue", {}),
            "Profit for the Year": inc.get("net_income_continuing", {}),
            "Total Assets": bs.get("total_assets", {}),
            "Total Liabilities": bs.get("total_liabilities", {}),
            "Total Equity": bs.get("total_equity", {}),
            "Net Cash From Operating Activities": cf.get("operating_cash_flow", {}),
        }

    return {
        "line_items": line_items,
        "fs_dictionary": {
            "STANDALONE": _metric_dict(normalized_standalone),
            "CONSOLIDATED": _metric_dict(normalized_consolidated),
        },
    }


def run_full_pipeline(
    pdf_path: str,
    ticker: str,
    fiscal_year_t: Optional[int] = None,
    fiscal_year_t_minus_1: Optional[int] = None,
    out_dir: str = "output",
) -> dict[str, Any]:
    """
    Step1 -> Step2 -> normalize_financials (Standalone + Consolidated) ->
    ForensicDiagnosticEngine, all in one call. Writes every intermediate
    and final artifact into out_dir and returns the diagnostic report.
    """
    import os
    import sys
    import datetime as _dt

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from pipeline.Step1 import extract_core_financial_statements
    from pipeline.Step2 import extract_financials
    from pipeline.normalize_financials import run_pipeline_from_step2

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set.")

    fy = fiscal_year_t or _dt.date.today().year
    fy_prev = fiscal_year_t_minus_1 or (fy - 1)

    os.makedirs(out_dir, exist_ok=True)
    trimmed_pdf = os.path.join(out_dir, f"{ticker}_trimmed.pdf")
    output_xlsx = os.path.join(out_dir, f"{ticker}.xlsx")
    taxonomy_json_path = os.path.join(out_dir, f"{ticker}_taxonomy.json")
    normalized_standalone_path = os.path.join(out_dir, f"{ticker}_normalized_standalone.json")
    normalized_consolidated_path = os.path.join(out_dir, f"{ticker}_normalized_consolidated.json")
    diagnostic_json_path = os.path.join(out_dir, f"{ticker}_diagnostics.json")
    diagnostic_md_path = os.path.join(out_dir, f"{ticker}_diagnostics.md")

    logger.info("Step 1: extracting core financial pages -> %s", trimmed_pdf)
    extract_core_financial_statements(pdf_path, trimmed_pdf, gemini_key)

    logger.info("Step 2: parsing & building Excel -> %s", output_xlsx)
    extract_financials(trimmed_pdf, output_xlsx)

    with open(taxonomy_json_path, "r") as f:
        taxonomy_json = json.load(f)

    logger.info("Step 3: normalizing (Standalone + Consolidated)")
    normalized_standalone = run_pipeline_from_step2(
        taxonomy_json_path, ticker, fy, fy_prev, report_type="Standalone"
    )
    normalized_consolidated = run_pipeline_from_step2(
        taxonomy_json_path, ticker, fy, fy_prev, report_type="Consolidated"
    )
    with open(normalized_standalone_path, "w") as f:
        json.dump(normalized_standalone, f, indent=2)
    with open(normalized_consolidated_path, "w") as f:
        json.dump(normalized_consolidated, f, indent=2)

    logger.info("Step 4: running forensic diagnostics")
    diagnostic_input = build_diagnostic_input_from_pipeline(
        taxonomy_json, normalized_standalone, normalized_consolidated
    )
    engine = ForensicDiagnosticEngine(diagnostic_input)
    report = engine.run_full_diagnostics()

    with open(diagnostic_json_path, "w") as f:
        json.dump(report, f, indent=2)
    with open(diagnostic_md_path, "w") as f:
        f.write(ForensicDiagnosticEngine.to_markdown(report))

    print(f"\n📄  Diagnostics JSON  -> {diagnostic_json_path}")
    print(f"📄  Diagnostics MD    -> {diagnostic_md_path}")
    print("\n" + ForensicDiagnosticEngine.to_markdown(report))

    return report


# ─────────────────────────────────────────────────────────────────────────
# EXAMPLE EXECUTION
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) >= 3:
        # Real usage: python pipeline/forensic_diagnostic_engine.py <pdf_path> <ticker> [fy] [fy_prev] [out_dir]
        _pdf_path = _sys.argv[1]
        _ticker = _sys.argv[2]
        _fy = int(_sys.argv[3]) if len(_sys.argv) > 3 else None
        _fy_prev = int(_sys.argv[4]) if len(_sys.argv) > 4 else None
        _out_dir = _sys.argv[5] if len(_sys.argv) > 5 else "output"
        run_full_pipeline(_pdf_path, _ticker, _fy, _fy_prev, _out_dir)
        _sys.exit(0)
    mock_data = {
        "line_items": [
            {"raw_line_item": "Miscellaneous Suspense Account", "value": 25000,
             "fs_statement": "", "match_score": 0.0},
            {"raw_line_item": "Revenue from Operations", "value": 500000,
             "fs_statement": "Profit and Loss", "match_score": 100.0},
        ],
        "fs_dictionary": {
            "STANDALONE": {
                "Revenue from Operations": {"current_year": 500000, "previous_year": 450000},
                "Profit for the Year": {"current_year": 80000, "previous_year": 70000},
                "Total Assets": {"current_year": 600000, "previous_year": 550000},
                "Total Liabilities": {"current_year": 350000, "previous_year": 320000},
                "Total Equity": {"current_year": 250000, "previous_year": 230000},
                "Net Cash From Operating Activities": {"current_year": -5000, "previous_year": 40000},
                "Net Cash From Investing Activities": {"current_year": -20000, "previous_year": -15000},
                "Net Cash From Financing Activities": {"current_year": 10000, "previous_year": -10000},
                "Cash and Cash Equivalents at the Beginning of the Year": {"current_year": 30000, "previous_year": 15000},
                "Cash and Cash Equivalents at the End of the Year": {"current_year": 15000, "previous_year": 30000},
                "Dividends Paid": {"current_year": 90000, "previous_year": 20000},
            },
            "CONSOLIDATED": {
                "Revenue from Operations": {"current_year": 480000, "previous_year": 430000},
                "Profit for the Year": {"current_year": 75000, "previous_year": 65000},
                "Total Assets": {"current_year": 620000, "previous_year": 560000},
                "Total Liabilities": {"current_year": 360000, "previous_year": 330000},
                "Total Equity": {"current_year": 260000, "previous_year": 230000},
            },
        },
    }

    engine = ForensicDiagnosticEngine(mock_data)
    result = engine.run_full_diagnostics()
    print(json.dumps(result, indent=2))
    print("\n" + "=" * 70 + "\n")
    print(ForensicDiagnosticEngine.to_markdown(result))