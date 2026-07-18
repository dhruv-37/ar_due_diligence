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
import math
import sys
from collections import Counter
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
    "receivables": ["Trade Receivables", "Sundry Debtors", "Accounts Receivable", "Receivables"],
    "total_current_assets": ["Total Current Assets", "Current Assets"],
    "net_ppe": [
        "Net Property Plant and Equipment",
        "Property, Plant and Equipment (Net)",
        "Property Plant and Equipment",
        "Net Block",
    ],
    "ppe": ["Property, Plant and Equipment", "PPE", "Fixed Assets", "Gross Block"],
    "retained_earnings": [
        "Retained Earnings",
        "Surplus in Statement of Profit and Loss",
        "Retained Earnings/Accumulated Losses",
        "Reserves and Surplus",
        "Other Equity",
    ],
    "inventory": ["Inventories", "Inventory", "Stock-in-trade"],
    "cogs": ["Cost of Materials Consumed", "Cost of Goods Sold", "COGS", "Purchases of Stock-in-Trade"],  # unused: see _compute_cogs_from_line_items
    "cogs_reported": ["Cost of Goods Sold", "COGS", "Cost of Materials Consumed", "Cost of Sales"],
    "profit_before_tax": ["Profit Before Tax", "Profit Before Exceptional Items and Tax", "PBT"],
    "tax_expense": ["Total Tax Expense", "Tax Expense", "Current Tax"],
    "cash_taxes_paid": ["Direct Taxes Paid", "Income Taxes Paid", "Taxes Paid"],
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


def _first_significant_digit(value: float) -> Optional[int]:
    """Return the first significant digit (1-9) of a positive number, or None."""
    v = abs(value)
    if v <= 0:
        return None
    while v < 1:
        v *= 10
    while v >= 10:
        v /= 10
    d = int(v)
    return d if 1 <= d <= 9 else None


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
        self.layer_a_signals: list[dict[str, Any]] = []
        self.layer_b_signals: list[dict[str, Any]] = []
        self.layer_c_signals: list[dict[str, Any]] = []

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
        if canonical == "cogs":
            # COGS has no reliable single line item across schemas — derive it
            # instead of keyword-matching (see _compute_cogs_from_line_items).
            return self._compute_cogs_from_line_items(scope, period)

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

    def _compute_cogs_from_line_items(self, scope: str, period: str = "current_year") -> Optional[float]:
        """
        COGS proxy = Revenue from Operations - (sum of all P&L expense line
        items that appear *before* Finance Costs in statement order).

        We don't keyword-match a "COGS" line item — Indian Schedule III P&Ls
        rarely have one. Instead we walk the ordered income-statement
        line_items, sum every expense line up to (but excluding) the Finance
        Costs line, and subtract that from Revenue from Operations. Works for
        both "current_year" and "previous_year" since line_items carries both.
        """
        if period not in ("current_year", "previous_year"):
            return None
        value_field = "value" if period == "current_year" else "previous_year"

        rfo = self._get_metric(scope, "revenue", period)
        if rfo is None:
            return None

        income_stmt_keywords = ("income statement", "profit and loss", "p&l", "statement of profit and loss")
        finance_cost_keywords = ("finance cost", "finance costs", "interest expense", "borrowing cost")
        revenue_side_keywords = {_norm(a) for a in _METRIC_ALIASES.get("revenue", [])} | {"other income", "total income"}

        expense_sum = 0.0
        matched_any = False

        for item in self.line_items:
            report_type = item.get("report_type")
            if report_type and _norm(report_type) != _norm(scope):
                continue

            fs_statement = _norm(item.get("fs_statement") or "")
            if not any(kw in fs_statement for kw in income_stmt_keywords):
                continue

            label = _norm(item.get("raw_line_item") or "")
            if not label:
                continue

            if any(kw in label for kw in finance_cost_keywords):
                # everything from Finance Costs onward is excluded
                break

            if label in revenue_side_keywords:
                continue

            val = _f(item.get(value_field))
            if val is None:
                continue

            expense_sum += abs(val)
            matched_any = True

        if not matched_any:
            return None

        return rfo - expense_sum

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
    # D. LAYER A — MATHEMATICAL SIGNATURES & MANIPULATION MODELS
    # ─────────────────────────────────────────────────────────────────

    _BENFORD_CRITICAL_CHI2 = 15.51  # d.f.=8, alpha=0.05

    def run_layer_a_diagnostics(self) -> list[dict[str, Any]]:
        """Benford's Law digit-distribution scan + Beneish M-Score core proxies."""
        signals: list[dict[str, Any]] = []
        signals.extend(self._check_benford_law())
        for scope in _SCOPES:
            signals.extend(self._check_dsri(scope))
            signals.extend(self._check_aqi(scope))

        self.layer_a_signals = signals
        return signals

    def _check_benford_law(self) -> list[dict[str, Any]]:
        """Chi-square goodness-of-fit test of first-digit distribution vs Benford's Law."""
        out: list[dict[str, Any]] = []

        digits = []
        for item in self.line_items:
            val = _f(item.get("value", item.get("current_year")))
            if val is None:
                continue
            d = _first_significant_digit(val)
            if d is not None:
                digits.append(d)

        n = len(digits)
        if n < 30:
            # Sample too small for a meaningful chi-square test — skip gracefully.
            return out

        observed_counts = Counter(digits)
        chi_square = 0.0
        for d in range(1, 10):
            expected_p = math.log10(1 + 1 / d)
            expected = expected_p * n
            observed = observed_counts.get(d, 0)
            if expected > 0:
                chi_square += ((observed - expected) ** 2) / expected

        if chi_square > self._BENFORD_CRITICAL_CHI2:
            out.append({
                "risk_level": "WARNING",
                "metric": "Benford's Law Digit Distribution",
                "description": (
                    f"First-significant-digit distribution across {n} line items deviates "
                    f"from Benford's Law (chi-square={chi_square:.2f} > critical value "
                    f"{self._BENFORD_CRITICAL_CHI2}) — Anomalous Digit Distribution."
                ),
                "variance": round(chi_square, 2),
            })
        return out

    def _check_dsri(self, scope: str) -> list[dict[str, Any]]:
        """Days Sales in Receivables Index — potential revenue inflation if > 1.2."""
        out: list[dict[str, Any]] = []

        rec_t = self._get_metric(scope, "receivables", "current_year")
        rec_t1 = self._get_metric(scope, "receivables", "previous_year")
        rev_t = self._get_metric(scope, "revenue", "current_year")
        rev_t1 = self._get_metric(scope, "revenue", "previous_year")

        if None in (rec_t, rec_t1, rev_t, rev_t1) or rev_t == 0 or rev_t1 == 0:
            return out

        ratio_t = rec_t / rev_t
        ratio_t1 = rec_t1 / rev_t1
        if ratio_t1 == 0:
            return out

        dsri = ratio_t / ratio_t1
        if dsri > 1.2:
            out.append({
                "risk_level": "CRITICAL",
                "metric": f"{scope} DSRI (Days Sales in Receivables Index)",
                "description": (
                    f"DSRI = {dsri:.2f} (> 1.2) — Receivables/Revenue grew disproportionately "
                    f"year-over-year, a potential indicator of revenue inflation."
                ),
                "variance": round(dsri, 4),
            })
        return out

    def _check_aqi(self, scope: str) -> list[dict[str, Any]]:
        """Asset Quality Index — potential expense capitalization if > 1.3."""
        out: list[dict[str, Any]] = []

        ta_t = self._get_metric(scope, "total_assets", "current_year")
        ta_t1 = self._get_metric(scope, "total_assets", "previous_year")
        ca_t = self._get_metric(scope, "total_current_assets", "current_year")
        ca_t1 = self._get_metric(scope, "total_current_assets", "previous_year")
        ppe_t = self._get_metric(scope, "net_ppe", "current_year")
        ppe_t1 = self._get_metric(scope, "net_ppe", "previous_year")

        if None in (ta_t, ta_t1, ca_t, ca_t1, ppe_t, ppe_t1) or ta_t == 0 or ta_t1 == 0:
            return out

        soft_t = ta_t - ca_t - ppe_t
        soft_t1 = ta_t1 - ca_t1 - ppe_t1

        ratio_t = soft_t / ta_t
        ratio_t1 = soft_t1 / ta_t1
        if ratio_t1 == 0:
            return out

        aqi = ratio_t / ratio_t1
        if aqi > 1.3:
            out.append({
                "risk_level": "CRITICAL",
                "metric": f"{scope} AQI (Asset Quality Index)",
                "description": (
                    f"AQI = {aqi:.2f} (> 1.3) — Soft Assets (Total Assets - Current Assets - "
                    f"PP&E) grew disproportionately relative to Total Assets, a potential "
                    f"indicator of expense capitalization."
                ),
                "variance": round(aqi, 4),
            })
        return out

    # ─────────────────────────────────────────────────────────────────
    # E. LAYER B — INTER-PERIOD ARTICULATION CHECKS
    # ─────────────────────────────────────────────────────────────────

    _SLOAN_WARNING_THRESHOLD = 0.10
    _SLOAN_CRITICAL_THRESHOLD = 0.15

    def run_layer_b_diagnostics(self) -> list[dict[str, Any]]:
        """Retained-earnings roll-forward + Sloan Ratio accruals anomaly checks."""
        signals: list[dict[str, Any]] = []
        for scope in _SCOPES:
            signals.extend(self._check_retained_earnings_rollforward(scope))
            signals.extend(self._check_sloan_ratio(scope))

        self.layer_b_signals = signals
        return signals

    def _check_retained_earnings_rollforward(self, scope: str) -> list[dict[str, Any]]:
        """RE_t = RE_t-1 + Net Profit_t - Dividends Paid_t."""
        out: list[dict[str, Any]] = []

        re_t = self._get_metric(scope, "retained_earnings", "current_year")
        re_t1 = self._get_metric(scope, "retained_earnings", "previous_year")
        profit_t = self._get_metric(scope, "profit_for_the_year", "current_year")
        dividends_t = self._get_metric(scope, "dividends_paid", "current_year")

        if None in (re_t, re_t1, profit_t):
            return out
        dividends_t = dividends_t or 0.0

        expected_re_t = re_t1 + profit_t - abs(dividends_t)
        diff = re_t - expected_re_t

        if abs(diff) > TOLERANCE:
            out.append({
                "risk_level": "CRITICAL",
                "metric": f"{scope} Retained Earnings Roll-Forward",
                "description": (
                    f"Retained Earnings ({re_t:,.2f}) does not equal prior-year Retained "
                    f"Earnings ({re_t1:,.2f}) + Net Profit ({profit_t:,.2f}) - Dividends Paid "
                    f"({abs(dividends_t):,.2f}) = {expected_re_t:,.2f} — Off-Statement Equity "
                    f"Leakage."
                ),
                "variance": round(diff, 2),
            })
        return out

    def _check_sloan_ratio(self, scope: str) -> list[dict[str, Any]]:
        """Sloan Ratio = (Net Profit - OCF - ICF) / Total Assets."""
        out: list[dict[str, Any]] = []

        profit = self._get_metric(scope, "profit_for_the_year", "current_year")
        ocf = self._get_metric(scope, "operating_cash_flow", "current_year")
        icf = self._get_metric(scope, "investing_cash_flow", "current_year")
        total_assets = self._get_metric(scope, "total_assets", "current_year")

        if None in (profit, ocf, icf, total_assets) or total_assets == 0:
            return out

        sloan_ratio = (profit - ocf - icf) / total_assets
        abs_ratio = abs(sloan_ratio)

        if abs_ratio > self._SLOAN_CRITICAL_THRESHOLD:
            out.append({
                "risk_level": "CRITICAL",
                "metric": f"{scope} Sloan Ratio (Accruals Anomaly)",
                "description": (
                    f"Sloan Ratio = {sloan_ratio:.4f} (|ratio| > {self._SLOAN_CRITICAL_THRESHOLD}) "
                    f"— high non-cash accruals relative to Total Assets, a significant "
                    f"earnings-quality risk."
                ),
                "variance": round(sloan_ratio, 4),
            })
        elif abs_ratio > self._SLOAN_WARNING_THRESHOLD:
            out.append({
                "risk_level": "WARNING",
                "metric": f"{scope} Sloan Ratio (Accruals Anomaly)",
                "description": (
                    f"Sloan Ratio = {sloan_ratio:.4f} (|ratio| > {self._SLOAN_WARNING_THRESHOLD}) "
                    f"— elevated non-cash accruals relative to Total Assets."
                ),
                "variance": round(sloan_ratio, 4),
            })
        return out

    # ─────────────────────────────────────────────────────────────────
    # F. LAYER C — CROSS-METRIC OPERATIONAL DRIFTS
    # ─────────────────────────────────────────────────────────────────

    _DSO_RECEIVABLES_VS_REVENUE_GAP = 0.30
    _DSO_DAYS_JUMP = 15
    _DSO_SLUGGISH_REVENUE_GROWTH = 0.10
    _INVENTORY_GROWTH_THRESHOLD = 0.15
    _INVENTORY_VS_COGS_GAP = 0.25

    def run_layer_c_diagnostics(self) -> list[dict[str, Any]]:
        """Cross-metric operational drift checks (DSO/Revenue, Inventory/COGS)."""
        signals: list[dict[str, Any]] = []
        for scope in _SCOPES:
            self._run_layer_c_operational_drifts(scope, signals)

        self.layer_c_signals = signals
        return signals

    def _run_layer_c_operational_drifts(self, scope: str, signals: list) -> None:
        """
        A. DSO vs. Revenue Growth Divergence (Channel Stuffing Guard)
        B. Inventory vs. COGS Divergence (Margin Manipulation Guard)
        Appends any triggered signals directly onto the caller-supplied list.
        """
        rev_t = self._get_metric(scope, "revenue", "current_year")
        rev_t1 = self._get_metric(scope, "revenue", "previous_year")
        rec_t = self._get_metric(scope, "receivables", "current_year")
        rec_t1 = self._get_metric(scope, "receivables", "previous_year")

        if None not in (rev_t, rev_t1, rec_t, rec_t1) and rev_t1 != 0 and rec_t1 != 0:
            dso_t = (rec_t / rev_t) * 365 if rev_t else None
            dso_t1 = (rec_t1 / rev_t1) * 365 if rev_t1 else None

            revenue_growth = (rev_t - rev_t1) / rev_t1
            receivables_growth = (rec_t - rec_t1) / rec_t1

            dso_jump = (dso_t - dso_t1) if (dso_t is not None and dso_t1 is not None) else None

            triggered = (receivables_growth - revenue_growth) > self._DSO_RECEIVABLES_VS_REVENUE_GAP
            if not triggered and dso_jump is not None:
                if dso_jump > self._DSO_DAYS_JUMP and revenue_growth < self._DSO_SLUGGISH_REVENUE_GROWTH:
                    triggered = True

            if triggered:
                signals.append({
                    "risk_level": "CRITICAL",
                    "metric": f"{scope} DSO vs. Revenue Growth Divergence",
                    "description": (
                        f"Receivables growth ({receivables_growth*100:.1f}%) is outpacing Revenue "
                        f"growth ({revenue_growth*100:.1f}%), and/or DSO jumped from {dso_t1:.1f} to "
                        f"{dso_t:.1f} days while revenue growth was sluggish — Potential Channel "
                        f"Stuffing / Aggressive Revenue Recognition."
                    ),
                    "variance": round(receivables_growth - revenue_growth, 4),
                })

        inv_t = self._get_metric(scope, "inventory", "current_year")
        inv_t1 = self._get_metric(scope, "inventory", "previous_year")
        cogs_t = self._get_metric(scope, "cogs_reported", "current_year")
        cogs_t1 = self._get_metric(scope, "cogs_reported", "previous_year")

        if None not in (inv_t, inv_t1) and inv_t1 != 0:
            inventory_growth = (inv_t - inv_t1) / inv_t1

            cogs_growth = None
            if None not in (cogs_t, cogs_t1) and cogs_t1 != 0:
                cogs_growth = (cogs_t - cogs_t1) / cogs_t1

            triggered = False
            if cogs_growth is not None:
                if inventory_growth > self._INVENTORY_GROWTH_THRESHOLD and cogs_growth <= 0:
                    triggered = True
                elif (inventory_growth - cogs_growth) > self._INVENTORY_VS_COGS_GAP:
                    triggered = True

            if triggered:
                cogs_growth_str = f"{cogs_growth*100:.1f}%" if cogs_growth is not None else "N/A"
                signals.append({
                    "risk_level": "HIGH",
                    "metric": f"{scope} Inventory vs. COGS Divergence",
                    "description": (
                        f"Inventory growth ({inventory_growth*100:.1f}%) is disproportionate to "
                        f"COGS growth ({cogs_growth_str}) — Potential Inventory Overvaluation / "
                        f"Artificial Margin Inflation."
                    ),
                    "variance": round(inventory_growth - (cogs_growth or 0.0), 4),
                })

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

        for signal in self.layer_a_signals:
            level = signal.get("risk_level")
            if level == "CRITICAL":
                score -= 15.0
            elif level == "WARNING":
                score -= 7.0
            else:
                score -= 2.0

        for signal in self.layer_b_signals:
            level = signal.get("risk_level")
            if level == "CRITICAL":
                score -= 15.0
            elif level == "WARNING":
                score -= 7.0
            else:
                score -= 2.0

        for signal in self.layer_c_signals:
            level = signal.get("risk_level")
            if level in ("CRITICAL", "HIGH"):
                score -= 15.0
            elif level == "WARNING":
                score -= 7.0
            else:
                score -= 2.0

        return max(0.0, round(score, 2))

    def run_full_diagnostics(self) -> dict[str, Any]:
        """Run all three pipelines and assemble the structured report."""
        logger.info("Running taxonomy validation...")
        self.validate_taxonomy()
        logger.info("Running accounting math verification...")
        self.verify_accounting_math()
        logger.info("Running forensic red-flag detection...")
        self.detect_red_flags()
        logger.info("Running Layer A mathematical signature models...")
        self.run_layer_a_diagnostics()
        logger.info("Running Layer B inter-period articulation checks...")
        self.run_layer_b_diagnostics()
        logger.info("Running Layer C cross-metric operational drift checks...")
        self.run_layer_c_diagnostics()

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "integrity_score": self._compute_integrity_score(),
            "taxonomy_gaps": self.taxonomy_gaps,
            "mathematical_discrepancies": self.mathematical_discrepancies,
            "forensic_red_flags": self.forensic_red_flags,
            "layer_a_signals": self.layer_a_signals,
            "layer_b_signals": self.layer_b_signals,
            "layer_c_signals": self.layer_c_signals,
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

        lines.append(f"\n## Layer A — Mathematical Signatures ({len(report.get('layer_a_signals', []))})")
        if report.get("layer_a_signals"):
            icon = {"CRITICAL": "🔴", "WARNING": "🟠", "INFO": "🔵"}
            for s in report["layer_a_signals"]:
                lines.append(
                    f"- {icon.get(s['risk_level'], '')} **[{s['risk_level']}] {s['metric']}**: "
                    f"{s['description']} (variance: {s['variance']})"
                )
        else:
            lines.append("- None")

        lines.append(f"\n## Layer B — Inter-Period Articulation ({len(report.get('layer_b_signals', []))})")
        if report.get("layer_b_signals"):
            icon = {"CRITICAL": "🔴", "WARNING": "🟠", "INFO": "🔵"}
            for s in report["layer_b_signals"]:
                lines.append(
                    f"- {icon.get(s['risk_level'], '')} **[{s['risk_level']}] {s['metric']}**: "
                    f"{s['description']} (variance: {s['variance']})"
                )
        else:
            lines.append("- None")

        lines.append(f"\n## Layer C — Cross-Metric Operational Drifts ({len(report.get('layer_c_signals', []))})")
        if report.get("layer_c_signals"):
            icon = {"CRITICAL": "🔴", "HIGH": "🟣", "WARNING": "🟠", "INFO": "🔵"}
            for s in report["layer_c_signals"]:
                lines.append(
                    f"- {icon.get(s['risk_level'], '')} **[{s['risk_level']}] {s['metric']}**: "
                    f"{s['description']} (variance: {s['variance']})"
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
            "previous_year": rec.get("previous_year"),
            "fs_statement": rec.get("fs_statement") or rec.get("statement"),
            "match_score": rec.get("match_score"),
            "report_type": rec.get("report_type"),
        })

    def _metric_dict(normalized: dict[str, Any]) -> dict[str, Any]:
        inc = normalized.get("income_statement") or normalized.get("Income Statement") or {}
        bs = normalized.get("balance_sheet") or normalized.get("Balance Sheet") or {}
        cf = normalized.get("cash_flow_statement") or normalized.get("Cash Flow Statement") or {}

        def _remap_periods(entry: Any) -> dict[str, Any]:
            # normalize_financials() emits {"t": ..., "t_minus_1": ...};
            # the engine reads {"current_year": ..., "previous_year": ...}.
            if not isinstance(entry, dict):
                return {}
            return {
                "current_year": entry.get("current_year", entry.get("t")),
                "previous_year": entry.get("previous_year", entry.get("t_minus_1")),
            }

        def safe_get(canonical_key: str, statement_dict: dict[str, Any]) -> dict[str, Any]:
            entry = statement_dict.get(canonical_key) or normalized.get(canonical_key) or {}
            return _remap_periods(entry)

        return {
            # Baseline Financials
            "Revenue from Operations": safe_get("revenue", inc),
            "Profit for the Year": safe_get("net_income_continuing", inc) or safe_get("profit_for_the_year", inc),
            "Total Assets": safe_get("total_assets", bs),
            "Total Liabilities": safe_get("total_liabilities", bs),
            "Total Equity": safe_get("total_equity", bs),

            # Core Cash Flows
            "Net Cash From Operating Activities": safe_get("operating_cash_flow", cf),
            "Net Cash From Investing Activities": safe_get("investing_cash_flow", cf),
            "Net Cash From Financing Activities": safe_get("financing_cash_flow", cf),
            "Cash and Cash Equivalents at the Beginning of the Year": safe_get("cash_opening", cf),
            "Cash and Cash Equivalents at the End of the Year": safe_get("cash_closing", cf),
            "Dividends Paid": safe_get("dividends_paid", cf),

            # Advanced Forensic Structural Mappings (Layer A & B)
            "Trade Receivables": safe_get("accounts_receivable_net", bs) or safe_get("receivables", bs) or safe_get("trade_receivables", bs),
            "Total Current Assets": safe_get("total_current_assets", bs) or safe_get("current_assets", bs),
            "Net Property Plant and Equipment": safe_get("property_plant_equipment_net", bs) or safe_get("net_ppe", bs) or safe_get("ppe", bs),
            "Retained Earnings": safe_get("retained_earnings", bs) or safe_get("reserves_surplus", bs),

            # Advanced Forensic Structural Mappings (Layer C)
            "Inventories": safe_get("inventories", bs) or safe_get("inventory", bs),
            "Cost of Goods Sold": safe_get("cost_of_goods_sold", inc) or safe_get("cogs", inc),
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
    diagnostic_input_path = os.path.join(out_dir, f"{ticker}_diagnostic_input.json")

    logger.info("Step 1: extracting core financial pages -> %s", trimmed_pdf)
    extract_core_financial_statements(pdf_path, trimmed_pdf, gemini_key)

    logger.info("Step 2: parsing & building Excel -> %s", output_xlsx)
    extract_financials(trimmed_pdf, output_xlsx)

    with open(taxonomy_json_path, "r", encoding="utf-8") as f:
        taxonomy_json = json.load(f)

    logger.info("Step 3: normalizing (Standalone + Consolidated)")
    normalized_standalone = run_pipeline_from_step2(
        taxonomy_json_path, ticker, fy, fy_prev, report_type="Standalone"
    )
    normalized_consolidated = run_pipeline_from_step2(
        taxonomy_json_path, ticker, fy, fy_prev, report_type="Consolidated"
    )
    with open(normalized_standalone_path, "w", encoding="utf-8") as f:
        json.dump(normalized_standalone, f, indent=2)
    with open(normalized_consolidated_path, "w", encoding="utf-8") as f:
        json.dump(normalized_consolidated, f, indent=2)

    logger.info("Step 4: running forensic diagnostics")
    diagnostic_input = build_diagnostic_input_from_pipeline(
        taxonomy_json, normalized_standalone, normalized_consolidated
    )
    # Persist the EXACT {line_items, fs_dictionary} object the engine
    # consumes — this is what audit_layer_inputs.py needs. Previously this
    # was only ever built in-memory, so there was no file that reflected
    # what Layer A/B/C actually saw as input; only the diagnostics OUTPUT
    # (which metrics fired) was saved, not the diagnostics INPUT (which
    # metrics were even available to fire).
    with open(diagnostic_input_path, "w", encoding="utf-8") as f:
        json.dump(diagnostic_input, f, indent=2, default=str)
    engine = ForensicDiagnosticEngine(diagnostic_input)
    report = engine.run_full_diagnostics()

    with open(diagnostic_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(diagnostic_md_path, "w", encoding="utf-8") as f:
        f.write(ForensicDiagnosticEngine.to_markdown(report))

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"\n📄  Diagnostic input   -> {diagnostic_input_path}   (feed this to audit_layer_inputs.py)")
    print(f"📄  Diagnostics JSON  -> {diagnostic_json_path}")
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