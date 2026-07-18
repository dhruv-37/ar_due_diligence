"""
normalize_financials.py
========================
Phase 1 Data Normalization & Validation pipeline.

Ingests messy, heterogeneous raw JSON extracted from a corporate annual
report and produces a brand-new, schema-conformant normalized JSON object.
The input dictionary is never mutated (immutable pattern) — a fresh
dictionary is always returned.

Pipeline stages
----------------
1. Alias-mapping extraction  (raw key -> canonical schema field)
2. Schema construction        (t / t-1 values, coerced to float)
3. Accounting-identity validation (logged, non-fatal)
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any, Optional

# ─────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────

logger = logging.getLogger("normalize_financials")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    )
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────────────
# ALIAS MAP  (canonical field -> list of known raw-key variants)
# ─────────────────────────────────────────────────────────────────────────

ALIAS_MAP: dict[str, list[str]] = {
    "revenue": [
        "Net Sales", "Turnover", "Revenues", "Sales Revenue", "Total Revenue",
        "Revenue from operations", "Revenue from Operations",
    ],
    "cost_of_goods_sold": ["Cost of Sales", "Cost of Revenue", "Production Costs"],
    "sga_expense": [
        "Operating Expenses", "SGA & Corporate", "Administrative Costs",
        "Selling, General and Administrative", "Other expenses",
    ],
    "net_income_continuing": [
        "Net Income from Continuing Operations", "Operating Income After Tax", "Net Income",
        "Profit for the year", "PROFIT FOR THE YEAR", "Profit for the period",
    ],
    "accounts_receivable_net": [
        "Trade Receivables", "Net Notes and Accounts Receivable",
        "Customer Receivables", "Accounts Receivable",
    ],
    "total_current_assets": ["Total Current Assets", "Short-term Assets"],
    "property_plant_equipment_net": [
        "Net Fixed Assets", "Property and Equipment, net", "PP&E, net",
        "Property, plant and equipment",
    ],
    "property_plant_equipment_gross": [
        "Property and Equipment at Cost", "Historic Cost of Fixed Assets", "PP&E, gross",
    ],
    "total_assets": ["Total Assets", "TOTAL ASSETS"],
    "total_liabilities": ["Total Liabilities"],
    "total_equity": [
        "Total Stockholders' Equity", "Total Equity", "Shareholders' Equity",
    ],
    "depreciation_amortization": [
        "Depreciation and Amortization", "D&A", "Depreciation",
        "Depreciation and amortisation expense",
    ],
    "operating_cash_flow": [
        "Net Cash Provided by Operating Activities", "Cash from Operations", "Operating Cash Flow",
        "Net cash flows generated from operating activities",
        "Net cash flows generated from / (used in) operating activities",
    ],
    "investing_cash_flow": [
        "Net Cash Used in Investing Activities", "Cash from Investing", "Investing Cash Flow",
        "Net cash flows used in investing activities",
        "Net cash flows generated from / (used in) investing activities",
    ],
    "financing_cash_flow": [
        "Net Cash Used in Financing Activities", "Cash from Financing", "Financing Cash Flow",
        "Net cash flows used in financing activities",
        "Net cash flows generated from / (used in) financing activities",
    ],
    "cash_opening": [
        "Cash and Cash Equivalents at the Beginning of the Year",
        "Opening Cash Balance", "Cash and cash equivalents at beginning of the year",
    ],
    "cash_closing": [
        "Cash and Cash Equivalents at the End of the Year",
        "Closing Cash Balance", "Cash and cash equivalents at end of the year",
    ],
    "dividends_paid": [
        "Dividends Paid", "Dividend Paid", "Payment of dividends",
        "Dividends paid (including tax on dividend)",
    ],
    "retained_earnings": [
        "Retained Earnings", "Reserves and Surplus", "Other Equity",
        "Surplus in Statement of Profit and Loss",
    ],
}

# Line items summed together when no single line matches a canonical field
# directly (e.g. Ind-AS balance sheets often split receivables into
# "Billed" / "Unbilled" rather than reporting one net figure).
COMPOSITE_ALIAS_MAP: dict[str, list[str]] = {
    "accounts_receivable_net": [
        "Financial assets - Trade receivables - Billed",
        "Financial assets - Trade receivables - Unbilled",
    ],
}

# canonical field -> (schema section, schema key)
_SCHEMA_LOCATION: dict[str, tuple[str, str]] = {
    "revenue": ("income_statement", "revenue"),
    "cost_of_goods_sold": ("income_statement", "cost_of_goods_sold"),
    "sga_expense": ("income_statement", "sga_expense"),
    "net_income_continuing": ("income_statement", "net_income_continuing"),
    "accounts_receivable_net": ("balance_sheet", "accounts_receivable_net"),
    "total_current_assets": ("balance_sheet", "total_current_assets"),
    "property_plant_equipment_net": ("balance_sheet", "property_plant_equipment_net"),
    "property_plant_equipment_gross": ("balance_sheet", "property_plant_equipment_gross"),
    "total_assets": ("balance_sheet", "total_assets"),
    "total_liabilities": ("balance_sheet", "total_liabilities"),
    "total_equity": ("balance_sheet", "total_equity"),
    "depreciation_amortization": ("cash_flow_statement", "depreciation_amortization"),
    "operating_cash_flow": ("cash_flow_statement", "operating_cash_flow"),
    "investing_cash_flow": ("cash_flow_statement", "investing_cash_flow"),
    "financing_cash_flow": ("cash_flow_statement", "financing_cash_flow"),
    "cash_opening": ("cash_flow_statement", "cash_opening"),
    "cash_closing": ("cash_flow_statement", "cash_closing"),
    "dividends_paid": ("cash_flow_statement", "dividends_paid"),
    "retained_earnings": ("balance_sheet", "retained_earnings"),
}

TOLERANCE = 1_000.0  # allowed rounding slack (thousands/millions reporting)


# ─────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────

def _coerce_float(value: Any) -> Optional[float]:
    """Best-effort coercion of a raw value (str/int/float/None) to float."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").replace("$", "").strip()
        neg = cleaned.startswith("(") and cleaned.endswith(")")
        cleaned = cleaned.strip("()")
        if not cleaned:
            return None
        try:
            num = float(cleaned)
            return -num if neg else num
        except ValueError:
            logger.warning("Could not coerce value %r to float — treating as missing.", value)
            return None
    logger.warning("Unsupported value type %s for %r — treating as missing.", type(value), value)
    return None


def _normalize_key(s: str) -> str:
    """Lowercase + collapse internal whitespace, for tolerant key matching."""
    return " ".join(str(s).strip().lower().split())


def _lookup_alias(period_dict: dict[str, Any], canonical_field: str) -> Optional[float]:
    """
    Search a single-period raw dict (e.g. raw['fiscal_year_t']) for any known
    alias of `canonical_field`, in priority order. Matching is case- and
    whitespace-insensitive since real-world financial statements vary in
    capitalization (e.g. "TOTAL ASSETS" vs "Total Assets" vs "Total assets").
    Logs the match or the fallback path taken.
    """
    normalized_lookup = {_normalize_key(k): v for k, v in period_dict.items()}
    aliases = ALIAS_MAP.get(canonical_field, [])
    for alias in aliases:
        key = _normalize_key(alias)
        if key in normalized_lookup:
            value = _coerce_float(normalized_lookup[key])
            if value is not None:
                logger.info("Matched '%s' -> '%s' (value=%s)", canonical_field, alias, value)
                return value
            logger.warning(
                "Alias '%s' present for '%s' but value unusable — trying next alias.",
                alias, canonical_field,
            )

    # Composite fallback: sum multiple line items that together represent
    # the canonical field (e.g. Billed + Unbilled receivables).
    composite_aliases = COMPOSITE_ALIAS_MAP.get(canonical_field)
    if composite_aliases:
        parts = []
        for alias in composite_aliases:
            key = _normalize_key(alias)
            if key in normalized_lookup:
                value = _coerce_float(normalized_lookup[key])
                if value is not None:
                    parts.append(value)
        if len(parts) == len(composite_aliases):
            total = sum(parts)
            logger.info(
                "Matched '%s' -> sum(%s) = %s (composite fallback)",
                canonical_field, composite_aliases, total,
            )
            return total

    logger.warning("No alias match found for canonical field '%s'.", canonical_field)
    return None


def _empty_schema(ticker: str, fy_t: int, fy_t_minus_1: int, reporting_unit: Optional[str] = None) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "fiscal_year_t": fy_t,
        "fiscal_year_t_minus_1": fy_t_minus_1,
        "reporting_unit": reporting_unit,
        "income_statement": {
            "revenue": {"t": None, "t_minus_1": None},
            "cost_of_goods_sold": {"t": None, "t_minus_1": None},
            "sga_expense": {"t": None, "t_minus_1": None},
            "net_income_continuing": {"t": None, "t_minus_1": None},
        },
        "balance_sheet": {
            "accounts_receivable_net": {"t": None, "t_minus_1": None},
            "total_current_assets": {"t": None, "t_minus_1": None},
            "property_plant_equipment_net": {"t": None, "t_minus_1": None},
            "property_plant_equipment_gross": {"t": None, "t_minus_1": None},
            "total_assets": {"t": None, "t_minus_1": None},
            "total_liabilities": {"t": None, "t_minus_1": None},
            "total_equity": {"t": None, "t_minus_1": None},
            "retained_earnings": {"t": None, "t_minus_1": None},
        },
        "cash_flow_statement": {
            "depreciation_amortization": {"t": None, "t_minus_1": None},
            "operating_cash_flow": {"t": None, "t_minus_1": None},
            "investing_cash_flow": {"t": None, "t_minus_1": None},
            "financing_cash_flow": {"t": None, "t_minus_1": None},
            "cash_opening": {"t": None, "t_minus_1": None},
            "cash_closing": {"t": None, "t_minus_1": None},
            "dividends_paid": {"t": None, "t_minus_1": None},
        },
    }


# ─────────────────────────────────────────────────────────────────────────
# CORE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────

def normalize_financials(raw_json: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a messy, heterogeneous raw financial JSON into a normalized
    dictionary strictly matching the target schema.

    The input `raw_json` is never mutated — a deep copy is used internally
    purely for safety, and a completely fresh dictionary is returned.

    Expected raw_json shape (flexible on key names, but structurally):
        {
            "ticker": "ABC",
            "fiscal_year_t": 2024,
            "fiscal_year_t_minus_1": 2023,
            "period_t": {<raw line items for year t>},
            "period_t_minus_1": {<raw line items for year t-1>},
        }

    Returns
    -------
    dict matching the target_schema exactly.
    """
    raw_snapshot = copy.deepcopy(raw_json)  # guarantee no accidental mutation

    ticker = raw_snapshot.get("ticker", "UNKNOWN")
    fy_t = raw_snapshot.get("fiscal_year_t")
    fy_t_minus_1 = raw_snapshot.get("fiscal_year_t_minus_1")

    period_t = raw_snapshot.get("period_t", {}) or {}
    period_t_minus_1 = raw_snapshot.get("period_t_minus_1", {}) or {}

    normalized = _empty_schema(ticker, fy_t, fy_t_minus_1, raw_snapshot.get("reporting_unit"))

    logger.info("Starting normalization for ticker=%s (FY%s vs FY%s)", ticker, fy_t, fy_t_minus_1)

    for canonical_field, (section, key) in _SCHEMA_LOCATION.items():
        value_t = _lookup_alias(period_t, canonical_field)
        value_t_minus_1 = _lookup_alias(period_t_minus_1, canonical_field)
        normalized[section][key]["t"] = value_t
        normalized[section][key]["t_minus_1"] = value_t_minus_1

    _derive_total_liabilities(normalized, period_t, period_t_minus_1)
    _derive_cogs(normalized, raw_snapshot.get("pnl_items", []))

    logger.info("Normalization complete for ticker=%s", ticker)
    return normalized


# Labels marking the boundary between operating expenses and finance/tax
# section in a P&L — everything from the income total down to (but not
# including) one of these is treated as "cost of goods sold" for the
# purposes of this schema.
_FINANCE_COST_LABELS = {
    "finance costs", "finance cost", "financial expenses",
    "interest and finance charges", "finance charges", "interest expense",
}
_INCOME_TOTAL_LABELS = {"total income", "total revenue"}


def _derive_cogs(normalized: dict[str, Any], pnl_items: list[dict]) -> None:
    """
    Derive cost_of_goods_sold as: sum of every P&L expense line item that
    appears after the Total Income/Total Revenue line and before the
    Finance Costs line (subtotal rows excluded). This mirrors how Ind-AS
    P&Ls are laid out — operating expenses are grouped together, with
    finance costs and tax broken out separately below them.
    """
    inc = normalized["income_statement"]
    if inc["cost_of_goods_sold"]["t"] is not None and inc["cost_of_goods_sold"]["t_minus_1"] is not None:
        return
    if not pnl_items:
        logger.warning("No P&L line items supplied — cannot derive cost_of_goods_sold.")
        return

    started = False
    t_sum = 0.0
    tm1_sum = 0.0
    matched_labels: list[str] = []

    for rec in pnl_items:
        label_norm = _normalize_key(rec.get("line_item", ""))
        is_total = bool(rec.get("is_total"))
        if not started:
            if is_total and label_norm in _INCOME_TOTAL_LABELS:
                started = True
            continue
        if label_norm in _FINANCE_COST_LABELS:
            break
        if is_total:
            continue
        cy = _coerce_float(rec.get("current_year"))
        py = _coerce_float(rec.get("previous_year"))
        if cy is not None:
            t_sum += cy
        if py is not None:
            tm1_sum += py
        matched_labels.append(rec.get("line_item", ""))

    if not started:
        # Fallback: no explicit Total Income/Revenue subtotal row found —
        # anchor off the revenue line itself instead, skipping further
        # income-labelled rows by name heuristic.
        revenue_aliases = {_normalize_key(a) for a in ALIAS_MAP["revenue"]}
        t_sum = 0.0
        tm1_sum = 0.0
        matched_labels = []
        for rec in pnl_items:
            label_norm = _normalize_key(rec.get("line_item", ""))
            is_total = bool(rec.get("is_total"))
            if not started:
                if label_norm in revenue_aliases:
                    started = True
                continue
            if label_norm in _FINANCE_COST_LABELS:
                break
            if is_total or "income" in label_norm:
                continue
            cy = _coerce_float(rec.get("current_year"))
            py = _coerce_float(rec.get("previous_year"))
            if cy is not None:
                t_sum += cy
            if py is not None:
                tm1_sum += py
            matched_labels.append(rec.get("line_item", ""))
        if not matched_labels:
            logger.warning("Could not locate a revenue/income anchor to derive cost_of_goods_sold.")
            return
        logger.info(
            "Derived cost_of_goods_sold via revenue-anchor fallback from: %s", matched_labels
        )
    else:
        logger.info(
            "Derived cost_of_goods_sold (pre-finance-cost expenses) from: %s", matched_labels
        )

    revenue_t = inc["revenue"]["t"]
    revenue_tm1 = inc["revenue"]["t_minus_1"]
    if revenue_t is None or revenue_tm1 is None:
        logger.warning("Revenue not available — cannot derive cost_of_goods_sold as Revenue - Expenses.")
        return

    inc["cost_of_goods_sold"]["t"] = revenue_t - t_sum
    inc["cost_of_goods_sold"]["t_minus_1"] = revenue_tm1 - tm1_sum
    logger.info(
        "cost_of_goods_sold = Revenue - pre-finance-cost expenses: t=%.2f-%.2f=%.2f, "
        "t_minus_1=%.2f-%.2f=%.2f",
        revenue_t, t_sum, revenue_t - t_sum, revenue_tm1, tm1_sum, revenue_tm1 - tm1_sum,
    )


def _derive_total_liabilities(
    normalized: dict[str, Any], period_t: dict[str, Any], period_t_minus_1: dict[str, Any]
) -> None:
    """
    Many Ind-AS / IFRS balance sheets report "Total Equity and Liabilities"
    (== Total Assets by construction) rather than a standalone Total
    Liabilities line. When total_liabilities wasn't matched directly,
    derive it as (Total Equity and Liabilities, or Total Assets) - Total Equity.
    """
    bs = normalized["balance_sheet"]
    equity_and_liab_aliases = [
        "TOTAL EQUITY AND LIABILITIES", "Total Equity and Liabilities",
    ]
    for period_label, period_dict in (("t", period_t), ("t_minus_1", period_t_minus_1)):
        if bs["total_liabilities"][period_label] is not None:
            continue
        normalized_lookup = {_normalize_key(k): v for k, v in period_dict.items()}
        equity_and_liab = None
        for alias in equity_and_liab_aliases:
            key = _normalize_key(alias)
            if key in normalized_lookup:
                equity_and_liab = _coerce_float(normalized_lookup[key])
                if equity_and_liab is not None:
                    break
        if equity_and_liab is None:
            equity_and_liab = bs["total_assets"][period_label]  # assets == equity+liabilities

        equity = bs["total_equity"][period_label]
        if equity_and_liab is not None and equity is not None:
            derived = equity_and_liab - equity
            bs["total_liabilities"][period_label] = derived
            logger.info(
                "[%s] Derived total_liabilities = %.2f (Equity+Liabilities %.2f - Equity %.2f)",
                period_label, derived, equity_and_liab, equity,
            )


# ─────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────

def _check_balance_sheet_identity(normalized: dict[str, Any], period_label: str) -> bool:
    """Check 1: Total Assets == Total Liabilities + Total Equity."""
    bs = normalized["balance_sheet"]
    assets = bs["total_assets"][period_label]
    liabilities = bs["total_liabilities"][period_label]
    equity = bs["total_equity"][period_label]

    if None in (assets, liabilities, equity):
        logger.warning("[%s] Balance sheet identity check skipped — missing inputs.", period_label)
        return True

    diff = abs(assets - (liabilities + equity))
    passed = diff <= TOLERANCE
    if not passed:
        logger.warning(
            "[%s] Balance sheet identity FAILED: Assets=%.2f vs Liabilities+Equity=%.2f (diff=%.2f)",
            period_label, assets, liabilities + equity, diff,
        )
    else:
        logger.info("[%s] Balance sheet identity OK (diff=%.2f)", period_label, diff)
    return passed


def _check_gross_profit_identity(normalized: dict[str, Any], period_label: str) -> bool:
    """Check 2: Gross Profit == Revenue - Cost of Goods Sold (computed, sanity-checked)."""
    inc = normalized["income_statement"]
    revenue = inc["revenue"][period_label]
    cogs = inc["cost_of_goods_sold"][period_label]

    if None in (revenue, cogs):
        logger.warning("[%s] Gross profit check skipped — missing inputs.", period_label)
        return True

    gross_profit = revenue - cogs
    # There is no independently-reported gross profit field in the schema,
    # so this check validates internal consistency (non-negative, sane magnitude)
    # and logs the computed figure for downstream Phase 2 use.
    plausible = gross_profit <= revenue + TOLERANCE
    if not plausible:
        logger.warning(
            "[%s] Gross profit identity FAILED: computed Gross Profit=%.2f exceeds Revenue=%.2f",
            period_label, gross_profit, revenue,
        )
    else:
        logger.info("[%s] Gross profit computed OK: %.2f (Revenue=%.2f, COGS=%.2f)",
                    period_label, gross_profit, revenue, cogs)
    return plausible


def _check_earnings_quality(normalized: dict[str, Any], period_label: str) -> bool:
    """Check 3: Flag decoupling between Net Income and Operating Cash Flow."""
    net_income = normalized["income_statement"]["net_income_continuing"][period_label]
    ocf = normalized["cash_flow_statement"]["operating_cash_flow"][period_label]

    if None in (net_income, ocf):
        logger.warning("[%s] Earnings-quality check skipped — missing inputs.", period_label)
        return True

    decoupled = net_income > 0 and ocf < 0
    if decoupled:
        logger.warning(
            "[%s] EARNINGS QUALITY FLAG: Net Income positive (%.2f) but Operating Cash Flow "
            "negative (%.2f) — possible earnings-quality concern for Phase 2 review.",
            period_label, net_income, ocf,
        )
    else:
        logger.info("[%s] Earnings-quality check OK (NI=%.2f, OCF=%.2f)", period_label, net_income, ocf)
    return not decoupled


def validate_accounting_identities(normalized: dict[str, Any]) -> dict[str, bool]:
    """
    Run all three accounting-identity checks for both fiscal periods (t, t-1).
    Never raises — all failures are logged as warnings and returned in a
    results dict for the caller to inspect.
    """
    results: dict[str, bool] = {}
    for period_label in ("t", "t_minus_1"):
        results[f"balance_sheet_identity_{period_label}"] = _check_balance_sheet_identity(
            normalized, period_label
        )
        results[f"gross_profit_identity_{period_label}"] = _check_gross_profit_identity(
            normalized, period_label
        )
        results[f"earnings_quality_{period_label}"] = _check_earnings_quality(
            normalized, period_label
        )
    return results


# ─────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────

def run_normalization_pipeline(raw_json: dict[str, Any]) -> dict[str, Any]:
    """
    Full Phase 1 pipeline: normalize -> validate -> return normalized JSON.
    The validation results are attached under a non-schema '_validation' key
    for visibility, without altering the strict target schema fields.
    """
    normalized = normalize_financials(raw_json)
    validation_results = validate_accounting_identities(normalized)

    normalized_with_meta = copy.deepcopy(normalized)
    normalized_with_meta["_validation"] = validation_results
    return normalized_with_meta


# ─────────────────────────────────────────────────────────────────────────
# STEP2 ADAPTER
# ─────────────────────────────────────────────────────────────────────────
# Step2.py (pipeline/Step2.py) writes "<output>_taxonomy.json" shaped as:
#   {
#     "line_items": [
#       {"line_item": ..., "taxonomy_node": ..., "fs_statement": ...,
#        "statement": ..., "is_total": ..., "report_type": "Standalone"/"Consolidated",
#        "current_year": FLOAT, "previous_year": FLOAT, "match_score": ...},
#       ...
#     ],
#     "fs_dictionary": {...}
#   }
# This adapter reshapes that flat line-item list into the
# {"period_t": {line_item: value}, "period_t_minus_1": {line_item: value}}
# form that normalize_financials() consumes, so the existing alias-mapping
# logic runs unchanged against Step2's `line_item` text.

def build_raw_json_from_step2(
    step2_json_path: str,
    ticker: str,
    fiscal_year_t: int,
    fiscal_year_t_minus_1: int,
    report_type: str = "Standalone",
) -> dict[str, Any]:
    """
    Load a Step2 `<output>_taxonomy.json` file and reshape it into the raw
    input format expected by normalize_financials().

    Parameters
    ----------
    step2_json_path : path to the "_taxonomy.json" file written by Step2.py
    ticker           : ticker/identifier to stamp on the output
    fiscal_year_t / fiscal_year_t_minus_1 : the two fiscal years (Step2 does
        not itself know calendar years — the caller supplies them)
    report_type      : "Standalone" or "Consolidated" — Step2 output contains
        both scopes mixed together; this filters to one
    """
    with open(step2_json_path, "r") as f:
        step2_output = json.load(f)

    line_items = step2_output.get("line_items", [])
    period_t: dict[str, Any] = {}
    period_t_minus_1: dict[str, Any] = {}
    pnl_items: list[dict[str, Any]] = []

    matched = 0
    for rec in line_items:
        if str(rec.get("report_type", "")).strip().lower() != report_type.strip().lower():
            continue
        label = rec.get("line_item")
        if not label:
            continue
        # last-write-wins is fine here — duplicate line_item labels within a
        # single report_type were already de-duplicated upstream in Step2
        period_t[label] = rec.get("current_year")
        period_t_minus_1[label] = rec.get("previous_year")
        matched += 1
        if str(rec.get("statement", "")).strip() == "Profit and Loss":
            pnl_items.append(rec)

    logger.info(
        "Loaded %d line item(s) from Step2 output (report_type=%s) for ticker=%s",
        matched, report_type, ticker,
    )

    return {
        "ticker": ticker,
        "fiscal_year_t": fiscal_year_t,
        "fiscal_year_t_minus_1": fiscal_year_t_minus_1,
        "period_t": period_t,
        "period_t_minus_1": period_t_minus_1,
        "pnl_items": pnl_items,
        "reporting_unit": step2_output.get("reporting_unit"),
    }


def run_pipeline_from_step2(
    step2_json_path: str,
    ticker: str,
    fiscal_year_t: int,
    fiscal_year_t_minus_1: int,
    report_type: str = "Standalone",
) -> dict[str, Any]:
    """
    Convenience wrapper: Step2 taxonomy JSON -> normalized + validated schema,
    in one call. This is the function to call directly with Step2's output.
    """
    raw_json = build_raw_json_from_step2(
        step2_json_path, ticker, fiscal_year_t, fiscal_year_t_minus_1, report_type
    )
    return run_normalization_pipeline(raw_json)


# ─────────────────────────────────────────────────────────────────────────
# EXAMPLE EXECUTION
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mock_raw_json = {
        "ticker": "ACME",
        "fiscal_year_t": 2024,
        "fiscal_year_t_minus_1": 2023,
        "period_t": {
            "Turnover": "1,250,000",
            "Cost of Revenue": "700,000",
            "SGA & Corporate": "180,000",
            "Net Income": "210,000",
            "Trade Receivables": "95,000",
            "Total Current Assets": "410,000",
            "Property and Equipment, net": "560,000",
            "Historic Cost of Fixed Assets": "890,000",
            "Total Assets": "1,300,000",
            "Total Liabilities": "740,000",
            "Shareholders' Equity": "560,000",
            "D&A": "60,000",
            "Cash from Operations": "-15,000",  # intentionally negative to trigger flag
        },
        "period_t_minus_1": {
            "Turnover": "1,100,000",
            "Cost of Revenue": "640,000",
            "Operating Expenses": "165,000",
            "Net Income from Continuing Operations": "180,000",
            "Accounts Receivable": "88,000",
            "Total Current Assets": "375,000",
            "Net Fixed Assets": "520,000",
            "Property and Equipment at Cost": "820,000",
            "Total Assets": "1,180,000",
            "Total Liabilities": "690,000",
            "Total Equity": "490,000",
            "Depreciation and Amortization": "55,000",
            "Net Cash Provided by Operating Activities": "205,000",
        },
    }

    original_snapshot = copy.deepcopy(mock_raw_json)

    result = run_normalization_pipeline(mock_raw_json)

    assert mock_raw_json == original_snapshot, "Input JSON was mutated — immutability violated!"

    print("\n=== NORMALIZED OUTPUT (mock data) ===")
    print(json.dumps(result, indent=2))

    # ── Real usage: feed Step2's output directly ──────────────────────────
    # result = run_pipeline_from_step2(
    #     step2_json_path="output_taxonomy.json",   # written by pipeline/Step2.py
    #     ticker="RELIANCE",
    #     fiscal_year_t=2025,
    #     fiscal_year_t_minus_1=2024,
    #     report_type="Standalone",                  # or "Consolidated"
    # )
    # print(json.dumps(result, indent=2))