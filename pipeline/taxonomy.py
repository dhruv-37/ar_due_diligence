"""
taxonomy.py  —  Context-Aware Financial Rules Engine: Upgrade 2
================================================================
Maps raw PDF-extracted line item strings to a fixed set of
Internal Taxonomy Nodes (ITNs).

Design principles
-----------------
* Nodes are named as a finance professional would name them,
  not as a programmer would (PROFIT_BEFORE_TAX, not PBT_CALC_NODE).
* Each node carries metadata: which statement it belongs to,
  whether it is a subtotal/total, and its sign convention
  (POSITIVE = adds to income/assets, NEGATIVE = deducts).
* Matching is three-tier:
    1. Exact match (lowercased, stripped)
    2. Fuzzy match via RapidFuzz token_sort_ratio (threshold = 82)
    3. Unmatched items are returned with node = UNRECOGNISED and
       logged so the next iteration can add them.
* The mapper is stateless and deterministic — given the same
  string it always returns the same node.

Usage
-----
    from taxonomy import map_line_item, TaxonomyNode, TAXONOMY

    node = map_line_item("Value of Services (Revenue)")
    print(node.name)          # REVENUE_GROSS
    print(node.statement)     # Profit and Loss
    print(node.is_total)      # False
    print(node.sign)          # POSITIVE
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import re

# ──────────────────────────────────────────────────────────────────────────────
# ENUMERATIONS
# ──────────────────────────────────────────────────────────────────────────────

class Statement(str, Enum):
    PROFIT_AND_LOSS    = "Profit and Loss"
    BALANCE_SHEET      = "Balance Sheet"
    CASH_FLOW          = "Cash Flow"
    CHANGES_IN_EQUITY  = "Changes in Equity"

class Sign(str, Enum):
    """
    POSITIVE  : item adds to the running total (income, asset, inflow)
    NEGATIVE  : item is deducted (expense, liability, outflow, contra)
    NEUTRAL   : item is a subtotal / total — sign depends on constituents
    """
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL  = "neutral"


# ──────────────────────────────────────────────────────────────────────────────
# TAXONOMY NODE DEFINITION
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TaxonomyNode:
    name: str                       # Internal node name (finance-friendly)
    statement: Statement            # Which financial statement
    section: str                    # Standard section heading
    sign: Sign                      # Sign convention
    is_total: bool = False          # True if this is a subtotal/grand total row
    is_contra: bool = False         # True if this reduces (e.g. Less: GST)
    consolidation_only: bool = False  # True if node only appears in Consolidated statements
    aliases: list[str] = field(default_factory=list)  # All known PDF string variants


# ──────────────────────────────────────────────────────────────────────────────
# THE MASTER TAXONOMY
# Organised by Statement → Section → Node
# Each alias list is built from REAL strings seen in the Reliance AR 2024 cache
# plus broad variants to cover other Indian companies.
# ──────────────────────────────────────────────────────────────────────────────

TAXONOMY: dict[str, TaxonomyNode] = {}

def _register(node: TaxonomyNode) -> TaxonomyNode:
    TAXONOMY[node.name] = node
    return node


# ── PROFIT AND LOSS ───────────────────────────────────────────────────────────

REVENUE_GROSS = _register(TaxonomyNode(
    name        = "REVENUE_GROSS",
    statement   = Statement.PROFIT_AND_LOSS,
    section     = "Income",
    sign        = Sign.POSITIVE,
    aliases     = [
        "value of services (revenue)",
        "value of services",
        "revenue from contracts with customers",
        "gross revenue from operations",
        "gross turnover",
        "turnover",
        "net sales",
        "sale of products",
        "sale of services",
        "income from operations",
    ]
))

REVENUE_GST_DEDUCTION = _register(TaxonomyNode(
    name        = "REVENUE_GST_DEDUCTION",
    statement   = Statement.PROFIT_AND_LOSS,
    section     = "Income",
    sign        = Sign.NEGATIVE,
    is_contra   = True,
    aliases     = [
        "less: gst recovered",
        "less: gst",
        "gst recovered",
        "goods and services tax",
        "indirect taxes recovered",
        "less: indirect taxes",
    ]
))

REVENUE_FROM_OPERATIONS = _register(TaxonomyNode(
    name        = "REVENUE_FROM_OPERATIONS",
    statement   = Statement.PROFIT_AND_LOSS,
    section     = "Income",
    sign        = Sign.POSITIVE,
    is_total    = False,
    aliases     = [
        "revenue from operations",
        "net revenue from operations",
        "total revenue from operations",
        "revenue from operations (net)",
        "revenue from operations (net of gst)",
        "total income from operations",
        "value of services",
        "turnover (net of indirect taxes)",
    ]
))

OTHER_INCOME = _register(TaxonomyNode(
    name        = "OTHER_INCOME",
    statement   = Statement.PROFIT_AND_LOSS,
    section     = "Income",
    sign        = Sign.POSITIVE,
    aliases     = [
        "other income",
        "non-operating income",
        "other operating income",
        "miscellaneous income",
        "interest and other income",
        "other revenues",
    ]
))

TOTAL_INCOME = _register(TaxonomyNode(
    name        = "TOTAL_INCOME",
    statement   = Statement.PROFIT_AND_LOSS,
    section     = "Income",
    sign        = Sign.POSITIVE,
    is_total    = True,
    aliases     = [
        "total income",
        "total revenue",
        "aggregate revenue",
        "total income from operations and other income",
        "total revenues",
        "net revenues",
        "gross income",
    ]
))

# ── Expenses ──

EMPLOYEE_BENEFITS_EXPENSE = _register(TaxonomyNode(
    name      = "EMPLOYEE_BENEFITS_EXPENSE",
    statement = Statement.PROFIT_AND_LOSS,
    section   = "Expenses",
    sign      = Sign.NEGATIVE,
    aliases   = [
        "employee benefits expense",
        "staff costs",
        "employee cost",
        "personnel expenses",
        "manpower cost",
        "salaries wages and bonus",
        "remuneration and benefits",
    ]
))

DEPRECIATION_AMORTISATION = _register(TaxonomyNode(
    name      = "DEPRECIATION_AMORTISATION",
    statement = Statement.PROFIT_AND_LOSS,
    section   = "Expenses",
    sign      = Sign.NEGATIVE,
    aliases   = [
        "depreciation / amortisation expense",
        "depreciation and amortisation expense",
        "depreciation amortisation and impairment",
        "depreciation",
        "depreciation and impairment",
        "amortisation of intangible assets",
        "depreciation on property plant and equipment",
    ]
))

OTHER_EXPENSES = _register(TaxonomyNode(
    name      = "OTHER_EXPENSES",
    statement = Statement.PROFIT_AND_LOSS,
    section   = "Expenses",
    sign      = Sign.NEGATIVE,
    aliases   = [
        "other expenses",
        "administrative and other expenses",
        "selling general and administrative expenses",
        "operating and other expenses",
        "miscellaneous expenses",
        "general and administrative expenses",
    ]
))

COST_OF_MATERIALS = _register(TaxonomyNode(
    name      = "COST_OF_MATERIALS",
    statement = Statement.PROFIT_AND_LOSS,
    section   = "Expenses",
    sign      = Sign.NEGATIVE,
    aliases   = [
        "cost of materials consumed",
        "raw materials consumed",
        "material costs",
        "cost of goods sold",
        "purchases of stock-in-trade",
        "cost of revenue",
    ]
))

FINANCE_COSTS = _register(TaxonomyNode(
    name      = "FINANCE_COSTS",
    statement = Statement.PROFIT_AND_LOSS,
    section   = "Expenses",
    sign      = Sign.NEGATIVE,
    aliases   = [
        "finance costs",
        "interest expense",
        "interest and finance charges",
        "borrowing costs",
        "interest on borrowings",
        "finance charges",
    ]
))

TOTAL_EXPENSES = _register(TaxonomyNode(
    name      = "TOTAL_EXPENSES",
    statement = Statement.PROFIT_AND_LOSS,
    section   = "Expenses",
    sign      = Sign.NEGATIVE,
    is_total  = True,
    aliases   = [
        "total expenses",
        "total operating expenses",
        "aggregate expenses",
        "total costs and expenses",
    ]
))

# ── Profit milestones ──

EBITDA = _register(TaxonomyNode(
    name      = "EBITDA",
    statement = Statement.PROFIT_AND_LOSS,
    section   = "Profit Milestones",
    sign      = Sign.POSITIVE,
    is_total  = True,
    aliases   = [
        "ebitda",
        "earnings before interest tax depreciation and amortisation",
        "operating profit",
        "profit before depreciation interest and tax",
    ]
))

SHARE_OF_PROFIT_OF_ASSOCIATES = _register(TaxonomyNode(
    name                 = "SHARE_OF_PROFIT_OF_ASSOCIATES",
    statement            = Statement.PROFIT_AND_LOSS,
    section              = "Profit Before Tax",
    sign                 = Sign.POSITIVE,
    consolidation_only   = True,
    aliases              = [
        "share of profit of associate",
        "share of profit/(loss) of associates",
        "share of profit of associates and joint ventures",
        "share of net profit of associates",
        "equity in earnings of associates",
        "share in profit of associate companies",
    ]
))

PROFIT_BEFORE_EXCEPTIONAL = _register(TaxonomyNode(
    name      = "PROFIT_BEFORE_EXCEPTIONAL",
    statement = Statement.PROFIT_AND_LOSS,
    section   = "Profit Before Tax",
    sign      = Sign.POSITIVE,
    is_total  = True,
    aliases   = [
        "profit before exceptional items and tax",
        "profit before share of profit of associates and tax",
        "profit before exceptional item and tax",
        "earnings before exceptional items and tax",
        "profit before extra-ordinary items and tax",
    ]
))

EXCEPTIONAL_ITEMS = _register(TaxonomyNode(
    name      = "EXCEPTIONAL_ITEMS",
    statement = Statement.PROFIT_AND_LOSS,
    section   = "Profit Before Tax",
    sign      = Sign.NEUTRAL,
    aliases   = [
        "exceptional items",
        "exceptional item",
        "extraordinary items",
        "special items",
    ]
))

PROFIT_BEFORE_TAX = _register(TaxonomyNode(
    name      = "PROFIT_BEFORE_TAX",
    statement = Statement.PROFIT_AND_LOSS,
    section   = "Profit Before Tax",
    sign      = Sign.POSITIVE,
    is_total  = True,
    aliases   = [
        "profit before tax",
        "profit/(loss) before tax",
        "income before income taxes",
        "profit before income tax",
        "earnings before tax",
        "pbt",
        "profit before taxation",
    ]
))

# ── Tax ──

CURRENT_TAX = _register(TaxonomyNode(
    name      = "CURRENT_TAX",
    statement = Statement.PROFIT_AND_LOSS,
    section   = "Tax Expenses",
    sign      = Sign.NEGATIVE,
    aliases   = [
        "current tax",
        "current income tax",
        "income tax - current",
        "current year tax",
    ]
))

DEFERRED_TAX = _register(TaxonomyNode(
    name      = "DEFERRED_TAX",
    statement = Statement.PROFIT_AND_LOSS,
    section   = "Tax Expenses",
    sign      = Sign.NEGATIVE,
    aliases   = [
        "deferred tax",
        "deferred income tax",
        "deferred tax expense/(benefit)",
        "deferred tax (credit)/charge",
    ]
))

PRIOR_PERIOD_TAX = _register(TaxonomyNode(
    name      = "PRIOR_PERIOD_TAX",
    statement = Statement.PROFIT_AND_LOSS,
    section   = "Tax Expenses",
    sign      = Sign.NEGATIVE,
    aliases   = [
        "provision for income tax of earlier years",
        "tax relating to earlier years",
        "prior period tax adjustment",
        "excess provision for tax written back",
        "income tax - earlier years",
    ]
))

TOTAL_TAX_EXPENSE = _register(TaxonomyNode(
    name      = "TOTAL_TAX_EXPENSE",
    statement = Statement.PROFIT_AND_LOSS,
    section   = "Tax Expenses",
    sign      = Sign.NEGATIVE,
    is_total  = True,
    aliases   = [
        "total tax expense",
        "tax expenses total",
        "tax expenses (total)",
        "total income tax expense",
        "aggregate tax expense",
    ]
))

# ── Bottom line ──

PROFIT_FOR_THE_YEAR = _register(TaxonomyNode(
    name      = "PROFIT_FOR_THE_YEAR",
    statement = Statement.PROFIT_AND_LOSS,
    section   = "Profit for the Year",
    sign      = Sign.POSITIVE,
    is_total  = True,
    aliases   = [
        "profit for the year",
        "profit/(loss) for the year",
        "profit after tax",
        "net profit for the year",
        "profit for the period",
        "net income",
        "net profit after tax",
        "pat",
    ]
))

# ── Other Comprehensive Income (OCI) ──

OCI_EQUITY_INVESTMENTS = _register(TaxonomyNode(
    name      = "OCI_EQUITY_INVESTMENTS",
    statement = Statement.PROFIT_AND_LOSS,
    section   = "Other Comprehensive Income",
    sign      = Sign.NEUTRAL,
    aliases   = [
        "equity investments through other comprehensive income",
        "items not reclassifiable to profit or loss: equity investments through other comprehensive income",
        "i) items not reclassifiable to profit or loss: equity investments through other comprehensive income",
        "i) items not reclassifiable to profit or loss - equity investments through other comprehensive income",
        "fair value changes on equity instruments",
    ]
))

OCI_REMEASUREMENT_DEFINED_BENEFIT = _register(TaxonomyNode(
    name      = "OCI_REMEASUREMENT_DEFINED_BENEFIT",
    statement = Statement.PROFIT_AND_LOSS,
    section   = "Other Comprehensive Income",
    sign      = Sign.NEUTRAL,
    aliases   = [
        "remeasurement of defined benefit plan",
        "items not reclassifiable to profit or loss: remeasurement of defined benefit plan",
        "i) items not reclassifiable to profit or loss: remeasurement of defined benefit plan",
        "i) items not reclassifiable to profit or loss - remeasurement of defined benefit plan",
        "remeasurements of defined benefit obligation",
        "actuarial gains/(losses) on defined benefit plans",
    ]
))

OCI_TAX_ON_NON_RECLASSIFIABLE = _register(TaxonomyNode(
    name      = "OCI_TAX_ON_NON_RECLASSIFIABLE",
    statement = Statement.PROFIT_AND_LOSS,
    section   = "Other Comprehensive Income",
    sign      = Sign.NEUTRAL,
    aliases   = [
        "ii) income tax relating to items not reclassifiable to profit or loss",
        "income tax relating to items not reclassifiable to profit or loss",
        "tax on items not reclassifiable to profit or loss",
        "income tax on non-reclassifiable oci items",
    ]
))

OCI_DEBT_INVESTMENTS = _register(TaxonomyNode(
    name      = "OCI_DEBT_INVESTMENTS",
    statement = Statement.PROFIT_AND_LOSS,
    section   = "Other Comprehensive Income",
    sign      = Sign.NEUTRAL,
    aliases   = [
        "iii) items reclassifiable to profit or loss debt investments through other comprehensive income",
        "items reclassifiable to profit or loss debt investments through other comprehensive income",
        "iii) items reclassifiable to profit or loss - debt investments through other comprehensive income",
        "debt investments through other comprehensive income",
        "fair value changes on debt instruments",
        "fair value changes on debt investments through oci",
    ]
))

OCI_TAX_ON_RECLASSIFIABLE = _register(TaxonomyNode(
    name      = "OCI_TAX_ON_RECLASSIFIABLE",
    statement = Statement.PROFIT_AND_LOSS,
    section   = "Other Comprehensive Income",
    sign      = Sign.NEUTRAL,
    aliases   = [
        "iv) income tax relating to items reclassifiable to profit or loss",
        "income tax relating to items reclassifiable to profit or loss",
        "tax on items reclassifiable to profit or loss",
        "income tax on reclassifiable oci items",
    ]
))

TOTAL_OCI = _register(TaxonomyNode(
    name      = "TOTAL_OCI",
    statement = Statement.PROFIT_AND_LOSS,
    section   = "Other Comprehensive Income",
    sign      = Sign.NEUTRAL,
    is_total  = True,
    aliases   = [
        "total other comprehensive income/ (loss) for the year (net of tax)",
        "total other comprehensive income for the year (net of tax)",
        "total other comprehensive income (net of tax)",
        "other comprehensive income for the year",
        "total oci",
        "net other comprehensive income",
    ]
))

TOTAL_COMPREHENSIVE_INCOME = _register(TaxonomyNode(
    name      = "TOTAL_COMPREHENSIVE_INCOME",
    statement = Statement.PROFIT_AND_LOSS,
    section   = "Total Comprehensive Income",
    sign      = Sign.POSITIVE,
    is_total  = True,
    aliases   = [
        "total comprehensive income/ (loss) for the year",
        "total comprehensive income for the year",
        "total comprehensive income/(loss) for the year",
        "total comprehensive income",
        "total comprehensive income / (loss)",
    ]
))

EARNINGS_PER_SHARE = _register(TaxonomyNode(
    name      = "EARNINGS_PER_SHARE",
    statement = Statement.PROFIT_AND_LOSS,
    section   = "Earnings per Share",
    sign      = Sign.POSITIVE,
    aliases   = [
        "basic and diluted (in ₹)",
        "earnings per equity share of face value of ₹10 each - basic and diluted (in ₹)",
        "earnings per share - basic and diluted",
        "basic eps",
        "diluted eps",
        "basic and diluted eps",
        "earnings per equity share (basic and diluted)",
        "basic earnings per share",
        "diluted earnings per share",
        # Currency-symbol-agnostic variants: PDF extraction (PyMuPDF / Gemini
        # round-trip) sometimes mangles the ₹ glyph into a backtick or other
        # stray character, e.g. "Basic and Diluted (in `)" instead of "(in ₹)".
        # These short, symbol-free forms still match via substring fallback
        # in get_row_by_node / get_rows_by_node regardless of what character
        # sits where the currency symbol should be.
        "basic and diluted",
        "basic and diluted (in",
    ]
))


# ── BALANCE SHEET — Assets ────────────────────────────────────────────────────

PROPERTY_PLANT_EQUIPMENT = _register(TaxonomyNode(
    name      = "PROPERTY_PLANT_EQUIPMENT",
    statement = Statement.BALANCE_SHEET,
    section   = "Non-Current Assets",
    sign      = Sign.POSITIVE,
    aliases   = [
        "property, plant and equipment",
        "property plant and equipment",
        "tangible assets",
        "fixed assets",
        "plant and machinery",
        "ppe",
    ]
))

NON_CURRENT_INVESTMENTS = _register(TaxonomyNode(
    name      = "NON_CURRENT_INVESTMENTS",
    statement = Statement.BALANCE_SHEET,
    section   = "Non-Current Assets",
    sign      = Sign.POSITIVE,
    aliases   = [
        "financial assets - investments",
        "non-current investments",
        "long-term investments",
        "investments (non-current)",
        "financial assets: investments",
    ]
))

NON_CURRENT_FINANCIAL_ASSETS_OTHER = _register(TaxonomyNode(
    name      = "NON_CURRENT_FINANCIAL_ASSETS_OTHER",
    statement = Statement.BALANCE_SHEET,
    section   = "Non-Current Assets",
    sign      = Sign.POSITIVE,
    aliases   = [
        "financial assets - other financial assets",
        "other financial assets (non-current)",
        "other non-current financial assets",
        "financial assets: other financial assets",
    ]
))

OTHER_NON_CURRENT_ASSETS = _register(TaxonomyNode(
    name      = "OTHER_NON_CURRENT_ASSETS",
    statement = Statement.BALANCE_SHEET,
    section   = "Non-Current Assets",
    sign      = Sign.POSITIVE,
    aliases   = [
        "other non-current assets",
        "other assets (non-current)",
        "non-current assets others",
    ]
))

TOTAL_NON_CURRENT_ASSETS = _register(TaxonomyNode(
    name      = "TOTAL_NON_CURRENT_ASSETS",
    statement = Statement.BALANCE_SHEET,
    section   = "Non-Current Assets",
    sign      = Sign.POSITIVE,
    is_total  = True,
    aliases   = [
        "total non-current assets",
        "total of non-current assets",
        "non-current assets total",
    ]
))

INVENTORIES = _register(TaxonomyNode(
    name      = "INVENTORIES",
    statement = Statement.BALANCE_SHEET,
    section   = "Current Assets",
    sign      = Sign.POSITIVE,
    aliases   = [
        "inventories",
        "stocks",
        "inventory",
        "stock in trade",
        "finished goods inventories",
    ]
))

CURRENT_INVESTMENTS = _register(TaxonomyNode(
    name      = "CURRENT_INVESTMENTS",
    statement = Statement.BALANCE_SHEET,
    section   = "Current Assets",
    sign      = Sign.POSITIVE,
    aliases   = [
        "financial assets - investments",
        "current investments",
        "short-term investments",
        "investments (current)",
        "financial assets: investments",
    ]
))

TRADE_RECEIVABLES = _register(TaxonomyNode(
    name      = "TRADE_RECEIVABLES",
    statement = Statement.BALANCE_SHEET,
    section   = "Current Assets",
    sign      = Sign.POSITIVE,
    aliases   = [
        "financial assets - trade receivables",
        "trade receivables",
        "sundry debtors",
        "debtors",
        "accounts receivable",
        "trade debtors",
    ]
))

CASH_AND_CASH_EQUIVALENTS = _register(TaxonomyNode(
    name      = "CASH_AND_CASH_EQUIVALENTS",
    statement = Statement.BALANCE_SHEET,
    section   = "Current Assets",
    sign      = Sign.POSITIVE,
    aliases   = [
        "financial assets - cash and cash equivalents",
        "cash and cash equivalents",
        "cash and bank balances",
        "cash at bank and in hand",
    ]
))

CURRENT_FINANCIAL_ASSETS_OTHER = _register(TaxonomyNode(
    name      = "CURRENT_FINANCIAL_ASSETS_OTHER",
    statement = Statement.BALANCE_SHEET,
    section   = "Current Assets",
    sign      = Sign.POSITIVE,
    aliases   = [
        "financial assets - other financial assets",
        "other financial assets (current)",
        "other current financial assets",
    ]
))

OTHER_CURRENT_ASSETS = _register(TaxonomyNode(
    name      = "OTHER_CURRENT_ASSETS",
    statement = Statement.BALANCE_SHEET,
    section   = "Current Assets",
    sign      = Sign.POSITIVE,
    aliases   = [
        "other current assets",
        "other assets (current)",
        "current assets others",
    ]
))

TOTAL_CURRENT_ASSETS = _register(TaxonomyNode(
    name      = "TOTAL_CURRENT_ASSETS",
    statement = Statement.BALANCE_SHEET,
    section   = "Current Assets",
    sign      = Sign.POSITIVE,
    is_total  = True,
    aliases   = [
        "total current assets",
        "current assets total",
        "total of current assets",
    ]
))

TOTAL_ASSETS = _register(TaxonomyNode(
    name      = "TOTAL_ASSETS",
    statement = Statement.BALANCE_SHEET,
    section   = "Total Assets",
    sign      = Sign.POSITIVE,
    is_total  = True,
    aliases   = [
        "total assets",
        "aggregate assets",
        "total of assets",
    ]
))

# ── Balance Sheet — Equity & Liabilities ──

EQUITY_SHARE_CAPITAL = _register(TaxonomyNode(
    name      = "EQUITY_SHARE_CAPITAL",
    statement = Statement.BALANCE_SHEET,
    section   = "Equity",
    sign      = Sign.POSITIVE,
    aliases   = [
        "equity share capital",
        "share capital",
        "paid-up equity share capital",
        "issued subscribed and paid up",
    ]
))

OTHER_EQUITY = _register(TaxonomyNode(
    name      = "OTHER_EQUITY",
    statement = Statement.BALANCE_SHEET,
    section   = "Equity",
    sign      = Sign.POSITIVE,
    aliases   = [
        "other equity",
        "reserves and surplus",
        "shareholders funds excluding share capital",
        "total reserves",
    ]
))

TOTAL_EQUITY = _register(TaxonomyNode(
    name      = "TOTAL_EQUITY",
    statement = Statement.BALANCE_SHEET,
    section   = "Equity",
    sign      = Sign.POSITIVE,
    is_total  = True,
    aliases   = [
        "total equity",
        "total shareholders equity",
        "shareholders funds",
        "net worth",
    ]
))

DEFERRED_TAX_LIABILITIES = _register(TaxonomyNode(
    name      = "DEFERRED_TAX_LIABILITIES",
    statement = Statement.BALANCE_SHEET,
    section   = "Non-Current Liabilities",
    sign      = Sign.POSITIVE,
    aliases   = [
        "deferred tax liabilities (net)",
        "deferred tax liability",
        "net deferred tax liabilities",
        "deferred tax liabilities net",
    ]
))

TOTAL_NON_CURRENT_LIABILITIES = _register(TaxonomyNode(
    name      = "TOTAL_NON_CURRENT_LIABILITIES",
    statement = Statement.BALANCE_SHEET,
    section   = "Non-Current Liabilities",
    sign      = Sign.POSITIVE,
    is_total  = True,
    aliases   = [
        "total non-current liabilities",
        "non-current liabilities total",
    ]
))

TRADE_PAYABLES_MSME = _register(TaxonomyNode(
    name      = "TRADE_PAYABLES_MSME",
    statement = Statement.BALANCE_SHEET,
    section   = "Current Liabilities",
    sign      = Sign.POSITIVE,
    aliases   = [
        "financial liabilities - trade payables due to: micro and small enterprises",
        "trade payables - micro and small enterprises",
        "trade payables due to msme",
        "dues to micro and small enterprises",
    ]
))

TRADE_PAYABLES_OTHERS = _register(TaxonomyNode(
    name      = "TRADE_PAYABLES_OTHERS",
    statement = Statement.BALANCE_SHEET,
    section   = "Current Liabilities",
    sign      = Sign.POSITIVE,
    aliases   = [
        "financial liabilities - trade payables due to: other than micro and small enterprises",
        "trade payables - other than micro and small enterprises",
        "trade payables - others",
        "trade payables due to other creditors",
        "other trade payables",
    ]
))

OTHER_FINANCIAL_LIABILITIES = _register(TaxonomyNode(
    name      = "OTHER_FINANCIAL_LIABILITIES",
    statement = Statement.BALANCE_SHEET,
    section   = "Current Liabilities",
    sign      = Sign.POSITIVE,
    aliases   = [
        "other financial liabilities",
        "other current financial liabilities",
        "current financial liabilities others",
    ]
))

OTHER_CURRENT_LIABILITIES = _register(TaxonomyNode(
    name      = "OTHER_CURRENT_LIABILITIES",
    statement = Statement.BALANCE_SHEET,
    section   = "Current Liabilities",
    sign      = Sign.POSITIVE,
    aliases   = [
        "other current liabilities",
        "current liabilities others",
    ]
))

PROVISIONS_CURRENT = _register(TaxonomyNode(
    name      = "PROVISIONS_CURRENT",
    statement = Statement.BALANCE_SHEET,
    section   = "Current Liabilities",
    sign      = Sign.POSITIVE,
    aliases   = [
        "provisions",
        "current provisions",
        "provision for employee benefits",
        "provisions (current)",
    ]
))

TOTAL_CURRENT_LIABILITIES = _register(TaxonomyNode(
    name      = "TOTAL_CURRENT_LIABILITIES",
    statement = Statement.BALANCE_SHEET,
    section   = "Current Liabilities",
    sign      = Sign.POSITIVE,
    is_total  = True,
    aliases   = [
        "total current liabilities",
        "current liabilities total",
    ]
))

TOTAL_LIABILITIES = _register(TaxonomyNode(
    name      = "TOTAL_LIABILITIES",
    statement = Statement.BALANCE_SHEET,
    section   = "Total Liabilities",
    sign      = Sign.POSITIVE,
    is_total  = True,
    aliases   = [
        "total liabilities",
        "aggregate liabilities",
    ]
))

TOTAL_EQUITY_AND_LIABILITIES = _register(TaxonomyNode(
    name      = "TOTAL_EQUITY_AND_LIABILITIES",
    statement = Statement.BALANCE_SHEET,
    section   = "Total Equity and Liabilities",
    sign      = Sign.POSITIVE,
    is_total  = True,
    aliases   = [
        "total equity and liabilities",
        "total of equity and liabilities",
        "balance sheet total",
    ]
))


# ── CASH FLOW STATEMENT ───────────────────────────────────────────────────────

CFO_NET_PROFIT_BEFORE_TAX = _register(TaxonomyNode(
    name      = "CFO_NET_PROFIT_BEFORE_TAX",
    statement = Statement.CASH_FLOW,
    section   = "Cash Flow from Operating Activities",
    sign      = Sign.POSITIVE,
    aliases   = [
        "net profit before tax as per statement of profit and loss",
        "profit before tax (cash flow)",
        "net profit before taxation",
        "net income before taxes",
    ]
))

CFO_DEPRECIATION_ADDBACK = _register(TaxonomyNode(
    name      = "CFO_DEPRECIATION_ADDBACK",
    statement = Statement.CASH_FLOW,
    section   = "Cash Flow from Operating Activities",
    sign      = Sign.POSITIVE,
    aliases   = [
        "adjusted for: depreciation / amortisation expense",
        "add: depreciation and amortisation",
        "depreciation and amortisation (cash flow)",
        "adjusted for: depreciation and amortisation expense",
    ]
))

CFO_GAIN_ON_DISPOSAL_PPE = _register(TaxonomyNode(
    name      = "CFO_GAIN_ON_DISPOSAL_PPE",
    statement = Statement.CASH_FLOW,
    section   = "Cash Flow from Operating Activities",
    sign      = Sign.NEGATIVE,
    aliases   = [
        "adjusted for: net gain on disposal/ sale of property, plant and equipments",
        "less: profit on sale of fixed assets",
        "net gain on disposal of ppe",
        "gain on sale of property plant and equipment",
    ]
))

CFO_GAIN_ON_FINANCIAL_ASSETS = _register(TaxonomyNode(
    name      = "CFO_GAIN_ON_FINANCIAL_ASSETS",
    statement = Statement.CASH_FLOW,
    section   = "Cash Flow from Operating Activities",
    sign      = Sign.NEGATIVE,
    aliases   = [
        "adjusted for: net gain on financial assets",
        "net gain on financial assets",
        "gain/(loss) on financial assets",
        "net profit on sale of investments",
    ]
))

CFO_INTEREST_INCOME = _register(TaxonomyNode(
    name      = "CFO_INTEREST_INCOME",
    statement = Statement.CASH_FLOW,
    section   = "Cash Flow from Operating Activities",
    sign      = Sign.NEGATIVE,
    aliases   = [
        "adjusted for: interest income",
        "less: interest income (cash flow)",
        "interest income (operating adjustment)",
    ]
))

CFO_DIVIDEND_INCOME = _register(TaxonomyNode(
    name      = "CFO_DIVIDEND_INCOME",
    statement = Statement.CASH_FLOW,
    section   = "Cash Flow from Operating Activities",
    sign      = Sign.NEGATIVE,
    aliases   = [
        "adjusted for: dividend income",
        "less: dividend income (cash flow)",
        "dividend income (operating adjustment)",
    ]
))

CFO_SHARE_IN_ASSOCIATE = _register(TaxonomyNode(
    name                 = "CFO_SHARE_IN_ASSOCIATE",
    statement            = Statement.CASH_FLOW,
    section              = "Cash Flow from Operating Activities",
    sign                 = Sign.NEUTRAL,
    consolidation_only   = True,
    aliases              = [
        "adjusted for: share in (profit)/loss of associate",
        "share of (profit)/loss in associate (cash flow)",
        "undistributed earnings of associates",
    ]
))

CFO_NON_CASH_ADJUSTMENTS_SUBTOTAL = _register(TaxonomyNode(
    name      = "CFO_NON_CASH_ADJUSTMENTS_SUBTOTAL",
    statement = Statement.CASH_FLOW,
    section   = "Cash Flow from Operating Activities",
    sign      = Sign.NEUTRAL,
    is_total  = True,
    aliases   = [
        "total adjustments for non-cash and non-operating items",
        "adjusted for: (subtotal)",
        "total non-cash adjustments",
    ]
))

CFO_OPERATING_PROFIT_BEFORE_WC = _register(TaxonomyNode(
    name      = "CFO_OPERATING_PROFIT_BEFORE_WC",
    statement = Statement.CASH_FLOW,
    section   = "Cash Flow from Operating Activities",
    sign      = Sign.POSITIVE,
    is_total  = True,
    aliases   = [
        "operating profit before working capital changes",
        "profit before working capital changes",
        "cash profit before working capital adjustments",
    ]
))

CFO_WC_TRADE_RECEIVABLES = _register(TaxonomyNode(
    name      = "CFO_WC_TRADE_RECEIVABLES",
    statement = Statement.CASH_FLOW,
    section   = "Cash Flow from Operating Activities",
    sign      = Sign.NEUTRAL,
    aliases   = [
        "adjusted for: trade and other receivables",
        "decrease/(increase) in trade receivables",
        "movement in trade receivables",
    ]
))

CFO_WC_INVENTORIES = _register(TaxonomyNode(
    name      = "CFO_WC_INVENTORIES",
    statement = Statement.CASH_FLOW,
    section   = "Cash Flow from Operating Activities",
    sign      = Sign.NEUTRAL,
    aliases   = [
        "adjusted for: inventories",
        "decrease/(increase) in inventories",
        "movement in inventories",
    ]
))

CFO_WC_TRADE_PAYABLES = _register(TaxonomyNode(
    name      = "CFO_WC_TRADE_PAYABLES",
    statement = Statement.CASH_FLOW,
    section   = "Cash Flow from Operating Activities",
    sign      = Sign.NEUTRAL,
    aliases   = [
        "adjusted for: trade and other payables",
        "increase/(decrease) in trade payables",
        "movement in trade payables",
    ]
))

CFO_WC_ADJUSTMENTS_SUBTOTAL = _register(TaxonomyNode(
    name      = "CFO_WC_ADJUSTMENTS_SUBTOTAL",
    statement = Statement.CASH_FLOW,
    section   = "Cash Flow from Operating Activities",
    sign      = Sign.NEUTRAL,
    is_total  = True,
    aliases   = [
        "total adjustments for working capital changes",
        "adjusted for: (subtotal)",
        "total working capital changes",
    ]
))

CFO_CASH_GENERATED_FROM_OPERATIONS = _register(TaxonomyNode(
    name      = "CFO_CASH_GENERATED_FROM_OPERATIONS",
    statement = Statement.CASH_FLOW,
    section   = "Cash Flow from Operating Activities",
    sign      = Sign.POSITIVE,
    is_total  = True,
    aliases   = [
        "cash generated from/ (used in) operations",
        "cash generated from operations",
        "cash from operations before tax",
        "cash inflow/(outflow) from operations",
    ]
))

CFO_TAXES_PAID = _register(TaxonomyNode(
    name      = "CFO_TAXES_PAID",
    statement = Statement.CASH_FLOW,
    section   = "Cash Flow from Operating Activities",
    sign      = Sign.NEGATIVE,
    aliases   = [
        "taxes paid (net)",
        "income taxes paid (net)",
        "direct taxes paid",
        "net taxes paid",
    ]
))

NET_CASH_FROM_OPERATING = _register(TaxonomyNode(
    name      = "NET_CASH_FROM_OPERATING",
    statement = Statement.CASH_FLOW,
    section   = "Cash Flow from Operating Activities",
    sign      = Sign.POSITIVE,
    is_total  = True,
    aliases   = [
        "net cash flow from / (used in) operating activities *",
        "net cash flow from operating activities",
        "net cash from/(used in) operating activities",
        "net cash generated from operating activities",
        "cash flow from operations (net)",
        # Punctuation-agnostic core phrase — real extractions vary spacing
        # around slashes and append footnote markers (e.g. "from / (used
        # in)" vs "from/(used in)" vs no parenthetical at all), which broke
        # every alias above against an actual extracted Cash Flow sheet.
        # Must include "operating" explicitly — a shorter "net cash flow
        # from" alone also substring-matches the Investing and Financing
        # total rows, which share the same "Net Cash Flow from/(used in)
        # ... Activities" phrasing.
        "net cash flow from / (used in) operating",
        "net cash flow from/ (used in) operating",
        "net cash flow from operating",
    ]
))

CFI_PROCEEDS_DISPOSAL_PPE = _register(TaxonomyNode(
    name      = "CFI_PROCEEDS_DISPOSAL_PPE",
    statement = Statement.CASH_FLOW,
    section   = "Cash Flow from Investing Activities",
    sign      = Sign.POSITIVE,
    aliases   = [
        "proceeds from disposal of property, plant and equipment",
        "sale proceeds of fixed assets",
        "proceeds from sale of ppe",
    ]
))

CFI_PURCHASE_OF_INVESTMENTS = _register(TaxonomyNode(
    name      = "CFI_PURCHASE_OF_INVESTMENTS",
    statement = Statement.CASH_FLOW,
    section   = "Cash Flow from Investing Activities",
    sign      = Sign.NEGATIVE,
    aliases   = [
        "purchase of investments",
        "acquisition of investments",
        "investment in securities",
        "purchase of financial instruments",
    ]
))

CFI_PROCEEDS_SALE_INVESTMENTS = _register(TaxonomyNode(
    name      = "CFI_PROCEEDS_SALE_INVESTMENTS",
    statement = Statement.CASH_FLOW,
    section   = "Cash Flow from Investing Activities",
    sign      = Sign.POSITIVE,
    aliases   = [
        "proceeds from sale of investments",
        "sale of investments",
        "redemption of investments",
    ]
))

CFI_FIXED_DEPOSITS = _register(TaxonomyNode(
    name      = "CFI_FIXED_DEPOSITS",
    statement = Statement.CASH_FLOW,
    section   = "Cash Flow from Investing Activities",
    sign      = Sign.NEUTRAL,
    aliases   = [
        "investment in fixed deposits",
        "fixed deposits (net)",
        "net movement in fixed deposits",
        "placement of fixed deposits",
    ]
))

CFI_INTEREST_RECEIVED = _register(TaxonomyNode(
    name      = "CFI_INTEREST_RECEIVED",
    statement = Statement.CASH_FLOW,
    section   = "Cash Flow from Investing Activities",
    sign      = Sign.POSITIVE,
    aliases   = [
        "interest received",
        "interest income received",
        "interest received on deposits",
    ]
))

CFI_DIVIDEND_INCOME = _register(TaxonomyNode(
    name      = "CFI_DIVIDEND_INCOME",
    statement = Statement.CASH_FLOW,
    section   = "Cash Flow from Investing Activities",
    sign      = Sign.POSITIVE,
    aliases   = [
        "dividend income",
        "dividends received",
        "dividend received from investments",
    ]
))

NET_CASH_FROM_INVESTING = _register(TaxonomyNode(
    name      = "NET_CASH_FROM_INVESTING",
    statement = Statement.CASH_FLOW,
    section   = "Cash Flow from Investing Activities",
    sign      = Sign.NEUTRAL,
    is_total  = True,
    aliases   = [
        "net cash flow from/ (used in) investing activities",
        "net cash from/(used in) investing activities",
        "net cash generated from/(used in) investing activities",
    ]
))

CFF_DIVIDEND_PAID = _register(TaxonomyNode(
    name      = "CFF_DIVIDEND_PAID",
    statement = Statement.CASH_FLOW,
    section   = "Cash Flow from Financing Activities",
    sign      = Sign.NEGATIVE,
    aliases   = [
        "dividend paid",
        "dividends paid to shareholders",
        "payment of dividends",
    ]
))

NET_CASH_FROM_FINANCING = _register(TaxonomyNode(
    name      = "NET_CASH_FROM_FINANCING",
    statement = Statement.CASH_FLOW,
    section   = "Cash Flow from Financing Activities",
    sign      = Sign.NEUTRAL,
    is_total  = True,
    aliases   = [
        "net cash flow from/ (used in) financing activities",
        "net cash from/(used in) financing activities",
        "net cash flow from financing activities",
    ]
))

NET_CHANGE_IN_CASH = _register(TaxonomyNode(
    name      = "NET_CHANGE_IN_CASH",
    statement = Statement.CASH_FLOW,
    section   = "Cash and Cash Equivalents",
    sign      = Sign.NEUTRAL,
    is_total  = True,
    aliases   = [
        "net (decrease) / increase in cash and cash equivalents",
        "net (decrease)/ increase in cash and cash equivalents",
        "net increase/(decrease) in cash and cash equivalents",
        "net change in cash and cash equivalents",
    ]
))

OPENING_CASH_BALANCE = _register(TaxonomyNode(
    name      = "OPENING_CASH_BALANCE",
    statement = Statement.CASH_FLOW,
    section   = "Cash and Cash Equivalents",
    sign      = Sign.POSITIVE,
    aliases   = [
        "opening balance of cash and cash equivalents",
        "cash and cash equivalents at the beginning of the year",
        "opening cash balance",
    ]
))

CLOSING_CASH_BALANCE = _register(TaxonomyNode(
    name      = "CLOSING_CASH_BALANCE",
    statement = Statement.CASH_FLOW,
    section   = "Cash and Cash Equivalents",
    sign      = Sign.POSITIVE,
    is_total  = True,
    aliases   = [
        "closing balance of cash and cash equivalents",
        "cash and cash equivalents at the end of the year",
        "closing cash balance",
        "cash and cash equivalents at year end",
    ]
))


# ── STATEMENT OF CHANGES IN EQUITY ───────────────────────────────────────────

EQUITY_SHARE_CAPITAL_OPENING = _register(TaxonomyNode(
    name      = "EQUITY_SHARE_CAPITAL_OPENING",
    statement = Statement.CHANGES_IN_EQUITY,
    section   = "Equity Share Capital",
    sign      = Sign.POSITIVE,
    aliases   = [
        "balance as at 1st april, 2023",
        "equity share capital - balance as at 1st april",
        "opening balance of equity share capital",
        "balance as at beginning of year",
    ]
))

EQUITY_SHARE_CAPITAL_CHANGES = _register(TaxonomyNode(
    name      = "EQUITY_SHARE_CAPITAL_CHANGES",
    statement = Statement.CHANGES_IN_EQUITY,
    section   = "Equity Share Capital",
    sign      = Sign.NEUTRAL,
    aliases   = [
        "changes during the year 2023-24",
        "changes during the year 2024-25",
        "equity share capital - changes during the year",
        "changes during the year",
    ]
))

EQUITY_SHARE_CAPITAL_CLOSING = _register(TaxonomyNode(
    name      = "EQUITY_SHARE_CAPITAL_CLOSING",
    statement = Statement.CHANGES_IN_EQUITY,
    section   = "Equity Share Capital",
    sign      = Sign.POSITIVE,
    is_total  = True,
    aliases   = [
        "balance as at 31st march, 2024",
        "balance as at 31st march, 2025",
        "equity share capital - balance as at 31st march",
        "closing balance of equity share capital",
    ]
))

RESERVE_OPENING_BALANCE = _register(TaxonomyNode(
    name      = "RESERVE_OPENING_BALANCE",
    statement = Statement.CHANGES_IN_EQUITY,
    section   = "Other Equity - Reserves and Surplus",
    sign      = Sign.POSITIVE,
    aliases   = [
        "capital reserve - balance as at 1st april",
        "securities premium - balance as at 1st april",
        "general reserve - balance as at 1st april",
        "retained earnings - balance as at 1st april",
        "other comprehensive income (oci) - balance as at 1st april",
        "total - balance as at 1st april",
    ]
))

RESERVE_COMPREHENSIVE_INCOME = _register(TaxonomyNode(
    name      = "RESERVE_COMPREHENSIVE_INCOME",
    statement = Statement.CHANGES_IN_EQUITY,
    section   = "Other Equity - Reserves and Surplus",
    sign      = Sign.POSITIVE,
    aliases   = [
        "capital reserve - total comprehensive income for the year",
        "securities premium - total comprehensive income for the year",
        "general reserve - total comprehensive income for the year",
        "retained earnings - total comprehensive income for the year",
        "other comprehensive income (oci) - total comprehensive income for the year",
        "total - total comprehensive income for the year",
    ]
))

RESERVE_DIVIDEND = _register(TaxonomyNode(
    name      = "RESERVE_DIVIDEND",
    statement = Statement.CHANGES_IN_EQUITY,
    section   = "Other Equity - Reserves and Surplus",
    sign      = Sign.NEGATIVE,
    aliases   = [
        "capital reserve - dividend",
        "securities premium - dividend",
        "general reserve - dividend",
        "retained earnings - dividend",
        "other comprehensive income (oci) - dividend",
        "total - dividend",
    ]
))

RESERVE_TRANSFER = _register(TaxonomyNode(
    name      = "RESERVE_TRANSFER",
    statement = Statement.CHANGES_IN_EQUITY,
    section   = "Other Equity - Reserves and Surplus",
    sign      = Sign.NEUTRAL,
    aliases   = [
        "capital reserve - transfer to / (from) retained earnings",
        "securities premium - transfer to / (from) retained earnings",
        "general reserve - transfer to / (from) retained earnings",
        "retained earnings - transfer to / (from) retained earnings",
        "other comprehensive income (oci) - transfer to / (from) retained earnings",
        "total - transfer to / (from) retained earnings",
    ]
))

RESERVE_CLOSING_BALANCE = _register(TaxonomyNode(
    name      = "RESERVE_CLOSING_BALANCE",
    statement = Statement.CHANGES_IN_EQUITY,
    section   = "Other Equity - Reserves and Surplus",
    sign      = Sign.POSITIVE,
    is_total  = True,
    aliases   = [
        "capital reserve - balance as at 31st march",
        "securities premium - balance as at 31st march",
        "general reserve - balance as at 31st march",
        "retained earnings - balance as at 31st march",
        "other comprehensive income (oci) - balance as at 31st march",
        "total - balance as at 31st march",
    ]
))


# ──────────────────────────────────────────────────────────────────────────────
# MATCHER
# ──────────────────────────────────────────────────────────────────────────────

# Build a flat lookup: lowercased alias → node name
_ALIAS_INDEX: dict[str, str] = {}
for _node in TAXONOMY.values():
    for _alias in _node.aliases:
        _ALIAS_INDEX[_alias.lower().strip()] = _node.name

# Sentinel for unrecognised items
UNRECOGNISED = TaxonomyNode(
    name      = "UNRECOGNISED",
    statement = Statement.PROFIT_AND_LOSS,  # placeholder
    section   = "Unrecognised",
    sign      = Sign.NEUTRAL,
)


def _normalise(text: str) -> str:
    """Lowercase, strip, collapse multiple spaces, remove special punctuation noise."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    # Remove common PDF artefacts: asterisks, trailing numbers, Roman numerals at start
    text = re.sub(r'^\s*(i{1,3}v?|v?i{0,3})\)\s*', '', text)  # Roman numeral prefix
    text = re.sub(r'\*+$', '', text).strip()
    return text


def map_line_item(raw_string: str, fuzzy_threshold: int = 82) -> TaxonomyNode:
    """
    Maps a raw PDF line item string to its Internal Taxonomy Node.

    Parameters
    ----------
    raw_string       : The extracted string from the PDF / cache.
    fuzzy_threshold  : RapidFuzz token_sort_ratio minimum score (0–100).
                       82 is tight enough to avoid false positives while
                       still catching minor formatting differences.

    Returns
    -------
    TaxonomyNode     : The matched node, or UNRECOGNISED if no match found.
    """
    normalised = _normalise(raw_string)

    # ── Tier 1: exact match ──
    if normalised in _ALIAS_INDEX:
        return TAXONOMY[_ALIAS_INDEX[normalised]]

    # ── Tier 2: fuzzy match ──
    try:
        from rapidfuzz import process, fuzz
        best_match, score, _ = process.extractOne(
            normalised,
            _ALIAS_INDEX.keys(),
            scorer=fuzz.token_sort_ratio
        )
        if score >= fuzzy_threshold:
            return TAXONOMY[_ALIAS_INDEX[best_match]]
    except ImportError:
        pass  # rapidfuzz not installed — fall through to UNRECOGNISED

    # ── Tier 3: unrecognised ──
    return UNRECOGNISED


def map_dataframe(df, line_item_col: str = "line_item") -> "pd.DataFrame":
    """
    Applies map_line_item across an entire DataFrame and appends
    taxonomy columns: taxonomy_node, taxonomy_section, is_total,
    sign, consolidation_only, match_status.

    Parameters
    ----------
    df             : pandas DataFrame with at least a line_item column.
    line_item_col  : Name of the column containing raw PDF strings.

    Returns
    -------
    DataFrame with 6 new columns appended.
    """
    import pandas as pd

    nodes = df[line_item_col].apply(map_line_item)

    df = df.copy()
    df["taxonomy_node"]        = nodes.apply(lambda n: n.name)
    df["taxonomy_section"]     = nodes.apply(lambda n: n.section)
    df["is_total"]             = nodes.apply(lambda n: n.is_total)
    df["sign"]                 = nodes.apply(lambda n: n.sign.value)
    df["consolidation_only"]   = nodes.apply(lambda n: n.consolidation_only)
    df["match_status"]         = nodes.apply(
        lambda n: "EXACT_OR_FUZZY" if n.name != "UNRECOGNISED" else "UNRECOGNISED"
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# VALIDATION UTILITY  —  run this directly to test against real cache data
# ──────────────────────────────────────────────────────────────────────────────

def validate_against_cache(cache_path: str) -> None:
    """
    Loads financials_cache.json, runs every line_item through the mapper,
    and prints a clean report showing matches and unrecognised items.
    """
    import json, textwrap

    with open(cache_path, "r") as f:
        data = json.load(f)

    matched      = []
    unrecognised = []

    seen = set()
    for record in data:
        raw = record.get("line_item", "")
        if raw in seen:
            continue
        seen.add(raw)

        node = map_line_item(raw)
        entry = {
            "raw"       : raw,
            "statement" : record.get("statement", ""),
            "node"      : node.name,
            "section"   : node.section,
            "is_total"  : node.is_total,
            "sign"      : node.sign.value,
        }
        if node.name == "UNRECOGNISED":
            unrecognised.append(entry)
        else:
            matched.append(entry)

    total = len(matched) + len(unrecognised)
    pct   = round(len(matched) / total * 100, 1) if total else 0

    print("=" * 72)
    print(f"  TAXONOMY VALIDATION REPORT")
    print(f"  Cache : {cache_path}")
    print(f"  Total unique line items : {total}")
    print(f"  Matched                 : {len(matched)}  ({pct}%)")
    print(f"  Unrecognised            : {len(unrecognised)}")
    print("=" * 72)

    if matched:
        print("\n✅  MATCHED\n")
        print(f"  {'RAW STRING':<60} {'NODE':<40} {'TOTAL?'}")
        print(f"  {'-'*60} {'-'*40} {'-'*6}")
        for m in matched:
            short = textwrap.shorten(m['raw'], width=58)
            print(f"  {short:<60} {m['node']:<40} {'yes' if m['is_total'] else ''}")

    if unrecognised:
        print("\n⚠️   UNRECOGNISED — add these to taxonomy.py\n")
        for u in unrecognised:
            print(f"  [{u['statement']}]  {u['raw']}")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    import sys
    cache = sys.argv[1] if len(sys.argv) > 1 else "financials_cache.json"
    validate_against_cache(cache)