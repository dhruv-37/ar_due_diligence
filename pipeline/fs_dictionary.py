"""
fs_dictionary.py
================
Predefined dictionary of financial-statement line items (ratio-analysis
essentials + standard FS captions), segregated by:
    scope (STANDALONE / CONSOLIDATED)
      -> statement (Profit and Loss / Balance Sheet / Cash Flow /
                    Statement of Changes in Equity)
        -> item name -> {"keywords": [...], "current_year": None, "previous_year": None}

This is a static reference dictionary, not an LLM output. taxonomy_mapper.py
fuzzy-matches extracted line items against the "keywords" lists here to
assign an item name; year values are filled in per-run from extracted data.
"""

from __future__ import annotations
from copy import deepcopy

_PNL_ITEMS = {
    "Revenue From Operations": ["revenue from operations", "net sales", "sale of products",
                                  "turnover", "gross revenue"],
    "Other Income": ["other income", "non-operating income", "miscellaneous income"],
    "Total Income": ["total income", "total revenue", "aggregate revenue"],
    "Cost Of Materials Consumed": ["cost of materials consumed", "raw material consumed"],
    "Purchases Of Stock In Trade": ["purchases of stock-in-trade", "purchase of stock in trade"],
    "Employee Benefits Expense": ["employee benefits expense", "staff costs", "employee cost", "personnel expenses"],
    "Finance Costs": ["finance costs", "interest expense", "finance cost"],
    "Depreciation And Amortisation Expense": ["depreciation and amortisation expense", "depreciation",
                                               "depreciation / amortisation expense"],
    "Other Expenses": ["other expenses", "administrative and other expenses",
                        "general and administrative expenses"],
    "Total Expenses": ["total expenses", "total operating expenses"],
    "Exceptional Items": ["exceptional items", "exceptional item", "extraordinary items"],
    "Profit Before Tax": ["profit before tax", "profit/(loss) before tax", "pbt", "earnings before tax"],
    "Tax Expense": ["tax expense", "total tax expense", "current tax", "provision for tax"],
    "Profit For The Year": ["profit for the year", "profit after tax", "net profit", "pat",
                             "profit/(loss) for the year"],
    "Other Comprehensive Income": ["other comprehensive income", "total other comprehensive income", "oci"],
    "Total Comprehensive Income": ["total comprehensive income", "total comprehensive income for the year"],
    "Earnings Per Share": ["earnings per equity share", "basic and diluted eps", "eps"],
    "Basic EPS": ["basic (in h)", "basic (in rs)", "earnings per equity share basic",
                  "basic earnings per share"],
    "Diluted EPS": ["diluted (in h)", "diluted (in rs)", "earnings per equity share diluted",
                    "diluted earnings per share"],
    "Changes In Inventories": ["changes in inventories of finished goods work-in-progress and stock-in-trade",
                                "changes in inventories", "change in inventories"],
    "Non Controlling Interest": ["non-controlling interest", "non controlling interest", "minority interest"],
}

_BS_ITEMS = {
    "Property Plant And Equipment": ["property, plant and equipment", "tangible assets", "fixed assets"],
    "Capital Work In Progress": ["capital work-in-progress", "cwip"],
    "Goodwill": ["goodwill"],
    "Other Intangible Assets": ["other intangible assets", "intangible assets"],
    "Intangible Assets Under Development": ["intangible assets under development",
                                             "other intangible assets under development"],
    "Non Current Investments": ["non-current investments", "investments (non-current)"],
    "Loans": ["loans"],
    "Deferred Tax Assets": ["deferred tax assets (net)", "deferred tax assets"],
    "Other Non Current Assets": ["other non-current assets"],
    "Total Non Current Assets": ["total non-current assets"],
    "Inventories": ["inventories", "stock in trade"],
    "Trade Receivables": ["trade receivables", "sundry debtors"],
    "Cash And Cash Equivalents": ["cash and cash equivalents"],
    "Current Investments": ["current investments", "investments (current)"],
    "Other Current Assets": ["other current assets"],
    "Total Current Assets": ["total current assets"],
    "Total Assets": ["total assets"],
    "Equity Share Capital": ["equity share capital", "share capital"],
    "Other Equity": ["other equity", "reserves and surplus"],
    "Total Equity": ["total equity", "total shareholders funds"],
    "Non Current Borrowings": ["non-current borrowings", "long term borrowings"],
    "Non Current Provisions": ["long-term provisions", "non-current provisions"],
    "Deferred Tax Liabilities": ["deferred tax liabilities (net)", "deferred tax liabilities"],
    "Total Non Current Liabilities": ["total non-current liabilities"],
    "Current Borrowings": ["current borrowings", "short term borrowings"],
    "Trade Payables": ["trade payables", "sundry creditors"],
    "Other Current Liabilities": ["other current liabilities"],
    "Current Provisions": ["short-term provisions", "current provisions"],
    "Total Current Liabilities": ["total current liabilities"],
    "Total Equity And Liabilities": ["total equity and liabilities", "total liabilities"],
    "Borrowings": ["borrowings"],
    "Lease Liabilities": ["lease liabilities"],
    "Provisions": ["provisions"],
    "Non Controlling Interest": ["non-controlling interest", "non controlling interest", "minority interest"],
}

_CF_ITEMS = {
    "Net Cash From Operating Activities": ["net cash from operating activities",
                                            "net cash generated from operating activities",
                                            "cash flow from operating activities"],
    "Net Cash From Investing Activities": ["net cash from investing activities",
                                            "net cash used in investing activities"],
    "Net Cash From Financing Activities": ["net cash from financing activities",
                                            "net cash used in financing activities"],
    "Purchase Of Property Plant And Equipment": ["purchase of property, plant and equipment",
                                                  "purchase of fixed assets"],
    "Proceeds From Borrowings": ["proceeds from borrowings"],
    "Repayment Of Borrowings": ["repayment of borrowings"],
    "Dividend Paid": ["dividend paid", "payment of dividends"],
    "Interest Paid": ["interest paid"],
    "Interest Received": ["interest received"],
    "Net Increase Decrease In Cash": ["net increase/(decrease) in cash and cash equivalents",
                                       "net change in cash and cash equivalents"],
    "Opening Cash And Cash Equivalents": ["cash and cash equivalents at the beginning of the year",
                                           "opening balance of cash and cash equivalents"],
    "Closing Cash And Cash Equivalents": ["cash and cash equivalents at the end of the year",
                                           "closing balance of cash and cash equivalents"],
}

_SOCE_ITEMS = {
    "Opening Balance Equity Share Capital": ["equity share capital - balance as at 1st april",
                                              "opening balance of equity share capital"],
    "Closing Balance Equity Share Capital": ["equity share capital - balance as at 31st march",
                                              "closing balance of equity share capital"],
    "Opening Balance Other Equity": ["other equity - balance as at 1st april", "opening balance of reserves"],
    "Closing Balance Other Equity": ["other equity - balance as at 31st march", "closing balance of reserves"],
    "Total Comprehensive Income For The Year": ["total comprehensive income for the year"],
    "Dividend Distributed": ["dividends", "dividend"],
    "Transfer To Reserves": ["transfer to / (from) retained earnings", "transfer to reserves"],
    "Equity Share Capital Change During The Year": ["equity share capital change during the year",
                                                      "equity share capital - change during the year"],
}

_STATEMENT_TEMPLATE = {
    "Profit and Loss": _PNL_ITEMS,
    "Balance Sheet": _BS_ITEMS,
    "Cash Flow": _CF_ITEMS,
    "Statement of Changes in Equity": _SOCE_ITEMS,
}


def _build_scope_dict() -> dict:
    scope_dict = {}
    for statement, items in _STATEMENT_TEMPLATE.items():
        scope_dict[statement] = {
            name: {"keywords": list(keywords), "current_year": None, "previous_year": None}
            for name, keywords in items.items()
        }
    return scope_dict


FS_DICTIONARY: dict = {
    "STANDALONE": _build_scope_dict(),
    "CONSOLIDATED": _build_scope_dict(),
}


def new_fs_dictionary() -> dict:
    """Returns a fresh deep copy of FS_DICTIONARY (per-run population target)."""
    return deepcopy(FS_DICTIONARY)