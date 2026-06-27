"""
tools/excel_reader_tool.py
==========================
Reads the structured Excel file produced by Step2 into a JSON-serialisable
dict so downstream agents (Red Flag, Narrative, Memo) can consume it.

Each sheet becomes a list of {line_item, current_year, previous_year,
taxonomy_node} records. Empty or missing sheets are skipped silently.
"""

import json
import os
from pathlib import Path

import pandas as pd
from langchain.tools import tool


@tool
def excel_reader_tool(xlsx_path: str) -> str:
    """
    Reads the structured financial Excel file produced by Step2.

    Args:
        xlsx_path: Absolute or relative path to the .xlsx file.

    Returns:
        JSON string with keys:
            status  — "success" | "error"
            sheets  — dict of sheet_name → list of row dicts
            error   — error message (only present on failure)
    """
    xlsx_path = str(Path(xlsx_path).resolve())

    if not Path(xlsx_path).exists():
        return json.dumps({"status": "error", "error": f"File not found: {xlsx_path}"})

    try:
        xf = pd.ExcelFile(xlsx_path)
        sheets = {}

        for sheet_name in xf.sheet_names:
            df = pd.read_excel(xf, sheet_name=sheet_name)

            # Normalise column names to lowercase stripped strings
            df.columns = [str(c).strip().lower() for c in df.columns]

            # Keep only the four core columns; fill missing ones with None
            core_cols = ["line_item", "current_year", "previous_year", "taxonomy_node"]
            for col in core_cols:
                if col not in df.columns:
                    df[col] = None

            df = df[core_cols].dropna(subset=["line_item"])

            if df.empty:
                continue

            # Convert NaN → None for JSON serialisation
            records = df.where(pd.notna(df), other=None).to_dict(orient="records")
            sheets[sheet_name] = records

        if not sheets:
            return json.dumps({"status": "error", "error": "No usable sheets found in Excel file."})

        return json.dumps({"status": "success", "sheets": sheets})

    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)})