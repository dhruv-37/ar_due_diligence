"""
tools/extractor_tool.py
=======================
LangChain tool that wraps Step1 (page filter) + Step2 (Gemini extraction → Excel).
Returns a JSON string with status and output_xlsx path.
"""

import json
import os
import sys
from pathlib import Path

from langchain_core.tools import tool

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


@tool
def extract_financials_tool(pdf_path: str, output_xlsx: str = "") -> str:
    """
    Run Step1 (PDF page filter) then Step2 (Gemini LLM extraction → Excel).
    Returns JSON: {"status": "ok"|"error", "output_xlsx": str, "error": str}.
    """
    try:
        from pipeline.Step1 import extract_core_financial_statements
        from pipeline.Step2 import extract_financials

        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            return json.dumps({"status": "error", "error": "GEMINI_API_KEY not set"})

        input_stem = Path(pdf_path).stem
        trimmed_pdf = str(Path(pdf_path).parent / f"{input_stem}_trimmed.pdf")

        print(f"\n── Step 1: Extracting core financial pages → {trimmed_pdf}")
        extract_core_financial_statements(pdf_path, trimmed_pdf, gemini_key)

        if not output_xlsx:
            output_dir = Path(_PROJECT_ROOT) / "output"
            output_dir.mkdir(exist_ok=True)
            output_xlsx = str(output_dir / f"{input_stem}_financials.xlsx")

        print(f"\n── Step 2: Parsing & building Excel → {output_xlsx}")
        extract_financials(trimmed_pdf, output_xlsx)

        return json.dumps({"status": "ok", "output_xlsx": output_xlsx})

    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)})