"""
run_pipeline.py
================
One command, full pipeline: Step1 (page extraction) -> Step2 (parse +
Excel + taxonomy JSON) -> normalize_financials (schema-normalized JSON).

Place this file at the repo root (next to main.py / server.py).

Usage (simplest form — everything else is auto-derived):
    python run_pipeline.py "data/pdfs/Tata Consultancy Services Annual Report.pdf" TCS

Optional overrides:
    python run_pipeline.py <pdf> <ticker> [--fy 2025] [--fy-prev 2024]
                            [--report-type Standalone|Consolidated] [--out-dir output]

Output files (all named after the ticker, written into --out-dir, default "output/"):
    output/TCS.xlsx                the formatted Excel workbook
    output/TCS_taxonomy.json       Step2's structured line-item JSON
    output/TCS_normalized.json     final normalized + validated schema
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.Step1 import extract_core_financial_statements
from pipeline.Step2 import extract_financials
from pipeline.normalize_financials import run_pipeline_from_step2


def main():
    parser = argparse.ArgumentParser(description="Run the full AR due-diligence pipeline in one command.")
    parser.add_argument("pdf_path", help="Path to the source annual report PDF")
    parser.add_argument("ticker", help="Ticker / company identifier, e.g. TCS")
    parser.add_argument("--fy", type=int, default=None, help="Current fiscal year (default: this year)")
    parser.add_argument("--fy-prev", type=int, default=None, help="Prior fiscal year (default: fy - 1)")
    parser.add_argument("--report-type", default="Consolidated", choices=["Standalone", "Consolidated"])
    parser.add_argument("--out-dir", default="output", help="Directory for all output files (default: output/)")
    args = parser.parse_args()

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        print("❌  GEMINI_API_KEY environment variable not set.")
        sys.exit(1)

    import datetime
    fy = args.fy or datetime.date.today().year
    fy_prev = args.fy_prev or (fy - 1)

    os.makedirs(args.out_dir, exist_ok=True)
    trimmed_pdf   = os.path.join(args.out_dir, f"{args.ticker}_trimmed.pdf")
    output_xlsx   = os.path.join(args.out_dir, f"{args.ticker}.xlsx")
    taxonomy_json = os.path.join(args.out_dir, f"{args.ticker}_taxonomy.json")
    normalized_json = os.path.join(args.out_dir, f"{args.ticker}_normalized.json")

    print(f"\n── Step 1: Extracting core financial pages → {trimmed_pdf}")
    extract_core_financial_statements(args.pdf_path, trimmed_pdf, gemini_key)

    print(f"\n── Step 2: Parsing & building Excel → {output_xlsx}")
    extract_financials(trimmed_pdf, output_xlsx)
    # extract_financials writes "<output_xlsx-without-ext>_taxonomy.json" itself;
    # that path is exactly `taxonomy_json` computed above.

    print(f"\n── Step 3: Normalizing → {normalized_json}")
    result = run_pipeline_from_step2(
        step2_json_path=taxonomy_json,
        ticker=args.ticker,
        fiscal_year_t=fy,
        fiscal_year_t_minus_1=fy_prev,
        report_type=args.report_type,
    )
    with open(normalized_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n✅  Done.")
    print(f"    Excel:      {output_xlsx}")
    print(f"    Taxonomy:   {taxonomy_json}")
    print(f"    Normalized: {normalized_json}")


if __name__ == "__main__":
    main()