"""
main.py
=======
Single entry point for the Indian AR Due Diligence Agent.

Usage
-----
    python main.py <pdf_path> <ticker> [--output-xlsx <path>]

Example
-------
    python main.py data/pdfs/Reliance_AR_2025.pdf RELIANCE
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Project root on sys.path ──────────────────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agents.memo_agent import run_due_diligence


def main():
    parser = argparse.ArgumentParser(description="Indian AR Due Diligence Agent")
    parser.add_argument("pdf_path",      help="Path to the Annual Report PDF")
    parser.add_argument("ticker",        help="NSE ticker without suffix, e.g. RELIANCE")
    parser.add_argument("--output-xlsx", default="", help="Optional Excel output path")
    args = parser.parse_args()

    # ── Validate env ──────────────────────────────────────────────────────────
    missing = [k for k in ["GEMINI_API_KEY"] if not os.environ.get(k)]
    if missing:
        print(f"❌  Missing environment variables: {', '.join(missing)}")
        print("    Add them to your .env file.")
        sys.exit(1)

    # ── Validate PDF ──────────────────────────────────────────────────────────
    if not Path(args.pdf_path).exists():
        print(f"❌  PDF not found: {args.pdf_path}")
        sys.exit(1)

    print(f"""
╔══════════════════════════════════════════════════════╗
║       Indian AR Due Diligence Agent                  ║
╠══════════════════════════════════════════════════════╣
║  PDF    : {args.pdf_path:<42}║
║  Ticker : {args.ticker:<42}║
╚══════════════════════════════════════════════════════╝
""")

    result = run_due_diligence(
        pdf_path     = args.pdf_path,
        ticker       = args.ticker,
        output_xlsx  = args.output_xlsx,
    )

    print(f"""
╔══════════════════════════════════════════════════════╗
║  ✅  Due Diligence Complete                          ║
╠══════════════════════════════════════════════════════╣
║  Memo  → {result.get('memo_path', 'N/A'):<43}║
║  Excel → {result.get('extraction', {}).get('output_xlsx', 'N/A'):<43}║
╚══════════════════════════════════════════════════════╝
""")

    if result.get("errors"):
        print("⚠️  Errors during pipeline:")
        for err in result["errors"]:
            print(f"   - {err}")


if __name__ == "__main__":
    main()