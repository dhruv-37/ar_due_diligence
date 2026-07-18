"""
audit_layer_inputs.py
======================
Standalone auditing tool for the forensic diagnostic engine (Layer A / B / C).

WHY THIS EXISTS
----------------
Layer A/B/C checks each have a guard clause like:

    if None in (rec_t, rec_t1, rev_t, rev_t1) or rev_t == 0 or rev_t1 == 0:
        return out

If any one of the metrics a check depends on is missing, the check
*silently returns no signal* — which looks identical to "the company's
numbers are clean." This script makes that distinction visible: for every
metric every Layer A/B/C check depends on, it shows you the exact value
pulled, which alias matched, and whether the check that needs it will
actually run or silently skip.

USAGE
-----
    python pipeline/audit_layer_inputs.py path/to/output.json

where output.json is the same {"line_items": [...], "fs_dictionary": {...}}
structure fed into ForensicDiagnosticEngine (e.g. VI_normalized_*.json's
sibling "full" extraction file, or whatever your pipeline writes before
calling run_full_diagnostics()).

For a quick manual cross-check against the source AR / Excel, also pass
--dump-fs-dictionary to print the full fs_dictionary contents (every raw
value the taxonomy mapper attached to each canonical node) so you can
diff it page-by-page against the PDF.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from pipeline.forensic_diagnostic_engine import (
        ForensicDiagnosticEngine, _METRIC_ALIASES, _SCOPES,
    )
except ImportError:
    from forensic_diagnostic_engine import (
        ForensicDiagnosticEngine, _METRIC_ALIASES, _SCOPES,
    )


# Which canonical metrics each Layer A/B/C check actually reads, so we can
# tell you exactly which checks are live vs. dead for this AR.
_CHECK_DEPENDENCIES: dict[str, list[str]] = {
    "Layer A — Benford's Law":              [],  # runs on all line_items directly, not fs_dictionary metrics
    "Layer A — DSRI (per scope)":           ["receivables", "revenue"],
    "Layer A — AQI (per scope)":            ["total_assets", "total_current_assets", "net_ppe"],
    "Layer B — RE Roll-Forward (per scope)": ["retained_earnings", "profit_for_the_year", "dividends_paid"],
    "Layer B — Sloan Ratio (per scope)":     ["profit_for_the_year", "operating_cash_flow",
                                               "investing_cash_flow", "total_assets"],
    "Layer C — DSO/Inventory drift (per scope)": ["receivables", "revenue", "inventory", "cogs"],
}


def _fmt(v) -> str:
    if v is None:
        return "MISSING"
    try:
        return f"{v:,.2f}"
    except (TypeError, ValueError):
        return str(v)


def audit_metrics(engine: ForensicDiagnosticEngine) -> dict[str, dict]:
    """
    For every canonical metric in _METRIC_ALIASES, pull CY/PY for both
    scopes via the engine's own _get_metric() (the exact same code path
    Layer A/B/C uses) and report what came back.
    """
    report: dict[str, dict] = {}
    for scope in _SCOPES:
        report[scope] = {}
        for canonical in _METRIC_ALIASES:
            cy = engine._get_metric(scope, canonical, "current_year")
            py = engine._get_metric(scope, canonical, "previous_year")
            report[scope][canonical] = {"current_year": cy, "previous_year": py}
    return report


def print_metric_table(report: dict[str, dict]) -> None:
    print("\n" + "=" * 78)
    print("METRIC-BY-METRIC DUMP  (exactly what _get_metric() returns to Layer A/B/C)")
    print("=" * 78)
    for scope, metrics in report.items():
        print(f"\n--- {scope} ---")
        print(f"{'metric':30s} {'current_year':>18s} {'previous_year':>18s}")
        for canonical, vals in metrics.items():
            cy, py = vals["current_year"], vals["previous_year"]
            flag = "  ⚠️  MISSING" if cy is None or py is None else ""
            print(f"{canonical:30s} {_fmt(cy):>18s} {_fmt(py):>18s}{flag}")


def print_check_liveness(report: dict[str, dict]) -> None:
    print("\n" + "=" * 78)
    print("WHICH LAYER A/B/C CHECKS WILL ACTUALLY RUN vs. SILENTLY SKIP")
    print("=" * 78)
    for check_name, deps in _CHECK_DEPENDENCIES.items():
        if not deps:
            print(f"\n{check_name}: always attempts (runs on raw line_items directly)")
            continue
        print(f"\n{check_name}  (needs: {', '.join(deps)})")
        for scope, metrics in report.items():
            missing = [
                d for d in deps
                if metrics.get(d, {}).get("current_year") is None
                or metrics.get(d, {}).get("previous_year") is None
            ]
            status = "❌ WILL SKIP" if missing else "✅ will run"
            detail = f" — missing: {', '.join(missing)}" if missing else ""
            print(f"    {scope:14s} {status}{detail}")


def print_fs_dictionary(engine: ForensicDiagnosticEngine) -> None:
    print("\n" + "=" * 78)
    print("FULL fs_dictionary CONTENTS (for manual page-by-page cross-check against the PDF)")
    print("=" * 78)
    print(json.dumps(engine.fs_dictionary, indent=2, default=str))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_path", help="Path to the {line_items, fs_dictionary} JSON fed to ForensicDiagnosticEngine")
    ap.add_argument("--dump-fs-dictionary", action="store_true",
                     help="Also print the full fs_dictionary for manual cross-checking against the source AR")
    args = ap.parse_args()

    financial_data = json.loads(Path(args.json_path).read_text())
    engine = ForensicDiagnosticEngine(financial_data)

    report = audit_metrics(engine)
    print_metric_table(report)
    print_check_liveness(report)

    if args.dump_fs_dictionary:
        print_fs_dictionary(engine)

    print("\n" + "=" * 78)
    print("NEXT STEP: open the source AR PDF next to this output and manually verify")
    print("every non-MISSING value above matches the actual page. Any check marked")
    print("'WILL SKIP' means Layer A/B/C is silently producing zero signals for it —")
    print("that is NOT the same as 'the company's numbers are clean.'")
    print("=" * 78)


if __name__ == "__main__":
    main()