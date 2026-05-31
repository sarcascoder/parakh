"""End-to-end Parakh demo — runs on CPU, no model required.

Simulates extracting structured data from 4 invoices with a local model, then:
  1. scores field-level accuracy vs ground truth,
  2. derives calibrated confidence from repeated (jittered) runs,
  3. computes the safe auto-accept threshold.

Run:  python -m examples.invoices.run_demo   (from the repo root)
"""
from __future__ import annotations

from parakh import (
    FieldSpec, FieldType, evaluate, consensus_extraction,
    reliability_table, safe_auto_accept_threshold, score_document,
)
from parakh.report import text_report

# --- Schema ---------------------------------------------------------------- #
SCHEMA = [
    FieldSpec("invoice_number", FieldType.EXACT),
    FieldSpec("vendor", FieldType.STRING, threshold=0.85),
    FieldSpec("invoice_date", FieldType.DATE),
    FieldSpec("total", FieldType.NUMBER, abs_tol=0.01),
    FieldSpec("line_items", FieldType.TABLE, threshold=0.8, columns=(
        FieldSpec("desc", FieldType.STRING, threshold=0.8),
        FieldSpec("amount", FieldType.NUMBER, abs_tol=0.01),
    )),
]

# --- Ground truth (human-verified) ---------------------------------------- #
GROUND_TRUTH = {
    "inv_001": {
        "invoice_number": "A-1001", "vendor": "Acme Inc.",
        "invoice_date": "2026-01-15", "total": 1200.00,
        "line_items": [{"desc": "Widget", "amount": 1000.0},
                       {"desc": "Shipping", "amount": 200.0}],
    },
    "inv_002": {
        "invoice_number": "B-2042", "vendor": "Globex LLC",
        "invoice_date": "2026-02-03", "total": 540.50,
        "line_items": [{"desc": "Consulting", "amount": 540.5}],
    },
    "inv_003": {
        "invoice_number": "C-3300", "vendor": "Initech",
        "invoice_date": "2026-03-21", "total": 99.99,
        "line_items": [{"desc": "License", "amount": 99.99}],
    },
    "inv_004": {
        "invoice_number": "D-7777", "vendor": "Umbrella Corp",
        "invoice_date": "2026-04-09", "total": 3150.00,
        "line_items": [{"desc": "Server", "amount": 3000.0},
                       {"desc": "Setup", "amount": 150.0}],
    },
}

# --- Predictions from a local model (with realistic mistakes) -------------- #
PREDICTIONS = {
    "inv_001": {  # perfect, but formatted differently — should still score 100%
        "invoice_number": "A-1001", "vendor": "ACME, Inc",
        "invoice_date": "01/15/2026", "total": "$1,200.00",
        "line_items": [{"desc": "Widget", "amount": "1000"},
                       {"desc": "Shipping", "amount": "200"}],
    },
    "inv_002": {  # wrong total (OCR misread), missing a line item is N/A here
        "invoice_number": "B-2042", "vendor": "Globex LLC",
        "invoice_date": "2026-02-03", "total": 504.50,
        "line_items": [{"desc": "Consulting", "amount": 540.5}],
    },
    "inv_003": {  # vendor typo, dropped a digit in invoice number
        "invoice_number": "C-330", "vendor": "Initrech",
        "invoice_date": "2026-03-21", "total": 99.99,
        "line_items": [{"desc": "License", "amount": 99.99}],
    },
    "inv_004": {  # missed a line item -> table recall drops
        "invoice_number": "D-7777", "vendor": "Umbrella Corp",
        "invoice_date": "2026-04-09", "total": 3150.00,
        "line_items": [{"desc": "Server", "amount": 3000.0}],
    },
}

# --- Repeated runs for self-consistency confidence (inv_003) --------------- #
# The model flip-flops on the typo'd fields across samples -> low confidence.
SAMPLES_003 = [
    {"invoice_number": "C-330", "vendor": "Initrech", "invoice_date": "2026-03-21",
     "total": 99.99, "line_items": [{"desc": "License", "amount": 99.99}]},
    {"invoice_number": "C-3300", "vendor": "Initech", "invoice_date": "2026-03-21",
     "total": 99.99, "line_items": [{"desc": "License", "amount": 99.99}]},
    {"invoice_number": "C-330", "vendor": "Initech", "invoice_date": "2026-03-21",
     "total": 99.99, "line_items": [{"desc": "License", "amount": 99.99}]},
]


def main() -> None:
    report = evaluate(SCHEMA, PREDICTIONS, GROUND_TRUTH)
    print(text_report(report))

    print("\nSELF-CONSISTENCY CONFIDENCE (inv_003, 3 runs):")
    cons = consensus_extraction(SCHEMA, SAMPLES_003)
    for f, c in cons.confidence.items():
        flag = "  <-- review" if c < 0.99 else ""
        print(f"    {f:<18} confidence {c:.0%}{flag}")
    print(f"    consensus invoice_number = {cons.consensus['invoice_number']}")

    # Build (confidence, correct) pairs to find a safe auto-accept threshold.
    pairs = []
    for doc_id, exp in GROUND_TRUTH.items():
        dr = score_document(SCHEMA, doc_id, PREDICTIONS[doc_id], exp)
        for fr in dr.fields:
            # Demo: use field score as a confidence proxy (replace with
            # self-consistency confidence in production).
            pairs.append((fr.score, fr.correct))
    thr = safe_auto_accept_threshold(pairs, target_precision=0.99)
    print(f"\nSAFE AUTO-ACCEPT THRESHOLD (>=99% precision): {thr:.2f}")
    print("Reliability table (confidence band -> measured accuracy):")
    for b in reliability_table(pairs, bins=5):
        if b.n:
            print(f"    {b.lo:.1f}-{b.hi:.1f}: n={b.n:<3} accuracy={b.accuracy:.0%}")


if __name__ == "__main__":
    main()
