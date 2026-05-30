"""Plumb CLI.

    plumb eval  --pred predictions.json --truth ground_truth.json --schema schema.json
    plumb demo

Schema JSON format (list of field specs):
    [
      {"name": "invoice_number", "type": "exact"},
      {"name": "total",          "type": "number", "abs_tol": 0.01},
      {"name": "invoice_date",   "type": "date"},
      {"name": "vendor",         "type": "string", "threshold": 0.9},
      {"name": "line_items",     "type": "table",
       "columns": [{"name": "desc", "type": "string"},
                   {"name": "amount", "type": "number"}]}
    ]
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Sequence

from .metrics import FieldSpec, FieldType, evaluate
from .report import markdown_report, text_report


def _spec_from_dict(d: dict) -> FieldSpec:
    cols = tuple(_spec_from_dict(c) for c in d.get("columns", []))
    return FieldSpec(
        name=d["name"],
        type=FieldType(d.get("type", "string")),
        threshold=float(d.get("threshold", 0.90)),
        abs_tol=float(d.get("abs_tol", 0.01)),
        rel_tol=float(d.get("rel_tol", 0.0)),
        columns=cols,
        weight=float(d.get("weight", 1.0)),
    )


def load_schema(path: str) -> List[FieldSpec]:
    with open(path) as f:
        return [_spec_from_dict(d) for d in json.load(f)]


def _cmd_eval(args: argparse.Namespace) -> int:
    specs = load_schema(args.schema)
    with open(args.pred) as f:
        preds = json.load(f)
    with open(args.truth) as f:
        truth = json.load(f)
    report = evaluate(specs, preds, truth)
    if args.format == "markdown":
        print(markdown_report(report))
    else:
        print(text_report(report))
    # Exit non-zero if below threshold — usable as a CI gate.
    if args.min_accuracy is not None and report.document_accuracy < args.min_accuracy:
        print(f"\nFAIL: document accuracy {report.document_accuracy:.1%} "
              f"< required {args.min_accuracy:.1%}", file=sys.stderr)
        return 1
    return 0


def _cmd_demo(_: argparse.Namespace) -> int:
    from examples.invoices.run_demo import main as demo_main  # type: ignore
    demo_main()
    return 0


def _cmd_review(args: argparse.Namespace) -> int:
    """Launch the local review UI.

    With --schema and --pred it serves YOUR data; with none of them it falls
    back to the bundled demo so you can try it instantly.
    """
    from .store import Store
    from .review import ReviewApp, serve

    if args.schema and args.pred:
        specs = load_schema(args.schema)
        with open(args.pred) as f:
            predictions = json.load(f)
        samples = None
        if args.samples:
            with open(args.samples) as f:
                samples = json.load(f)   # {doc_id: [prediction, ...]}
        store = Store(args.db)
    else:
        from examples.invoices.run_demo import SCHEMA, PREDICTIONS, GROUND_TRUTH, SAMPLES_003
        specs, predictions = SCHEMA, PREDICTIONS
        samples = {"inv_003": SAMPLES_003}
        store = Store(args.db)
        for doc_id, gt in GROUND_TRUTH.items():
            if doc_id != "inv_003":
                store.set_ground_truth(doc_id, gt)

    app = ReviewApp(specs=specs, predictions=predictions, store=store, samples=samples)
    serve(app, host=args.host, port=args.port)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="plumb", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("eval", help="evaluate predictions against ground truth")
    pe.add_argument("--pred", required=True)
    pe.add_argument("--truth", required=True)
    pe.add_argument("--schema", required=True)
    pe.add_argument("--format", choices=["text", "markdown"], default="text")
    pe.add_argument("--min-accuracy", type=float, default=None,
                    help="fail (exit 1) if document accuracy is below this (0..1)")
    pe.set_defaults(func=_cmd_eval)

    pd = sub.add_parser("demo", help="run the offline invoice demo")
    pd.set_defaults(func=_cmd_demo)

    pr = sub.add_parser("review", help="launch the local review UI (your data, or demo)")
    pr.add_argument("--schema", help="schema JSON (omit to use demo data)")
    pr.add_argument("--pred", help="predictions JSON {doc_id: {...}} (omit to use demo)")
    pr.add_argument("--samples", help="optional {doc_id: [pred,...]} for confidence")
    pr.add_argument("--host", default="127.0.0.1")
    pr.add_argument("--port", type=int, default=8000)
    pr.add_argument("--db", default="plumb.db")
    pr.set_defaults(func=_cmd_review)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
