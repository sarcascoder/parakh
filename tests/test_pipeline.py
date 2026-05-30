"""Pipeline facade: the end-to-end flow people drop into their own code."""
import os

from plumb import Pipeline, FieldSpec, FieldType
from plumb.extractors import FixedExtractor


SCHEMA = [
    FieldSpec("invoice_number", FieldType.EXACT),
    FieldSpec("total", FieldType.NUMBER, abs_tol=0.01),
]


def _pipe(tmp_path, predictions, runs=1) -> Pipeline:
    # FixedExtractor treats `document` as the doc_id and replays predictions.
    extractor = FixedExtractor(predictions, name="fixed")
    return Pipeline(schema=SCHEMA, extractor=extractor,
                    store_path=os.path.join(str(tmp_path), "p.db"),
                    consistency_runs=runs)


def test_extract_evaluate_and_correct_loop(tmp_path):
    preds = {"d1": {"invoice_number": "A-1", "total": "$100.00"}}
    pipe = _pipe(tmp_path, preds)
    try:
        out = pipe.extract("d1", "d1")            # document==doc_id for FixedExtractor
        assert out["invoice_number"] == "A-1"

        # No ground truth yet -> evaluating against empty truth gives no docs.
        assert len(pipe.evaluate().docs) == 0

        # A reviewer records the correct answer; now eval has something to score.
        pipe.record_correction("d1", {"invoice_number": "A-1", "total": 100.0})
        report = pipe.evaluate()
        assert report.document_accuracy == 1.0    # "$100.00" matches 100.0
    finally:
        pipe.close()


def test_self_consistency_flags_unstable_field(tmp_path):
    # Build an extractor whose output we vary per call via a tiny stateful shim.
    seq = [
        {"invoice_number": "A-1", "total": 100},
        {"invoice_number": "A-9", "total": 100},  # invoice_number wobbles
        {"invoice_number": "A-1", "total": 100},
    ]

    class Wobbler:
        name = "wobbler"
        def __init__(self): self.i = 0
        def extract(self, document, specs):
            v = seq[self.i % len(seq)]; self.i += 1; return dict(v)

    pipe = Pipeline(schema=SCHEMA, extractor=Wobbler(),
                    store_path=os.path.join(str(tmp_path), "p.db"),
                    consistency_runs=3)
    try:
        consensus = pipe.extract("d1", "ignored")
        assert consensus["invoice_number"] == "A-1"        # 2 of 3
        conf = pipe.confidence("d1")
        assert conf["invoice_number"] < 0.99                # wobbled -> low confidence
        assert conf["total"] == 1.0                         # stable -> full confidence
        queue = pipe.review_queue()
        assert queue[0].needs_review
    finally:
        pipe.close()


def test_fewshot_block_uses_recorded_corrections(tmp_path):
    preds = {"d1": {"invoice_number": "A-1", "total": 100}}
    pipe = _pipe(tmp_path, preds)
    try:
        pipe.extract("d1", "d1")
        pipe.record_correction("d1", {"invoice_number": "A-1", "total": 100})
        block = pipe.fewshot_block({"d1": "Invoice A-1 total 100"})
        assert "A-1" in block and "verified examples" in block
    finally:
        pipe.close()
