from parakh.metrics import FieldSpec, FieldType
from parakh.queue import build_review_queue

SPECS = [
    FieldSpec("invoice_number", FieldType.EXACT),
    FieldSpec("vendor", FieldType.STRING, threshold=0.85),
    FieldSpec("total", FieldType.NUMBER, abs_tol=0.01),
]


def test_wrong_field_enters_queue():
    preds = {"d1": {"invoice_number": "X-1", "vendor": "Acme", "total": 100}}
    gt = {"d1": {"invoice_number": "X-2", "vendor": "Acme", "total": 100}}
    q = build_review_queue(SPECS, preds, ground_truth=gt)
    assert q[0].needs_review
    flagged = [f for f in q[0].fields if f.reason]
    assert any(f.name == "invoice_number" for f in flagged)


def test_low_confidence_enters_queue_without_ground_truth():
    preds = {"d1": {"invoice_number": "X-1", "vendor": "Acme", "total": 100}}
    samples = {"d1": [
        {"invoice_number": "X-1", "vendor": "Acme", "total": 100},
        {"invoice_number": "X-7", "vendor": "Acme", "total": 100},
    ]}
    q = build_review_queue(SPECS, preds, samples=samples)
    inv = [f for f in q[0].fields if f.name == "invoice_number"][0]
    assert "low confidence" in inv.reason


def test_clean_doc_not_flagged_and_priority_ordering():
    preds = {
        "clean": {"invoice_number": "A", "vendor": "Acme", "total": 1},
        "dirty": {"invoice_number": "B", "vendor": "Acme", "total": 1},
    }
    gt = {
        "clean": {"invoice_number": "A", "vendor": "Acme", "total": 1},
        "dirty": {"invoice_number": "WRONG", "vendor": "Acme", "total": 1},
    }
    q = build_review_queue(SPECS, preds, ground_truth=gt)
    assert q[0].doc_id == "dirty"          # higher priority first
    clean = [d for d in q if d.doc_id == "clean"][0]
    assert not clean.needs_review
