from parakh.metrics import FieldSpec, FieldType
from parakh.confidence import (
    consensus_extraction, field_consistency,
    reliability_table, safe_auto_accept_threshold,
)

SPECS = [FieldSpec("num", FieldType.EXACT), FieldSpec("vendor", FieldType.STRING)]


def test_consensus_picks_modal_value():
    samples = [
        {"num": "C-330", "vendor": "Initrech"},
        {"num": "C-3300", "vendor": "Initech"},
        {"num": "C-330", "vendor": "Initech"},
    ]
    cons = consensus_extraction(SPECS, samples)
    assert cons.consensus["num"] == "C-330"        # 2/3
    assert cons.consensus["vendor"] == "Initech"   # 2/3
    assert cons.confidence["num"] == 2 / 3
    assert "num" in cons.low_confidence_fields(0.99)


def test_full_agreement_is_confident():
    samples = [{"num": "A-1", "vendor": "Acme"}] * 4
    assert field_consistency(SPECS[0], samples) == 1.0


def test_reliability_and_threshold():
    # High confidence correct, low confidence wrong -> threshold should sit high.
    pairs = [(0.95, True), (0.92, True), (0.30, False), (0.40, False), (0.99, True)]
    thr = safe_auto_accept_threshold(pairs, target_precision=0.99)
    assert thr <= 0.92                       # accepting >=0.92 yields 100% precision
    buckets = reliability_table(pairs, bins=5)
    assert sum(b.n for b in buckets) == len(pairs)
