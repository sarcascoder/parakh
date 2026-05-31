from parakh.metrics import FieldSpec, FieldType, evaluate, score_field


def test_exact_match_normalizes_case_and_space():
    spec = FieldSpec("id", FieldType.EXACT)
    assert score_field(spec, {"id": " A-1001 "}, {"id": "a-1001"}).correct


def test_number_tolerance_and_currency_formatting():
    spec = FieldSpec("total", FieldType.NUMBER, abs_tol=0.01)
    assert score_field(spec, {"total": "$1,200.00"}, {"total": 1200.0}).correct
    assert not score_field(spec, {"total": 504.50}, {"total": 540.50}).correct


def test_relative_tolerance():
    spec = FieldSpec("amt", FieldType.NUMBER, abs_tol=0.0, rel_tol=0.05)
    assert score_field(spec, {"amt": 102}, {"amt": 100}).correct      # within 5%
    assert not score_field(spec, {"amt": 110}, {"amt": 100}).correct  # outside 5%


def test_date_format_invariance():
    spec = FieldSpec("d", FieldType.DATE)
    assert score_field(spec, {"d": "01/15/2026"}, {"d": "2026-01-15"}).correct
    assert score_field(spec, {"d": "January 15, 2026"}, {"d": "2026-01-15"}).correct


def test_string_fuzzy_threshold():
    spec = FieldSpec("vendor", FieldType.STRING, threshold=0.85)
    assert score_field(spec, {"vendor": "ACME, Inc"}, {"vendor": "Acme Inc."}).correct
    assert not score_field(spec, {"vendor": "Globex"}, {"vendor": "Initech"}).correct


def test_table_recall_penalizes_missing_rows():
    spec = FieldSpec("items", FieldType.TABLE, threshold=0.8, columns=(
        FieldSpec("desc", FieldType.STRING, threshold=0.8),
        FieldSpec("amount", FieldType.NUMBER),
    ))
    exp = {"items": [{"desc": "A", "amount": 10}, {"desc": "B", "amount": 20}]}
    pred_missing = {"items": [{"desc": "A", "amount": 10}]}
    pred_full = {"items": [{"desc": "A", "amount": 10}, {"desc": "B", "amount": 20}]}
    assert score_field(spec, pred_full, exp).correct
    r = score_field(spec, pred_missing, exp)
    assert not r.correct and 0 < r.score < 1   # F1 ~0.67


def test_dataset_aggregation_and_weakest_fields():
    specs = [FieldSpec("a", FieldType.EXACT), FieldSpec("b", FieldType.EXACT)]
    truth = {"d1": {"a": "x", "b": "y"}, "d2": {"a": "x", "b": "y"}}
    preds = {"d1": {"a": "x", "b": "WRONG"}, "d2": {"a": "x", "b": "y"}}
    rep = evaluate(specs, preds, truth)
    pf = rep.per_field_accuracy()
    assert pf["a"] == 1.0
    assert pf["b"] == 0.5
    assert rep.document_accuracy == 0.5          # only d2 fully correct
    assert rep.weakest_fields(1)[0][0] == "b"
