from parakh.metrics import FieldSpec, FieldType
from parakh.leaderboard import compare_models

SPECS = [FieldSpec("a", FieldType.EXACT), FieldSpec("b", FieldType.EXACT)]
GT = {"d1": {"a": "x", "b": "y"}, "d2": {"a": "x", "b": "y"}}


def test_ranking_and_best_per_field():
    preds = {
        # model_a perfect on a, wrong on b
        "model_a": {"d1": {"a": "x", "b": "no"}, "d2": {"a": "x", "b": "no"}},
        # model_b wrong on a, perfect on b
        "model_b": {"d1": {"a": "no", "b": "y"}, "d2": {"a": "no", "b": "y"}},
    }
    lb = compare_models(SPECS, GT, preds)
    bpf = lb.best_per_field()
    assert bpf["a"][0] == "model_a"
    assert bpf["b"][0] == "model_b"
    # neither model gets a full document right -> doc accuracy 0 for both
    assert all(e.document_accuracy == 0.0 for e in lb.entries)
