import os

from plumb.store import Store


def test_store_roundtrip_predictions_and_truth(tmp_path):
    db = os.path.join(str(tmp_path), "plumb.db")
    s = Store(db)
    try:
        s.add_document("inv_001", "invoices/inv_001.pdf")
        s.add_prediction("inv_001", {"total": 100}, model="qwen", run="r1")
        s.add_prediction("inv_001", {"total": 101}, model="qwen", run="r2")
        s.set_ground_truth("inv_001", {"total": 100})

        assert s.predictions(model="qwen", run="r1") == {"inv_001": {"total": 100}}
        assert len(s.samples("inv_001", model="qwen")) == 2
        assert s.ground_truth() == {"inv_001": {"total": 100}}
    finally:
        s.close()
