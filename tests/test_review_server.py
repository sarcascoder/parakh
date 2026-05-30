"""End-to-end test of the stdlib review server: real socket, real HTTP."""
import json
import os
import threading
import urllib.request
from http.server import HTTPServer

from plumb.metrics import FieldSpec, FieldType
from plumb.review import ReviewApp, make_handler
from plumb.store import Store

SPECS = [FieldSpec("invoice_number", FieldType.EXACT),
         FieldSpec("total", FieldType.NUMBER, abs_tol=0.01)]


def _make_app(tmp_path) -> ReviewApp:
    store = Store(os.path.join(str(tmp_path), "plumb.db"))
    preds = {"d1": {"invoice_number": "A-1", "total": 100}}
    samples = {"d1": [
        {"invoice_number": "A-1", "total": 100},
        {"invoice_number": "A-9", "total": 100},  # flip -> low confidence
    ]}
    return ReviewApp(specs=SPECS, predictions=preds, store=store, samples=samples)


def test_queue_flags_low_confidence(tmp_path):
    app = _make_app(tmp_path)
    q = app.queue()
    assert q[0]["needs_review"]
    inv = [f for f in q[0]["fields"] if f["name"] == "invoice_number"][0]
    assert "low confidence" in inv["reason"]


def test_save_correction_writes_ground_truth(tmp_path):
    app = _make_app(tmp_path)
    res = app.save_correction("d1", {"invoice_number": "A-1"})
    assert res["ok"]
    assert app.store.ground_truth()["d1"]["invoice_number"] == "A-1"


def test_http_round_trip(tmp_path):
    app = _make_app(tmp_path)
    httpd = HTTPServer(("127.0.0.1", 0), make_handler(app))
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.handle_request)  # serve one GET
    t.start()
    body = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/queue", timeout=5).read()
    t.join(timeout=5)
    httpd.server_close()
    data = json.loads(body)
    assert data and data[0]["doc_id"] == "d1"
