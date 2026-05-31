"""Local, zero-dependency review UI built on the stdlib http.server.

The reviewer sees the worst-first queue, each field annotated with why it's
flagged (wrong vs. low confidence), edits the values, and saves. Saving writes
the corrected extraction back to the Store as ground truth — which immediately
improves future evals and can be exported as few-shot examples.

`ReviewApp` holds all logic and is unit-testable without binding a socket;
`serve()` wires it to an HTTPServer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..metrics import FieldSpec, FieldType
from ..queue import build_review_queue
from ..store import Store


@dataclass
class ReviewApp:
    specs: Sequence[FieldSpec]
    predictions: Dict[str, Dict]
    store: Store
    samples: Optional[Dict[str, List[Dict]]] = None
    confidence_threshold: float = 0.99

    # -- data ------------------------------------------------------------- #
    def queue(self) -> List[dict]:
        gt = self.store.ground_truth()
        items = build_review_queue(
            self.specs, self.predictions, ground_truth=gt,
            samples=self.samples, confidence_threshold=self.confidence_threshold,
        )
        type_by_name = {s.name: s.type.value for s in self.specs}
        out: List[dict] = []
        for it in items:
            out.append({
                "doc_id": it.doc_id,
                "priority": round(it.priority, 3),
                "needs_review": it.needs_review,
                "fields": [{
                    "name": f.name,
                    "type": type_by_name.get(f.name, "string"),
                    "value": f.value,
                    "confidence": f.confidence,
                    "correct": f.correct,
                    "reason": f.reason,
                } for f in it.fields],
            })
        return out

    def save_correction(self, doc_id: str, corrected: Dict) -> dict:
        """Persist a corrected extraction as ground truth."""
        if doc_id not in self.predictions:
            return {"ok": False, "error": f"unknown doc_id {doc_id!r}"}
        # Start from current prediction, overlay reviewer edits.
        merged = dict(self.predictions[doc_id])
        merged.update(corrected)
        self.store.set_ground_truth(doc_id, merged, reviewed_by="ui")
        return {"ok": True, "doc_id": doc_id, "saved_fields": list(corrected.keys())}

    # -- routing (returns status, content_type, body) --------------------- #
    def handle(self, method: str, path: str, body: bytes = b"") -> Tuple[int, str, bytes]:
        if method == "GET" and path in ("/", "/index.html"):
            return 200, "text/html; charset=utf-8", _PAGE.encode()
        if method == "GET" and path == "/api/queue":
            return 200, "application/json", json.dumps(self.queue()).encode()
        if method == "POST" and path == "/api/correct":
            try:
                payload = json.loads(body or b"{}")
                result = self.save_correction(payload["doc_id"], payload.get("fields", {}))
                code = 200 if result.get("ok") else 400
                return code, "application/json", json.dumps(result).encode()
            except Exception as e:  # noqa: BLE001 - surface error to client
                return 400, "application/json", json.dumps({"ok": False, "error": str(e)}).encode()
        return 404, "text/plain", b"not found"


def make_handler(app: ReviewApp):
    from http.server import BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def _respond(self, method: str) -> None:
            length = int(self.headers.get("Content-Length", 0) or 0)
            body = self.rfile.read(length) if length else b""
            code, ctype, payload = app.handle(method, self.path, body)
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):   # noqa: N802
            self._respond("GET")

        def do_POST(self):  # noqa: N802
            self._respond("POST")

        def log_message(self, *_):  # silence
            pass

    return Handler


def serve(app: ReviewApp, host: str = "127.0.0.1", port: int = 8000) -> None:  # pragma: no cover
    from http.server import HTTPServer

    httpd = HTTPServer((host, port), make_handler(app))
    print(f"Parakh review UI → http://{host}:{port}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.server_close()


_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Parakh Review</title>
<style>
 body{font:14px/1.5 system-ui,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
 header{padding:16px 24px;background:#161922;border-bottom:1px solid #262b36}
 header h1{margin:0;font-size:18px} header span{color:#8a93a6}
 main{padding:24px;max-width:880px;margin:0 auto}
 .doc{background:#161922;border:1px solid #262b36;border-radius:10px;padding:16px;margin-bottom:18px}
 .doc h2{margin:0 0 10px;font-size:15px}
 .pri{float:right;font-size:12px;color:#f0a020}
 .f{display:grid;grid-template-columns:140px 1fr;gap:10px;align-items:center;margin:6px 0}
 .f label{color:#8a93a6}
 .f input,.f textarea{width:100%;background:#0f1115;border:1px solid #333a47;color:#e6e6e6;border-radius:6px;padding:6px 8px;font:13px monospace}
 .flag{font-size:11px;border-radius:10px;padding:1px 8px;margin-left:6px}
 .bad{background:#3a1620;color:#ff6b81} .warn{background:#3a3216;color:#f0c020}
 button{background:#3b82f6;color:#fff;border:0;border-radius:6px;padding:8px 16px;cursor:pointer;font-size:13px}
 .ok{color:#46c46a;margin-left:10px} .empty{color:#8a93a6;text-align:center;padding:40px}
</style></head><body>
<header><h1>Parakh — review queue <span>· corrections saved as ground truth, locally</span></h1></header>
<main id="app"><p class="empty">Loading…</p></main>
<script>
async function load(){
 const q = await (await fetch('/api/queue')).json();
 const app = document.getElementById('app');
 const todo = q.filter(d=>d.needs_review);
 if(!todo.length){app.innerHTML='<p class="empty">Nothing to review — all clear.</p>';return;}
 app.innerHTML='';
 for(const d of todo){
   const div=document.createElement('div');div.className='doc';
   let h='<span class="pri">priority '+d.priority+'</span><h2>'+d.doc_id+'</h2>';
   for(const f of d.fields){
     const isTable=f.type==='table';
     const val=isTable?JSON.stringify(f.value,null,1):(f.value==null?'':f.value);
     let flag='';
     if(f.correct===false)flag+='<span class="flag bad">wrong</span>';
     if(f.confidence!=null&&f.confidence<0.99)flag+='<span class="flag warn">conf '+Math.round(f.confidence*100)+'%</span>';
     h+='<div class="f"><label>'+f.name+flag+'</label>'+
        (isTable?'<textarea rows="4" data-f="'+f.name+'" data-t="table">'+val+'</textarea>'
                :'<input data-f="'+f.name+'" value="'+String(val).replace(/"/g,'&quot;')+'">')+'</div>';
   }
   h+='<div style="margin-top:10px"><button onclick="save(this,\\''+d.doc_id+'\\')">Save as ground truth</button><span class="ok" id="ok-'+d.doc_id+'"></span></div>';
   div.innerHTML=h;app.appendChild(div);
 }
}
async function save(btn,docId){
 const root=btn.closest('.doc');const fields={};
 root.querySelectorAll('[data-f]').forEach(el=>{
   fields[el.dataset.f]= el.dataset.t==='table'? JSON.parse(el.value||'[]') : el.value;
 });
 const r=await (await fetch('/api/correct',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({doc_id:docId,fields})})).json();
 document.getElementById('ok-'+docId).textContent = r.ok? '✓ saved':'✗ '+r.error;
}
load();
</script></body></html>"""
