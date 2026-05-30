"""Local SQLite store for documents, predictions, ground truth and corrections.

Everything stays on disk, on the user's machine — no data leaves. Corrections
made in the review UI are written back here and double as (a) ground truth for
future evals and (b) few-shot examples for the extractor.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    source_path TEXT,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS predictions (
    doc_id TEXT,
    model TEXT,
    run TEXT,
    payload TEXT,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS ground_truth (
    doc_id TEXT PRIMARY KEY,
    payload TEXT,
    reviewed_by TEXT,
    updated_at REAL
);
"""


@dataclass
class Store:
    path: str

    def __post_init__(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False + a lock: the review server handles requests
        # from a different thread than the one that opened the connection.
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(_SCHEMA)
            self._db.commit()

    # -- documents -------------------------------------------------------- #
    def add_document(self, doc_id: str, source_path: str = "") -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO documents(doc_id, source_path, created_at) VALUES (?,?,?)",
                (doc_id, source_path, time.time()),
            )
            self._db.commit()

    # -- predictions ------------------------------------------------------ #
    def add_prediction(self, doc_id: str, payload: Dict, model: str = "default",
                       run: str = "default") -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO predictions(doc_id, model, run, payload, created_at) VALUES (?,?,?,?,?)",
                (doc_id, model, run, json.dumps(payload), time.time()),
            )
            self._db.commit()

    def predictions(self, model: str = "default", run: str = "default") -> Dict[str, Dict]:
        with self._lock:
            cur = self._db.execute(
                "SELECT doc_id, payload FROM predictions WHERE model=? AND run=?", (model, run)
            )
            rows = cur.fetchall()
        return {doc_id: json.loads(p) for doc_id, p in rows}

    def samples(self, doc_id: str, model: str = "default") -> List[Dict]:
        """All prediction runs for one doc/model — used for self-consistency."""
        with self._lock:
            cur = self._db.execute(
                "SELECT payload FROM predictions WHERE doc_id=? AND model=?", (doc_id, model)
            )
            rows = cur.fetchall()
        return [json.loads(p) for (p,) in rows]

    # -- ground truth ----------------------------------------------------- #
    def set_ground_truth(self, doc_id: str, payload: Dict, reviewed_by: str = "human") -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO ground_truth(doc_id, payload, reviewed_by, updated_at) VALUES (?,?,?,?)",
                (doc_id, json.dumps(payload), reviewed_by, time.time()),
            )
            self._db.commit()

    def ground_truth(self) -> Dict[str, Dict]:
        with self._lock:
            cur = self._db.execute("SELECT doc_id, payload FROM ground_truth")
            rows = cur.fetchall()
        return {doc_id: json.loads(p) for doc_id, p in rows}

    def close(self) -> None:
        with self._lock:
            self._db.close()
