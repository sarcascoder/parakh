"""Offline extractors for testing and demos — no model required.

`FixedExtractor` replays pre-computed predictions so the whole eval/correction
pipeline runs on CPU with zero dependencies. This is what the demo and tests use.
"""
from __future__ import annotations

from typing import Dict, Sequence

from ..metrics import FieldSpec


class FixedExtractor:
    """Replays a {doc_id: prediction} map. `document` is treated as the doc_id."""

    def __init__(self, predictions: Dict[str, Dict], name: str = "fixed"):
        self.name = name
        self._preds = predictions

    def extract(self, document: str, specs: Sequence[FieldSpec]) -> Dict:
        return dict(self._preds.get(document, {}))
