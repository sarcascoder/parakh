"""Extractor adapter interface.

Parakh does not do extraction itself — it measures and corrects whatever
extractor you already use. Adapters wrap any source of `dict` predictions:
a local VLM via an OpenAI-compatible endpoint, Docling/Marker output, or a
fixed set of pre-computed predictions for offline evaluation.
"""
from __future__ import annotations

from typing import Dict, Protocol, Sequence

from ..metrics import FieldSpec


class Extractor(Protocol):
    name: str

    def extract(self, document: str, specs: Sequence[FieldSpec]) -> Dict:
        """Return a dict of {field_name: value} for one document.

        `document` is text or a path, depending on the adapter.
        """
        ...
