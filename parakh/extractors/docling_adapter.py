"""Adapters that plug existing extractors into Parakh.

`MappingExtractor` adapts ANY extractor that already produced a dict: you give
it pre-computed output keyed by doc id, plus an optional key-rename map so its
field names line up with your schema. This is how you evaluate Docling / Marker
/ LlamaParse output without re-running them.

`DoclingExtractor` calls IBM Docling live if it's installed; it imports lazily
so Parakh's core stays dependency-free.
"""
from __future__ import annotations

from typing import Dict, Mapping, Sequence

from ..metrics import FieldSpec


class MappingExtractor:
    """Replay pre-computed extractor output, optionally renaming keys to the schema.

    rename: {source_field_name: schema_field_name}
    """

    def __init__(self, outputs: Dict[str, Dict], rename: Mapping[str, str] | None = None,
                 name: str = "mapping"):
        self.name = name
        self._outputs = outputs
        self._rename = dict(rename or {})

    def extract(self, document: str, specs: Sequence[FieldSpec]) -> Dict:
        raw = self._outputs.get(document, {})
        if not self._rename:
            return dict(raw)
        return {self._rename.get(k, k): v for k, v in raw.items()}


class DoclingExtractor:
    """Live Docling extraction. Requires `pip install docling` (lazy import)."""

    def __init__(self, name: str = "docling"):
        self.name = name

    def extract(self, document: str, specs: Sequence[FieldSpec]) -> Dict:
        try:
            from docling.document_converter import DocumentConverter  # type: ignore
        except ImportError as e:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "DoclingExtractor needs docling: pip install docling"
            ) from e
        converter = DocumentConverter()
        result = converter.convert(document)
        # Docling returns a rich document; expose markdown for a downstream
        # schema-extraction step. Most users pair this with MappingExtractor
        # after their own structuring pass.
        return {"_markdown": result.document.export_to_markdown()}
