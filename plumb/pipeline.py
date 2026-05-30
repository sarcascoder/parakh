"""High-level Pipeline facade — the one object you drop into your code.

Wraps an extractor + schema + local store so a real pipeline is a few lines:

    from plumb import Pipeline, FieldSpec, FieldType
    from plumb.extractors import OpenAICompatExtractor

    pipe = Pipeline(
        schema=[FieldSpec("total", FieldType.NUMBER), ...],
        extractor=OpenAICompatExtractor(model="qwen2.5-vl"),
        store_path="plumb.db",
        consistency_runs=3,          # >1 enables self-consistency confidence
    )

    pipe.extract("inv_001", document_text)   # runs model, stores prediction(s)
    queue  = pipe.review_queue()             # what a human should check, worst-first
    report = pipe.evaluate()                 # field-level accuracy vs saved ground truth
    pipe.record_correction("inv_001", {...}) # reviewer fix -> ground truth + few-shot
    block  = pipe.fewshot_block({"inv_001": document_text})  # prime the next run

Everything is local and offline-friendly; the only network call is the
extractor itself, and only when you call extract().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .confidence import consensus_extraction
from .extractors.base import Extractor
from .fewshot import build_examples, examples_covering_fields, render_block
from .leaderboard import Leaderboard, compare_models
from .metrics import DatasetReport, FieldSpec, evaluate
from .queue import DocReview, build_review_queue
from .store import Store


@dataclass
class Pipeline:
    schema: Sequence[FieldSpec]
    extractor: Extractor
    store_path: str = "plumb.db"
    consistency_runs: int = 1
    confidence_threshold: float = 0.99

    store: Store = field(init=False)

    def __post_init__(self) -> None:
        if self.consistency_runs < 1:
            raise ValueError("consistency_runs must be >= 1")
        self.store = Store(self.store_path)

    # -- extraction ------------------------------------------------------- #
    def extract(self, doc_id: str, document: str, *, source_path: str = "") -> Dict:
        """Run the extractor (consistency_runs times) and persist the result(s).

        Returns the consensus prediction (modal value per field) when
        consistency_runs > 1, otherwise the single prediction.
        """
        self.store.add_document(doc_id, source_path)
        samples: List[Dict] = []
        for i in range(self.consistency_runs):
            pred = self.extractor.extract(document, self.schema)
            self.store.add_prediction(doc_id, pred, model=self.extractor.name, run=f"r{i}")
            samples.append(pred)
        if self.consistency_runs == 1:
            return samples[0]
        return consensus_extraction(self.schema, samples).consensus

    def extract_batch(self, documents: Dict[str, str]) -> Dict[str, Dict]:
        return {doc_id: self.extract(doc_id, text) for doc_id, text in documents.items()}

    # -- confidence & review --------------------------------------------- #
    def confidence(self, doc_id: str) -> Dict[str, float]:
        samples = self.store.samples(doc_id, model=self.extractor.name)
        if not samples:
            return {}
        return consensus_extraction(self.schema, samples).confidence

    def predictions(self) -> Dict[str, Dict]:
        """Best (consensus) prediction per document from stored samples."""
        out: Dict[str, Dict] = {}
        # Gather every doc that has at least one stored prediction.
        seen: Dict[str, List[Dict]] = {}
        for run in self._runs():
            for doc_id, pred in self.store.predictions(model=self.extractor.name, run=run).items():
                seen.setdefault(doc_id, []).append(pred)
        for doc_id, samples in seen.items():
            out[doc_id] = consensus_extraction(self.schema, samples).consensus
        return out

    def _runs(self) -> List[str]:
        return [f"r{i}" for i in range(self.consistency_runs)]

    def review_queue(self) -> List[DocReview]:
        preds = self.predictions()
        samples = {doc_id: self.store.samples(doc_id, model=self.extractor.name)
                   for doc_id in preds}
        return build_review_queue(
            self.schema, preds, ground_truth=self.store.ground_truth(),
            samples=samples, confidence_threshold=self.confidence_threshold,
        )

    # -- ground truth & evaluation --------------------------------------- #
    def record_correction(self, doc_id: str, corrected: Dict, *, merge: bool = True) -> Dict:
        """Persist a reviewer's corrected extraction as ground truth."""
        if merge:
            base = self.predictions().get(doc_id, {})
            merged = {**base, **corrected}
        else:
            merged = dict(corrected)
        self.store.set_ground_truth(doc_id, merged, reviewed_by="pipeline")
        return merged

    def evaluate(self, ground_truth: Optional[Dict[str, Dict]] = None) -> DatasetReport:
        gt = ground_truth if ground_truth is not None else self.store.ground_truth()
        return evaluate(self.schema, self.predictions(), gt)

    # -- few-shot feedback ------------------------------------------------ #
    def fewshot_block(self, documents: Dict[str, str], *,
                      weak_fields: Optional[Sequence[str]] = None, limit: int = 5) -> str:
        gt = self.store.ground_truth()
        if weak_fields:
            examples = examples_covering_fields(gt, documents, weak_fields, limit=limit)
        else:
            examples = build_examples(gt, documents, limit=limit)
        return render_block(examples)

    def close(self) -> None:
        self.store.close()
