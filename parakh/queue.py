"""Review queue — decide which documents/fields a human should look at, and why.

Two independent signals push a field into the queue:
  * it disagrees with existing ground truth (a known error), and/or
  * it has low self-consistency confidence (the model flip-flops across runs).

Documents are ranked worst-first so a reviewer's time goes where it matters.
This is pure logic — no server, no model — so it is fully unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .confidence import consensus_extraction
from .metrics import FieldSpec, score_document


@dataclass
class FieldReview:
    name: str
    value: object
    confidence: Optional[float]      # None when no repeated samples available
    correct: Optional[bool]          # None when no ground truth yet
    reason: str


@dataclass
class DocReview:
    doc_id: str
    fields: List[FieldReview]
    priority: float = 0.0            # higher = review sooner

    @property
    def needs_review(self) -> bool:
        return any(f.reason for f in self.fields)


def build_review_queue(
    specs: Sequence[FieldSpec],
    predictions: Dict[str, Dict],
    *,
    ground_truth: Optional[Dict[str, Dict]] = None,
    samples: Optional[Dict[str, List[Dict]]] = None,
    confidence_threshold: float = 0.99,
) -> List[DocReview]:
    """Build a worst-first review queue.

    predictions          : {doc_id: prediction}
    ground_truth         : optional {doc_id: verified extraction}
    samples              : optional {doc_id: [prediction, ...]} for self-consistency
    """
    ground_truth = ground_truth or {}
    samples = samples or {}
    reviews: List[DocReview] = []

    for doc_id, pred in predictions.items():
        conf: Dict[str, float] = {}
        if doc_id in samples:
            conf = consensus_extraction(specs, samples[doc_id]).confidence

        correctness: Dict[str, bool] = {}
        if doc_id in ground_truth:
            dr = score_document(specs, doc_id, pred, ground_truth[doc_id])
            correctness = {f.name: f.correct for f in dr.fields}

        field_reviews: List[FieldReview] = []
        priority = 0.0
        for s in specs:
            c = conf.get(s.name)
            ok = correctness.get(s.name)
            reasons: List[str] = []
            if ok is False:
                reasons.append("disagrees with ground truth")
                priority += 1.0
            if c is not None and c < confidence_threshold:
                reasons.append(f"low confidence ({c:.0%})")
                priority += (1.0 - c)
            field_reviews.append(FieldReview(
                name=s.name, value=pred.get(s.name),
                confidence=c, correct=ok, reason="; ".join(reasons),
            ))
        reviews.append(DocReview(doc_id=doc_id, fields=field_reviews, priority=priority))

    reviews.sort(key=lambda d: d.priority, reverse=True)
    return reviews
