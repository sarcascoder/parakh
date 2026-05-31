"""Calibrated confidence — without trusting the model's self-report.

Research is consistent that LLM self-reported confidence is poorly calibrated
("overconfident regardless of prompting strategy"). Parakh instead derives
confidence from signals that actually correlate with correctness:

1. Self-consistency: run the extractor N times (temperature > 0) and measure
   agreement per field. Fields that flip between runs are exactly the fields a
   human should review.
2. Calibration check: bucket fields by confidence and compare to measured
   accuracy on your labeled set (a reliability table). This tells you whether a
   confidence threshold is safe to auto-accept.

All offline-computable from a set of predictions — no model call needed here.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from .metrics import FieldSpec, FieldType
from .normalize import normalize_text, parse_number


def _canonical(spec: FieldSpec, value: Any) -> str:
    """Canonical key for agreement counting."""
    if spec.type == FieldType.NUMBER:
        n = parse_number(value)
        return "∅" if n is None else f"{n:.4f}"
    if spec.type == FieldType.TABLE:
        rows = value or []
        return f"rows:{len(rows)}"  # coarse agreement signal for tables
    return normalize_text(value) or "∅"


def field_consistency(spec: FieldSpec, samples: Sequence[Dict]) -> float:
    """Agreement = share of samples equal to the modal value for this field."""
    if not samples:
        return 0.0
    keys = [_canonical(spec, s.get(spec.name)) for s in samples]
    _, count = Counter(keys).most_common(1)[0]
    return count / len(keys)


@dataclass
class ConsistencyResult:
    consensus: Dict[str, Any]            # modal value per field
    confidence: Dict[str, float]         # agreement per field (0..1)

    def low_confidence_fields(self, threshold: float = 0.99) -> List[str]:
        return [f for f, c in self.confidence.items() if c < threshold]


def consensus_extraction(specs: Sequence[FieldSpec],
                         samples: Sequence[Dict]) -> ConsistencyResult:
    """Pick the modal value per field across repeated extractions and report
    the per-field agreement as confidence."""
    consensus: Dict[str, Any] = {}
    confidence: Dict[str, float] = {}
    for spec in specs:
        if not samples:
            consensus[spec.name], confidence[spec.name] = None, 0.0
            continue
        keyed: Dict[str, Any] = {}
        keys: List[str] = []
        for s in samples:
            v = s.get(spec.name)
            k = _canonical(spec, v)
            keys.append(k)
            keyed.setdefault(k, v)
        modal_key, count = Counter(keys).most_common(1)[0]
        consensus[spec.name] = keyed[modal_key]
        confidence[spec.name] = count / len(keys)
    return ConsistencyResult(consensus=consensus, confidence=confidence)


@dataclass
class CalibrationBucket:
    lo: float
    hi: float
    n: int
    accuracy: float


def reliability_table(pairs: Sequence[tuple[float, bool]],
                      bins: int = 5) -> List[CalibrationBucket]:
    """Given (confidence, was_correct) pairs, bucket and report measured
    accuracy per confidence band. A well-calibrated system has accuracy rising
    with confidence and roughly matching the band.
    """
    buckets: List[CalibrationBucket] = []
    width = 1.0 / bins
    for b in range(bins):
        lo, hi = b * width, (b + 1) * width
        in_bucket = [c for (conf, c) in pairs if (lo <= conf < hi) or (b == bins - 1 and conf == 1.0)]
        n = len(in_bucket)
        acc = (sum(1 for c in in_bucket if c) / n) if n else 0.0
        buckets.append(CalibrationBucket(lo=lo, hi=hi, n=n, accuracy=acc))
    return buckets


def safe_auto_accept_threshold(pairs: Sequence[tuple[float, bool]],
                               target_precision: float = 0.99) -> float:
    """Lowest confidence threshold at which accuracy of auto-accepted fields
    meets target precision. Returns 1.0 if no threshold is safe.

    This is the headline number for an AP/compliance team: "above X confidence
    you can auto-accept; below it, route to review."
    """
    if not pairs:
        return 1.0
    thresholds = sorted({round(c, 3) for c, _ in pairs})
    for t in thresholds:
        accepted = [ok for (c, ok) in pairs if c >= t]
        if accepted and (sum(1 for ok in accepted if ok) / len(accepted)) >= target_precision:
            return t
    return 1.0
