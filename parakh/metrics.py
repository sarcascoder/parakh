"""Field-level extraction metrics — the core of Parakh.

Given a predicted extraction and a ground-truth extraction (both plain dicts
following a schema), compute per-field correctness using a comparison strategy
appropriate to each field's type, then aggregate to document- and dataset-level
scores.

This is deliberately model-agnostic and dependency-free: it works on the output
of *any* extractor (Docling, Marker, a local VLM, a cloud API).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from .normalize import normalize_text, parse_date, parse_number


class FieldType(str, Enum):
    STRING = "string"      # fuzzy + normalized comparison
    EXACT = "exact"        # ids, codes, enums — must match after light normalization
    NUMBER = "number"      # numeric/currency with tolerance
    DATE = "date"          # parsed date equality
    TABLE = "table"        # list[dict] line items


@dataclass
class FieldSpec:
    name: str
    type: FieldType = FieldType.STRING
    # For STRING: similarity at/above this counts as correct.
    threshold: float = 0.90
    # For NUMBER: absolute OR relative tolerance (whichever is looser).
    abs_tol: float = 0.01
    rel_tol: float = 0.0
    # For TABLE: the per-row sub-fields and how to compare them.
    columns: Sequence["FieldSpec"] = field(default_factory=tuple)
    weight: float = 1.0


@dataclass
class FieldResult:
    name: str
    score: float            # 0..1
    correct: bool
    predicted: Any
    expected: Any
    detail: str = ""


@dataclass
class DocResult:
    doc_id: str
    fields: List[FieldResult]

    @property
    def mean_score(self) -> float:
        if not self.fields:
            return 0.0
        return sum(f.score for f in self.fields) / len(self.fields)

    @property
    def exact(self) -> bool:
        """True only if every field is correct (document-level accuracy)."""
        return all(f.correct for f in self.fields) if self.fields else False


# --------------------------------------------------------------------------- #
# Scalar comparisons
# --------------------------------------------------------------------------- #

def _string_score(pred: Any, exp: Any) -> float:
    p, e = normalize_text(pred), normalize_text(exp)
    if p == e:
        return 1.0
    if not p and not e:
        return 1.0
    if not p or not e:
        return 0.0
    return SequenceMatcher(None, p, e).ratio()


def _exact_score(pred: Any, exp: Any) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(exp) else 0.0


def _number_score(pred: Any, exp: Any, abs_tol: float, rel_tol: float) -> float:
    p, e = parse_number(pred), parse_number(exp)
    if p is None and e is None:
        return 1.0
    if p is None or e is None:
        return 0.0
    diff = abs(p - e)
    tol = max(abs_tol, rel_tol * abs(e))
    return 1.0 if diff <= tol else 0.0


def _date_score(pred: Any, exp: Any) -> float:
    p, e = parse_date(pred), parse_date(exp)
    if p is None and e is None:
        return 1.0
    if p is None or e is None:
        return 0.0
    return 1.0 if p == e else 0.0


def _scalar_score(spec: FieldSpec, pred: Any, exp: Any) -> float:
    if spec.type == FieldType.EXACT:
        return _exact_score(pred, exp)
    if spec.type == FieldType.NUMBER:
        return _number_score(pred, exp, spec.abs_tol, spec.rel_tol)
    if spec.type == FieldType.DATE:
        return _date_score(pred, exp)
    return _string_score(pred, exp)


# --------------------------------------------------------------------------- #
# Table comparison (line items)
# --------------------------------------------------------------------------- #

def _row_score(columns: Sequence[FieldSpec], pred_row: Dict, exp_row: Dict) -> float:
    if not columns:
        return _string_score(str(pred_row), str(exp_row))
    total = sum(c.weight for c in columns) or 1.0
    acc = 0.0
    for c in columns:
        s = _scalar_score(c, (pred_row or {}).get(c.name), (exp_row or {}).get(c.name))
        acc += c.weight * s
    return acc / total


def _table_score(spec: FieldSpec, pred: Any, exp: Any) -> tuple[float, str]:
    """Greedy best-match row alignment, then row precision/recall -> F1.

    A row counts as a true positive when its weighted column score >= 0.5.
    Returns (f1, detail-string).
    """
    pred_rows: List[Dict] = list(pred or [])
    exp_rows: List[Dict] = list(exp or [])

    if not pred_rows and not exp_rows:
        return 1.0, "no rows expected, none predicted"
    if not exp_rows:
        return 0.0, f"{len(pred_rows)} rows predicted, 0 expected"
    if not pred_rows:
        return 0.0, f"0 rows predicted, {len(exp_rows)} expected"

    used_pred: set[int] = set()
    matched = 0
    cell_scores: List[float] = []
    for er in exp_rows:
        best_i, best_s = -1, -1.0
        for i, pr in enumerate(pred_rows):
            if i in used_pred:
                continue
            s = _row_score(spec.columns, pr, er)
            if s > best_s:
                best_s, best_i = s, i
        if best_i >= 0:
            used_pred.add(best_i)
            cell_scores.append(best_s)
            if best_s >= 0.5:
                matched += 1

    precision = matched / len(pred_rows)
    recall = matched / len(exp_rows)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    avg_cell = sum(cell_scores) / len(cell_scores) if cell_scores else 0.0
    detail = (
        f"rows matched {matched}/{len(exp_rows)} (P={precision:.2f} R={recall:.2f} "
        f"F1={f1:.2f}); avg matched-row cell score {avg_cell:.2f}"
    )
    return f1, detail


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def score_field(spec: FieldSpec, predicted: Dict, expected: Dict) -> FieldResult:
    pred_v = predicted.get(spec.name)
    exp_v = expected.get(spec.name)
    if spec.type == FieldType.TABLE:
        score, detail = _table_score(spec, pred_v, exp_v)
        correct = score >= spec.threshold
    else:
        score = _scalar_score(spec, pred_v, exp_v)
        correct = score >= (1.0 if spec.type in (FieldType.EXACT, FieldType.NUMBER, FieldType.DATE) else spec.threshold)
        detail = f"{spec.type.value} score {score:.2f}"
    return FieldResult(
        name=spec.name, score=score, correct=correct,
        predicted=pred_v, expected=exp_v, detail=detail,
    )


def score_document(specs: Sequence[FieldSpec], doc_id: str,
                   predicted: Dict, expected: Dict) -> DocResult:
    return DocResult(doc_id=doc_id,
                     fields=[score_field(s, predicted, expected) for s in specs])


@dataclass
class DatasetReport:
    docs: List[DocResult]
    specs: Sequence[FieldSpec]

    @property
    def document_accuracy(self) -> float:
        """Fraction of documents that are 100% correct end-to-end."""
        if not self.docs:
            return 0.0
        return sum(1 for d in self.docs if d.exact) / len(self.docs)

    @property
    def mean_field_score(self) -> float:
        scores = [f.score for d in self.docs for f in d.fields]
        return sum(scores) / len(scores) if scores else 0.0

    def per_field_accuracy(self) -> Dict[str, float]:
        """Field name -> fraction of docs where that field was correct.

        Surfacing the weakest fields is the single most actionable output:
        it tells you exactly where to spend prompt/model effort.
        """
        out: Dict[str, float] = {}
        for s in self.specs:
            vals = [f.correct for d in self.docs for f in d.fields if f.name == s.name]
            out[s.name] = (sum(1 for v in vals if v) / len(vals)) if vals else 0.0
        return out

    def weakest_fields(self, k: int = 3) -> List[tuple[str, float]]:
        return sorted(self.per_field_accuracy().items(), key=lambda kv: kv[1])[:k]


def evaluate(specs: Sequence[FieldSpec],
             predictions: Dict[str, Dict],
             ground_truth: Dict[str, Dict]) -> DatasetReport:
    """Evaluate predictions against ground truth, keyed by document id."""
    docs: List[DocResult] = []
    for doc_id, exp in ground_truth.items():
        pred = predictions.get(doc_id, {})
        docs.append(score_document(specs, doc_id, pred, exp))
    return DatasetReport(docs=docs, specs=list(specs))
