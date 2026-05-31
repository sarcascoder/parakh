"""Model/prompt leaderboard — on *your* documents, not a generic benchmark.

Given several candidate prediction sets (e.g. qwen2.5-vl vs granite-docling vs
your fine-tune) evaluated against the same ground truth, rank them and, crucially,
report the **best model per field**. Real pipelines are often won by routing:
model A nails totals, model B nails vendor names.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .metrics import DatasetReport, FieldSpec, evaluate


@dataclass
class Entry:
    model: str
    document_accuracy: float
    mean_field_score: float
    per_field: Dict[str, float]


@dataclass
class Leaderboard:
    entries: List[Entry]

    @property
    def ranked(self) -> List[Entry]:
        return sorted(self.entries, key=lambda e: (e.document_accuracy, e.mean_field_score),
                      reverse=True)

    def best_per_field(self) -> Dict[str, tuple[str, float]]:
        """field -> (winning model, its accuracy on that field)."""
        out: Dict[str, tuple[str, float]] = {}
        if not self.entries:
            return out
        for fld in self.entries[0].per_field:
            best_model, best_acc = "", -1.0
            for e in self.entries:
                acc = e.per_field.get(fld, 0.0)
                if acc > best_acc:
                    best_model, best_acc = e.model, acc
            out[fld] = (best_model, best_acc)
        return out


def compare_models(
    specs: Sequence[FieldSpec],
    ground_truth: Dict[str, Dict],
    predictions_by_model: Dict[str, Dict[str, Dict]],
) -> Leaderboard:
    entries: List[Entry] = []
    for model, preds in predictions_by_model.items():
        rep: DatasetReport = evaluate(specs, preds, ground_truth)
        entries.append(Entry(
            model=model,
            document_accuracy=rep.document_accuracy,
            mean_field_score=rep.mean_field_score,
            per_field=rep.per_field_accuracy(),
        ))
    return Leaderboard(entries=entries)


def leaderboard_text(lb: Leaderboard) -> str:
    lines = ["MODEL LEADERBOARD (your documents)", "-" * 48]
    for i, e in enumerate(lb.ranked, 1):
        lines.append(f"{i}. {e.model:<18} doc-acc {e.document_accuracy:.0%}  "
                     f"field {e.mean_field_score:.0%}")
    lines.append("\nBest model per field (route for max accuracy):")
    for fld, (model, acc) in lb.best_per_field().items():
        lines.append(f"  {fld:<18} -> {model} ({acc:.0%})")
    return "\n".join(lines)
