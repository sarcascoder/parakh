"""Few-shot export — the 'gets smarter the more you use it' loop.

Every correction a reviewer saves becomes ground truth. Those verified
(document -> correct extraction) pairs are exactly what an LLM extractor needs
as few-shot examples. Feeding a handful back into the prompt measurably lifts
accuracy on the fields people kept getting wrong — without any fine-tuning.

Pure stdlib; fully testable without a model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .metrics import FieldSpec


@dataclass
class FewShotExample:
    input_text: str
    output: Dict


def build_examples(
    ground_truth: Dict[str, Dict],
    documents: Dict[str, str],
    *,
    limit: int = 5,
    prefer_docs: Optional[Sequence[str]] = None,
) -> List[FewShotExample]:
    """Pair verified extractions with their source document text.

    ground_truth : {doc_id: correct extraction}   (from the review loop)
    documents    : {doc_id: raw document text}
    prefer_docs  : doc ids to prioritise (e.g. ones that exercise weak fields)

    Only docs present in BOTH maps can become examples.
    """
    order: List[str] = []
    if prefer_docs:
        order.extend([d for d in prefer_docs if d in ground_truth and d in documents])
    for d in ground_truth:
        if d in documents and d not in order:
            order.append(d)
    return [FewShotExample(documents[d], ground_truth[d]) for d in order[:limit]]


def render_block(examples: Sequence[FewShotExample]) -> str:
    """Render examples as a prompt-injectable block."""
    if not examples:
        return ""
    parts: List[str] = ["Here are verified examples. Follow the same JSON shape exactly.\n"]
    for i, ex in enumerate(examples, 1):
        parts.append(f"Example {i} document:\n\"\"\"\n{ex.input_text}\n\"\"\"")
        parts.append(f"Example {i} correct output:\n{json.dumps(ex.output, ensure_ascii=False)}\n")
    return "\n".join(parts)


def examples_covering_fields(
    ground_truth: Dict[str, Dict],
    documents: Dict[str, str],
    weak_fields: Sequence[str],
    *,
    limit: int = 5,
) -> List[FewShotExample]:
    """Pick examples that actually contain the fields the model gets wrong, so
    the few-shot budget is spent where it helps most."""
    scored: List[tuple[int, str]] = []
    for doc_id, gt in ground_truth.items():
        if doc_id not in documents:
            continue
        coverage = sum(1 for f in weak_fields if gt.get(f) not in (None, "", []))
        scored.append((coverage, doc_id))
    scored.sort(reverse=True)
    ordered = [doc_id for _, doc_id in scored][:limit]
    return [FewShotExample(documents[d], ground_truth[d]) for d in ordered]
