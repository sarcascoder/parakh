"""Parakh — self-hosted accuracy evals + correction loop for document extraction.

Model-agnostic. Bring your own extractor (local VLM, Docling, Marker, an API).
Parakh measures field-level accuracy, derives calibrated confidence, and turns
human corrections into ground truth + few-shot examples — all on your machine.
"""
from .metrics import (
    DatasetReport,
    DocResult,
    FieldResult,
    FieldSpec,
    FieldType,
    evaluate,
    score_document,
    score_field,
)
from .confidence import (
    consensus_extraction,
    field_consistency,
    reliability_table,
    safe_auto_accept_threshold,
)
from .store import Store
from .queue import build_review_queue, DocReview, FieldReview
from .leaderboard import compare_models, Leaderboard, leaderboard_text
from .fewshot import build_examples, render_block, examples_covering_fields, FewShotExample
from .pipeline import Pipeline

__version__ = "0.1.0"

__all__ = [
    "FieldSpec", "FieldType", "FieldResult", "DocResult", "DatasetReport",
    "evaluate", "score_document", "score_field",
    "consensus_extraction", "field_consistency", "reliability_table",
    "safe_auto_accept_threshold", "Store",
    "build_review_queue", "DocReview", "FieldReview",
    "compare_models", "Leaderboard", "leaderboard_text",
    "build_examples", "render_block", "examples_covering_fields", "FewShotExample",
    "Pipeline",
    "__version__",
]
