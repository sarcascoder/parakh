"""Human-readable reports from a DatasetReport."""
from __future__ import annotations

from typing import List

from .metrics import DatasetReport


def _bar(score: float, width: int = 20) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def text_report(report: DatasetReport) -> str:
    lines: List[str] = []
    lines.append("=" * 56)
    lines.append("  PARAKH EXTRACTION REPORT")
    lines.append("=" * 56)
    lines.append(f"  Documents evaluated : {len(report.docs)}")
    lines.append(f"  Document accuracy   : {report.document_accuracy:.1%}  "
                 "(docs that are 100% correct)")
    lines.append(f"  Mean field score    : {report.mean_field_score:.1%}")
    lines.append("")
    lines.append("  Per-field accuracy (worst first):")
    for name, acc in sorted(report.per_field_accuracy().items(), key=lambda kv: kv[1]):
        lines.append(f"    {name:<18} {_bar(acc)} {acc:.0%}")
    lines.append("")
    weakest = report.weakest_fields(3)
    if weakest:
        names = ", ".join(f"{n} ({a:.0%})" for n, a in weakest)
        lines.append(f"  ► Focus your prompt/model effort here: {names}")
    lines.append("=" * 56)
    return "\n".join(lines)


def markdown_report(report: DatasetReport) -> str:
    md: List[str] = []
    md.append("# Parakh Extraction Report\n")
    md.append(f"- **Documents evaluated:** {len(report.docs)}")
    md.append(f"- **Document accuracy (100% correct):** {report.document_accuracy:.1%}")
    md.append(f"- **Mean field score:** {report.mean_field_score:.1%}\n")
    md.append("## Per-field accuracy\n")
    md.append("| Field | Accuracy |")
    md.append("|---|---|")
    for name, acc in sorted(report.per_field_accuracy().items(), key=lambda kv: kv[1]):
        md.append(f"| {name} | {acc:.0%} |")
    return "\n".join(md)
