"""Value normalization helpers.

Document-extraction comparison fails on superficial differences: "$1,200.00" vs
"1200", "01/02/2026" vs "2026-01-02", "  ACME  Inc. " vs "Acme Inc.". These
helpers canonicalize values *before* comparison so the metrics engine measures
real disagreement, not formatting noise.

Pure stdlib, no third-party deps.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from typing import Optional

__all__ = [
    "normalize_text",
    "parse_number",
    "parse_date",
]

_WS = re.compile(r"\s+")


def normalize_text(value: object) -> str:
    """Lowercase, strip, collapse internal whitespace, drop accents/punctuation noise."""
    if value is None:
        return ""
    s = str(value)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = _WS.sub(" ", s)
    return s


_NUM_RE = re.compile(r"[-+]?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?|[-+]?\d*\.?\d+")


def parse_number(value: object) -> Optional[float]:
    """Extract a float from messy numeric/currency strings.

    Handles thousands separators, currency symbols, and trailing text.
    Returns None when no number can be found.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    # Drop currency symbols and spaces used as thousands separators.
    s = s.replace(" ", " ")
    m = _NUM_RE.search(s)
    if not m:
        return None
    token = m.group(0).replace(" ", "").replace(",", "")
    try:
        return float(token)
    except ValueError:
        return None


_DATE_FORMATS = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%d %b %Y",
    "%Y/%m/%d",
)


def parse_date(value: object) -> Optional[date]:
    """Parse a date from many common human formats. Returns None on failure.

    Ambiguous numeric dates are parsed with the first matching format; callers
    that need a fixed locale should normalize upstream.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = str(value).strip()
    if not s:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None
