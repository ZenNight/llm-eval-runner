"""Scoring a model answer against the expected answers of a case.

Scoring is deliberately deterministic. There is no LLM judge here on purpose:
a judge adds cost, nondeterminism and a second model to debug. If a case needs
fuzzy judgement, write a regex or a numeric tolerance instead.
"""

from __future__ import annotations

import re

from .dataset import Case

_NUMBER = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


def normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip surrounding punctuation."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,;:!?\"'`")


def _first_number(text: str) -> float | None:
    m = _NUMBER.search(text.replace(",", "."))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", "."))
    except ValueError:
        return None


def score(case: Case, answer: str) -> bool:
    """Return True if the answer satisfies at least one expected value."""
    ans = normalize(answer)

    if case.match == "exact":
        return any(ans == normalize(e) for e in case.expected)

    if case.match == "contains":
        return any(normalize(e) in ans for e in case.expected)

    if case.match == "regex":
        return any(re.search(e, answer, flags=re.IGNORECASE | re.DOTALL) for e in case.expected)

    if case.match == "numeric":
        got = _first_number(answer)
        if got is None:
            return False
        for e in case.expected:
            want = _first_number(e)
            if want is not None and abs(got - want) <= case.tolerance:
                return True
        return False

    raise ValueError(f"unknown match type: {case.match}")
