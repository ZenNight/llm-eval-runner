"""Loading and validating evaluation datasets.

A dataset is a JSONL file. Each line is one case:

    {"id": "q01", "prompt": "What is 2 + 2?", "expected": "4", "match": "exact"}

Fields:
    id        unique string
    prompt    text sent to the model
    expected  reference answer (string, or list of acceptable strings)
    match     "exact" | "contains" | "regex" | "numeric" (default "contains")
    tolerance optional float for "numeric" (absolute, default 0.0)
    system    optional system prompt override for this case
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

VALID_MATCH = {"exact", "contains", "regex", "numeric"}


@dataclass
class Case:
    id: str
    prompt: str
    expected: list[str]
    match: str = "contains"
    tolerance: float = 0.0
    system: str | None = None
    tags: list[str] = field(default_factory=list)


def parse_case(raw: dict, line_no: int) -> Case:
    for key in ("id", "prompt", "expected"):
        if key not in raw:
            raise ValueError(f"line {line_no}: missing required field '{key}'")
    match = raw.get("match", "contains")
    if match not in VALID_MATCH:
        raise ValueError(f"line {line_no}: unknown match type '{match}' (valid: {sorted(VALID_MATCH)})")
    expected = raw["expected"]
    if isinstance(expected, (str, int, float)):
        expected = [str(expected)]
    elif isinstance(expected, list):
        expected = [str(e) for e in expected]
    else:
        raise ValueError(f"line {line_no}: 'expected' must be a string or a list of strings")
    return Case(
        id=str(raw["id"]),
        prompt=str(raw["prompt"]),
        expected=expected,
        match=match,
        tolerance=float(raw.get("tolerance", 0.0)),
        system=raw.get("system"),
        tags=list(raw.get("tags", [])),
    )


def load_jsonl(path: str | Path) -> list[Case]:
    cases: list[Case] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            case = parse_case(json.loads(line), line_no)
            if case.id in seen:
                raise ValueError(f"line {line_no}: duplicate id '{case.id}'")
            seen.add(case.id)
            cases.append(case)
    if not cases:
        raise ValueError(f"{path}: dataset is empty")
    return cases


def filter_by_tag(cases: Iterable[Case], tag: str | None) -> list[Case]:
    if tag is None:
        return list(cases)
    return [c for c in cases if tag in c.tags]
