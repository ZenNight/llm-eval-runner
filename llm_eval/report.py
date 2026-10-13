"""Rendering a RunReport as a terminal table, Markdown or JSON."""

from __future__ import annotations

import json
from pathlib import Path

from .runner import RunReport


def summary_lines(rep: RunReport) -> list[str]:
    return [
        f"provider      {rep.provider}",
        f"model         {rep.model}",
        f"cases         {rep.total}  (passed {rep.passed}, failed {rep.failed}, errors {rep.errors})",
        f"accuracy      {rep.accuracy * 100:.1f}%",
        f"latency       p50 {rep.latency_p50_ms:.0f} ms, p95 {rep.latency_p95_ms:.0f} ms",
        f"tokens        in {rep.input_tokens}, out {rep.output_tokens}",
        f"cost          ${rep.cost_usd:.4f}",
        f"wall time     {rep.wall_time_s:.1f} s",
    ]


def render_text(rep: RunReport, show_failures: bool = True) -> str:
    out = list(summary_lines(rep))
    if show_failures:
        bad = [r for r in rep.results if not r.passed]
        if bad:
            out.append("")
            out.append("failures:")
            for r in bad:
                reason = r.error or f"got: {r.answer[:120]!r}"
                out.append(f"  {r.id:<10} expected {r.expected}  {reason}")
    return "\n".join(out)


def render_markdown(rep: RunReport) -> str:
    rows = [
        "| id | passed | latency ms | answer |",
        "|---|---|---|---|",
    ]
    for r in rep.results:
        ans = (r.error or r.answer).replace("|", "\\|").replace("\n", " ")[:80]
        rows.append(f"| {r.id} | {'yes' if r.passed else 'no'} | {r.latency_ms:.0f} | {ans} |")
    head = [f"## {rep.provider} / {rep.model}", "", "```", *summary_lines(rep), "```", ""]
    return "\n".join(head + rows) + "\n"


def write_json(rep: RunReport, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rep.to_dict(), fh, ensure_ascii=False, indent=2)
