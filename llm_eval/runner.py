"""Run a list of cases against a provider and aggregate the results."""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass, field

from .dataset import Case
from .providers import Provider
from .scoring import score


@dataclass
class CaseResult:
    id: str
    prompt: str
    expected: list[str]
    answer: str
    passed: bool
    latency_ms: float
    input_tokens: int
    output_tokens: int
    error: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class RunReport:
    provider: str
    model: str
    total: int
    passed: int
    failed: int
    errors: int
    accuracy: float
    latency_p50_ms: float
    latency_p95_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    wall_time_s: float
    results: list[CaseResult]

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = max(0, min(len(values) - 1, round(pct / 100 * (len(values) - 1))))
    return values[k]


def run(
    cases: list[Case],
    provider: Provider,
    model: str,
    system: str | None = None,
    price_in_per_million: float = 0.0,
    price_out_per_million: float = 0.0,
    on_case=None,
) -> RunReport:
    results: list[CaseResult] = []
    t0 = time.perf_counter()
    for case in cases:
        try:
            comp = provider.complete(case.prompt, system=case.system or system)
            passed = score(case, comp.text)
            res = CaseResult(
                id=case.id, prompt=case.prompt, expected=case.expected, answer=comp.text,
                passed=passed, latency_ms=comp.latency_ms,
                input_tokens=comp.input_tokens, output_tokens=comp.output_tokens, tags=case.tags,
            )
        except Exception as exc:  # noqa: BLE001  a failing endpoint must not kill the whole run
            res = CaseResult(
                id=case.id, prompt=case.prompt, expected=case.expected, answer="",
                passed=False, latency_ms=0.0, input_tokens=0, output_tokens=0,
                error=f"{type(exc).__name__}: {exc}", tags=case.tags,
            )
        results.append(res)
        if on_case:
            on_case(res)
    wall = time.perf_counter() - t0

    ok = [r for r in results if r.error is None]
    latencies = [r.latency_ms for r in ok]
    passed = sum(r.passed for r in results)
    errors = sum(r.error is not None for r in results)
    tin = sum(r.input_tokens for r in results)
    tout = sum(r.output_tokens for r in results)
    cost = tin / 1e6 * price_in_per_million + tout / 1e6 * price_out_per_million

    return RunReport(
        provider=provider.name,
        model=model,
        total=len(results),
        passed=passed,
        failed=len(results) - passed - errors,
        errors=errors,
        accuracy=(passed / len(results)) if results else 0.0,
        latency_p50_ms=statistics.median(latencies) if latencies else 0.0,
        latency_p95_ms=_percentile(latencies, 95),
        input_tokens=tin,
        output_tokens=tout,
        cost_usd=cost,
        wall_time_s=wall,
        results=results,
    )
