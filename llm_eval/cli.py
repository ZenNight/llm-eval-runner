"""Command line entry point.

Examples:

    llm-eval run examples/arithmetic.jsonl --provider mock
    llm-eval run examples/aspen.jsonl --provider lmstudio --model glm-4-9b-chat
    llm-eval run examples/aspen.jsonl --provider openai --model gpt-4o-mini \
        --price-in 0.15 --price-out 0.60 --json results/gpt4o-mini.json
    llm-eval compare results/*.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .dataset import filter_by_tag, load_jsonl
from .providers import make_provider
from .report import render_markdown, render_text, write_json
from .runner import run


def _cmd_run(args: argparse.Namespace) -> int:
    cases = filter_by_tag(load_jsonl(args.dataset), args.tag)
    if args.limit:
        cases = cases[: args.limit]
    provider = make_provider(
        args.provider, model=args.model, base_url=args.base_url,
        **({} if args.provider == "mock" else {"temperature": args.temperature, "max_tokens": args.max_tokens}),
    )

    def progress(res):
        if not args.quiet:
            mark = "ok " if res.passed else ("ERR" if res.error else "FAIL")
            print(f"[{mark}] {res.id}", file=sys.stderr)

    rep = run(
        cases, provider, model=args.model, system=args.system,
        price_in_per_million=args.price_in, price_out_per_million=args.price_out,
        on_case=progress,
    )
    if args.markdown:
        print(render_markdown(rep))
    else:
        print(render_text(rep))
    if args.json:
        write_json(rep, args.json)
        print(f"\nsaved {args.json}", file=sys.stderr)
    return 0 if rep.errors == 0 else 2


def _cmd_compare(args: argparse.Namespace) -> int:
    rows = []
    for p in args.files:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        rows.append((d["provider"], d["model"], d["accuracy"], d["latency_p50_ms"], d["cost_usd"], d["total"]))
    rows.sort(key=lambda r: -r[2])
    print("| provider | model | accuracy | p50 ms | cost USD | cases |")
    print("|---|---|---|---|---|---|")
    for prov, model, acc, p50, cost, n in rows:
        print(f"| {prov} | {model} | {acc * 100:.1f}% | {p50:.0f} | {cost:.4f} | {n} |")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llm-eval", description="Evaluate an LLM endpoint on a JSONL question set.")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a dataset against one model")
    r.add_argument("dataset", help="path to a .jsonl file")
    r.add_argument("--provider", default="mock", choices=["mock", "openai", "lmstudio", "anthropic"])
    r.add_argument("--model", default="mock")
    r.add_argument("--base-url", dest="base_url", default=None, help="override the endpoint base URL")
    r.add_argument("--system", default=None, help="system prompt applied to every case")
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--max-tokens", dest="max_tokens", type=int, default=512)
    r.add_argument("--price-in", dest="price_in", type=float, default=0.0, help="USD per 1M input tokens")
    r.add_argument("--price-out", dest="price_out", type=float, default=0.0, help="USD per 1M output tokens")
    r.add_argument("--tag", default=None, help="only run cases carrying this tag")
    r.add_argument("--limit", type=int, default=0, help="run only the first N cases")
    r.add_argument("--json", default=None, help="write the full report to this path")
    r.add_argument("--markdown", action="store_true", help="print a Markdown table instead of plain text")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=_cmd_run)

    c = sub.add_parser("compare", help="print a leaderboard from saved JSON reports")
    c.add_argument("files", nargs="+")
    c.set_defaults(func=_cmd_compare)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
