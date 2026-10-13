import json
from llm_eval.cli import main
from llm_eval.dataset import load_jsonl
from llm_eval.providers import MockProvider
from llm_eval.runner import run


class BrokenProvider(MockProvider):
    def complete(self, prompt, system=None):
        raise RuntimeError("endpoint down")


def test_run_aggregates_accuracy_tokens_and_cost():
    cases = load_jsonl("examples/arithmetic.jsonl")
    answers = {c.prompt: c.expected[0] for c in cases[:3]}  # answer 3 of 5 correctly
    prov = MockProvider(answers=answers, fallback="no idea")
    rep = run(cases, prov, model="mock", price_in_per_million=1_000_000, price_out_per_million=2_000_000)
    assert rep.total == 5 and rep.passed == 3 and rep.failed == 2 and rep.errors == 0
    assert abs(rep.accuracy - 0.6) < 1e-9
    assert rep.cost_usd == rep.input_tokens * 1.0 + rep.output_tokens * 2.0
    assert prov.calls == 5


def test_provider_errors_are_recorded_not_raised():
    cases = load_jsonl("examples/arithmetic.jsonl")
    rep = run(cases, BrokenProvider(), model="mock")
    assert rep.errors == 5 and rep.passed == 0
    assert all("endpoint down" in r.error for r in rep.results)


def test_cli_run_writes_json_and_compare_reads_it(tmp_path, capsys):
    out = tmp_path / "r.json"
    code = main(["run", "examples/arithmetic.jsonl", "--provider", "mock", "--quiet", "--json", str(out)])
    assert code == 0
    data = json.loads(out.read_text())
    assert data["total"] == 5 and data["provider"] == "mock"
    main(["compare", str(out)])
    assert "| mock | mock |" in capsys.readouterr().out
