import json
import pytest
from llm_eval.dataset import filter_by_tag, load_jsonl, parse_case


def test_parse_case_defaults_and_scalar_expected():
    c = parse_case({"id": "x", "prompt": "p", "expected": 4}, 1)
    assert c.expected == ["4"]
    assert c.match == "contains"
    assert c.tolerance == 0.0


def test_parse_case_rejects_bad_match():
    with pytest.raises(ValueError, match="unknown match type"):
        parse_case({"id": "x", "prompt": "p", "expected": "a", "match": "fuzzy"}, 3)


def test_load_jsonl_skips_comments_and_detects_duplicates(tmp_path):
    f = tmp_path / "d.jsonl"
    f.write_text("# comment\n" + json.dumps({"id": "a", "prompt": "p", "expected": "x"}) + "\n"
                 + json.dumps({"id": "a", "prompt": "q", "expected": "y"}) + "\n")
    with pytest.raises(ValueError, match="duplicate id"):
        load_jsonl(f)


def test_examples_load_and_tag_filter():
    cases = load_jsonl("examples/aspen.jsonl")
    assert len(cases) >= 5
    hysys = filter_by_tag(cases, "hysys")
    assert all("hysys" in c.tags for c in hysys)
    assert 0 < len(hysys) < len(cases)
