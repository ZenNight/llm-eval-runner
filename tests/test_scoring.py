from llm_eval.dataset import Case
from llm_eval.scoring import normalize, score


def mk(expected, match="contains", tolerance=0.0):
    return Case(id="t", prompt="p", expected=expected if isinstance(expected, list) else [expected], match=match, tolerance=tolerance)


def test_normalize_strips_case_space_and_punctuation():
    assert normalize("  Brasília.  ") == "brasília"
    assert normalize("YES!") == "yes"


def test_exact_match_ignores_case_and_trailing_period():
    assert score(mk("yes", "exact"), "Yes.")
    assert not score(mk("yes", "exact"), "yes, it is prime")


def test_contains_accepts_any_alternative():
    c = mk(["Brasilia", "Brasília"], "contains")
    assert score(c, "The capital is Brasília, not Rio.")
    assert not score(c, "The capital is Rio de Janeiro.")


def test_regex_is_case_insensitive():
    c = mk([r"peng[- ]?robinson", r"\bPR\b"], "regex")
    assert score(c, "Use Peng-Robinson for gas systems.")
    assert score(c, "PR is the usual default.")
    assert not score(c, "Use NRTL.")


def test_numeric_with_tolerance_and_comma_decimal():
    c = mk("373.15", "numeric", tolerance=0.2)
    assert score(c, "373.15 K")
    assert score(c, "About 373,1 kelvin")
    assert not score(c, "373.5")
    assert not score(c, "no number here")


def test_numeric_takes_first_number_in_answer():
    c = mk("45", "numeric")
    assert score(c, "45. Because 15 times 3 is 45.")
    assert not score(c, "15 times 3 equals 45.")  # first number is 15; documents a known limitation
