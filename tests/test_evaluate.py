import json
from pathlib import Path

from foodsafety_rag.evaluate import deterministic_pass, extract_values, match_rank
from foodsafety_rag.schemas import Passage

ROOT = Path(__file__).parent.parent


def test_extract_values_normalizes_units():
    got = extract_values("Cook to 165°F (74°C) for 15 seconds, or 130°F for 112 minutes.")
    assert (165.0, "F") in got and (74.0, "C") in got
    assert (15.0, "sec") in got and (112.0, "min") in got


def test_extract_values_handles_plain_and_plural_units():
    got = extract_values("Hold for 2 hours at 135 F, then check in 1 minute.")
    assert (2.0, "hr") in got and (135.0, "F") in got and (1.0, "min") in got


def test_deterministic_pass_requires_all_reference_values():
    ok, missing = deterministic_pass(
        "Poultry must reach 165°F (74°C), effectively instantaneous.",
        "165°F (74°C), held for less than one second")
    assert ok and not missing

    bad, missing = deterministic_pass(
        "Poultry must reach 160°F.",
        "165°F (74°C), held for less than one second")
    assert not bad and (165.0, "F") in missing


def _p(doc, heading):
    return Passage(doc=doc, heading=heading, text="x", score=0.9)


def test_match_rank_finds_expected_passage():
    passages = [_p("Other Doc", "Other"), _p("Target Doc", "Target Heading")]
    assert match_rank(passages, "Target Doc", "Target Heading") == 2
    assert match_rank(passages, "Target Doc", "Missing") is None


def test_eval_set_is_well_formed():
    cases = json.loads((ROOT / "evals" / "eval_set.json").read_text(encoding="utf-8"))
    assert len(cases) >= 10
    assert sum(1 for c in cases if not c["answerable"]) >= 2
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    for c in cases:
        assert c["question"] and c["reference_answer"]
        if c["expected_chunk"] is not None:
            path = ROOT / "corpus" / c["expected_chunk"]["path"]
            assert path.exists(), f"{c['id']}: {path}"
            text = path.read_text(encoding="utf-8")
            assert f"## {c['expected_chunk']['heading']}" in text, (
                f"{c['id']}: heading not in {path.name}")
