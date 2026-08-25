from pathlib import Path

CORPUS = Path(__file__).parent.parent / "corpus"


def _curated():
    return sorted(f for f in CORPUS.rglob("*.md")
                  if "bulk" not in f.parts and f.name.lower() != "readme.md")


def _bulk():
    return sorted((CORPUS / "bulk").rglob("*.md"))


def _text(files):
    return "\n".join(f.read_text(encoding="utf-8") for f in files)


def test_corpus_tier_sizes():
    assert 12 <= len(_curated()) <= 40, len(_curated())
    assert len(_bulk()) >= 200, len(_bulk())
    for folder in ("fda", "fda-2017", "hawaii", "sop"):
        assert (CORPUS / folder).is_dir(), folder
    assert (CORPUS / "README.md").exists()


def test_curated_docs_have_title_and_sections():
    for f in _curated():
        lines = f.read_text(encoding="utf-8").splitlines()
        assert len([l for l in lines if l.startswith("# ")]) == 1, f.name
        assert len([l for l in lines if l.startswith("## ")]) >= 2, f.name


def test_bulk_docs_have_title_and_a_section():
    for f in _bulk():
        text = f.read_text(encoding="utf-8")
        assert text.startswith("# "), f.name
        assert "\n## " in text, f.name
        assert "Do not hand-edit" in text, f.name


def test_key_demo_facts_present():
    text = _text(_curated())
    assert "165°F" in text          # poultry — canonical demo question
    assert "41°F" in text           # cold holding
    assert "20 seconds" in text     # handwashing duration


def test_conflict_case_facts_present():
    fda = _text((CORPUS / "fda").glob("*.md"))
    fda17 = _text((CORPUS / "fda-2017").glob("*.md"))
    hawaii = _text((CORPUS / "hawaii").glob("*.md"))
    sop = _text((CORPUS / "sop").glob("*.md"))

    # State vs. federal: FDA 2022 sets a minimum handwashing water
    # temperature; Hawaii's current rule sets none (hot water is optional).
    assert "85°F" in fda
    assert "does not set a minimum" in hawaii

    # 2017 vs. 2022 edition: the handwashing minimum changed 100°F -> 85°F,
    # and sesame became the ninth major allergen (2017 lists eight).
    assert "100°F" in fda17
    assert "sesame" in fda.lower() and "sesame" in fda17.lower()
    assert "nine" in fda.lower() and "eight" in fda17.lower()

    # SOP stricter than code: company hot holding at 140°F vs. the code's
    # 135°F, and a 48-hour illness return vs. the code's 24 hours.
    assert "140°F" in sop
    assert "135°F" in fda
    assert "48 hours" in sop
    assert "asymptomatic for at least 24 hours" in fda


def test_curated_docs_cite_their_source():
    for f in _curated():
        text = f.read_text(encoding="utf-8")
        if f.parent.name in ("fda", "fda-2017"):
            assert "Food Code" in text and "edition" in text, f.name
        elif f.parent.name == "hawaii":
            assert "11-50" in text, f.name
        elif f.parent.name == "sop":
            # The cafe is fictional, but every SOP is built on a real,
            # publicly distributed ICN template and must name it.
            assert "Pacific Market Cafe" in text, f.name
            assert "Institute of Child Nutrition" in text, f.name
