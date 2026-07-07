from pathlib import Path

CORPUS = Path(__file__).parent.parent / "corpus"


def _md_files():
    return sorted(CORPUS.rglob("*.md"))


def test_corpus_has_expected_size():
    files = _md_files()
    assert 6 <= len(files) <= 12, f"spec calls for ~6-12 docs, found {len(files)}"
    assert (CORPUS / "fda").is_dir() and (CORPUS / "sop").is_dir()


def test_every_doc_has_title_and_sections():
    for f in _md_files():
        lines = f.read_text(encoding="utf-8").splitlines()
        titles = [l for l in lines if l.startswith("# ")]
        sections = [l for l in lines if l.startswith("## ")]
        assert len(titles) == 1, f"{f.name}: needs exactly one '# ' title"
        assert len(sections) >= 2, f"{f.name}: needs >= 2 '## ' sections"


def test_key_demo_facts_present():
    text = "\n".join(f.read_text(encoding="utf-8") for f in _md_files())
    assert "165°F" in text        # poultry — canonical demo question
    assert "41°F" in text         # cold holding
    assert "20 seconds" in text   # handwashing
