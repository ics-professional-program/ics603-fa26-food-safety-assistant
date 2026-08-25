from pathlib import Path

from foodsafety_rag.ingest import (
    OVERLAP_CHARS,
    RawChunk,
    chunk_markdown,
    load_corpus,
    overlap_tail,
)

SAMPLE = """# Sample Rules

Intro line that belongs to no section and is dropped.

## First section
Alpha bravo charlie. This is the body of section one.

## Second section
Delta echo foxtrot. This is the body of section two.
"""


def test_chunk_markdown_one_chunk_per_section():
    chunks = chunk_markdown(SAMPLE, source="fda", path="fda/sample.md")
    assert [c.heading for c in chunks] == ["First section", "Second section"]
    assert all(isinstance(c, RawChunk) for c in chunks)
    assert all(c.title == "Sample Rules" for c in chunks)
    assert all(c.source == "fda" and c.path == "fda/sample.md" for c in chunks)


def test_chunk_text_contains_heading_and_body():
    chunks = chunk_markdown(SAMPLE, source="fda", path="fda/sample.md")
    assert chunks[0].text.startswith("First section")
    assert "Alpha bravo charlie" in chunks[0].text


def test_second_chunk_has_overlap_from_first():
    chunks = chunk_markdown(SAMPLE, source="fda", path="fda/sample.md")
    # small overlap: tail of section one is carried into section two's text
    assert "section one" in chunks[1].text
    assert "Delta echo foxtrot" in chunks[1].text


def test_overlap_tail_never_opens_mid_word():
    text = "word " * 100                      # far longer than the overlap window
    assert overlap_tail(text).startswith("word")
    # a slice landing inside a long word drops the fragment, not the next word
    text = "x" * (OVERLAP_CHARS + 10) + " keeps this"
    assert overlap_tail(text) == "keeps this"


def test_overlap_tail_keeps_short_text_whole():
    assert overlap_tail("Short body.") == "Short body."


def test_real_corpus_chunks_contain_no_word_fragments():
    """Every word in a chunk is a whole word from its source document. A chunk
    built by slicing mid-word introduces a token the document never contained."""
    corpus = Path(__file__).parent.parent / "corpus"
    for md_file in sorted(corpus.rglob("*.md")):
        md = md_file.read_text(encoding="utf-8")
        for chunk in chunk_markdown(md, source=md_file.parent.name,
                                    path=md_file.name):
            invented = set(chunk.text.split()) - set(md.split())
            assert not invented, f"{md_file.name} / {chunk.heading}: {invented}"


def test_load_corpus_reads_real_corpus():
    chunks = load_corpus(Path(__file__).parent.parent / "corpus")
    assert len(chunks) >= 16  # 8 docs x >=2 sections
    assert {"fda", "sop"} <= {c.source for c in chunks}
    poultry = [c for c in chunks if "165°F" in c.text and "Poultry" in c.heading]
    assert poultry, "expected the poultry cooking-temperature chunk"


def test_load_corpus_excludes_readme_and_filters_bulk(tmp_path):
    (tmp_path / "fda").mkdir()
    (tmp_path / "bulk" / "fda-2022-full").mkdir(parents=True)
    (tmp_path / "README.md").write_text(
        "# About\n\n## Provenance\nnotes\n", encoding="utf-8")
    (tmp_path / "fda" / "a.md").write_text(
        "# A\n\n## S1\nbody one.\n\n## S2\nbody two.\n", encoding="utf-8")
    (tmp_path / "bulk" / "fda-2022-full" / "b.md").write_text(
        "# B\n\n## T1\nbulk body.\n", encoding="utf-8")

    everything = load_corpus(tmp_path)
    assert {c.source for c in everything} == {"fda", "fda-2022-full"}
    assert not any("README" in c.path for c in everything)

    core = load_corpus(tmp_path, include_bulk=False)
    assert {c.source for c in core} == {"fda"}
    assert {c.path for c in core} == {"fda/a.md"}
