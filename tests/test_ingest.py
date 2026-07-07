from pathlib import Path

from foodsafety_rag.ingest import RawChunk, chunk_markdown, load_corpus

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


def test_load_corpus_reads_real_corpus():
    chunks = load_corpus(Path(__file__).parent.parent / "corpus")
    assert len(chunks) >= 16  # 8 docs x >=2 sections
    assert {c.source for c in chunks} == {"fda", "sop"}
    poultry = [c for c in chunks if "165°F" in c.text and "Poultry" in c.heading]
    assert poultry, "expected the poultry cooking-temperature chunk"
