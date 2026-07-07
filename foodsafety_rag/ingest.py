"""Corpus loading and chunking: markdown -> RawChunk passages (chunk by heading)."""

from dataclasses import dataclass
from pathlib import Path

OVERLAP_CHARS = 200  # small overlap: tail of the previous section


@dataclass
class RawChunk:
    source: str   # corpus subfolder: "fda" or "sop"
    title: str    # document title (the single "# " line)
    path: str     # path relative to the corpus dir, posix-style
    heading: str  # the "## " section heading
    text: str     # heading + overlap + section body (what gets embedded)


def chunk_markdown(md: str, *, source: str, path: str) -> list[RawChunk]:
    lines = md.splitlines()
    title = next(
        (line[2:].strip() for line in lines if line.startswith("# ")),
        Path(path).stem,
    )

    sections: list[tuple[str, list[str]]] = []
    heading: str | None = None
    body: list[str] = []
    for line in lines:
        if line.startswith("# ") and not line.startswith("## "):
            continue
        if line.startswith("## "):
            if heading is not None:
                sections.append((heading, body))
            heading = line[3:].strip()
            body = []
        elif heading is not None:
            body.append(line)
    if heading is not None:
        sections.append((heading, body))

    chunks: list[RawChunk] = []
    prev_body = ""
    for heading, body_lines in sections:
        body_text = "\n".join(body_lines).strip()
        overlap = prev_body[-OVERLAP_CHARS:] if prev_body else ""
        text = f"{heading}\n{overlap}\n{body_text}" if overlap else f"{heading}\n{body_text}"
        chunks.append(RawChunk(source=source, title=title, path=path,
                               heading=heading, text=text))
        prev_body = body_text
    return chunks


def load_corpus(corpus_dir: Path) -> list[RawChunk]:
    chunks: list[RawChunk] = []
    for md_file in sorted(corpus_dir.rglob("*.md")):
        chunks.extend(
            chunk_markdown(
                md_file.read_text(encoding="utf-8"),
                source=md_file.parent.name,
                path=md_file.relative_to(corpus_dir).as_posix(),
            )
        )
    return chunks
