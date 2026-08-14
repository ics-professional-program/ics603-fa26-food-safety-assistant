"""Corpus loading and chunking: markdown -> RawChunk passages (chunk by heading)."""

import re
from dataclasses import dataclass
from pathlib import Path

OVERLAP_CHARS = 200  # small overlap: tail of the previous section

_LEADING_WORD_FRAGMENT = re.compile(r"^\S*\s*")


@dataclass
class RawChunk:
    source: str   # corpus subfolder: "fda" or "sop"
    title: str    # document title (the single "# " line)
    path: str     # path relative to the corpus dir, posix-style
    heading: str  # the "## " section heading
    text: str     # heading + overlap + section body (what gets embedded)


def overlap_tail(text: str) -> str:
    """The tail of `text` to carry into the next chunk, started on a word
    boundary. A plain character slice can open mid-word ("...ernal temperature"),
    and that fragment ends up in the source panel students are asked to inspect.
    """
    if len(text) <= OVERLAP_CHARS:
        return text.lstrip()
    tail = text[-OVERLAP_CHARS:]
    if text[-OVERLAP_CHARS - 1].isspace():
        return tail.lstrip()          # the slice already landed on a boundary
    # The slice split a word: drop the fragment and start at the next whole one.
    return _LEADING_WORD_FRAGMENT.sub("", tail, count=1)


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
        overlap = overlap_tail(prev_body) if prev_body else ""
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
