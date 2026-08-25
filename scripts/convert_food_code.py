"""Convert extracted FDA Food Code text into corpus/bulk/ markdown documents.

One document per code section (e.g. 3-401.11), one ``##`` heading per lettered
provision, so the existing heading-based chunker applies unchanged. This is the
provenance record for the bulk corpus tier: the generated files are committed,
and each carries the edition, source URL, and pull date in its header.

Input is the plain text of the official PDF (extract with pypdf). The parser
truncates at the first Annex page header (annexes repeat chapter section
numbers with commentary), skips table-of-contents entries (dot leaders), and
drops page furniture.

Usage:
    uv run python scripts/convert_food_code.py foodcode-2022.txt \
        --edition 2022 --chapters 2 3 4 5 \
        --out corpus/bulk/fda-2022-full \
        --url https://www.fda.gov/media/164194/download --pulled 2026-08-24
"""

import argparse
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

SECTION_RE = re.compile(r"^(\d)-(\d{3})\.(\d{2})\s+(.+?)\s*$")
PROVISION_RE = re.compile(r"^\(([A-Z])\)\s")
FURNITURE_RE = re.compile(
    r"^\s*(\d{1,4}|FDA Food Code \d{4}.*|Food Code \d{4}.*|"
    r"Chapter\s+\d+\s*[-–—]\s*.*|ANNEXES?.*)\s*$")
# The parse stops at the first of these: an annex page header (2022 pages
# carry "FDA Food Code 2022 ... Annex N"; 2017 pages carry a bare
# "Annex N - Title") or the standalone INDEX heading that precedes the 2017
# annexes. Annexes restate chapter section numbers, and the index would pour
# thousands of entries into the last open provision. A TOC "Annex N" line
# without a dash must not truncate.
ANNEX_HEADER_RE = re.compile(
    r"^\s*((FDA )?Food Code \d{4}\s+.*Annex\s+\d|"
    r"Annex\s+\d+\s*[-–—]|INDEX\s*$)")
TOC_LEADER_RE = re.compile(r"\.{3,}|\.\s(\.\s)+")
DEGREE_RE = re.compile(r"(\d)\s*o([CF])\b")
MIN_PROVISION_WORDS = 25


@dataclass
class Section:
    number: str
    title: str
    provisions: list[str]


def truncate_at_annex(text: str) -> str:
    """Cut the text where the annex pages begin: annexes restate chapter
    section numbers with commentary, which would collide with the real ones."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if ANNEX_HEADER_RE.match(line):
            return "\n".join(lines[:i])
    return text


def clean_lines(text: str) -> str:
    """Drop page furniture (page numbers, running headers) and normalize the
    PDF extraction's ``63oC (145oF)`` degree artifacts to ``63°C (145°F)``."""
    kept = [line for line in text.splitlines() if not FURNITURE_RE.match(line)]
    return DEGREE_RE.sub(r"\1°\2", "\n".join(kept))


def _merge_short(provisions: list[str]) -> list[str]:
    """Fold provisions under MIN_PROVISION_WORDS into a neighbor so chunks
    stay in a readable size range."""
    merged: list[str] = []
    for prov in provisions:
        if merged and len(merged[-1].split()) < MIN_PROVISION_WORDS:
            merged[-1] = merged[-1] + "\n" + prov
        else:
            merged.append(prov)
    if len(merged) > 1 and len(merged[-1].split()) < MIN_PROVISION_WORDS:
        tail = merged.pop()
        merged[-1] = merged[-1] + "\n" + tail
    return merged


def parse_sections(text: str, chapters: set[int]) -> list[Section]:
    text = clean_lines(truncate_at_annex(text))
    sections: list[Section] = []
    seen: set[str] = set()
    current: Section | None = None
    provision_lines: list[str] = []
    last_letter = ""

    def close_provision():
        if current is not None and provision_lines:
            paragraph = " ".join(l.strip() for l in provision_lines).strip()
            if paragraph:
                current.provisions.append(paragraph)
        provision_lines.clear()

    def close_section():
        nonlocal current
        close_provision()
        if current is not None:
            current.provisions = _merge_short(
                [p for p in current.provisions if p.split()])
            if current.provisions:
                sections.append(current)
        current = None

    for line in text.splitlines():
        m = SECTION_RE.match(line)
        if m:
            number = f"{m.group(1)}-{m.group(2)}.{m.group(3)}"
            title = m.group(4).rstrip(".")
            if TOC_LEADER_RE.search(title):
                continue                      # table-of-contents entry
            close_section()
            last_letter = ""
            if int(m.group(1)) in chapters and number not in seen:
                seen.add(number)
                current = Section(number=number, title=title, provisions=[])
            continue
        if current is None:
            continue
        pm = PROVISION_RE.match(line)
        # Provisions ascend strictly (A), (B), (C)... A letter out of order is
        # a cross-reference at a line start (page-break artifact), not a new
        # provision.
        if pm and ord(pm.group(1)) == (ord(last_letter) + 1 if last_letter
                                       else ord("A")):
            close_provision()
            last_letter = pm.group(1)
        provision_lines.append(line)
    close_section()
    return sections


def write_docs(sections: list[Section], out_dir: Path, *, edition: str,
               url: str, pulled: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.md"):
        stale.unlink()
    written = 0
    for section in sections:
        lines = [
            f"# FDA Food Code {edition} {section.number} {section.title}",
            "",
            f"Generated from the FDA Food Code {edition} edition "
            f"({url}, pulled {pulled}) by scripts/convert_food_code.py.",
            "Teaching data, not current food-safety guidance. Do not hand-edit.",
            "",
        ]
        used: set[str] = set()
        for prov in section.provisions:
            m = PROVISION_RE.match(prov)
            heading = (f"{section.number}({m.group(1)})" if m
                       else section.number)
            while heading in used:
                heading += " cont."
            used.add(heading)
            lines.append(f"## {heading}")
            lines.append(textwrap.fill(prov, width=78))
            lines.append("")
        (out_dir / f"{section.number}.md").write_text(
            "\n".join(lines), encoding="utf-8")
        written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text_file", type=Path)
    parser.add_argument("--edition", required=True)
    parser.add_argument("--chapters", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--pulled", required=True)
    args = parser.parse_args()

    text = args.text_file.read_text(encoding="utf-8")
    sections = parse_sections(text, chapters=set(args.chapters))
    count = write_docs(sections, args.out, edition=args.edition,
                       url=args.url, pulled=args.pulled)
    headings = sum(len(s.provisions) for s in sections)
    print(f"Wrote {count} section docs ({headings} passages) to {args.out}")


if __name__ == "__main__":
    main()
