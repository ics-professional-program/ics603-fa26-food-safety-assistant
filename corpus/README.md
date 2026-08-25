# Corpus provenance

**These passages are teaching data for ICS 603, adapted for a course demo.
They are not current food-safety guidance.** Regulations change; anyone who
needs the actual rules should read the source documents below or contact the
relevant authority.

## Layout

Two tiers. The curated folders hold hand-adapted documents sized for
grounded-answer demonstrations; `bulk/` holds mechanically converted chapters
of the real FDA Food Code so the vector index has enough rows to make an
index-vs-sequential-scan comparison measurable (course session 11.1).

| Folder | Contents | Kind |
|---|---|---|
| `fda/` | FDA Food Code **2022 edition** excerpts | curated |
| `fda-2017/` | FDA Food Code **2017 edition** excerpts, where the editions differ | curated |
| `hawaii/` | Hawaii Administrative Rules Title 11, Chapter 50 (Food Safety Code) excerpts | curated |
| `sop/` | Pacific Market Cafe standard operating procedures — a **fictional** establishment, written on the real ICN HACCP-based SOP templates | curated |
| `bulk/fda-2022-full/` | FDA Food Code 2022, chapters 2-8, one doc per code section | generated |
| `bulk/fda-2017-full/` | FDA Food Code 2017, chapters 2-8, one doc per code section | generated |

`bulk/` is **generated — do not hand-edit**. Regenerate with:

```bash
uv run --with pypdf python -c "..."   # extract the PDF text (see below)
uv run python scripts/convert_food_code.py <extracted.txt> \
    --edition 2022 --chapters 2 3 4 5 6 7 8 \
    --out corpus/bulk/fda-2022-full \
    --url https://www.fda.gov/media/164194/download --pulled YYYY-MM-DD
```

`scripts/ingest_corpus.py --skip-bulk` ingests the curated tier only, which
is the fast path for the small-corpus demonstrations. Measured 2026-08-24 on
an Apple Silicon laptop: the full ingest (1,478 passages, local embeddings)
takes about 9 seconds end to end; the curated-only ingest about 2 seconds.
Budget more on slower hardware, and run the full ingest before class rather
than during it.

## Sources

All pulled 2026-08-24. The FDA Food Code is a public-domain U.S. federal
publication; Hawaii Administrative Rules are public state rules. The ICN
SOPs are federally funded (USDA/FNS) training materials, published for free
distribution and republished wholesale by state child-nutrition agencies.

The `sop/` documents are **adaptations, not copies**: each one names the ICN
procedure it is built on, follows that template's section contract (purpose
and scope, instructions, monitoring, corrective action, verification and
record keeping), and is rewritten for a table-service cafe rather than a
school kitchen. Pacific Market Cafe remains fictional on purpose — a real
establishment's SOPs are its own internal documents, which is exactly why a
grounded assistant for one is a plausible application. Suggested citation:
Institute of Child Nutrition. (2018). *HACCP-based standard operating
procedures*. University, MS: Author.

| Source | Document | URL |
|---|---|---|
| `fda/`, `bulk/fda-2022-full/` | FDA Food Code, 2022 edition (full PDF) | https://www.fda.gov/media/164194/download |
| `fda-2017/`, `bulk/fda-2017-full/` | FDA Food Code, 2017 edition (full PDF) | https://www.fda.gov/media/110822/download |
| `hawaii/` | HAR Title 11, Ch. 50, November 2024 revision text (adopted effective August 2025) | https://health.hawaii.gov/san/files/2024/11/Ch-50-revision-11.12.2024.pdf |
| `sop/` | Institute of Child Nutrition, HACCP-Based Standard Operating Procedures (published 2005, updated 2018) | https://theicn.org/icn-resources-a-z/food-safety-standard-operating-procedures/ |

The Hawaii signed rules (effective August 2025) are a scanned PDF
(https://health.hawaii.gov/news/files/2025/08/Food-Safety-Code-HAR-11-50-Effective-August-2025.pdf);
the November 2024 revision text above is the machine-readable version of the
same adoption, which aligned the state code with the 2022 FDA model Food Code
with Hawaii-specific amendments.

## Deliberate disagreements

The curated tier contains three real, citable divergences, used by sessions
11.2 (conflicting sources) and 13.0 (evaluation):

1. **State vs. federal** — the FDA 2022 code requires handwashing-sink water
   of at least 85°F; Hawaii's current rule sets **no minimum** (hot water is
   optional). `fda/handwashing-sink-requirements.md` vs.
   `hawaii/handwashing-water-temperature.md`.
2. **2017 vs. 2022 edition** — the handwashing-water minimum changed from
   100°F to 85°F, and sesame became the ninth major food allergen (2017
   lists eight). `fda-2017/` vs. `fda/`.
3. **SOP stricter than code** — Pacific Market Cafe holds hot food at 140°F
   where the code requires 135°F (`sop/hot-holding-line-checks.md` vs.
   `fda/holding-and-cooling.md`); the SOP's 48-hour illness return rule is
   also stricter than the code's 24 hours (`sop/employee-illness.md` vs.
   `fda/employee-health-and-exclusions.md`). Both SOPs are built on ICN
   templates that simply restate the code's numbers, so the house standards
   are visible as deliberate deviations rather than as invented facts.
