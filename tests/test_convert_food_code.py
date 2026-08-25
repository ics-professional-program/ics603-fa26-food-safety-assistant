from scripts.convert_food_code import Section, clean_lines, parse_sections

SAMPLE = """\
3-401.11 Raw Animal Foods.
(A) Except as specified under (B), raw animal foods such as eggs, fish,
meat, and poultry shall be cooked to heat all parts of the food to a
temperature and for a time that complies with one of the following methods
based on the food that is being cooked, considering its origin and history.
(B) Whole meat roasts shall be cooked as specified in the corresponding
table of temperatures and holding times, using an oven that is preheated
and held at the temperature specified for the roast's weight.
3-401.12 Microwave Cooking.
Raw animal foods cooked in a microwave oven shall be rotated and stirred
throughout or midway during cooking to compensate for uneven distribution
of heat, covered to retain surface moisture, and heated to a temperature
of at least 165 degrees Fahrenheit in all parts of the food.
"""


def test_parse_sections_splits_on_section_numbers():
    sections = parse_sections(SAMPLE, chapters={3})
    assert [s.number for s in sections] == ["3-401.11", "3-401.12"]
    assert sections[0].title == "Raw Animal Foods"
    assert len(sections[0].provisions) == 2          # (A) and (B)
    assert sections[0].provisions[0].startswith("(A)")


def test_parse_sections_filters_by_chapter():
    assert parse_sections(SAMPLE, chapters={5}) == []


def test_clean_lines_drops_page_furniture():
    noisy = ("Food Code 2022               Chapter 3. Food \n"
             "Chapter 3 - 19 \n"
             "143\n"
             "3-401.11 Raw Animal Foods.\n"
             "Body text here.")
    cleaned = clean_lines(noisy)
    assert "143" not in cleaned
    assert "Chapter 3 - 19" not in cleaned
    assert "Body text here." in cleaned


def test_out_of_order_letter_is_a_continuation_not_a_provision():
    # A page break can put a cross-reference like "(B) of this section" at a
    # line start inside provision (D). Provisions ascend strictly, so an
    # out-of-order letter belongs to the current provision.
    text = ("3-401.11 Raw Animal Foods.\n"
            "(A) Raw animal foods shall be cooked to the required temperature\n"
            "for the required time, as listed for each category of food below,\n"
            "unless a variance as described in this chapter applies instead.\n"
            "(B) Whole meat roasts shall be cooked as specified in the chart,\n"
            "using holding times that correspond to the chosen temperature and\n"
            "accounting for postoven heat rise where the chart notes it.\n"
            "(C) The consumer shall be informed that to ensure its safety, the\n"
            "food should be cooked as specified under paragraph (A) or\n"
            "(B) of this section; otherwise a variance is required under the\n"
            "provisions that govern such requests and their documentation.\n")
    sections = parse_sections(text, chapters={3})
    assert [p[:3] for p in sections[0].provisions] == ["(A)", "(B)", "(C)"]
    assert "of this section" in sections[0].provisions[2]


def test_truncates_at_2017_style_annex_header():
    # The 2017 extraction's annex page header is a bare "Annex 1 - Compliance
    # & Enforcement" line (no "Food Code" prefix). Annex provisions reuse
    # chapter-8 numbering, so everything from that header on must be dropped.
    # A TOC line like "Annex 1" alone (no dash) must NOT truncate.
    text = ("Annex 1  \n"
            "8-101.10 Real Chapter Eight Provision.\n"
            "The regulatory authority shall apply this code in a manner that\n"
            "protects public health while recognizing the operational realities\n"
            "of food establishments and their differing sizes and menus.\n"
            "Annex 1 – Compliance & Enforcement \n"
            "8-909.40 Annex Only Provision.\n"
            "This provision exists only in the annex and must not be parsed as\n"
            "part of chapter eight, or the two editions' corpora will diverge\n"
            "in ways the source documents do not.\n")
    sections = parse_sections(text, chapters={8})
    assert [s.number for s in sections] == ["8-101.10"]


def test_truncates_at_the_index():
    # The 2017 layout puts the book's INDEX between chapter 8 and Annex 1.
    # Without a cut, thousands of index entries pour into the last open
    # provision. A standalone INDEX line ends the parse.
    text = ("8-501.40 Removal of Exclusions and Restrictions.\n"
            "The regulatory authority shall release a food employee or\n"
            "conditional employee from restriction or exclusion according to\n"
            "law and the conditions specified under the referenced section.\n"
            "INDEX\n"
            "Access\n"
            "allowed after due notice, 218\n"
            "application for inspection order, 219\n")
    sections = parse_sections(text, chapters={8})
    assert len(sections) == 1
    assert "due notice" not in sections[0].provisions[0]


def test_short_provisions_merge():
    text = ("3-101.11 Safe Food.\n"
            "(A) Food shall be safe.\n"
            "(B) Food shall be unadulterated and, as specified under the sections\n"
            "that follow, honestly presented to the consumer in a way that does\n"
            "not mislead or misinform the consumer about its character, origin,\n"
            "quantity, or the substances it contains, including major allergens\n"
            "declared in a manner the consumer can reasonably notice and read.\n")
    sections = parse_sections(text, chapters={3})
    assert len(sections) == 1
    # (A) is far under the merge threshold, so it folds into (B)
    assert len(sections[0].provisions) == 1
    assert "Food shall be safe" in sections[0].provisions[0]
