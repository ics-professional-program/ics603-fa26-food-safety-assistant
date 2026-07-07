"""Generate notebooks/first_llm_call.ipynb (run once; commit the output)."""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# Your First LLM Call\n"
    "*ICS 603 — lecture 1.3 'Course Arc Preview and First LLM Taste'*\n\n"
    "One tiny call to a large language model — the same `ask_model()` function "
    "the food-safety assistant uses in production. No retrieval, no grounding "
    "yet: just prompt in, text out."
))

cells.append(nbf.v4.new_code_cell(
    "# Load the API key from ../.env into the environment (never printed).\n"
    "import os\n"
    "from pathlib import Path\n"
    "\n"
    "for line in Path('../.env').read_text(encoding='utf-8').splitlines():\n"
    "    if '=' in line and not line.lstrip().startswith('#'):\n"
    "        key, _, value = line.partition('=')\n"
    "        os.environ.setdefault(key.strip(), value.strip())\n"
    "\n"
    "print('key loaded:', 'GEMINI_API_KEY' in os.environ)  # True/False only - never the key"
))

cells.append(nbf.v4.new_code_cell(
    "# THE LIVE CALL - one prompt, one response. Also saves the response so the\n"
    "# fallback cell below works offline next time.\n"
    "from pathlib import Path\n"
    "from foodsafety_rag.generate import ask_model\n"
    "\n"
    "response = ask_model('In one sentence, what is retrieval-augmented generation?')\n"
    "Path('../fixtures/first_llm_call.txt').write_text(response, encoding='utf-8')\n"
    "print(response)"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Fallback: captured response (no key / no network)\n"
    "If the live call above fails in class (no key, no wifi, quota), the cell "
    "below replays the response captured on a previous successful run. Same "
    "text, zero network."
))

cells.append(nbf.v4.new_code_cell(
    "from pathlib import Path\n"
    "\n"
    "captured = Path('../fixtures/first_llm_call.txt').read_text(encoding='utf-8')\n"
    "print('[captured response - replayed offline]')\n"
    "print(captured)"
))

cells.append(nbf.v4.new_markdown_cell(
    "## What to notice\n\n"
    "| | This call |\n"
    "|---|---|\n"
    "| **Input** | One sentence of plain English - no code, no schema. |\n"
    "| **Context** | None. The model answered from training memory alone - this is exactly what the RAG app changes. |\n"
    "| **Output** | Fluent prose. Fluent is not the same as verified. |\n"
    "| **Cost** | Fractions of a cent (free tier here) - but it meters per token, unlike ordinary function calls. |\n"
    "| **Latency** | ~1-3 seconds - orders of magnitude slower than a normal function call; design around it. |\n"
    "| **Uncertainty** | Re-run the live cell: the wording can change. Same input, different output - ordinary software never does this. |\n\n"
    "The rest of the course wraps this one call in retrieval (M8), structure "
    "(Pydantic, M4/M6), storage (M7), and deployment (Docker/Jetstream) - "
    "turning 'useful' into 'trustworthy'."
))

nb.cells = cells
out = Path(__file__).parent.parent / "notebooks" / "first_llm_call.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print(f"wrote {out}")
