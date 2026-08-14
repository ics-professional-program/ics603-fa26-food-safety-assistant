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
    "# Load the endpoint settings from ../.env into the environment (key never printed).\n"
    "import os\n"
    "from pathlib import Path\n"
    "\n"
    "for line in Path('../.env').read_text(encoding='utf-8').splitlines():\n"
    "    if '=' in line and not line.lstrip().startswith('#'):\n"
    "        key, _, value = line.partition('=')\n"
    "        os.environ.setdefault(key.strip(), value.strip())\n"
    "\n"
    "print('endpoint:', os.environ.get('LLM_BASE_URL', '(default)'))\n"
    "print('model:', os.environ.get('LLM_MODEL', '(default)'))\n"
    "print('key loaded:', 'LLM_API_KEY' in os.environ)  # True/False only - never the key"
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
    "## What the response object carries\n"
    "`ask_model` hands back a string, but the endpoint returns more than that: "
    "how many tokens the call used and why the model stopped. Before running the "
    "cell, predict the prompt-token count for a question of about ten words."
))

cells.append(nbf.v4.new_code_cell(
    "# The same call, one level down - the response object instead of the text.\n"
    "import time\n"
    "from foodsafety_rag.config import get_settings\n"
    "from foodsafety_rag.generate import get_client\n"
    "\n"
    "client, model = get_client(), get_settings().llm_model\n"
    "prompt = 'In one sentence, what is retrieval-augmented generation?'\n"
    "\n"
    "start = time.monotonic()\n"
    "raw = client.chat.completions.create(\n"
    "    model=model, messages=[{'role': 'user', 'content': prompt}])\n"
    "elapsed = time.monotonic() - start\n"
    "\n"
    "print('prompt tokens:    ', raw.usage.prompt_tokens)\n"
    "print('completion tokens:', raw.usage.completion_tokens)\n"
    "print('latency:           %.2f s' % elapsed)\n"
    "print('finish reason:    ', raw.choices[0].finish_reason)"
))

cells.append(nbf.v4.new_code_cell(
    "# How much of that context did we write? Send one word and compare.\n"
    "brief = client.chat.completions.create(\n"
    "    model=model, messages=[{'role': 'user', 'content': 'Hi'}], max_tokens=1)\n"
    "print('prompt tokens for the single word \"Hi\":', brief.usage.prompt_tokens)\n"
    "# The difference from zero is the chat template the server adds for us."
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
    "| **Context** | Not zero. The server wraps our message in a chat template before the model sees it - measure it with the cells above. RAG is how we choose the rest. |\n"
    "| **Output** | Fluent prose. Fluent is not the same as verified. |\n"
    "| **Cost** | Metered per token, unlike an ordinary function call - and the prompt is charged too, not just the answer. |\n"
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
