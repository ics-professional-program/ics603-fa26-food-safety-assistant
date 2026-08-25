"""Capture real /ask responses into fixtures/ for offline replay (REPLAY=1).

Prereqs: db up, index built, app running with a real key:
  python -m uvicorn app.main:app --port 8000
Run:  python scripts/capture_fixtures.py

Override the target with ASK_URL if the app is on another port, e.g.
  ASK_URL=http://localhost:8010/ask python scripts/capture_fixtures.py
"""

import json
import os
import urllib.request
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
BASE_URL = os.environ.get("ASK_URL", "http://localhost:8000/ask")

CANONICAL = {
    "poultry-temperature": "What is the minimum internal temperature for poultry?",
    "cooling-rice": "How long can I take to cool cooked rice?",
    "handwashing-water": "How warm does handwashing water need to be?",
    "off-corpus-dog": "Can I bring my dog into the kitchen?",
}


def main() -> None:
    FIXTURES_DIR.mkdir(exist_ok=True)
    for name, question in CANONICAL.items():
        request = urllib.request.Request(
            BASE_URL,
            data=json.dumps({"question": question}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read())
        path = FIXTURES_DIR / f"{name}.json"
        path.write_text(json.dumps(body, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        print(f"captured {path.name}: grounded={body['grounded']}")


if __name__ == "__main__":
    main()
