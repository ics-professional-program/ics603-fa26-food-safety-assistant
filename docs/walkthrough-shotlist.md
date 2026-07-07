# Walkthrough recording — shot list (lecture 1.3, Aug 27)

Fallback video in case the live demo fails. Record at 1080p; keep it under
5 minutes. Prep: `docker compose up -d`, index built, `.env` loaded.

1. **The page** (10s). http://localhost:8000 — point out the title bar and
   the empty question box. "A staff member's tool, not a chatbot."
2. **Grounded hit** (60s). Type: *What is the minimum internal temperature
   for poultry?* → point at the `grounded ✓` badge, read the answer (165°F /
   74°C / 15 s), then scroll the Sources panel: "the model saw exactly these
   passages — you can check its work."
3. **Second hit** (30s). Type: *How long can I take to cool cooked rice?* →
   135→70 in 2h, 70→41 in 4h, citation to §3-501.
4. **The decline** (45s). Type: *Can I bring my dog into the kitchen?* →
   `not in sources` badge, decline text, weak-scored passages. "It refuses
   to bluff. Useful is not the same as correct."
5. **The contrast** (45s). Terminal: `python scripts/contrast.py "Can I
   bring my dog into the kitchen?"` — plain Gemini answers anyway; the
   pipeline declines. Split-screen both outputs.
6. **The map** (30s). Show the README module-map table, then the directory
   tree: "every file here is a module you'll build this semester; the
   capstone is your own version of this app."
