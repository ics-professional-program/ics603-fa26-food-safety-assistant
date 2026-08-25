# Retrieval reference numbers (session 11.1)

Measured over the full corpus: **1462 chunks** (curated + generated Food Code chapters). Exact scan = no vector index; HNSW = `CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)`. Median of 10 runs each; index build took **0.13s**.

| Question | exact (ms) | HNSW (ms) | speedup | recall@4 |
|---|---|---|---|---|
| What is the minimum internal temperature for cooking poultry? | 1.25 | 0.38 | 3.3x | 1.00 |
| How hot does the water at a handwashing sink need to be? | 1.21 | 0.37 | 3.3x | 1.00 |
| What temperature must hot food be held at? | 1.14 | 0.37 | 3.1x | 1.00 |
| How do I calibrate a probe thermometer? | 1.12 | 0.38 | 3.0x | 1.00 |

## Expected top-4 (the Predict step)

Exact-scan results - what students should predict before running the query:

**What is the minimum internal temperature for cooking poultry?**

1. FDA Food Code §3-401 — Cooking Temperatures — Poultry and stuffed foods
2. FDA Food Code §3-401 — Cooking Temperatures — Ground meats
3. FDA Food Code §3-401 — Cooking Temperatures — Whole-muscle meats, seafood, and eggs
4. FDA Food Code 2022 3-501.16 Time/Temperature Control for Safety Food, Hot and Cold — 3-501.16

**How hot does the water at a handwashing sink need to be?**

1. FDA Food Code 2022 5-202.12 Handwashing Sink, Installation — 5-202.12(A)
2. FDA Food Code 2017 §5-202.12 — Handwashing Sink Requirements — Water temperature at handwashing sinks
3. FDA Food Code 2017 5-202.12 Handwashing Sink, Installation — 5-202.12(A)
4. FDA Food Code §5-202.12 — Handwashing Sink Requirements — Water temperature at handwashing sinks

**What temperature must hot food be held at?**

1. FDA Food Code 2017 3-403.11 Reheating for Hot Holding — 3-403.11(A)
2. FDA Food Code 2022 3-403.11 Reheating for Hot Holding — 3-403.11(A)
3. FDA Food Code §3-501 — Holding and Cooling — Hot holding
4. FDA Food Code 2022 3-403.11 Reheating for Hot Holding — 3-403.11(C)

**How do I calibrate a probe thermometer?**

1. SOP-07 — Thermometer Calibration (Pacific Market Cafe) — Records and tolerance
2. SOP-07 — Thermometer Calibration (Pacific Market Cafe) — Calibration frequency
3. SOP-07 — Thermometer Calibration (Pacific Market Cafe) — Ice-point method
4. FDA Food Code 2022 4-302.12 Food Temperature Measuring Devices — 4-302.12(B)
