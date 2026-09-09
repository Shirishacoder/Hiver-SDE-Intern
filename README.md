# Hiver SDE Intern Take-Home — AI Support Agent (Brand: AmazonHelp)

An AI agent that reads an inbound customer tweet, (1) classifies intent, (2) drafts a
reply grounded in how the brand has historically resolved similar issues, and
(3) decides auto-handle vs. escalate-to-human, with a stated reason.

Brand chosen: **@AmazonHelp** (large volume, varied intents, good mix of
solvable-by-template and must-escalate cases in the Kaggle Customer Support on
Twitter dataset).

---

## 0. Folder map

```
hiver-support-agent/
├── README.md                  <- you are here
├── requirements.txt
├── .env.example                <- copy to .env, add your GROQ_API_KEY (free, no card, see §3)
├── data/
│   ├── README.md               <- how to get the real Kaggle data
│   ├── raw/                    <- put twcs.csv (Kaggle) here; ships empty
│   └── processed/               <- pipeline writes cleaned pairs here
│       └── sample_pairs.csv     <- 30 synthetic demo conversation pairs (see note below)
├── src/
│   ├── llm_client.py            <- thin wrapper around Anthropic API + a --mock offline mode
│   ├── data_prep.py             <- filters raw twcs.csv to one brand, reconstructs threads
│   ├── intents.py                <- intent taxonomy + classifier (LLM + keyword baseline)
│   ├── retrieval.py               <- TF-IDF grounding retrieval over past resolved threads
│   ├── escalation.py              <- auto-handle vs escalate policy + reason
│   └── agent.py                   <- glues the above into one pipeline call
├── eval/
│   ├── golden_set.csv              <- golden evaluation set (see §4 — REPLACE before grading)
│   ├── judge_rubric.md              <- LLM-as-judge rubric for reply quality
│   ├── metrics.py                   <- intent acc/F1, escalation P/R, judge-human agreement
│   └── run_eval.py                   <- runs agent over golden set, prints/saves metrics
├── report/
│   ├── REPORT.md                     <- problem framing, baselines, failure analysis, etc.
│   └── DECISION_LOG.md               <- 15 non-obvious decisions and why
└── scripts/
    └── run_pipeline.py               <- single entry point, see §2
```

## 1. IMPORTANT — read this first

This repo ships in two states glued together, and you must know which parts are
which before you present it:

1. **Fully working, real code**: `src/`, `eval/metrics.py`, `eval/run_eval.py`,
   `scripts/run_pipeline.py`, the retrieval/classification/escalation logic. This
   is real, runs, and will work on the actual Kaggle dataset once you drop
   `twcs.csv` into `data/raw/`.
2. **Placeholders you must replace with your own work before submitting**:
   - `eval/golden_set.csv` currently contains **30 synthetic, clearly-fake demo
     rows** so the eval harness is runnable out of the box. The assignment
     requires **150–250 examples you hand-label yourself from real data**. See
     `eval/golden_set.csv` header comment and `report/REPORT.md §2` for the
     sampling/labeling protocol to follow — the protocol is written for you,
     the labeling is not done for you.
   - `data/processed/sample_pairs.csv` is a **30-row synthetic sample** of
     brand↔customer pairs (fake, generated to match the Kaggle schema) so
     `scripts/run_pipeline.py` runs end-to-end in under a minute without
     downloading anything. Swap in the real filtered/reconstructed pairs from
     `src/data_prep.py` once you have `twcs.csv`.
   - `report/REPORT.md` has every required section stubbed with the analysis
     framework and prompts for what to fill in, plus one worked example per
     section using the synthetic data so you can see the intended shape of the
     answer. The actual numbers/failure examples must come from your real run.

Do not submit this as-is. The graders will notice fabricated data pretending
to be real customer tweets — that is worse than an honest smaller golden set.

## 2. Reproduce the headline pipeline in under 15 minutes

```bash
git clone <your-repo-url> && cd hiver-support-agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# no API key needed for this smoke test — runs in --mock mode against the
# 30-row synthetic sample already in data/processed/sample_pairs.csv
python scripts/run_pipeline.py --input data/processed/sample_pairs.csv --mock

# with a real Anthropic key, real LLM classification + drafting:
cp .env.example .env   # add GROQ_API_KEY=gsk_...
python scripts/run_pipeline.py --input data/processed/sample_pairs.csv

# evaluation harness against the golden set (swap in your real one first):
python eval/run_eval.py --golden eval/golden_set.csv --mock
```

Expect ~1-2 minutes in `--mock` mode, ~5-10 minutes with real API calls at
30 rows depending on rate limits.

## 3. Getting the real dataset

See `data/README.md`. Short version: download
`thoughtvector/customer-support-on-twitter` from Kaggle, place `twcs.csv` in
`data/raw/`, then:

```bash
python src/data_prep.py --brand AmazonHelp --raw data/raw/twcs.csv --out data/processed/pairs.csv --max-threads 5000
python scripts/run_pipeline.py --input data/processed/pairs.csv
```

`--max-threads` subsamples — the assignment explicitly says a subsample is
fine and expected; we default to 5,000 reconstructed threads (~2 min to
process on a laptop).

## 4. Golden set protocol (what you need to actually do)

1. Run `src/data_prep.py` on the real data.
2. Stratified-sample ~200 inbound customer messages: aim for spread across
   message length (short/long), the intent buckets in `src/intents.py`
   (don't let one intent dominate), and whether the thread eventually got a
   human agent's `@` reply with an order/case number (proxy for "needed
   escalation").
3. For each, hand-label: `true_intent`, `gold_escalate` (0/1), a 1-2 sentence
   `reference_resolution_summary` of how the brand actually resolved it in
   the thread (used to judge groundedness later).
4. Save as `eval/golden_set.csv` with the same columns as the current
   placeholder file.

## 5. Model/config notes

- LLM: Claude (`claude-sonnet-4-6` by default, see `src/llm_client.py`) for
  intent classification and reply drafting. Swappable — the client is a thin
  wrapper, not tied to Anthropic's SDK internals.
- Retrieval: TF-IDF + cosine similarity over historically-resolved
  `(customer_msg, brand_reply)` pairs (see `src/retrieval.py` and
  `report/DECISION_LOG.md` for why not embeddings).
- Escalation: rule-based policy layered on top of LLM confidence + intent
  risk tier (see `src/escalation.py`), not a black-box LLM judgment — this is
  a deliberate choice, explained in the decision log.

## 6. Citations / borrowed material

- Dataset: Axel Springer / thoughtvector, "Customer Support on Twitter",
  Kaggle.
- TF-IDF retrieval pattern: standard scikit-learn `TfidfVectorizer` +
  cosine similarity, no external code copied.
- No other external code, prompts, or libraries beyond what's in
  `requirements.txt` were used verbatim from a third party.
