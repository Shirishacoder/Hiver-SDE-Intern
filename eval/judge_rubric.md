# LLM-as-judge rubric for reply quality

The judge is a separate Claude call (see `eval/metrics.py::judge_reply`) that
scores each `(customer_message, reference_resolution_summary, reply_draft)`
triple on 4 dimensions, 1-5 each. We deliberately do NOT ask a single
"overall quality 1-10" score — a single number invites the judge to conflate
tone with correctness, and gives us nothing to debug when scores are low.

## Dimensions

1. **Groundedness (1-5)** — Does the reply's proposed resolution match the
   pattern in `reference_resolution_summary` (or make sense given it), rather
   than inventing a policy/promise not supported by precedent?
   - 5: matches precedent exactly in substance
   - 3: plausible but not clearly supported by the reference
   - 1: contradicts the reference or invents unsupported specifics (e.g. a
     dollar amount or delivery date not implied by precedent)

2. **Correctness / no false promises (1-5)** — Does the reply avoid promising
   something the agent can't actually guarantee (exact refund date, "your
   package will arrive tomorrow", etc.)? Hedged, honest language scores high.

3. **Tone (1-5)** — Brief, empathetic without over-apologizing, matches the
   brand's actual voice (see examples in `data/processed/*_pairs.csv`).

4. **Actionability (1-5)** — Does the customer know what happens next (who
   does what, roughly when), or is it a non-answer ("we'll look into it")?

## Judge prompt (implemented in `eval/metrics.py`)

```
You are scoring a draft customer-support reply. You will be given the
customer's message, a short reference summary of how this brand actually
resolved a similar issue historically, and the draft reply to score.

Score each dimension 1-5 (integers only):
- groundedness
- correctness
- tone
- actionability

Return ONLY JSON: {"groundedness": int, "correctness": int, "tone": int,
"actionability": int, "one_line_reason": str}
```

## Judge-human agreement (mandatory, per assignment)

Protocol used in `report/REPORT.md §3`:
1. Take 40 replies from the golden-set run (stratified across intents and
   escalate/auto-handle).
2. I hand-score the same 4 dimensions myself, blind to the judge's scores.
3. Report per-dimension agreement as:
   - Exact match rate
   - Within-1-point agreement rate (more meaningful for a 1-5 scale)
   - Cohen's kappa treating each dimension as ordinal categories
4. If agreement on any dimension is weak (<0.4 kappa or <70% within-1), that
   dimension's judge scores are flagged as unreliable in the report and not
   used to support any headline claim.

`eval/metrics.py::human_judge_agreement()` computes these once you fill in
`eval/results/human_scores.csv` (columns: thread_id, groundedness, correctness,
tone, actionability — your own scores on the same 40 replies the judge saw).
