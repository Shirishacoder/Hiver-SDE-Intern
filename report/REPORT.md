# Report — AI Support Agent for @AmazonHelp

> **Status of this document**: every section below is fully structured and
> the reasoning frameworks are real. The numbers and failure examples shown
> inline are from a **mock-mode smoke test on 30 synthetic rows**
> (`data/processed/sample_pairs.csv` + `eval/golden_set.csv` placeholder),
> run to prove the harness works end-to-end and to show what the analysis
> should look like. **Before submitting, replace `eval/golden_set.csv` with
> a real 150-250 row hand-labeled set from the actual Kaggle data, re-run
> `eval/run_eval.py` without `--mock`, and swap these numbers for real ones.**
> Leaving synthetic numbers in a final submission would itself be the kind
> of "misleading headline number" §4 warns about.

## 1. Problem framing

**Brand**: @AmazonHelp. Chosen because it has high volume, a genuine mix of
intents (not just one recurring complaint type), and enough historically
resolved threads to make grounding-by-retrieval meaningful.

**What "good" means here, concretely:**
- **Escalation recall matters more than escalation precision.** The cost of
  auto-handling something that should have gone to a human (angry customer,
  money, account security) is much higher than the cost of a human
  reviewing something that was actually fine. So the system is deliberately
  tuned to over-escalate rather than under-escalate — see
  `src/escalation.py` and the risk-tier design.
- **Groundedness beats fluency.** A reply that sounds great but promises a
  refund timeline or policy the brand doesn't actually follow is worse than
  a slightly stiff reply that matches precedent. This is why the judge
  rubric scores groundedness and correctness as separate dimensions from
  tone (§3, `eval/judge_rubric.md`).
- **A small, human-auditable intent taxonomy** (8 classes, see
  `src/intents.py`) rather than a large fine-grained one. A human reviewing
  escalations needs the taxonomy to fit in their head.

**Intent taxonomy derivation**: sampled ~200 inbound AmazonHelp-directed
tweets, read them, and grouped by hand into recurring resolution patterns
(not just topic) — e.g. "product_defect" and "return_request" are separate
because the brand resolves them differently (replace-in-place vs.
return-then-refund), even though both are "something's wrong with my item."

**What I chose not to build:**
- **No multi-turn dialogue state.** The agent handles one inbound message at
  a time, grounded in historical single-turn (customer→brand) pairs. Real
  threads are often 3-5 turns; handling that well needs conversation memory
  and a different data-reconstruction approach. Explicitly out of scope for
  the take-home window — flagged as the top "next week" item (§5).
- **No sentiment model as a separate component.** Frustration signals are
  folded into the `complaint_escalation` intent and its keywords rather than
  a separate sentiment score feeding two different systems — one signal,
  one place it's used, easier to audit.
- **No fine-tuning.** Everything is prompted, not trained, given the token
  budget and timeline — see decision log.
- **No autonomous sending.** This produces *drafts* and an *escalation
  decision with a reason*, not an agent that posts replies. That's a
  deliberate scope boundary given the assignment says "draft a reply."

## 2. Golden set — sampling and labeling

**Sampling** (protocol to follow on the real data; the current file is a
30-row synthetic stand-in):
- Stratify by intent bucket (aim for ≥15 examples per intent) using the
  keyword baseline as a rough pre-filter, then hand-verify/correct the label
  — this avoids a golden set that's 80% `delivery_status` because that's
  the most common tweet type.
- Within each intent, deliberately include: short messages (<10 words),
  long/rambling ones, messages with typos/slang, and messages that mention
  an order number vs. don't.
- Include ~20% "hard negatives": messages that superficially look like one
  intent but are actually another (e.g. "refund" mentioned inside a
  `general_inquiry` policy question) — these are where classifiers actually
  fail.

**Labeling**: for each sampled inbound tweet, read the *rest of the thread*
(via `in_response_to_tweet_id`/`response_tweet_id` chains) to see how
AmazonHelp actually resolved it, then label `true_intent`, `gold_escalate`
(1 if the brand's real response pattern needed a specialist/manual
investigation/apology-for-repeated-issue, 0 if it was a standard templated
resolution), and a 1-2 sentence `reference_resolution_summary` used later by
the judge for groundedness scoring.

## 3. Results vs. baselines

Three systems compared (see `src/baselines.py`, `src/agent.py`,
`eval/run_eval.py`):

| System | Intent accuracy | Escalation P / R / F1 | Notes |
|---|---|---|---|
| Trivial (majority-class intent, always escalate) | 13.3% | 0.47 / 1.00 / 0.64 | Escalation recall is trivially perfect but precision is bad — 100% human load, no automation |
| Simple (keyword rules, template replies) | 73.3% | 0.90 / 0.64 / 0.75 | Surprisingly strong on this **synthetic, keyword-friendly** golden set — see §4 |
| **Agent** (LLM classify + retrieval-grounded draft + rule-based escalation) | 60.0%* | 0.58 / 0.79 / 0.67* | *from mock-mode smoke test — not representative, see below |

*The agent numbers above are from `--mock` mode (a deterministic keyword
stand-in for the LLM, used only to prove the harness runs — see
`src/llm_client.py::_mock_reply`). They are **not** the real headline
number and must not be quoted as such. Re-run with a real Anthropic key on
the real golden set for the number that goes in the actual submission.*

**Reply quality (LLM-judge, `eval/judge_rubric.md`)**: run
`python eval/run_eval.py --run-judge` (requires a real API key — judge
scoring is meaningless on mock replies) then fill in
`eval/results/human_scores.csv` with your own scores on the same 40 threads
and run `python eval/metrics.py` for the judge-vs-human agreement numbers
(exact-match rate, within-1 rate, Cohen's kappa per dimension — see
`eval/metrics.py::human_judge_agreement`). Report the agreement numbers here
and flag any dimension below 0.4 kappa as unreliable before citing its
scores anywhere else in this report.

## 4. Failure analysis — top 5 failure modes

(Framework + real examples from the mock smoke test below; replace/extend
with real-LLM failures once you have a real run.)

1. **Grounding retrieval finds superficially similar but semantically wrong
   precedent.** TF-IDF matches on shared vocabulary ("order", "refund") not
   on which resolution actually applies. Example: a `cancellation` message
   can retrieve a `refund_or_billing` example if both mention "order
   #xxx-xxxx cancel my card charge." *Hypothesis*: needs either a
   retrieval filter by predicted intent first, or an embedding model that
   captures resolution-type similarity better than surface tokens.

2. **Confidence is a poor escalation signal for a mocked/heuristic
   classifier** — the mock classifier returns a flat 0.6 confidence for any
   keyword hit, which is meaningless for calibration. *Hypothesis*: with a
   real LLM, confidence needs calibration-checking (does "0.9 confidence"
   actually correlate with being right 90% of the time on the golden set?)
   before it's trusted as an escalation input — see `eval/metrics.py`,
   nothing currently checks this and it should be added.

3. **Short, ambiguous messages ("Cancel order 556-1290") give the
   classifier nothing to disambiguate urgency from.** A one-line cancel
   request and a one-line cancel-after-repeated-failed-attempts request look
   identical without thread history. *Hypothesis*: this is the single-turn
   scope limitation from §1 showing up directly — multi-turn context would
   likely fix a meaningful fraction of these.

4. **Keyword baseline occasionally beats the agent on accuracy in this
   synthetic golden set** (see §3 table) — because the synthetic examples
   were generated with clean, keyword-obvious intent signals, which flatters
   a keyword matcher unfairly. *Hypothesis*: on real messy tweets (typos,
   sarcasm, indirect phrasing) this gap should reverse; if it doesn't,
   that's a real finding worth reporting honestly rather than hiding.

5. **"Please DM us" deflection replies were filtered out of the grounding
   corpus** (`src/data_prep.py`), which is good for reply quality but means
   the corpus under-represents how often the *real* brand behavior is "ask
   to move to DM" rather than resolve in-thread. *Hypothesis*: this could
   make the agent look more capable of in-thread resolution than the brand
   actually is — worth measuring what fraction of real threads end in a DM
   deflection and stating that limitation explicitly.

## 5. "What is misleading about my headline number?" (mandatory)

- **The comparison table numbers above are from mock mode, not a real LLM.**
  This is the single biggest thing that would mislead a reader skimming
  just the table — flagged in bold in §3, but worth repeating: any headline
  "agent beats baselines" claim must come from a `--mock`-free run.
- **The golden set (as shipped) is synthetic and hand-designed to be
  intent-separable**, which inflates every system's intent accuracy
  relative to real, messy tweets. Real accuracy will very likely be lower
  for all three systems, and the *gap* between agent and baselines could
  shrink or grow — don't assume the ranking holds.
- **Escalation recall/precision on a 150-250 example golden set has wide
  confidence intervals.** A single-digit swing in true positives moves
  recall by several points. Report a bootstrap or Wilson confidence
  interval alongside any escalation P/R number, not just the point estimate.
- **The retrieval corpus and the golden set overlap risk**: if grounding
  examples used at inference time were drawn from the same pool the golden
  set examples were sampled from, similarity between a golden-set message
  and its *own* near-duplicate in the corpus could inflate groundedness
  scores. `src/retrieval.py::retrieve(..., exclude_thread_id=...)` guards
  against retrieving a message's exact own thread, but doesn't guard against
  near-duplicates from the same customer/incident — worth checking for on
  the real dataset.
- **"Please DM us" filtering (§4.5)** means the corpus and therefore the
  agent's sense of "how this brand resolves things" is systematically
  biased toward the subset of issues actually resolved in-thread.
- **Judge-human agreement, if not reported, would make the reply-quality
  number unfalsifiable.** A judge score with no agreement check is not
  evidence, it's a second unverified opinion — this is why §3 makes running
  the agreement check mandatory before quoting judge scores.

## 6. What I'd do next with one more week

1. Real multi-turn context (§1) — biggest expected accuracy/groundedness
   lift, and closes failure mode #3.
2. Calibration check for classifier confidence (failure mode #2) before
   trusting it as an escalation input, possibly replacing raw LLM confidence
   with a learned/calibrated score from the golden set.
3. Swap TF-IDF for an embedding-based retriever and A/B the two on the same
   golden set to see if failure mode #1 (semantically-wrong grounding)
   actually improves — right now that's a hypothesis, not a measured result.
4. Expand the golden set to the full 150-250 target with real data and rerun
   everything — everything in this report is currently a dry run.
5. Add a "please DM us" rate as a tracked metric so the corpus bias in
   failure mode #5 is visible over time, not just a one-off caveat.
6. Cost/latency tracking per request (tokens, $ per resolved ticket) — not
   built at all yet, and matters for "would a team actually deploy this."
