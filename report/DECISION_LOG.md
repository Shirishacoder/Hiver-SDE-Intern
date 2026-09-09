# Decision log

Plain list of non-obvious decisions and why, per assignment requirement.

1. **Brand: @AmazonHelp, not a smaller brand.** Needed enough volume for
   retrieval grounding to have real precedent to draw on; a low-volume brand
   would make the TF-IDF corpus too sparse to be meaningful.

2. **8 intents, not Banking77's 77 or a from-scratch large taxonomy.** A
   human reviewing escalations needs to hold the taxonomy in their head; a
   77-way taxonomy is built for a different problem (fine-grained banking
   FAQ routing) and would fragment support-relevant categories (e.g.
   product_defect vs return_request) across many overlapping banking-style
   classes that don't map to how a Twitter support team actually works.

3. **TF-IDF retrieval, not embeddings, for the first version.** Free, fast,
   deterministic (matters for reproducible eval numbers), and the corpus is
   short templated support English where lexical overlap is a reasonable
   proxy for topical similarity. Isolated to `src/retrieval.py` so swapping
   in embeddings later is a small, contained change — treated as the #1
   "next week" experiment, not assumed to be already-optimal.

4. **Escalation is a separate rule-based module, not "ask the LLM if this
   needs a human."** An LLM asked to judge its own draft's adequacy tends to
   rate itself favorably (the model that wrote the reply is a biased judge
   of whether the reply is good enough to send unsupervised). A rule layer
   on top of *measurable* signals (classifier confidence, retrieval
   similarity, intent risk tier) is auditable and its thresholds can be
   tuned against the golden set directly.

5. **Escalation is tuned for recall over precision.** Missing an escalation
   (false negative) sends an unsupervised reply to a fragile/high-stakes
   situation; a false positive just costs a human 30 seconds of review. The
   asymmetry in cost justifies the asymmetric tuning — stated explicitly
   rather than left implicit in a threshold number.

6. **Separate `trivial_baseline` and `simple_baseline`, not one baseline.**
   The trivial baseline (majority class + always escalate) tests "is the
   agent better than doing nothing intelligent at all." The simple baseline
   (keyword rules) tests "is the agent's LLM+retrieval complexity actually
   buying anything over a rule-based system a junior engineer could ship in
   a day." These answer different questions and collapsing them into one
   baseline would hide which one the agent is failing to beat.

7. **"Please DM us" deflection replies are filtered out of the grounding
   corpus** (`src/data_prep.py`) because they're not useful *resolution*
   examples for the reply-drafting prompt — but this is flagged explicitly
   in the report (§4/§5) as a source of corpus bias, not silently done.

8. **Judge scores 4 separate dimensions, not one overall score.** A single
   "quality 1-10" lets a judge trade off tone against factual groundedness
   invisibly. Separating them means a fluent-but-ungrounded reply can't hide
   behind a good tone score.

9. **Judge-human agreement is a required step before trusting judge scores
   anywhere else in the report**, not an optional appendix — an unvalidated
   LLM judge is not evidence on its own.

10. **Mock mode exists and is clearly labeled as non-representative**,
    rather than omitted or silently used to generate headline numbers. The
    risk of a mock mode is exactly that someone reports its numbers as real
    — addressed by making the report's §3/§5 explicitly call this out.

11. **Synthetic demo data ships in the repo, clearly marked as fake,**
    instead of requiring a Kaggle download before anything can be verified
    to even run. Reviewers can confirm the pipeline is wired correctly in
    under a minute; real results still require the real dataset.

12. **`--max-threads` subsampling default of 5,000** rather than the full
    brand volume (often 50k+ threads for a brand like AmazonHelp) — the
    assignment explicitly permits and expects subsampling, and 5,000 keeps
    `data_prep.py` and retrieval index-building under ~2 minutes on a
    laptop.

13. **Retrieval excludes a message's own thread_id but not near-duplicates**
    from the same customer/incident. Fixing the latter needs either a
    customer-id join or near-duplicate detection, both left as a stated
    "would do with more data/time" item rather than silently assumed solved.

14. **Escalation reason is a concatenated list of triggered rules, not a
    free-text LLM explanation.** A human reviewing an escalation needs to
    know *which specific threshold* fired (confidence floor / grounding
    floor / risk tier) to trust or override it — a free-text LLM
    justification is harder to audit and could rationalize a bad call
    fluently.

15. **No autonomous sending, no fine-tuning, no multi-turn state** — three
    explicit scope cuts stated in the report (§1) rather than left
    ambiguous, since "what you chose not to build" is a required section
    and vague scoping is itself a red flag in a take-home like this.
