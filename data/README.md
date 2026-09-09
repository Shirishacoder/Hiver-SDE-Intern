# Data

## Real dataset (required for real results)

1. Download `thoughtvector/customer-support-on-twitter` from Kaggle
   (https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter).
2. Unzip and place `twcs.csv` in `data/raw/`. Schema (Kaggle's, unchanged):
   `tweet_id, author_id, inbound, created_at, text, response_tweet_id,
   in_response_to_tweet_id`.
3. Run:
   ```bash
   python src/data_prep.py --brand AmazonHelp --raw data/raw/twcs.csv \
       --out data/processed/pairs.csv --max-threads 5000
   ```
   This filters to rows where `author_id == AmazonHelp` (outbound) or whose
   `in_response_to_tweet_id` chains back to an AmazonHelp reply, reconstructs
   `(customer_message, brand_reply)` pairs by following the
   `in_response_to_tweet_id` links, strips `@handles`/URLs, and drops pairs
   where the brand reply is just "please DM us" with no content (these are
   not useful grounding examples — see decision log).

`--max-threads 5000` is a deliberate subsample (full brand volume is often
50k+ threads); the assignment explicitly says a subsample is fine.

## Synthetic sample (ships with repo, for smoke-testing only)

`data/processed/sample_pairs.csv` — 30 rows I generated to match the same
schema `src/data_prep.py` outputs, so `scripts/run_pipeline.py` can be run
immediately without a Kaggle account. **These are not real tweets.** They
exist only so a grader can run the pipeline in <15 minutes without first
doing a Kaggle download. Do not use this file as your golden set or cite it
as real customer data anywhere in the report.
