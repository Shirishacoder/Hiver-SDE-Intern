"""
Samples rows from the real (customer_message, brand_reply) pairs for hand
labeling into the golden set. Pre-tags each row with the keyword baseline's
guess as a *starting suggestion only* — you must read the actual message and
correct/confirm it, not trust the guess blindly (that's the whole point of a
hand-labeled set).
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
from src.intents import classify_intent_keyword_baseline, INTENT_LIST  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/processed/pairs.csv")
    ap.add_argument("--out", default="eval/to_label.csv")
    ap.add_argument("--per-intent", type=int, default=27)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.pairs)
    df = df.dropna(subset=["customer_message", "brand_reply"])
    df = df[df["customer_message"].str.len() > 8]

    df["suggested_intent"] = df["customer_message"].apply(
        lambda m: classify_intent_keyword_baseline(m)["intent"])

    samples = []
    for intent in INTENT_LIST:
        bucket = df[df["suggested_intent"] == intent]
        n = min(args.per_intent, len(bucket))
        if n < args.per_intent:
            print(f"WARNING: only {len(bucket)} rows matched '{intent}', wanted {args.per_intent}.")
        samples.append(bucket.sample(n=n, random_state=args.seed))

    sampled = pd.concat(samples).sample(frac=1, random_state=args.seed)

    sampled["true_intent"] = ""
    sampled["gold_escalate"] = ""
    sampled["reference_resolution_summary"] = ""

    out_cols = ["thread_id", "customer_message", "brand_reply", "suggested_intent",
                "true_intent", "gold_escalate", "reference_resolution_summary"]
    sampled[out_cols].to_csv(args.out, index=False)
    print(f"\nWrote {len(sampled)} rows to {args.out}")
    print(sampled["suggested_intent"].value_counts())


if __name__ == "__main__":
    main()