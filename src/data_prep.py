"""
Filters the raw Kaggle `twcs.csv` (Customer Support on Twitter) down to one
brand and reconstructs (customer_message, brand_reply) pairs by following
`in_response_to_tweet_id`.

Usage:
    python src/data_prep.py --brand AmazonHelp --raw data/raw/twcs.csv \
        --out data/processed/pairs.csv --max-threads 5000

Output columns: thread_id, customer_message, brand_reply
(same schema as data/processed/sample_pairs.csv so downstream code doesn't
care whether it's reading real or synthetic data)
"""
import argparse
import re
import pandas as pd


def clean_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = re.sub(r"@\w+", "", t)          # strip @mentions (brand + customer handles)
    t = re.sub(r"http\S+", "", t)        # strip URLs
    t = re.sub(r"\s+", " ", t).strip()
    return t


def build_pairs(raw_path: str, brand: str, max_threads: int) -> pd.DataFrame:
    df = pd.read_csv(raw_path, dtype={"tweet_id": str, "in_response_to_tweet_id": str,
                                        "response_tweet_id": str, "author_id": str})
    by_id = df.set_index("tweet_id", drop=False)

    brand_replies = df[(df["author_id"] == brand) & (df["inbound"] == False)]  # noqa: E712

    rows = []
    for _, reply_row in brand_replies.iterrows():
        parent_id = reply_row.get("in_response_to_tweet_id")
        if pd.isna(parent_id) or parent_id not in by_id.index:
            continue
        parent = by_id.loc[parent_id]
        if isinstance(parent, pd.DataFrame):  # duplicate ids, just take first
            parent = parent.iloc[0]
        if not parent.get("inbound", False):
            continue  # only keep genuine customer -> brand pairs

        cust_msg = clean_text(parent["text"])
        brand_msg = clean_text(reply_row["text"])

        # Drop non-informative "please DM us" deflections — not useful as
        # grounding examples for reply drafting, and they'd make every
        # retrieved example look the same. See DECISION_LOG.md.
        if len(brand_msg) < 15 or "dm" in brand_msg.lower()[:40] and len(brand_msg) < 40:
            continue
        if not cust_msg or not brand_msg:
            continue

        rows.append({
            "thread_id": reply_row["tweet_id"],
            "customer_message": cust_msg,
            "brand_reply": brand_msg,
        })
        if len(rows) >= max_threads:
            break

    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", default="AmazonHelp")
    ap.add_argument("--raw", default="data/raw/twcs.csv")
    ap.add_argument("--out", default="data/processed/pairs.csv")
    ap.add_argument("--max-threads", type=int, default=5000)
    args = ap.parse_args()

    pairs = build_pairs(args.raw, args.brand, args.max_threads)
    pairs.to_csv(args.out, index=False)
    print(f"Wrote {len(pairs)} pairs to {args.out}")
