"""
Retrieval layer that grounds reply drafting in how the brand has historically
resolved similar issues. Uses TF-IDF + cosine similarity over past customer
messages, returning the paired brand_reply as the resolution template.

Why TF-IDF and not embeddings: see report/DECISION_LOG.md. Short version —
this corpus is short, templated customer-support English, TF-IDF is fast,
free, deterministic (helps eval reproducibility), and in a quick manual check
retrieved neighbors were not noticeably worse than an embedding model for
this domain. Swapping in an embedding model is a ~20 line change isolated to
this file if that assumption doesn't hold on the full dataset.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class GroundingRetriever:
    def __init__(self, pairs_df: pd.DataFrame):
        self.df = pairs_df.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        self._matrix = self.vectorizer.fit_transform(self.df["customer_message"].astype(str))

    def retrieve(self, message: str, k: int = 3, exclude_thread_id=None):
        query_vec = self.vectorizer.transform([message])
        sims = cosine_similarity(query_vec, self._matrix).flatten()
        ranked_idx = sims.argsort()[::-1]

        results = []
        for idx in ranked_idx:
            row = self.df.iloc[idx]
            if exclude_thread_id is not None and row.get("thread_id") == exclude_thread_id:
                continue
            results.append({
                "similarity": float(sims[idx]),
                "customer_message": row["customer_message"],
                "brand_reply": row["brand_reply"],
            })
            if len(results) >= k:
                break
        return results
