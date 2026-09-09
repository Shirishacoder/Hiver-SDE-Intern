"""
All quantitative metrics for the report:
 - intent classification accuracy / per-class precision-recall-F1
 - escalation precision/recall/F1 against gold_escalate
 - LLM-as-judge reply quality scores (see judge_rubric.md)
 - judge-vs-human agreement (Cohen's kappa + within-1 rate)
"""
import json
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    cohen_kappa_score,
)

JUDGE_SYSTEM_PROMPT = """You are scoring a draft customer-support reply. You will be given the
customer's message, a short reference summary of how this brand actually resolved a similar issue
historically, and the draft reply to score.

Score each dimension 1-5 (integers only):
- groundedness: does the reply's resolution match the pattern in the reference, without inventing
  unsupported specifics?
- correctness: does it avoid false/overconfident promises (exact dates, amounts, guarantees)?
- tone: brief, empathetic without over-apologizing, matches a real support brand's voice?
- actionability: does the customer know what happens next?

Return ONLY JSON: {"groundedness": int, "correctness": int, "tone": int, "actionability": int,
"one_line_reason": str}"""


def intent_metrics(y_true, y_pred, labels=None) -> dict:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0, labels=labels)
    report = classification_report(y_true, y_pred, zero_division=0, labels=labels)
    return {
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "per_class_report": report,
    }


def escalation_metrics(gold_escalate, pred_escalate) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold_escalate, pred_escalate, average="binary", zero_division=0)
    accuracy = accuracy_score(gold_escalate, pred_escalate)
    # Escalation recall is the metric we care about most: missing an
    # escalation (false negative) means a fragile/angry customer gets an
    # auto-reply. See report/REPORT.md for why we weight this over precision.
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def judge_reply(client, customer_message: str, reference_summary: str, reply_draft: str) -> dict:
    user_prompt = (
        f"Customer message: {customer_message}\n"
        f"Reference resolution (how this was actually resolved historically): "
        f"{reference_summary}\n"
        f"Draft reply to score: {reply_draft}"
    )
    resp = client.complete(system=JUDGE_SYSTEM_PROMPT, user=user_prompt, max_tokens=150)
    try:
        return json.loads(resp.text.strip())
    except (json.JSONDecodeError, AttributeError):
        return {"groundedness": None, "correctness": None, "tone": None,
                 "actionability": None, "one_line_reason": "judge parse failure"}


def human_judge_agreement(judge_scores_path: str, human_scores_path: str) -> dict:
    """
    judge_scores_path / human_scores_path: CSVs with columns
    thread_id, groundedness, correctness, tone, actionability
    """
    judge_df = pd.read_csv(judge_scores_path).set_index("thread_id")
    human_df = pd.read_csv(human_scores_path).set_index("thread_id")
    common = judge_df.index.intersection(human_df.index)
    judge_df, human_df = judge_df.loc[common], human_df.loc[common]

    results = {}
    for dim in ["groundedness", "correctness", "tone", "actionability"]:
        j, h = judge_df[dim], human_df[dim]
        exact = (j == h).mean()
        within_1 = (abs(j - h) <= 1).mean()
        try:
            kappa = cohen_kappa_score(j, h)
        except ValueError:
            kappa = float("nan")
        results[dim] = {"exact_match_rate": exact, "within_1_rate": within_1, "cohen_kappa": kappa}
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Compute judge-human agreement from two score CSVs")
    ap.add_argument("--judge-scores", default="eval/results/judge_scores.csv")
    ap.add_argument("--human-scores", default="eval/results/human_scores.csv")
    args = ap.parse_args()
    print(json.dumps(human_judge_agreement(args.judge_scores, args.human_scores), indent=2))
