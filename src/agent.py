"""
Main pipeline: given an inbound customer message, produce
{intent, reply_draft, escalate, escalation_reason, grounding_examples}.
"""
import json
from src.llm_client import LLMClient
from src.intents import classify_intent_llm, INTENTS
from src.retrieval import GroundingRetriever
from src.escalation import decide_escalation

DRAFT_SYSTEM_PROMPT = """You are an AmazonHelp customer support agent drafting a reply to a
customer tweet. You will be given the customer's message, the classified intent, and 1-3
examples of how AmazonHelp has resolved similar issues in the past.

Rules:
- Match the brand's tone from the examples: brief, empathetic, action-oriented, no over-apologizing.
- Ground your reply in the pattern of resolution shown in the examples where relevant — don't
  invent a policy that isn't reflected in them.
- Keep it to 1-3 sentences, tweet-appropriate.
- Do not promise specific dates/amounts you can't know; use the same hedging style as the examples.
- Output ONLY the reply text, nothing else."""


class SupportAgent:
    def __init__(self, pairs_df, mock: bool = False):
        self.llm = LLMClient(mock=mock)
        self.retriever = GroundingRetriever(pairs_df)

    def handle(self, message: str, thread_id=None) -> dict:
        intent_result = classify_intent_llm(self.llm, message)
        intent = intent_result["intent"]
        confidence = intent_result.get("confidence", 0.0)

        grounding = self.retriever.retrieve(message, k=3, exclude_thread_id=thread_id)
        top_sim = grounding[0]["similarity"] if grounding else 0.0

        esc = decide_escalation(intent, confidence, top_sim, message)

        examples_block = "\n\n".join(
            f"Example {i+1} (similarity {g['similarity']:.2f}):\n"
            f"Customer: {g['customer_message']}\nAmazonHelp: {g['brand_reply']}"
            for i, g in enumerate(grounding)
        ) or "No closely matching historical examples found."

        user_prompt = (
            f"Customer message: {message}\n"
            f"Classified intent: {intent} ({INTENTS.get(intent, '')})\n\n"
            f"Historical resolution examples:\n{examples_block}\n\n"
            f"Draft the reply now."
        )
        draft = self.llm.complete(system=DRAFT_SYSTEM_PROMPT, user=user_prompt, max_tokens=200)

        return {
            "customer_message": message,
            "intent": intent,
            "intent_confidence": confidence,
            "reply_draft": draft.text.strip(),
            "escalate": esc["escalate"],
            "escalation_reason": esc["reason"],
            "grounding_examples": grounding,
            "mocked": draft.mocked,
        }


if __name__ == "__main__":
    import argparse
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/processed/sample_pairs.csv")
    ap.add_argument("--message", required=True)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    pairs = pd.read_csv(args.pairs)
    agent = SupportAgent(pairs, mock=args.mock)
    result = agent.handle(args.message)
    print(json.dumps(result, indent=2))
