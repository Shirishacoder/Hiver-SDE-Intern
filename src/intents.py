"""
Intent taxonomy for AmazonHelp, derived by skimming ~200 sampled inbound
tweets and clustering by hand (see report/REPORT.md §1 for the process).
Kept deliberately small (8 intents) — see decision log for why we didn't
use something like Banking77's 77-way taxonomy.
"""
import json
import re

INTENTS = {
    "delivery_status": "Where's my order / tracking / late or lost package",
    "refund_or_billing": "Refund requests, duplicate/incorrect charges, billing disputes",
    "product_defect": "Item arrived broken, doesn't work, wrong item received",
    "return_request": "How to return or exchange an item",
    "account_access": "Login issues, locked account, 2FA/verification problems",
    "cancellation": "Cancel an order or subscription",
    "complaint_escalation": "Repeated unresolved issue / explicit frustration / demands a human",
    "general_inquiry": "Policy questions, shipping info, thanks, anything not above",
}

INTENT_LIST = list(INTENTS.keys())

CLASSIFY_SYSTEM_PROMPT = f"""You are classifying an inbound customer support tweet for AmazonHelp
into exactly one intent from this fixed list:

{json.dumps(INTENTS, indent=2)}

Respond with ONLY a JSON object, no other text:
{{"intent": "<one of the keys above>", "confidence": <float 0-1>}}

confidence reflects how unambiguous the message is, not how important it is."""


def classify_intent_llm(client, message: str) -> dict:
    resp = client.complete(system=CLASSIFY_SYSTEM_PROMPT, user=message, max_tokens=100)
    try:
        parsed = json.loads(resp.text.strip())
        if parsed.get("intent") not in INTENT_LIST:
            parsed["intent"] = "general_inquiry"
        return parsed
    except (json.JSONDecodeError, AttributeError):
        return {"intent": "general_inquiry", "confidence": 0.0}


# --- Simple baseline: keyword rules, no LLM call at all -------------------
_KEYWORD_RULES = [
    ("complaint_escalation", ["furious", "unacceptable", "third time", "done with", "on hold"]),
    ("account_access", ["password", "login", "log in", "locked out", "2fa", "verification code"]),
    ("cancellation", ["cancel my", "cancel order", "cancel the"]),
    ("return_request", ["return", "exchange"]),
    ("product_defect", ["broken", "defective", "doesn't work", "cracked", "damaged"]),
    ("refund_or_billing", ["refund", "money back", "charged twice", "duplicate charge", "overcharged"]),
    ("delivery_status", ["where is my", "tracking", "delivered", "hasn't arrived", "still in transit"]),
]


def classify_intent_keyword_baseline(message: str) -> dict:
    """The 'simple' baseline referenced in the report — pure keyword rules."""
    text = message.lower()
    for intent, keywords in _KEYWORD_RULES:
        if any(re.search(re.escape(k), text) for k in keywords):
            return {"intent": intent, "confidence": 1.0}
    return {"intent": "general_inquiry", "confidence": 1.0}


def trivial_baseline(message: str) -> dict:
    """The 'trivial' baseline — always predict the majority class."""
    return {"intent": "delivery_status", "confidence": 1.0}
