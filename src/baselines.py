"""
Two baselines the agent must beat, per assignment requirements.

Trivial baseline:
    - intent: always predict majority class
    - reply: fixed canned response regardless of message
    - escalation: always escalate (safest possible policy, 0 automation)

Simple baseline:
    - intent: keyword rules (src/intents.py::classify_intent_keyword_baseline)
    - reply: template lookup keyed on intent, no grounding/personalization
    - escalation: escalate iff keyword hit for HIGH_RISK_INTENTS, else auto-handle
"""
from src.intents import (
    trivial_baseline,
    classify_intent_keyword_baseline,
)
from src.escalation import HIGH_RISK_INTENTS

TRIVIAL_CANNED_REPLY = ("Thanks for reaching out. A member of our support team will review your "
                         "message and get back to you shortly.")

SIMPLE_TEMPLATES = {
    "delivery_status": "We're sorry for the delay — we're checking the latest tracking status "
                        "and will follow up shortly.",
    "refund_or_billing": "We're looking into this charge/refund now and will follow up shortly.",
    "product_defect": "Sorry to hear that — we can arrange a replacement or refund, please "
                       "confirm your order number.",
    "return_request": "You can start a return from Your Orders — select the item and choose "
                       "'Return items'.",
    "account_access": "Please try the 'forgot password' link on the sign-in page; let us know "
                       "if you're still locked out.",
    "cancellation": "We'll take a look at cancelling this for you — please confirm the order "
                     "number.",
    "complaint_escalation": "We're sorry for the frustration — a specialist will follow up with "
                             "you directly.",
    "general_inquiry": "Thanks for your message — let us know if you have a specific order or "
                        "question we can help with.",
}


def run_trivial_baseline(message: str) -> dict:
    intent_result = trivial_baseline(message)
    return {
        "customer_message": message,
        "intent": intent_result["intent"],
        "reply_draft": TRIVIAL_CANNED_REPLY,
        "escalate": True,
        "escalation_reason": "trivial baseline always escalates",
    }


def run_simple_baseline(message: str) -> dict:
    intent_result = classify_intent_keyword_baseline(message)
    intent = intent_result["intent"]
    escalate = intent in HIGH_RISK_INTENTS
    return {
        "customer_message": message,
        "intent": intent,
        "reply_draft": SIMPLE_TEMPLATES.get(intent, SIMPLE_TEMPLATES["general_inquiry"]),
        "escalate": escalate,
        "escalation_reason": (
            f"keyword-matched high-risk intent '{intent}'" if escalate
            else f"keyword-matched low-risk intent '{intent}'"
        ),
    }
