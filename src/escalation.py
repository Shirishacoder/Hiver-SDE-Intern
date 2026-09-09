"""
Escalation policy: decides auto-handle vs escalate-to-human, with a reason.

Deliberately rule-based on top of the classifier's confidence and a per-intent
risk tier, rather than "ask the LLM if this needs a human" — an LLM asked
that question tends to rubber-stamp its own draft as fine. See
report/DECISION_LOG.md for the reasoning and report/REPORT.md for the
false-negative risk this still carries (escalation recall is the headline
metric we care most about, precision is secondary).
"""

# Intents where getting it wrong is costly (money, account security, already-
# angry customer) default to escalate unless the model is very confident AND
# there's a strong grounding match.
HIGH_RISK_INTENTS = {"refund_or_billing", "account_access", "complaint_escalation"}

CONFIDENCE_FLOOR = 0.55          # below this, always escalate regardless of intent
GROUNDING_SIM_FLOOR = 0.25       # weak retrieval match -> not enough precedent to auto-answer


def decide_escalation(intent: str, intent_confidence: float, top_grounding_similarity: float,
                       message: str) -> dict:
    reasons = []

    if intent_confidence < CONFIDENCE_FLOOR:
        reasons.append(f"intent classifier confidence {intent_confidence:.2f} below floor "
                        f"{CONFIDENCE_FLOOR}")

    if top_grounding_similarity < GROUNDING_SIM_FLOOR:
        reasons.append(f"no closely-matching historical resolution found "
                        f"(best similarity {top_grounding_similarity:.2f})")

    if intent in HIGH_RISK_INTENTS:
        reasons.append(f"intent '{intent}' is high-risk (money/security/frustrated customer) "
                        f"-> requires human sign-off by policy")

    if intent == "complaint_escalation":
        reasons.append("customer explicitly signaling repeated/unresolved frustration")

    escalate = len(reasons) > 0
    return {
        "escalate": escalate,
        "reason": "; ".join(reasons) if reasons else "high-confidence, low-risk intent with "
                                                       "strong historical precedent",
    }
