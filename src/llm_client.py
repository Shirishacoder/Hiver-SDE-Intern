"""
Thin wrapper around an LLM API used for both intent classification and
reply drafting. Supports a --mock mode (no API key, no network) so the
pipeline can be smoke-tested by graders instantly.

Provider: Groq (free tier, no credit card, OpenAI-compatible chat API) —
see report/DECISION_LOG.md for why. Swapping providers only touches this
file; nothing else in src/ or eval/ knows which provider is behind
LLMClient.complete().

Why a wrapper instead of calling the SDK everywhere: keeps the rest of the
codebase provider-agnostic and gives us one place to add retries/rate-limit
backoff.
"""
import os
import json
import time
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")# Groq free-tier model; see README for alternatives


@dataclass
class LLMResponse:
    text: str
    mocked: bool = False


class LLMClient:
    def __init__(self, mock: bool = False, model: str = DEFAULT_MODEL, max_retries: int = 3):
        self.mock = mock
        self.model = model
        self.max_retries = max_retries
        self._client = None
        if not mock:
            try:
                from groq import Groq
                api_key = os.environ.get("GROQ_API_KEY")
                if not api_key:
                    raise RuntimeError(
                        "GROQ_API_KEY not set. Either export it, put it in "
                        ".env, or pass --mock to run without API calls. Get a "
                        "free key (no card needed) at console.groq.com."
                    )
                self._client = Groq(api_key=api_key)
            except ImportError as e:
                raise RuntimeError(
                    "groq package not installed. `pip install groq` "
                    "or run with --mock."
                ) from e

    def complete(self, system: str, user: str, max_tokens: int = 500) -> LLMResponse:
        if self.mock:
            return LLMResponse(text=self._mock_reply(system, user), mocked=True)

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    max_tokens=max(max_tokens, 400),
                    reasoning_effort="low",
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                text = (resp.choices[0].message.content or "").strip()
                if not text:
                    raise RuntimeError("model returned empty content (reasoning likely ate the token budget)")
                return LLMResponse(text=text, mocked=False)
            except Exception as e:  # noqa: BLE001 - want to retry on anything transient
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_err}")

    @staticmethod
    def _mock_reply(system: str, user: str) -> str:
        """
        Deterministic, no-network stand-in used for smoke tests and CI.
        Not a real model — good enough to exercise every code path
        (JSON parsing, downstream logic) but NOT a substitute for real
        LLM output when reporting quality numbers. See README §1.
        """
        if '"intent"' in system or "classify" in system.lower():
            text_lower = user.lower()
            if any(k in text_lower for k in ["refund", "money back", "charged twice", "charge"]):
                intent, conf = "refund_or_billing", 0.6
            elif any(k in text_lower for k in ["where is", "tracking", "delivered", "arrive"]):
                intent, conf = "delivery_status", 0.6
            elif any(k in text_lower for k in ["broken", "doesn't work", "defective", "cracked"]):
                intent, conf = "product_defect", 0.6
            elif any(k in text_lower for k in ["login", "password", "account", "locked"]):
                intent, conf = "account_access", 0.6
            elif any(k in text_lower for k in ["cancel"]):
                intent, conf = "cancellation", 0.6
            elif any(k in text_lower for k in ["return", "exchange"]):
                intent, conf = "return_request", 0.6
            elif any(k in text_lower for k in ["furious", "third time", "hold for an hour"]):
                intent, conf = "complaint_escalation", 0.55
            elif any(k in text_lower for k in ["thanks", "thank you"]):
                intent, conf = "general_inquiry", 0.7
            else:
                intent, conf = "general_inquiry", 0.4
            return json.dumps({"intent": intent, "confidence": conf})
        return (
            "[MOCK REPLY] Thanks for reaching out — a support specialist will "
            "review the details above and follow up shortly."
        )
