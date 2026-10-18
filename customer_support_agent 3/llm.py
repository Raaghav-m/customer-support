"""
Pluggable "LLM" interface used throughout the agent graph.

Two backends are provided:

1. RuleBasedLLM (default) - a deterministic, keyword-driven stand-in for a
   GPT-4-class model. It exists so that the full LangGraph pipeline can be
   built, run, tested, and evaluated end-to-end WITHOUT an OpenAI API key or
   network access - which matters for reproducible grading/demo environments.
   It is intentionally simple: it is not a claim of NLU accuracy, only a way
   to exercise every node, edge, and interrupt in the graph with real,
   inspectable inputs/outputs.

2. OpenAILLM - a thin wrapper around langchain_openai.ChatOpenAI, used
   automatically if OPENAI_API_KEY is set in the environment. This is the
   production backend referenced in the Proposed Methodology (GPT-4-class
   API / open-weight alternative).

Swapping backends requires no changes anywhere else in the codebase, since
every node calls `get_llm()` and only depends on the .classify_intent(),
.classify_sentiment(), and .generate_response() interface below.
"""

import os
import re


KNOWN_INTENTS = {
    "flight_booking", "hotel_booking", "car_rental", "policy_query",
    "cancel_flight", "cancel_hotel", "cancel_car", "cancel_generic", "general_query",
}

_INTENT_FOR_DOMAIN = {
    "flight": "flight_booking",
    "hotel": "hotel_booking",
    "car": "car_rental",
}


class BaseLLM:
    def classify_intent(self, text: str) -> str:
        raise NotImplementedError

    def classify_sentiment(self, text: str):
        raise NotImplementedError

    def generate_response(self, prompt: str) -> str:
        raise NotImplementedError


class RuleBasedLLM(BaseLLM):
    """Deterministic offline stand-in (see module docstring)."""

    POLICY_KEYWORDS = ["policy", "refund", "refunds", "baggage", "allowance",
                       "reschedule", "rescheduling", "rule", "rules", "fee", "fees"]
    FLIGHT_KEYWORDS = ["flight", "flights", "fly", "airport", "plane", "boarding"]
    HOTEL_KEYWORDS = ["hotel", "hotels", "room", "rooms", "checkin", "check-in", "stay"]
    CAR_KEYWORDS = ["car", "cars", "rental", "rentals", "vehicle", "pickup", "drive"]

    NEGATIVE_WORDS = ["angry", "furious", "terrible", "worst", "unacceptable",
                       "frustrated", "disappointed", "horrible", "scam", "never"]
    URGENT_WORDS = ["urgent", "immediately", "asap", "now", "emergency"]

    # A request to perform a cancellation, not a mention of the word inside a policy question.
    _IMPERATIVE_CANCEL = re.compile(
        r"(?:^|[.!?]\s*)cancel\b"
        r"|\bplease\s+cancel\b"
        r"|\b(?:want|need|like)\s+to\s+cancel\b"
        r"|\bcancel\s+(?:my|the|this|booking|it)\b",
        re.IGNORECASE,
    )

    def _hits(self, text: str, keywords) -> int:
        return sum(1 for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", text))

    def _domain_from_id(self, text: str):
        if re.search(r"\bHTL-\d{4}\b", text, re.IGNORECASE):
            return "hotel"
        if re.search(r"\bCAR-\d{3}\b", text, re.IGNORECASE):
            return "car"
        if re.search(r"\b[A-Za-z]{3}-[A-Za-z]{3}-\d{2}\b", text):
            return "flight"
        return None

    def _domain_from_keywords(self, text: str):
        scores = {
            "flight": self._hits(text, self.FLIGHT_KEYWORDS),
            "hotel": self._hits(text, self.HOTEL_KEYWORDS),
            "car": self._hits(text, self.CAR_KEYWORDS),
        }
        best_score = max(scores.values())
        if best_score == 0:
            return None
        winners = [domain for domain, score in scores.items() if score == best_score]
        if len(winners) == 1:
            return winners[0]
        id_domain = self._domain_from_id(text)
        if id_domain in winners:
            return id_domain
        return winners[0]

    def classify_intent(self, text: str) -> str:
        text_l = text.lower()
        policy_score = self._hits(text_l, self.POLICY_KEYWORDS)
        domain = self._domain_from_keywords(text_l) or self._domain_from_id(text)
        imperative_cancel = self._IMPERATIVE_CANCEL.search(text_l) is not None

        # Policy questions ("what is the cancellation fee / refund") are not cancel actions,
        # even when they mention a domain or a booking id.
        if policy_score > 0 and not imperative_cancel:
            return "policy_query"
        if imperative_cancel:
            return f"cancel_{domain}" if domain else "cancel_generic"
        if domain:
            return _INTENT_FOR_DOMAIN[domain]
        return "general_query"

    def classify_sentiment(self, text: str):
        text_l = text.lower()
        negative = any(re.search(rf"\b{re.escape(w)}\b", text_l) for w in self.NEGATIVE_WORDS)
        urgent = any(re.search(rf"\b{re.escape(w)}\b", text_l) for w in self.URGENT_WORDS) or "!" in text
        sentiment = "negative" if negative else "neutral"
        urgency = "high" if (negative and urgent) else ("medium" if (negative or urgent) else "low")
        return sentiment, urgency

    def generate_response(self, prompt: str) -> str:
        # Extremely small templated generator: echoes the structured facts
        # it was given rather than "hallucinating" free text. Real system
        # would call an LLM here with the same prompt.
        return prompt


class OpenAILLM(BaseLLM):
    """Production backend - requires OPENAI_API_KEY and langchain-openai."""

    def __init__(self, model: str = "gpt-4o-mini"):
        from langchain_openai import ChatOpenAI
        self.client = ChatOpenAI(model=model, temperature=0)

    def classify_intent(self, text: str) -> str:
        resp = self.client.invoke(
            f"Classify the customer support intent of this message into one of: "
            f"flight_booking, hotel_booking, car_rental, policy_query, "
            f"cancel_flight, cancel_hotel, cancel_car, cancel_generic, general_query. "
            f"Reply with only the label.\n\nMessage: {text}"
        )
        label = re.sub(r"[^a-z_]", "", resp.content.strip().lower())
        return label if label in KNOWN_INTENTS else "general_query"

    def classify_sentiment(self, text: str):
        resp = self.client.invoke(
            f"Classify sentiment (positive/neutral/negative) and urgency "
            f"(low/medium/high) of this message as 'sentiment,urgency'.\n\n{text}"
        )
        parts = resp.content.strip().split(",")
        sentiment = parts[0].strip().lower()
        urgency = parts[1].strip().lower() if len(parts) > 1 else "low"
        if sentiment not in {"positive", "neutral", "negative"}:
            sentiment = "neutral"
        if urgency not in {"low", "medium", "high"}:
            urgency = "low"
        return sentiment, urgency

    def generate_response(self, prompt: str) -> str:
        resp = self.client.invoke(prompt)
        return resp.content.strip()


def get_llm() -> BaseLLM:
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAILLM()
        except Exception:
            pass
    return RuleBasedLLM()
