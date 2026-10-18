"""
Baseline agent - mirrors the reference LangGraph Customer Support tutorial's
single-graph pattern: one generic assistant node with a shared tool node and
no specialist routing, no sentiment/urgency triage, and no policy grounding.
Used as the comparison point for the Result Analysis / comparative evaluation.
"""

import re
from langgraph.graph import StateGraph, END

from state import SupportState
from tools import backend

BOOKING_ID_RE = re.compile(
    r"\b([A-Z]{3}-[A-Z]{3}-\d{2}|HTL-\d{4}|CAR-\d{3})\b",
    re.IGNORECASE,
)


def _guess_tool(booking_id: str):
    """The single generic assistant has one shared tool node and must guess
    which backend a booking id belongs to purely from its prefix - it has no
    domain-specific specialist and no intent classification."""
    if booking_id.startswith("HTL"):
        return backend.search_hotel, backend.cancel_hotel
    if booking_id.startswith("CAR"):
        return backend.search_car, backend.cancel_car
    return backend.search_flight, backend.cancel_flight


def assistant(state: SupportState) -> SupportState:
    text = state["user_message"]
    match = BOOKING_ID_RE.search(text)
    booking_id = match.group(0).upper() if match else None
    is_cancel = "cancel" in text.lower()
    return {"booking_id": booking_id, "intent": "cancel_generic" if is_cancel else "lookup_generic"}


def shared_tool_node(state: SupportState) -> SupportState:
    booking_id = state.get("booking_id")
    if not booking_id:
        return {"tool_result": {"ok": False, "message": "No booking id found in message."},
                "needs_approval": False}
    search_fn, cancel_fn = _guess_tool(booking_id)
    if state["intent"] == "cancel_generic":
        # NOTE: the baseline has no human-in-the-loop interrupt on sensitive
        # actions - it cancels immediately, unlike the proposed agent.
        result = cancel_fn(booking_id)
    else:
        record = search_fn(booking_id)
        result = {"ok": True, "record": record} if record else \
            {"ok": False, "message": f"No booking found with id {booking_id}."}
    return {"tool_result": result, "needs_approval": False}


def respond(state: SupportState) -> SupportState:
    result = state.get("tool_result") or {}
    if result.get("ok") is True and "record" in result:
        response = f"Found booking: {result['record']}."
        return {"response": response, "resolved": True, "escalated": False}
    if result.get("ok") is True:
        return {"response": result.get("message", "Done."), "resolved": True, "escalated": False}
    return {"response": result.get("message", "Could not complete that request."),
            "resolved": False, "escalated": False}


def build_baseline_graph():
    g = StateGraph(SupportState)
    g.add_node("assistant", assistant)
    g.add_node("tools", shared_tool_node)
    g.add_node("respond", respond)
    g.set_entry_point("assistant")
    g.add_edge("assistant", "tools")
    g.add_edge("tools", "respond")
    g.add_edge("respond", END)
    return g.compile()


BASELINE_GRAPH = build_baseline_graph()
