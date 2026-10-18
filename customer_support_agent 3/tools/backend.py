"""
Mock backend for the Customer Support Agent.

In production this module would call real airline/hotel/car-rental APIs and a
policy-document vector store. For assignment purposes it is implemented as an
in-memory database so the whole agent graph can be run, tested, and evaluated
without external network access or paid API calls.
"""

import copy
import random
import re
import time

FLIGHTS_DB = {
    "BLR-DEL-01": {"id": "BLR-DEL-01", "from": "BLR", "to": "DEL", "date": "2026-09-10", "status": "confirmed"},
    "BLR-BOM-02": {"id": "BLR-BOM-02", "from": "BLR", "to": "BOM", "date": "2026-09-15", "status": "confirmed"},
    "DEL-MAA-03": {"id": "DEL-MAA-03", "from": "DEL", "to": "MAA", "date": "2026-09-20", "status": "confirmed"},
}

HOTELS_DB = {
    "HTL-1001": {"id": "HTL-1001", "city": "DEL", "checkin": "2026-09-10", "nights": 3, "status": "confirmed"},
    "HTL-1002": {"id": "HTL-1002", "city": "BOM", "checkin": "2026-09-15", "nights": 2, "status": "confirmed"},
}

CARS_DB = {
    "CAR-501": {"id": "CAR-501", "city": "DEL", "pickup_date": "2026-09-10", "status": "confirmed"},
}

POLICY_DB = {
    "cancellation": (
        "Bookings can be cancelled free of charge up to 24 hours before departure/check-in. "
        "Cancellations within 24 hours incur a 20% fee."
    ),
    "refund": (
        "Approved refunds are credited to the original payment method within 5-7 business days."
    ),
    "baggage": (
        "Economy fares include one 15kg checked bag. Excess baggage is charged at INR 500/kg."
    ),
    "reschedule": (
        "Bookings may be rescheduled once, free of charge, up to 48 hours before departure."
    ),
}


_INITIAL_FLIGHTS = copy.deepcopy(FLIGHTS_DB)
_INITIAL_HOTELS = copy.deepcopy(HOTELS_DB)
_INITIAL_CARS = copy.deepcopy(CARS_DB)

_CITIES = (
    ("new delhi", "DEL"),
    ("delhi", "DEL"),
    ("mumbai", "BOM"),
    ("bombay", "BOM"),
    ("bengaluru", "BLR"),
    ("bangalore", "BLR"),
    ("chennai", "MAA"),
)

POLICY_ALIASES = {
    "cancellation": ["cancellation", "cancellations", "cancel", "cancelling", "fee", "fees"],
    "refund": ["refund", "refunds", "refunded"],
    "baggage": ["baggage", "bag", "bags", "allowance"],
    "reschedule": ["reschedule", "rescheduling", "rescheduled"],
}


def _simulated_latency():
    """Simulate a small, realistic network/API round-trip delay."""
    time.sleep(random.uniform(0.01, 0.04))


def reset() -> None:
    """Restore the mock databases so independent conversations do not leak into each other."""
    for live, initial in (
        (FLIGHTS_DB, _INITIAL_FLIGHTS),
        (HOTELS_DB, _INITIAL_HOTELS),
        (CARS_DB, _INITIAL_CARS),
    ):
        live.clear()
        live.update(copy.deepcopy(initial))


def _city_from_text(text: str, default: str = "DEL") -> str:
    lowered = text.lower()
    for name, code in _CITIES:
        if name in lowered:
            return code
    return default


def _next_id(prefix: str, start: int, db: dict) -> str:
    seq = start
    while f"{prefix}{seq}" in db:
        seq += 1
    return f"{prefix}{seq}"


def search_flight(booking_id: str):
    _simulated_latency()
    record = FLIGHTS_DB.get(booking_id)
    return copy.deepcopy(record) if record else None


def cancel_flight(booking_id: str):
    _simulated_latency()
    record = FLIGHTS_DB.get(booking_id)
    if not record:
        return {"ok": False, "message": f"No flight booking found with id {booking_id}."}
    record["status"] = "cancelled"
    return {"ok": True, "message": f"Flight {booking_id} cancelled."}


def book_flight(text: str = "") -> dict:
    _simulated_latency()
    booking_id = _next_id("FLT-", 9001, FLIGHTS_DB)
    record = {
        "id": booking_id,
        "from": "BLR",
        "to": "DEL",
        "date": "2026-10-01",
        "status": "confirmed",
    }
    FLIGHTS_DB[booking_id] = record
    return {
        "ok": True,
        "message": f"Flight {booking_id} booked ({record['from']} to {record['to']} on {record['date']}).",
    }


def search_hotel(booking_id: str):
    _simulated_latency()
    record = HOTELS_DB.get(booking_id)
    return copy.deepcopy(record) if record else None


def cancel_hotel(booking_id: str):
    _simulated_latency()
    record = HOTELS_DB.get(booking_id)
    if not record:
        return {"ok": False, "message": f"No hotel booking found with id {booking_id}."}
    record["status"] = "cancelled"
    return {"ok": True, "message": f"Hotel booking {booking_id} cancelled."}


def book_hotel(text: str = "") -> dict:
    _simulated_latency()
    city = _city_from_text(text)
    booking_id = _next_id("HTL-", 1001, HOTELS_DB)
    record = {
        "id": booking_id,
        "city": city,
        "checkin": "2026-10-01",
        "nights": 1,
        "status": "confirmed",
    }
    HOTELS_DB[booking_id] = record
    return {
        "ok": True,
        "message": f"Hotel booking {booking_id} confirmed in {city} for {record['nights']} night(s).",
    }


def search_car(booking_id: str):
    _simulated_latency()
    record = CARS_DB.get(booking_id)
    return copy.deepcopy(record) if record else None


def cancel_car(booking_id: str):
    _simulated_latency()
    record = CARS_DB.get(booking_id)
    if not record:
        return {"ok": False, "message": f"No car rental found with id {booking_id}."}
    record["status"] = "cancelled"
    return {"ok": True, "message": f"Car rental {booking_id} cancelled."}


def book_car(text: str = "") -> dict:
    _simulated_latency()
    city = _city_from_text(text)
    booking_id = _next_id("CAR-", 501, CARS_DB)
    record = {
        "id": booking_id,
        "city": city,
        "pickup_date": "2026-10-01",
        "status": "confirmed",
    }
    CARS_DB[booking_id] = record
    return {
        "ok": True,
        "message": f"Car rental {booking_id} confirmed in {city}, pickup {record['pickup_date']}.",
    }


def search_policy(query: str):
    """Keyword retriever standing in for a Sentence-BERT + FAISS/Chroma pipeline."""
    _simulated_latency()
    query_lower = query.lower()
    best_key, best_score = None, 0
    for key, aliases in POLICY_ALIASES.items():
        score = 0
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", query_lower):
                score += 3 if alias == key or alias.startswith(key) else 1
        if score > best_score:
            best_key, best_score = key, score
    if best_key is None:
        return (
            "No matching policy passage found. "
            "I can help with cancellation, refunds, baggage, and rescheduling."
        )
    return POLICY_DB[best_key]
