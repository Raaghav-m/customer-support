"""End-to-end checks for routing, confirm-before-cancel, and isolated bookings."""

import copy
import unittest

from graph_baseline import BASELINE_GRAPH
from graph_proposed import PROPOSED_GRAPH
from tests.test_conversations import TEST_CASES
from tools import backend


# intent, resolved, human_in_loop, escalated
EXPECTED = {
    1: ("flight_booking", True, False, False),
    2: ("hotel_booking", True, False, False),
    3: ("car_rental", True, False, False),
    4: ("cancel_flight", True, True, False),
    5: ("cancel_hotel", True, True, False),
    6: ("cancel_car", True, True, False),
    7: ("policy_query", True, False, False),
    8: ("policy_query", True, False, False),
    9: ("policy_query", True, False, False),
    10: ("policy_query", True, False, False),
    11: ("flight_booking", True, False, True),
    12: ("cancel_hotel", True, True, True),
    13: ("flight_booking", True, False, False),
    14: ("hotel_booking", True, False, False),
    15: ("car_rental", True, False, False),
    16: ("cancel_flight", True, True, False),
    17: ("policy_query", True, False, False),
    18: ("car_rental", True, False, False),
    19: ("cancel_hotel", True, True, False),
    20: ("general_query", False, False, True),
    21: ("general_query", False, False, True),
    22: ("cancel_car", True, True, False),
    23: ("flight_booking", True, False, False),
    24: ("policy_query", True, False, False),
    25: ("policy_query", True, False, False),
}

POLICY_SNIPPETS = {
    7: "20%",
    8: "5-7 business days",
    9: "15kg",
    10: "rescheduled once",
    17: "20%",
    24: "5-7 business days",
    25: "5-7 business days",
}


def _run_case(case):
    backend.reset()
    result = PROPOSED_GRAPH.invoke({"user_message": case["text"]})
    human_in_loop = bool(result.get("needs_approval"))
    if human_in_loop:
        followup = copy.deepcopy(result)
        followup["user_approved"] = True
        result = PROPOSED_GRAPH.invoke(followup)
    return result, human_in_loop


class EndToEndTests(unittest.TestCase):
    def test_proposed_conversations(self):
        for case in TEST_CASES:
            with self.subTest(case=case["id"], text=case["text"]):
                result, human_in_loop = _run_case(case)
                intent, resolved, hitl, escalated = EXPECTED[case["id"]]
                self.assertEqual(result.get("intent"), intent)
                self.assertEqual(bool(result.get("resolved")), resolved)
                self.assertEqual(human_in_loop, hitl)
                self.assertEqual(bool(result.get("escalated")), escalated)
                snippet = POLICY_SNIPPETS.get(case["id"])
                if snippet:
                    self.assertIn(snippet, result.get("response", ""))

    def test_new_bookings_are_confirmed(self):
        hotel, _ = _run_case(TEST_CASES[13])
        car, _ = _run_case(TEST_CASES[14])
        self.assertIn("HTL-1003", hotel["response"])
        self.assertIn("DEL", hotel["response"])
        self.assertIn("CAR-502", car["response"])

    def test_lookup_is_not_poisoned_by_an_earlier_cancel(self):
        _run_case(TEST_CASES[5])  # cancels CAR-501
        result, human_in_loop = _run_case(TEST_CASES[17])  # status of CAR-501
        self.assertFalse(human_in_loop)
        self.assertIn("'status': 'confirmed'", result["response"])

    def test_cancel_waits_for_approval(self):
        backend.reset()
        result = PROPOSED_GRAPH.invoke({"user_message": "Please cancel my flight BLR-DEL-01"})
        self.assertTrue(result["needs_approval"])
        self.assertFalse(result["resolved"])
        self.assertEqual(backend.FLIGHTS_DB["BLR-DEL-01"]["status"], "confirmed")
        self.assertIn("confirm", result["response"])

        result["user_approved"] = True
        result = PROPOSED_GRAPH.invoke(result)
        self.assertTrue(result["resolved"])
        self.assertIn("cancelled", result["response"])
        self.assertEqual(backend.FLIGHTS_DB["BLR-DEL-01"]["status"], "cancelled")

    def test_baseline_cancels_without_confirmation(self):
        backend.reset()
        result = BASELINE_GRAPH.invoke({"user_message": "Please cancel my flight BLR-DEL-01"})
        self.assertTrue(result["resolved"])
        self.assertFalse(result["needs_approval"])
        self.assertEqual(backend.FLIGHTS_DB["BLR-DEL-01"]["status"], "cancelled")

    def test_reset_restores_bookings(self):
        backend.cancel_flight("BLR-DEL-01")
        backend.reset()
        self.assertEqual(backend.FLIGHTS_DB["BLR-DEL-01"]["status"], "confirmed")
        self.assertNotIn("HTL-1003", backend.HOTELS_DB)


if __name__ == "__main__":
    unittest.main()
