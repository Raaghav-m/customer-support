"""
25 synthetic support queries with ground-truth intent labels, used to
evaluate both agents. Domains mirror MultiWOZ-style task-oriented dialogue
(flight/hotel/car booking + policy) plus negative-sentiment and ambiguous
cases to test triage and escalation behaviour.
"""

TEST_CASES = [
    {"id": 1, "text": "Can you look up my flight BLR-DEL-01?", "gold_intent": "flight_booking"},
    {"id": 2, "text": "I need details on my hotel booking HTL-1001 please.", "gold_intent": "hotel_booking"},
    {"id": 3, "text": "What's the status of my car rental CAR-501?", "gold_intent": "car_rental"},
    {"id": 4, "text": "Please cancel my flight BLR-DEL-01, I don't need it anymore.", "gold_intent": "cancel_flight"},
    {"id": 5, "text": "Cancel my hotel booking HTL-1001 immediately.", "gold_intent": "cancel_hotel"},
    {"id": 6, "text": "I want to cancel the car rental CAR-501.", "gold_intent": "cancel_car"},
    {"id": 7, "text": "What is your cancellation policy?", "gold_intent": "policy_query"},
    {"id": 8, "text": "How long does a refund take to process?", "gold_intent": "policy_query"},
    {"id": 9, "text": "What's the baggage allowance on economy flights?", "gold_intent": "policy_query"},
    {"id": 10, "text": "Can I reschedule my booking for free?", "gold_intent": "policy_query"},
    {"id": 11, "text": "This is the WORST service ever, my flight BLR-BOM-02 got messed up, fix it NOW!", "gold_intent": "flight_booking"},
    {"id": 12, "text": "I am furious, cancel my hotel HTL-1002 right now, this is unacceptable!", "gold_intent": "cancel_hotel"},
    {"id": 13, "text": "hey can u check flight DEL-MAA-03", "gold_intent": "flight_booking"},
    {"id": 14, "text": "Looking to book a new hotel room in Delhi next week", "gold_intent": "hotel_booking"},
    {"id": 15, "text": "Need a rental car pickup at the airport tomorrow", "gold_intent": "car_rental"},
    {"id": 16, "text": "Cancel booking BLR-DEL-01", "gold_intent": "cancel_flight"},
    {"id": 17, "text": "Is there a fee for cancelling within 24 hours?", "gold_intent": "policy_query"},
    {"id": 18, "text": "Please check my car CAR-501 pickup date", "gold_intent": "car_rental"},
    {"id": 19, "text": "I'm disappointed, my hotel HTL-1001 reservation was wrong, please cancel it", "gold_intent": "cancel_hotel"},
    {"id": 20, "text": "Random question about the weather today", "gold_intent": "general_query"},
    {"id": 21, "text": "Can you help me understand something unrelated to bookings?", "gold_intent": "general_query"},
    {"id": 22, "text": "cancel car CAR-501 asap this is an emergency", "gold_intent": "cancel_car"},
    {"id": 23, "text": "Check status of flight BLR-BOM-02", "gold_intent": "flight_booking"},
    {"id": 24, "text": "What happens to my refund after I cancel a flight?", "gold_intent": "policy_query"},
    {"id": 25, "text": "terrible experience, want a refund for my stay HTL-1002", "gold_intent": "policy_query"},
]
