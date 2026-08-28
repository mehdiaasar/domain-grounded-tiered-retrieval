"""Routers are pure functions of state, so they test without any model."""

from dgtr.nodes import (
    route_after_constraints,
    route_after_constraints_regen,
    route_after_grading,
    route_on_score,
    route_on_score_intrinsic,
)


class TestRouteAfterConstraints:
    def test_clean_claims_go_to_closed_book_scoring(self):
        assert route_after_constraints({"constraint_labels": ["PASS", "PASS"]}) == "score_intrinsic"

    def test_violation_skips_closed_book_scoring(self):
        state = {"constraint_labels": ["PASS", "VIOLATION"]}
        assert route_after_constraints(state) == "detect_domain"

    def test_both_tiers_exhausted_stops(self):
        state = {
            "constraint_labels": ["VIOLATION"],
            "trusted_search_done": True,
            "general_search_done": True,
        }
        assert route_after_constraints(state) == "stop"

    def test_missing_labels_default_to_scoring(self):
        assert route_after_constraints({}) == "score_intrinsic"


class TestIntrinsicScoreRouting:
    def test_confident_answer_bypasses_retrieval(self):
        assert route_on_score_intrinsic({"score": 85.0}, 70.0) == "generate_answer"

    def test_threshold_is_inclusive(self):
        assert route_on_score_intrinsic({"score": 70.0}, 70.0) == "generate_answer"

    def test_low_confidence_escalates(self):
        assert route_on_score_intrinsic({"score": 69.9}, 70.0) == "detect_domain"

    def test_threshold_is_configurable(self):
        assert route_on_score_intrinsic({"score": 50.0}, 40.0) == "generate_answer"
        assert route_on_score_intrinsic({"score": 50.0}, 90.0) == "detect_domain"


class TestGradingRouting:
    def test_relevant_docs_regenerate(self):
        assert route_after_grading({"relevant_docs_found": True}) == "regenerate"

    def test_irrelevant_trusted_docs_fall_back_to_open_web(self):
        state = {"relevant_docs_found": False, "general_search_done": False}
        assert route_after_grading(state) == "general_web_search"

    def test_exhausted_tiers_decline(self):
        state = {"relevant_docs_found": False, "general_search_done": True}
        assert route_after_grading(state) == "generate_apology"


class TestPostRetrievalRouting:
    def test_clean_regenerated_claims_get_scored(self):
        assert route_after_constraints_regen({"constraint_labels": ["PASS"]}) == "score_retrieved"

    def test_violation_falls_back_to_open_web(self):
        state = {"constraint_labels": ["VIOLATION"], "general_search_done": False}
        assert route_after_constraints_regen(state) == "general_web_search"

    def test_violation_after_open_web_stops(self):
        state = {"constraint_labels": ["VIOLATION"], "general_search_done": True}
        assert route_after_constraints_regen(state) == "stop"

    def test_grounded_answer_is_accepted(self):
        assert route_on_score({"score": 100.0}, 70.0) == "generate_answer"

    def test_ungrounded_answer_escalates_then_stops(self):
        pending = {"score": 10.0, "general_search_done": False}
        assert route_on_score(pending, 70.0) == "general_web_search"
        assert route_on_score({"score": 10.0, "general_search_done": True}, 70.0) == "stop"
