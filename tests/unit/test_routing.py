"""Tests for graph routing predicates — pure functions, no VLM needed."""
from __future__ import annotations

from src.agent.nodes.routing import route_after_guard, route_after_judge, route_on_error


class TestRouteAfterGuard:
    def test_blocked_routes_to_blocked(self) -> None:
        assert route_after_guard({"blocked": True}) == "blocked"

    def test_not_blocked_routes_to_ok(self) -> None:
        assert route_after_guard({"blocked": False}) == "ok"

    def test_missing_key_defaults_to_ok(self) -> None:
        assert route_after_guard({}) == "ok"


class TestRouteOnError:
    def test_error_present_routes_to_error(self) -> None:
        assert route_on_error({"error": "retrieve_sections: boom"}) == "error"

    def test_no_error_routes_to_ok(self) -> None:
        assert route_on_error({"error": None}) == "ok"

    def test_missing_key_defaults_to_ok(self) -> None:
        assert route_on_error({}) == "ok"


class TestRouteAfterJudge:
    def test_safety_block_never_refines_even_on_fail(self) -> None:
        state = {
            "answer_blocked": True,
            "judge_result": {"verdict": "FAIL"},
            "judge_attempts": 0,
        }
        assert route_after_judge(state) == "done"

    def test_fail_under_attempt_limit_refines(self) -> None:
        state = {"judge_result": {"verdict": "FAIL"}, "judge_attempts": 1}
        assert route_after_judge(state) == "refine"

    def test_fail_at_attempt_limit_does_not_refine_again(self) -> None:
        state = {"judge_result": {"verdict": "FAIL"}, "judge_attempts": 2}
        assert route_after_judge(state) == "done"

    def test_pass_routes_to_done(self) -> None:
        state = {"judge_result": {"verdict": "PASS"}, "judge_attempts": 1}
        assert route_after_judge(state) == "done"

    def test_warn_routes_to_done_not_refine(self) -> None:
        state = {"judge_result": {"verdict": "WARN"}, "judge_attempts": 1}
        assert route_after_judge(state) == "done"
