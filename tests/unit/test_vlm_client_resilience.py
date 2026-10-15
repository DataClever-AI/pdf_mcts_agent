"""call_vlm: retry transient failures, never retry a 4xx, and pace requests."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from src.vlm import client as c

_REQ = httpx.Request("POST", "https://vlm.example/v1/chat/completions")


def _status_error(cls: type[openai.APIStatusError], code: int) -> openai.APIStatusError:
    return cls("boom", response=httpx.Response(code, request=_REQ), body=None)


def _ok() -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )


class _FakeClient:
    """Pops one outcome per call: an exception to raise, or a response to return."""

    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **_: Any) -> Any:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def vlm(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Install a fake client; no real sleeping and no pacing delays."""
    async def _no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(c, "_VLM_API_KEY", "test-key")
    monkeypatch.setattr(c.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(c._limiter, "reserve", lambda: 0.0)

    def install(outcomes: list[Any]) -> _FakeClient:
        fake = _FakeClient(outcomes)
        monkeypatch.setattr(c, "_get_client", lambda: fake)
        return fake

    return install


def _call() -> Any:
    return asyncio.run(c.call_vlm("hi", task="query_guard", max_tokens=5))


def test_recovers_from_a_short_connection_outage(vlm) -> None:  # type: ignore[no-untyped-def]
    fake = vlm([openai.APIConnectionError(request=_REQ),
                openai.APIConnectionError(request=_REQ), _ok()])

    response = _call()

    assert response.content == "hello"
    assert fake.calls == 3


def test_retries_http_503_from_the_proxy_rate_limit(vlm) -> None:  # type: ignore[no-untyped-def]
    fake = vlm([_status_error(openai.InternalServerError, 503), _ok()])

    assert _call().content == "hello"
    assert fake.calls == 2


def test_does_not_retry_a_4xx(vlm) -> None:  # type: ignore[no-untyped-def]
    fake = vlm([_status_error(openai.AuthenticationError, 401), _ok()])

    with pytest.raises(c.VLMClientError, match="HTTP 401"):
        _call()
    assert fake.calls == 1


def test_gives_up_after_the_retry_budget_and_names_the_cause(vlm) -> None:  # type: ignore[no-untyped-def]
    err = openai.APIConnectionError(request=_REQ)
    err.__cause__ = httpx.ConnectError("reset")
    fake = vlm([err] * (c._VLM_MAX_RETRIES + 1))

    with pytest.raises(c.VLMClientError, match=r"unreachable.*\(ConnectError\)"):
        _call()
    assert fake.calls == c._VLM_MAX_RETRIES + 1


def test_a_timeout_is_retried_only_once(vlm) -> None:  # type: ignore[no-untyped-def]
    fake = vlm([openai.APITimeoutError(request=_REQ)] * 4)

    with pytest.raises(c.VLMTimeoutError):
        _call()
    assert fake.calls == 2


class TestRateLimiter:
    def test_allows_a_burst_then_spaces_requests(self) -> None:
        limiter = c._RateLimiter(rate=10, burst=3)

        delays = [limiter.reserve() for _ in range(6)]

        assert delays[:3] == [0.0, 0.0, 0.0]          # the burst goes out at once
        assert 0.0 < delays[3] < delays[4] < delays[5]  # then one slot every 0.1s
        assert delays[5] - delays[4] == pytest.approx(0.1, abs=0.02)

    def test_recovers_after_idle_time(self, monkeypatch: pytest.MonkeyPatch) -> None:
        limiter = c._RateLimiter(rate=10, burst=2)
        clock = [100.0]
        monkeypatch.setattr(c.time, "monotonic", lambda: clock[0])
        for _ in range(5):
            limiter.reserve()
        clock[0] += 10.0  # long pause refills the bucket

        assert limiter.reserve() == 0.0
