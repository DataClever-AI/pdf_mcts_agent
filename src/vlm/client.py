"""
VLM client — single entry point for all Spark VLM calls.

6 permitted tasks:
  query_understanding  — expand/reformulate query before retrieval
  section_ranking      — rate section relevance in parallel batch
  rollout_scoring      — score MCTS leaf nodes during simulation
  answer_generation    — synthesize final answer from evidence
  judge_answer         — grade answer (grounding, completeness, safety, clarity)
  feedback_summarization — summarize human corrections into patterns

Rules:
- Only this module instantiates AsyncOpenAI.
- Unknown tasks raise VLMTaskError immediately (no network call).
- Retry transient failures only (connection errors, HTTP 5xx/429, one retry on a
  timeout) with exponential backoff. Never retry VLMTaskError or a 4xx (bad
  key/request). The tunnel to the VLM can drop for a few seconds while
  cloudflared reconnects to the edge, and a cold start may 502.
- Requests are paced below the proxy's rate limit (nginx: 10 r/s, burst 20);
  bursting past it returns 503 and silently degrades every VLM-backed stage.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time
import weakref
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse


def _load_env() -> None:
    """Load .env from project hierarchy into os.environ (stdlib only)."""
    for parent in Path(__file__).resolve().parents:
        env_file = parent / ".env"
        if env_file.is_file():
            with open(env_file, encoding="utf-8") as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val
            return


_load_env()

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI  # noqa: E402
from pydantic import BaseModel  # noqa: E402

VLMTask = Literal[
    "query_understanding",
    "query_guard",
    "section_ranking",
    "rollout_scoring",
    "answer_generation",
    "judge_answer",
    "feedback_summarization",
]

PERMITTED_TASKS: frozenset[str] = frozenset({
    "query_understanding",
    "query_guard",
    "section_ranking",
    "rollout_scoring",
    "answer_generation",
    "judge_answer",
    "feedback_summarization",
})


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class VLMError(Exception):
    """Base for all VLM errors."""


class VLMTaskError(VLMError):
    """Unknown task — never retried."""


class VLMClientError(VLMError):
    """Server unreachable or HTTP error."""


class VLMTimeoutError(VLMError):
    """Request exceeded timeout."""


class VLMParseError(VLMError):
    """Response cannot be parsed into expected format."""


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------

class VLMResponse(BaseModel):
    """Output of a single call_vlm() invocation."""

    content: str
    task: VLMTask
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


# ---------------------------------------------------------------------------
# Runtime config
# ---------------------------------------------------------------------------

_VLM_API_KEY: str = os.getenv("LLM_API_KEY") or os.getenv("VLM_API_KEY") or ""
_VLM_BASE_URL: str = (
    os.getenv("LLM_BASE_URL")
    or os.getenv("VLM_BASE_URL")
    or "https://<VLM_HOST>/v1"
)
_VLM_MODEL: str = os.getenv("LLM_MODEL") or os.getenv("VLM_MODEL") or "Qwen2.5-VL-7B"
_VLM_TIMEOUT: float = float(os.getenv("VLM_TIMEOUT_SECONDS", "120.0"))
_VLM_MAX_RETRIES: int = int(os.getenv("VLM_MAX_RETRIES", "3"))
_VLM_RETRY_DELAY: float = float(os.getenv("VLM_RETRY_DELAY_SECONDS", "1.0"))  # doubles each retry
# Stay under the proxy's limit (nginx limit_req: rate=10r/s burst=20).
_VLM_MAX_RPS: float = float(os.getenv("VLM_MAX_REQUESTS_PER_SECOND", "8"))
_VLM_BURST: int = int(os.getenv("VLM_BURST_REQUESTS", "15"))

logger = logging.getLogger(__name__)


class _RateLimiter:
    """
    Thread- and event-loop-independent pacing (GCRA / token bucket).

    reserve() books the next request slot and returns how long the caller must
    wait before sending — up to `burst` requests go out immediately, after that
    one every 1/rate seconds. Streamlit runs a fresh event loop per
    interaction, so the state is a plain monotonic-clock value behind a lock.
    """

    def __init__(self, rate: float, burst: int) -> None:
        self._interval = 1.0 / max(rate, 0.001)
        self._burst = max(burst, 1)
        self._tat = 0.0  # theoretical arrival time of the next request
        self._lock = threading.Lock()

    def reserve(self) -> float:
        with self._lock:
            now = time.monotonic()
            tat = max(self._tat, now)
            delay = max(0.0, tat - now - (self._burst - 1) * self._interval)
            self._tat = tat + self._interval
            return delay


_limiter = _RateLimiter(_VLM_MAX_RPS, _VLM_BURST)

_parsed_url = urlparse(_VLM_BASE_URL)
if _parsed_url.password:
    raise RuntimeError("VLM base URL must not contain embedded credentials.")

# One client per event loop: an AsyncOpenAI's connection pool is bound to the loop
# it first ran on, so a shared global fails with APIConnectionError once that loop
# closes (Streamlit runs a fresh asyncio loop per interaction).
_clients: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, AsyncOpenAI]" = (
    weakref.WeakKeyDictionary()
)


def _get_client() -> AsyncOpenAI:
    loop = asyncio.get_running_loop()
    client = _clients.get(loop)
    if client is None:
        client = AsyncOpenAI(
            api_key=_VLM_API_KEY or "UNSET",
            base_url=_VLM_BASE_URL,
            timeout=_VLM_TIMEOUT,
            max_retries=0,
        )
        _clients[loop] = client
    return client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_messages(
    prompt: str,
    context: str | None,
    task: VLMTask,
) -> list[dict[str, str]]:
    """Build chat messages from prompt, optional context, and task."""
    system_content = f"You are a technical document reasoning assistant. Task: {task}."
    messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
    if context:
        messages.append(
            {"role": "user", "content": f"Context:\n{context}\n\nQuery:\n{prompt}"}
        )
    else:
        messages.append({"role": "user", "content": prompt})
    return messages


def strip_json_fences(content: str) -> str:
    """Strip ```json ... ``` or ``` ... ``` fences from VLM output."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    return match.group(1) if match else content.strip()


def parse_float_response(content: str, default: float = 0.5) -> float:
    """Parse a VLM response as a float score clamped to [0.0, 1.0]."""
    try:
        return max(0.0, min(1.0, float(content.strip())))
    except ValueError:
        # Try extracting first float from text
        match = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\b", content)
        if match:
            try:
                return max(0.0, min(1.0, float(match.group(1))))
            except ValueError:
                pass
    return default


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

async def call_vlm(
    prompt: str,
    context: str | None = None,
    task: VLMTask = "answer_generation",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    frequency_penalty: float = 0.0,
    conversation_id: str | None = None,
) -> VLMResponse:
    """
    Single entry point for all Spark VLM calls.

    Args:
        prompt:            Instruction or question for the VLM.
        context:           Optional supporting text (evidence, history).
        task:              One of the 6 permitted task literals.
        max_tokens:        Maximum tokens in the completion.
        temperature:       Sampling temperature. 0.0 = deterministic.
        frequency_penalty: OpenAI-compatible penalty (vLLM supports it) on
                           token repetition, 0.0-2.0. Only needed for long
                           completions (answer_generation) — small 7B models
                           can fall into a greedy-decoding loop and repeat one
                           line until max_tokens even at temperature=0.0.
        conversation_id:   Optional ID for tracing.

    Returns:
        VLMResponse with content, token counts, and latency.

    Raises:
        VLMTaskError:    Unknown task (never retried).
        VLMClientError:  Server unreachable or HTTP error.
        VLMTimeoutError: Request timed out.
    """
    if not _VLM_API_KEY:
        raise RuntimeError(
            "No VLM API key. Set LLM_API_KEY or VLM_API_KEY environment variable."
        )
    if task not in PERMITTED_TASKS:
        raise VLMTaskError(f"Task '{task}' not permitted. Allowed: {sorted(PERMITTED_TASKS)}")

    messages = _build_messages(prompt, context, task)
    last_exc: VLMClientError | VLMTimeoutError | None = None
    host = _parsed_url.netloc.split("@")[-1]

    for attempt in range(_VLM_MAX_RETRIES + 1):
        if attempt > 0:
            await asyncio.sleep(_VLM_RETRY_DELAY * 2 ** (attempt - 1))
        wait = _limiter.reserve()
        if wait > 0:
            await asyncio.sleep(wait)
        try:
            start = time.monotonic()
            response = await _get_client().chat.completions.create(
                model=_VLM_MODEL,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
                frequency_penalty=frequency_penalty,
            )
            elapsed_ms = (time.monotonic() - start) * 1000
            content = response.choices[0].message.content or ""
            usage = response.usage
            return VLMResponse(
                content=content,
                task=task,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                latency_ms=elapsed_ms,
            )
        except APITimeoutError:
            last_exc = VLMTimeoutError(f"VLM timed out after {_VLM_TIMEOUT}s")
            if attempt >= 1:
                break  # a request that already timed out twice won't succeed on a third try
        except APIConnectionError as exc:
            cause = type(exc.__cause__).__name__ if exc.__cause__ else type(exc).__name__
            last_exc = VLMClientError(f"VLM unreachable at {host} ({cause})")
        except APIStatusError as exc:
            last_exc = VLMClientError(f"VLM returned HTTP {exc.status_code}")
            if 400 <= exc.status_code < 500 and exc.status_code != 429:
                break  # bad key / bad request: retrying cannot help
        logger.info("VLM %s attempt %d/%d failed: %s", task, attempt + 1,
                    _VLM_MAX_RETRIES + 1, last_exc)

    assert last_exc is not None
    raise last_exc
