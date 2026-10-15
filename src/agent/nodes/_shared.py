"""Small helpers shared across node modules — logging, error passthrough."""
from __future__ import annotations

import time

from src.agent.state import AgentState


def _log(node: str, detail: str, t0: float | None = None) -> dict[str, list[str]]:
    """
    Build a {"logs": [...]} partial update for one audit-trail line.

    `logs` is Annotated[list[str], add] in AgentState — LangGraph concatenates
    each node's returned delta into the running list automatically. This
    replaces a prior node_trace/step_metrics pair that mutated the `state`
    argument in place without including the key in the node's return dict —
    confirmed via direct repro that LangGraph discards such mutations (a node
    that does `state[k]=...; return {}` leaves the final state's `k`
    unchanged), so those two fields never actually accumulated in practice.
    """
    ms = f" ({(time.monotonic() - t0) * 1000:.0f}ms)" if t0 is not None else ""
    return {"logs": [f"{node}: {detail}{ms}"]}


def _is_error(state: AgentState) -> bool:
    return bool(state.get("error"))


def _binary_reply_is(content: str, positive: str) -> bool:
    """Parse a one-word binary VLM reply (e.g. GROUNDED/YES), tolerant of stray text."""
    return content.strip().upper().startswith(positive)


__all__: list[str] = []  # internal helpers — not part of the package's public API
