"""SSE Event Bus — real-time progress push for Offline Admin UI."""

from __future__ import annotations
import asyncio
import json
from datetime import datetime, UTC
from typing import Any

_subscribers: list[asyncio.Queue] = []


def subscribe() -> asyncio.Queue:
    """Register a new subscriber queue (one per SSE connection)."""
    q: asyncio.Queue = asyncio.Queue()
    _subscribers.append(q)
    return q


def unsubscribe(q: asyncio.Queue) -> None:
    """Remove a subscriber queue."""
    try:
        _subscribers.remove(q)
    except ValueError:
        pass


async def emit(event: dict[str, Any]) -> None:
    """Emit an event to all connected subscribers."""
    event["_ts"] = datetime.now(UTC).isoformat()
    dead: list[asyncio.Queue] = []
    for q in _subscribers:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        unsubscribe(q)


def sse_format(event: dict[str, Any]) -> str:
    """Format an event as SSE text."""
    payload = json.dumps(event, ensure_ascii=False)
    return f"data: {payload}\n\n"
