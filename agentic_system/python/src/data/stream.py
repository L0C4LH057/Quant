"""
Real-time market data streaming via WebSocket.

Provides a ``MarketStream`` class that maintains a persistent
WebSocket connection to the MT5 bridge (or any compatible WS endpoint)
and dispatches incoming ticks to registered callbacks.

GAP-04 fix: Previously no real-time data path existed — agents
relied on periodic REST polling.

Usage::

    stream = MarketStream("ws://192.168.1.100:8888/ws/quotes")
    stream.subscribe("EURUSD", on_tick)
    await stream.start()
    # ... later
    await stream.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Type alias for tick callbacks (async)
TickCallback = Callable[["Tick"], Coroutine[Any, Any, None]]


@dataclass(frozen=True)
class Tick:
    """Single price tick from the market data stream."""

    symbol: str
    bid: float
    ask: float
    spread: float
    volume: int
    timestamp: datetime

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tick":
        return cls(
            symbol=data["symbol"],
            bid=float(data["bid"]),
            ask=float(data["ask"]),
            spread=float(data.get("spread", 0.0)),
            volume=int(data.get("volume", 0)),
            timestamp=datetime.fromisoformat(data["time"])
            if isinstance(data.get("time"), str)
            else datetime.utcnow(),
        )


class MarketStream:
    """
    Persistent WebSocket market data stream.

    Parameters
    ----------
    ws_url : str
        WebSocket endpoint (e.g. ``ws://host:port/ws/quotes``).
    api_key : str, optional
        Sent as ``X-API-Key`` header on the WS handshake.
    reconnect_delay : float
        Seconds between reconnection attempts (default 5).
    max_reconnects : int
        Maximum consecutive reconnect attempts before giving up (-1 = infinite).
    """

    def __init__(
        self,
        ws_url: str,
        api_key: Optional[str] = None,
        reconnect_delay: float = 5.0,
        max_reconnects: int = -1,
    ) -> None:
        self._ws_url = ws_url
        self._api_key = api_key
        self._reconnect_delay = reconnect_delay
        self._max_reconnects = max_reconnects

        self._subscribers: Dict[str, List[TickCallback]] = {}
        self._symbols: Set[str] = set()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def subscribe(self, symbol: str, callback: TickCallback) -> None:
        """Register *callback* for ticks on *symbol*."""
        symbol = symbol.upper()
        self._subscribers.setdefault(symbol, []).append(callback)
        self._symbols.add(symbol)

    def unsubscribe(self, symbol: str, callback: Optional[TickCallback] = None) -> None:
        """Remove *callback* (or all callbacks) for *symbol*."""
        symbol = symbol.upper()
        if callback is None:
            self._subscribers.pop(symbol, None)
            self._symbols.discard(symbol)
        else:
            cbs = self._subscribers.get(symbol, [])
            self._subscribers[symbol] = [cb for cb in cbs if cb is not callback]
            if not self._subscribers[symbol]:
                self._symbols.discard(symbol)

    async def start(self) -> None:
        """Begin streaming in the background."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_forever())
        logger.info("MarketStream started — symbols: %s", self._symbols)

    async def stop(self) -> None:
        """Gracefully shut down the stream."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("MarketStream stopped.")

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    async def _run_forever(self) -> None:
        """Reconnection loop."""
        try:
            import websockets  # type: ignore[import-untyped]
        except ImportError:
            logger.error(
                "websockets package not installed — "
                "run `pip install websockets` to enable MarketStream."
            )
            self._running = False
            return

        consecutive_failures = 0
        while self._running:
            try:
                extra_headers = {}
                if self._api_key:
                    extra_headers["X-API-Key"] = self._api_key

                async with websockets.connect(
                    self._ws_url,
                    additional_headers=extra_headers,
                    ping_interval=20,
                    ping_timeout=10,
                ) as ws:
                    logger.info("WebSocket connected to %s", self._ws_url)
                    consecutive_failures = 0

                    # Send subscription message
                    await ws.send(
                        json.dumps(
                            {"action": "subscribe", "symbols": sorted(self._symbols)}
                        )
                    )

                    async for raw in ws:
                        if not self._running:
                            break
                        await self._dispatch(raw)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                consecutive_failures += 1
                logger.warning(
                    "WebSocket error (%d): %s — reconnecting in %.0fs",
                    consecutive_failures,
                    exc,
                    self._reconnect_delay,
                )
                if 0 < self._max_reconnects <= consecutive_failures:
                    logger.error(
                        "Max reconnect attempts (%d) reached. Stopping stream.",
                        self._max_reconnects,
                    )
                    break
                await asyncio.sleep(self._reconnect_delay)

        self._running = False

    async def _dispatch(self, raw: str) -> None:
        """Parse a raw WS message and notify subscribers."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("Non-JSON WS message ignored: %s", raw[:120])
            return

        symbol = data.get("symbol", "").upper()
        callbacks = self._subscribers.get(symbol, [])
        if not callbacks:
            return

        try:
            tick = Tick.from_dict(data)
        except (KeyError, ValueError) as e:
            logger.debug("Bad tick payload for %s: %s", symbol, e)
            return

        for cb in callbacks:
            try:
                await cb(tick)
            except Exception:
                logger.exception("Error in tick callback for %s", symbol)
