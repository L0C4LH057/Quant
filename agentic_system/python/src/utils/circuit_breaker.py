"""
Circuit breaker pattern for external service calls.

GAP-12 fix: Previously, transient failures in the MT5 bridge, LLM APIs,
or data providers would cascade into the agentic pipeline with no
automatic back-off or fail-open behaviour.

The ``CircuitBreaker`` transitions through three states::

    CLOSED  ──(failure threshold)──>  OPEN
    OPEN    ──(recovery timeout)───>  HALF_OPEN
    HALF_OPEN ──(success)──────────>  CLOSED
    HALF_OPEN ──(failure)──────────>  OPEN

Usage::

    from src.utils.circuit_breaker import CircuitBreaker, CircuitOpenError

    cb = CircuitBreaker("mt5-bridge", failure_threshold=5, recovery_timeout=30)

    async with cb:
        result = await call_mt5_api()

    # Or as a decorator:
    @cb.protect
    async def call_mt5_api(): ...
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a call is attempted while the circuit is OPEN."""

    def __init__(self, name: str, remaining: float) -> None:
        self.name = name
        self.remaining = remaining
        super().__init__(
            f"Circuit '{name}' is OPEN — retry in {remaining:.1f}s"
        )


class CircuitBreaker:
    """
    Async-friendly circuit breaker.

    Parameters
    ----------
    name : str
        Human-readable label (used in logs and errors).
    failure_threshold : int
        Consecutive failures before the circuit opens (default 5).
    recovery_timeout : float
        Seconds the circuit stays open before moving to half-open (default 30).
    success_threshold : int
        Successes in half-open state required to fully close (default 2).
    excluded_exceptions : tuple
        Exception types that should NOT count as failures (e.g. ``ValueError``).
    on_state_change : callable, optional
        Async callback ``(name, old_state, new_state)`` fired on transitions.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 2,
        excluded_exceptions: tuple = (),
        on_state_change: Optional[Callable] = None,
    ) -> None:
        self.name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._success_threshold = success_threshold
        self._excluded_exceptions = excluded_exceptions
        self._on_state_change = on_state_change

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    # ------------------------------------------------------------------
    # Context manager (async with cb: ...)
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "CircuitBreaker":
        await self._before_call()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            await self._on_success()
        elif not issubclass(exc_type, self._excluded_exceptions):
            await self._on_failure()
        # Don't suppress the exception
        return False

    # ------------------------------------------------------------------
    # Decorator
    # ------------------------------------------------------------------

    def protect(self, fn: Callable) -> Callable:
        """Decorate an async function with this circuit breaker."""

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with self:
                return await fn(*args, **kwargs)

        return wrapper

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    async def _before_call(self) -> None:
        async with self._lock:
            if self._state == CircuitState.OPEN:
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self._recovery_timeout:
                    await self._transition(CircuitState.HALF_OPEN)
                else:
                    remaining = self._recovery_timeout - elapsed
                    raise CircuitOpenError(self.name, remaining)

    async def _on_success(self) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._success_threshold:
                    self._failure_count = 0
                    self._success_count = 0
                    await self._transition(CircuitState.CLOSED)
            else:
                self._failure_count = 0

    async def _on_failure(self) -> None:
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                self._success_count = 0
                await self._transition(CircuitState.OPEN)
            elif self._failure_count >= self._failure_threshold:
                await self._transition(CircuitState.OPEN)

    async def _transition(self, new_state: CircuitState) -> None:
        old = self._state
        self._state = new_state
        logger.warning(
            "Circuit '%s' state: %s -> %s (failures=%d)",
            self.name,
            old.value,
            new_state.value,
            self._failure_count,
        )
        if self._on_state_change:
            try:
                await self._on_state_change(self.name, old, new_state)
            except Exception:
                logger.exception("Error in circuit breaker state-change callback")

    # ------------------------------------------------------------------
    # Manual controls
    # ------------------------------------------------------------------

    async def reset(self) -> None:
        """Manually reset the breaker to CLOSED."""
        async with self._lock:
            self._failure_count = 0
            self._success_count = 0
            await self._transition(CircuitState.CLOSED)

    async def trip(self) -> None:
        """Manually open the breaker."""
        async with self._lock:
            self._last_failure_time = time.monotonic()
            await self._transition(CircuitState.OPEN)
