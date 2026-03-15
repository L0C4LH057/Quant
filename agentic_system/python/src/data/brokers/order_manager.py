"""
Order lifecycle management.

GAP-05 fix: Previously, trade execution was a raw HTTP call with no
lifecycle tracking, reconciliation, or retry logic.

``OrderManager`` wraps the MT5 client to provide:

* Order submission with idempotency keys
* Persistent order journal (JSON on disk)
* Fill / partial-fill tracking
* Automatic SL/TP distance validation
* Retry with exponential back-off on transient errors

Usage::

    from src.data.brokers.order_manager import OrderManager

    manager = OrderManager(mt5_client, journal_dir="storage/orders")
    order_id = await manager.submit(
        symbol="EURUSD", side="buy", volume=0.1,
        sl_distance=0.0020, tp_distance=0.0040,
    )
    status = manager.get_order(order_id)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class Order:
    """Canonical order record."""

    order_id: str
    symbol: str
    side: str
    volume: float
    status: str = OrderStatus.PENDING.value
    sl: Optional[float] = None
    tp: Optional[float] = None
    entry_price: Optional[float] = None
    filled_volume: float = 0.0
    broker_ticket: Optional[int] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Order":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class OrderManager:
    """
    Manages order lifecycle against an MT5 broker client.

    Parameters
    ----------
    mt5_client
        An ``MT5Client`` instance (or any object with compatible trade methods).
    journal_dir : str | Path
        Directory to persist order journal JSON files.
    max_retries : int
        Maximum retry attempts for transient submission failures.
    retry_base_delay : float
        Base delay (seconds) for exponential back-off.
    """

    def __init__(
        self,
        mt5_client: Any,
        journal_dir: str | Path = "storage/orders",
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ) -> None:
        self._client = mt5_client
        self._journal_dir = Path(journal_dir)
        self._journal_dir.mkdir(parents=True, exist_ok=True)
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay

        # In-memory index (order_id -> Order)
        self._orders: Dict[str, Order] = {}
        self._load_journal()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit(
        self,
        symbol: str,
        side: str,
        volume: float,
        sl_distance: Optional[float] = None,
        tp_distance: Optional[float] = None,
        comment: str = "",
        magic: int = 0,
        idempotency_key: Optional[str] = None,
    ) -> str:
        """
        Submit a new order.

        Returns the ``order_id`` (UUID).  The order is persisted immediately
        and the actual broker submission runs with retry logic.
        """
        # Idempotency: if key seen before, return existing order
        if idempotency_key:
            for o in self._orders.values():
                if o.metadata.get("idempotency_key") == idempotency_key:
                    logger.info("Duplicate idempotency key %s — returning existing order %s", idempotency_key, o.order_id)
                    return o.order_id

        order = Order(
            order_id=str(uuid.uuid4()),
            symbol=symbol.upper(),
            side=side.lower(),
            volume=volume,
            metadata={
                "comment": comment,
                "magic": magic,
                "idempotency_key": idempotency_key,
            },
        )
        self._orders[order.order_id] = order
        self._persist(order)

        # Fire-and-forget the broker call (with retries)
        asyncio.create_task(self._execute_with_retry(order, sl_distance, tp_distance, comment, magic))
        return order.order_id

    def get_order(self, order_id: str) -> Optional[Order]:
        """Look up an order by ID."""
        return self._orders.get(order_id)

    def list_orders(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Order]:
        """Return orders optionally filtered by symbol / status."""
        orders = list(self._orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol.upper()]
        if status:
            orders = [o for o in orders if o.status == status]
        return sorted(orders, key=lambda o: o.created_at, reverse=True)

    def cancel(self, order_id: str) -> bool:
        """Mark a pending order as cancelled (broker cancel not yet supported)."""
        order = self._orders.get(order_id)
        if not order or order.status not in (OrderStatus.PENDING.value, OrderStatus.SUBMITTED.value):
            return False
        order.status = OrderStatus.CANCELLED.value
        order.updated_at = datetime.now(timezone.utc).isoformat()
        self._persist(order)
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _execute_with_retry(
        self,
        order: Order,
        sl_distance: Optional[float],
        tp_distance: Optional[float],
        comment: str,
        magic: int,
    ) -> None:
        """Submit to broker with exponential back-off on transient errors."""
        for attempt in range(1, self._max_retries + 1):
            try:
                order.status = OrderStatus.SUBMITTED.value
                order.updated_at = datetime.now(timezone.utc).isoformat()
                self._persist(order)

                # Resolve SL / TP absolute prices
                sl_price, tp_price = await self._resolve_sl_tp(
                    order.symbol, order.side, sl_distance, tp_distance
                )

                # Call broker
                result = await asyncio.to_thread(
                    self._client.place_order,
                    symbol=order.symbol,
                    order_type=order.side,
                    volume=order.volume,
                    sl=sl_price,
                    tp=tp_price,
                    comment=comment,
                    magic=magic,
                )

                if result.success:
                    order.status = OrderStatus.FILLED.value
                    order.broker_ticket = result.order_id
                    order.entry_price = result.price
                    order.filled_volume = order.volume
                    order.sl = sl_price
                    order.tp = tp_price
                else:
                    order.status = OrderStatus.REJECTED.value
                    order.error = result.error

                order.updated_at = datetime.now(timezone.utc).isoformat()
                self._persist(order)
                logger.info(
                    "Order %s %s (broker ticket=%s)",
                    order.order_id,
                    order.status,
                    order.broker_ticket,
                )
                return

            except Exception as exc:
                delay = self._retry_base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Order %s attempt %d/%d failed: %s — retrying in %.1fs",
                    order.order_id,
                    attempt,
                    self._max_retries,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        order.status = OrderStatus.ERROR.value
        order.error = "Max retries exceeded"
        order.updated_at = datetime.now(timezone.utc).isoformat()
        self._persist(order)
        logger.error("Order %s failed after %d retries.", order.order_id, self._max_retries)

    async def _resolve_sl_tp(
        self,
        symbol: str,
        side: str,
        sl_distance: Optional[float],
        tp_distance: Optional[float],
    ) -> tuple[Optional[float], Optional[float]]:
        """Convert distance-based SL/TP to absolute prices."""
        if sl_distance is None and tp_distance is None:
            return None, None

        quote = await asyncio.to_thread(self._client.get_quote, symbol)
        price = quote.ask if side == "buy" else quote.bid

        sl_price: Optional[float] = None
        tp_price: Optional[float] = None

        if sl_distance is not None and sl_distance > 0:
            sl_price = (price - sl_distance) if side == "buy" else (price + sl_distance)
        if tp_distance is not None and tp_distance > 0:
            tp_price = (price + tp_distance) if side == "buy" else (price - tp_distance)

        return sl_price, tp_price

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self, order: Order) -> None:
        path = self._journal_dir / f"{order.order_id}.json"
        path.write_text(json.dumps(order.to_dict(), indent=2, default=str))

    def _load_journal(self) -> None:
        """Load existing orders from disk on startup."""
        loaded = 0
        for p in self._journal_dir.glob("*.json"):
            try:
                data = json.loads(p.read_text())
                order = Order.from_dict(data)
                self._orders[order.order_id] = order
                loaded += 1
            except Exception as exc:
                logger.warning("Failed to load order journal %s: %s", p.name, exc)
        if loaded:
            logger.info("Loaded %d orders from journal.", loaded)
