"""
Alerting module — notify operators of trading events, anomalies, and errors.

GAP-11 fix: No alerting existed; failures were silently logged.

Supported channels:
    * **Webhook** — generic HTTP POST (Discord, Slack incoming-webhook, etc.)
    * **Console** — structured log at WARNING+ (always enabled)

Usage::

    from src.utils.alerting import AlertManager, Severity

    alerts = AlertManager(webhook_url="https://hooks.slack.com/services/...")
    await alerts.send("EURUSD position hit stop-loss", severity=Severity.HIGH)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Severity(IntEnum):
    """Alert severity levels."""

    INFO = 0
    WARNING = 1
    HIGH = 2
    CRITICAL = 3


class AlertManager:
    """
    Central alert dispatcher.

    Parameters
    ----------
    webhook_url : str | None
        HTTP(S) endpoint to POST alerts to.  Falls back to
        ``ALERT_WEBHOOK_URL`` env var.
    min_severity : Severity
        Minimum severity to actually dispatch (default WARNING).
    cooldown : float
        Seconds to suppress duplicate alerts with the same title.
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        min_severity: Severity = Severity.WARNING,
        cooldown: float = 60.0,
    ) -> None:
        self._webhook_url = webhook_url or os.getenv("ALERT_WEBHOOK_URL")
        self._min_severity = min_severity
        self._cooldown = cooldown
        self._recent: Dict[str, float] = {}  # title -> last_sent_ts
        self._history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send(
        self,
        title: str,
        detail: str = "",
        severity: Severity = Severity.WARNING,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Dispatch an alert.

        Returns ``True`` if the alert was actually sent (not suppressed).
        """
        if severity < self._min_severity:
            return False

        now = datetime.now(timezone.utc).timestamp()
        if title in self._recent and (now - self._recent[title]) < self._cooldown:
            logger.debug("Alert suppressed (cooldown): %s", title)
            return False

        self._recent[title] = now

        payload = {
            "title": title,
            "detail": detail,
            "severity": severity.name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        }
        self._history.append(payload)

        # Always log
        log_level = {
            Severity.INFO: logging.INFO,
            Severity.WARNING: logging.WARNING,
            Severity.HIGH: logging.ERROR,
            Severity.CRITICAL: logging.CRITICAL,
        }.get(severity, logging.WARNING)
        logger.log(log_level, "[ALERT:%s] %s — %s", severity.name, title, detail)

        # Webhook dispatch
        if self._webhook_url:
            asyncio.create_task(self._post_webhook(payload))

        return True

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Return recent alert history (in-memory only)."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _post_webhook(self, payload: Dict[str, Any]) -> None:
        try:
            import httpx

            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    self._webhook_url,  # type: ignore[arg-type]
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                if resp.status_code >= 400:
                    logger.warning(
                        "Webhook returned %d: %s", resp.status_code, resp.text[:200]
                    )
        except Exception as exc:
            logger.warning("Webhook delivery failed: %s", exc)
