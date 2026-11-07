"""Phone alerts through ntfy.

Alerts go through a database outbox, so a network drop delays them rather than
losing them, and a failed send can never interrupt scheduling.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import urllib.error
import urllib.request

from labq import db
from labq.config import AlertConfig

log = logging.getLogger("labq")

GIVE_UP_AFTER_SECONDS = 24 * 3600


def send(config: AlertConfig, title: str, body: str, priority: int, tags: list[str]) -> None:
    payload = json.dumps(
        {
            "topic": config.ntfy_topic,
            "title": title,
            "message": body,
            "priority": priority,
            "tags": tags,
        }
    ).encode()
    request = urllib.request.Request(
        config.ntfy_server.rstrip("/") + "/",
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "labq"},
        method="POST",
    )
    if config.ntfy_token:
        request.add_header("Authorization", f"Bearer {config.ntfy_token}")
    with urllib.request.urlopen(request, timeout=10):
        pass


def flush(conn: sqlite3.Connection, config: AlertConfig, now: float) -> None:
    """Send waiting alerts in order, stopping at the first failure."""
    for alert in db.unsent_alerts(conn):
        if not config.ntfy_topic:  # alerts are off, so nothing may leave this machine
            db.mark_alert_done(conn, alert["id"], now)
            continue
        if now - alert["created_at"] > GIVE_UP_AFTER_SECONDS:
            log.warning("dropping stale alert: %s", alert["title"])
            db.mark_alert_done(conn, alert["id"], now)
            continue
        try:
            send(
                config,
                alert["title"],
                alert["body"],
                alert["priority"],
                json.loads(alert["tags"]),
            )
        except (urllib.error.URLError, OSError, ValueError) as exc:
            log.warning("could not send alert, will retry: %s", exc)
            return
        db.mark_alert_done(conn, alert["id"], now)
