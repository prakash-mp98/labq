"""SQLite storage for the queue and the alert outbox."""

from __future__ import annotations

import json
import secrets
import sqlite3
import time
from pathlib import Path

QUEUED = "queued"
STARTING = "starting"  # claimed; the launch may or may not have reached the host
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELLED = "cancelled"
LOST = "lost"  # the wrapper vanished without an exit code, e.g. a reboot

ACTIVE = (STARTING, RUNNING)
FINISHED = (SUCCEEDED, FAILED, CANCELLED, LOST)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token TEXT NOT NULL,
    name TEXT NOT NULL,
    command TEXT NOT NULL,
    cwd TEXT NOT NULL,
    hosts TEXT,
    gpus INTEGER NOT NULL DEFAULT 1,
    min_mem_mib INTEGER NOT NULL DEFAULT 0,
    priority INTEGER NOT NULL DEFAULT 0,
    retries_left INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 1,
    host TEXT,
    gpu_ids TEXT,
    exit_code INTEGER,
    reason TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    submitted_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS jobs_state ON jobs (state);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    priority INTEGER NOT NULL,
    tags TEXT NOT NULL,
    created_at REAL NOT NULL,
    sent_at REAL
);
"""


def connect(path: Path | str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


def job_dir(job: sqlite3.Row) -> str:
    """Unique per attempt, so a retry can never collide with an earlier launch."""
    return f"{job['id']}-{job['token']}-{job['attempt']}"


def submit(
    conn: sqlite3.Connection,
    command: str,
    cwd: str,
    name: str | None = None,
    hosts: list[str] | None = None,
    gpus: int = 1,
    min_mem_mib: int = 0,
    priority: int = 0,
    retries: int = 0,
) -> int:
    cursor = conn.execute(
        "INSERT INTO jobs (token, name, command, cwd, hosts, gpus, min_mem_mib, priority,"
        " retries_left, state, submitted_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            secrets.token_hex(4),
            name or "",
            command,
            cwd,
            json.dumps(hosts) if hosts else None,
            gpus,
            min_mem_mib,
            priority,
            retries,
            QUEUED,
            time.time(),
        ),
    )
    job_id = int(cursor.lastrowid)
    if not name:
        # Commands can hold paths or secrets and alerts may cross a public topic,
        # so the default name reveals nothing about the job.
        conn.execute("UPDATE jobs SET name = ? WHERE id = ?", (f"job-{job_id}", job_id))
    return job_id


def get_job(conn: sqlite3.Connection, job_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()


def jobs_in(conn: sqlite3.Connection, states: tuple[str, ...]) -> list[sqlite3.Row]:
    marks = ",".join("?" * len(states))
    return conn.execute(
        f"SELECT * FROM jobs WHERE state IN ({marks}) ORDER BY priority DESC, id", states
    ).fetchall()


def all_jobs(conn: sqlite3.Connection, limit: int = 50) -> list[sqlite3.Row]:
    rows = conn.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return list(reversed(rows))


def claim(
    conn: sqlite3.Connection, job_id: int, host: str, gpu_ids: list[int], now: float
) -> bool:
    """Atomically move a queued job to starting. False if it was cancelled meanwhile."""
    cursor = conn.execute(
        "UPDATE jobs SET state = ?, host = ?, gpu_ids = ?, started_at = ?"
        " WHERE id = ? AND state = ?",
        (STARTING, host, json.dumps(gpu_ids), now, job_id, QUEUED),
    )
    return cursor.rowcount == 1


def mark_running(conn: sqlite3.Connection, job_id: int) -> None:
    conn.execute("UPDATE jobs SET state = ? WHERE id = ?", (RUNNING, job_id))


def finish(
    conn: sqlite3.Connection,
    job_id: int,
    state: str,
    exit_code: int | None,
    reason: str,
    now: float,
) -> None:
    conn.execute(
        "UPDATE jobs SET state = ?, exit_code = ?, reason = ?, finished_at = ? WHERE id = ?",
        (state, exit_code, reason, now, job_id),
    )


def requeue(
    conn: sqlite3.Connection, job_id: int, reason: str, spend_retry: bool = True
) -> None:
    """Send a job back to the queue as a new attempt with a fresh directory."""
    spend = 1 if spend_retry else 0
    conn.execute(
        "UPDATE jobs SET state = ?, attempt = attempt + 1, host = NULL, gpu_ids = NULL,"
        " exit_code = NULL, started_at = NULL, finished_at = NULL, reason = ?,"
        " cancel_requested = 0, retries_left = MAX(retries_left - ?, 0)"
        " WHERE id = ?",
        (QUEUED, reason, spend, job_id),
    )


def cancel(conn: sqlite3.Connection, job_id: int, now: float) -> str:
    """Returns the outcome: cancelled, requested, finished or missing."""
    job = get_job(conn, job_id)
    if job is None:
        return "missing"
    done = conn.execute(
        "UPDATE jobs SET state = ?, reason = ?, finished_at = ? WHERE id = ? AND state = ?",
        (CANCELLED, "cancelled before it started", now, job_id, QUEUED),
    )
    if done.rowcount:
        return "cancelled"
    requested = conn.execute(
        "UPDATE jobs SET cancel_requested = 1 WHERE id = ? AND state IN (?, ?)",
        (job_id, *ACTIVE),
    )
    return "requested" if requested.rowcount else "finished"


def add_alert(
    conn: sqlite3.Connection, title: str, body: str, priority: int, tags: list[str], now: float
) -> None:
    conn.execute(
        "INSERT INTO alerts (title, body, priority, tags, created_at) VALUES (?, ?, ?, ?, ?)",
        (title, body, priority, json.dumps(tags), now),
    )


def unsent_alerts(conn: sqlite3.Connection, limit: int = 10) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM alerts WHERE sent_at IS NULL ORDER BY id LIMIT ?", (limit,)
    ).fetchall()


def mark_alert_done(conn: sqlite3.Connection, alert_id: int, now: float) -> None:
    conn.execute("UPDATE alerts SET sent_at = ? WHERE id = ?", (now, alert_id))
