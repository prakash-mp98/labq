"""Command line interface."""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import shlex
import signal
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from labq import __version__, alerts, db, remote
from labq.config import (
    SAMPLE_CONFIG,
    Config,
    ConfigError,
    config_path,
    db_path,
    labq_home,
    load_config,
    normalize_cwd,
    parse_mem,
)
from labq.scheduler import Scheduler, block_reason, format_duration
from labq.transport import transport_for

log = logging.getLogger("labq")


def _table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(str(r[i])) for r in [headers, *rows]) for i in range(len(headers))]
    for row in [headers, *rows]:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)).rstrip())


def _open() -> sqlite3.Connection:
    return db.connect(db_path())


def cmd_init(_args: argparse.Namespace) -> int:
    path = config_path()
    if path.exists():
        print(f"{path} already exists, leaving it alone.")
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SAMPLE_CONFIG)
    path.chmod(0o600)
    print(f"Wrote {path}. Add your servers, then run `labq hosts` to check them.")
    return 0


def cmd_hosts(_args: argparse.Namespace) -> int:
    config = load_config()
    conn = _open()
    occupied: dict[tuple[str, int], int] = {}
    for job in db.jobs_in(conn, db.ACTIVE):
        for gpu in json.loads(job["gpu_ids"] or "[]"):
            occupied[(job["host"], gpu)] = job["id"]

    def check(name: str) -> remote.Probe:
        result = transport_for(config.hosts[name]).run(remote.probe_script([]))
        if result.transport_failed:
            return remote.Probe(False, error=result.stderr.strip().splitlines()[-1]
                                if result.stderr.strip() else "unreachable")
        return remote.parse_probe(result.stdout)

    names = list(config.hosts)
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        probes = dict(zip(names, pool.map(check, names)))

    rows: list[list[str]] = []
    healthy = True
    for name in names:
        probe = probes[name]
        if not probe.reachable:
            rows.append([name, "-", "-", "-", f"UNREACHABLE: {probe.error}"])
            healthy = False
            continue
        if probe.gpu_error:
            rows.append([name, "-", "-", "-", f"will not schedule: {probe.gpu_error}"])
            healthy = False
        allowed = config.hosts[name].gpus
        for gpu in probe.gpus:
            if allowed is not None and gpu.index not in allowed:
                status = "excluded by config"
            else:
                status = block_reason(gpu, config, occupied.get((name, gpu.index))) or "free"
            rows.append([
                name, str(gpu.index), f"{gpu.mem_used}/{gpu.mem_total} MiB",
                f"{gpu.util}%", status,
            ])
    _table(["HOST", "GPU", "MEMORY", "UTIL", "STATUS"], rows)
    return 0 if healthy else 1


def cmd_submit(args: argparse.Namespace) -> int:
    config = load_config()
    parts = args.command
    if parts and parts[0] == "--":
        parts = parts[1:]
    if not parts:
        print("nothing to run. Usage: labq submit [options] -- COMMAND", file=sys.stderr)
        return 2
    # One argument is a shell command as typed. Several are quoted one by one.
    command = parts[0] if len(parts) == 1 else shlex.join(parts)

    hosts = args.host or None
    for name in hosts or []:
        if name not in config.hosts:
            print(f"unknown host {name!r}. Known: {', '.join(config.hosts)}", file=sys.stderr)
            return 2
    if args.gpus < 1:
        print("--gpus must be at least 1", file=sys.stderr)
        return 2
    if args.retries < 0:
        print("--retries cannot be negative", file=sys.stderr)
        return 2
    candidates = [config.hosts[n] for n in hosts] if hosts else list(config.hosts.values())
    if all(h.gpus is not None and len(h.gpus) < args.gpus for h in candidates):
        print(f"no allowed host offers {args.gpus} GPUs", file=sys.stderr)
        return 2
    try:
        min_mem = parse_mem(args.mem) if args.mem else 0
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    cwd = normalize_cwd(args.cwd or os.getcwd())
    job_id = db.submit(
        _open(), command, cwd, name=args.name, hosts=hosts, gpus=args.gpus,
        min_mem_mib=min_mem, priority=args.priority, retries=args.retries,
    )
    print(f"Queued job {job_id} (runs in {cwd}).")
    return 0


def _when(stamp: float | None) -> str:
    return datetime.fromtimestamp(stamp).strftime("%d %b %H:%M") if stamp else "-"


def _elapsed(job: sqlite3.Row) -> str:
    if not job["started_at"]:
        return "-"
    return format_duration((job["finished_at"] or time.time()) - job["started_at"])


def _where(job: sqlite3.Row) -> str:
    if not job["host"]:
        return "-"
    gpus = ",".join(str(g) for g in json.loads(job["gpu_ids"] or "[]"))
    return f"{job['host']}:{gpus}"


def cmd_list(args: argparse.Namespace) -> int:
    conn = _open()
    jobs = db.all_jobs(conn, args.limit) if args.all else (
        db.jobs_in(conn, db.ACTIVE) + db.jobs_in(conn, (db.QUEUED,))
    )
    if not jobs:
        print("No jobs." if args.all else "No active or queued jobs. `labq list --all` shows finished ones.")
        return 0
    rows = [
        [
            str(j["id"]), j["name"], j["state"], _where(j), _when(j["submitted_at"]),
            _elapsed(j), j["reason"] or "",
        ]
        for j in jobs
    ]
    _table(["ID", "NAME", "STATE", "WHERE", "SUBMITTED", "ELAPSED", "NOTE"], rows)
    return 0


def _require_job(conn: sqlite3.Connection, job_id: int) -> sqlite3.Row | None:
    job = db.get_job(conn, job_id)
    if job is None:
        print(f"no job {job_id}", file=sys.stderr)
    return job


def cmd_show(args: argparse.Namespace) -> int:
    job = _require_job(_open(), args.job)
    if job is None:
        return 1
    print(f"job:        {job['id']} ({job['name']})")
    print(f"state:      {job['state']}" + (f" ({job['reason']})" if job["reason"] else ""))
    print(f"command:    {job['command']}")
    print(f"directory:  {job['cwd']}")
    print(f"where:      {_where(job)}")
    print(f"attempt:    {job['attempt']}, retries left {job['retries_left']}")
    print(f"needs:      {job['gpus']} GPU(s), {job['min_mem_mib']} MiB free, priority {job['priority']}")
    print(f"submitted:  {_when(job['submitted_at'])}")
    print(f"elapsed:    {_elapsed(job)}")
    if job["exit_code"] is not None:
        print(f"exit code:  {job['exit_code']}")
    if job["host"]:
        print(f"files:      ~/.labq/jobs/{db.job_dir(job)}/ on {job['host']}")
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    config = load_config()
    job = _require_job(_open(), args.job)
    if job is None:
        return 1
    if not job["host"]:
        print(f"job {args.job} has not started yet.", file=sys.stderr)
        return 1
    if job["host"] not in config.hosts:
        print(f"host {job['host']} is no longer in your config.", file=sys.stderr)
        return 1
    transport = transport_for(config.hosts[job["host"]])
    return transport.stream(remote.tail_script(db.job_dir(job), args.lines, args.follow))


def cmd_cancel(args: argparse.Namespace) -> int:
    conn = _open()
    worst = 0
    for job_id in args.jobs:
        outcome = db.cancel(conn, job_id, time.time())
        messages = {
            "cancelled": f"Cancelled job {job_id}.",
            "requested": f"Job {job_id} will be stopped on the scheduler's next check.",
            "finished": f"Job {job_id} had already finished.",
            "missing": f"No job {job_id}.",
        }
        print(messages[outcome])
        worst = max(worst, outcome == "missing")
    return worst


def cmd_retry(args: argparse.Namespace) -> int:
    conn = _open()
    job = _require_job(conn, args.job)
    if job is None:
        return 1
    if job["state"] not in db.FINISHED:
        print(f"job {args.job} is still {job['state']}.", file=sys.stderr)
        return 1
    db.requeue(conn, args.job, "retried by user", spend_retry=False)
    print(f"Job {args.job} is queued again.")
    return 0


def cmd_test_alert(_args: argparse.Namespace) -> int:
    config = load_config().alerts
    if not config.ntfy_topic:
        print(f"Set alerts.ntfy_topic in {config_path()} first.", file=sys.stderr)
        return 1
    try:
        alerts.send(config, "labq test", "Alerts are working.", 3, ["tada"])
    except Exception as exc:  # the user asked to see any failure directly
        print(f"Could not send: {exc}", file=sys.stderr)
        return 1
    print(f"Sent to topic {config.ntfy_topic}.")
    return 0


def run_daemon(config: Config, stop: threading.Event) -> None:
    conn = _open()
    scheduler = Scheduler(config, conn)
    while not stop.is_set():
        try:
            scheduler.tick()
        except Exception:  # one bad pass must not take the whole queue down
            log.exception("scheduler pass failed, will try again")
        stop.wait(config.scheduler.poll_seconds)


def cmd_daemon(_args: argparse.Namespace) -> int:
    config = load_config()
    labq_home().mkdir(parents=True, exist_ok=True)
    lock = open(labq_home() / "daemon.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("Another labq daemon is already running here.", file=sys.stderr)
        return 1

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    log.info("watching %s every %gs. Running jobs are unaffected if this stops.",
             ", ".join(config.hosts), config.scheduler.poll_seconds)
    run_daemon(config, stop)
    log.info("stopped")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="labq", description="One GPU job queue across the SSH servers you already have."
    )
    parser.add_argument("--version", action="version", version=f"labq {__version__}")
    commands = parser.add_subparsers(dest="cmd", required=True)

    commands.add_parser("init", help="write a starter config").set_defaults(func=cmd_init)
    commands.add_parser("hosts", help="check servers and show each GPU").set_defaults(func=cmd_hosts)

    submit = commands.add_parser("submit", help="add a job to the queue")
    submit.add_argument("--name", help="short name, shown in alerts")
    submit.add_argument("--host", action="append", help="allowed server (repeatable, default any)")
    submit.add_argument("--gpus", type=int, default=1, help="GPUs on one server (default 1)")
    submit.add_argument("--mem", help="minimum free GPU memory, such as 8G")
    submit.add_argument("--cwd", help="directory to run in (default: this one)")
    submit.add_argument("--priority", type=int, default=0, help="higher runs first")
    submit.add_argument("--retries", type=int, default=0, help="automatic retries on failure")
    submit.add_argument("command", nargs=argparse.REMAINDER, help="-- COMMAND")
    submit.set_defaults(func=cmd_submit)

    listing = commands.add_parser("list", aliases=["ls"], help="show jobs")
    listing.add_argument("-a", "--all", action="store_true", help="include finished jobs")
    listing.add_argument("-n", "--limit", type=int, default=50, help="how many with --all")
    listing.set_defaults(func=cmd_list)

    show = commands.add_parser("show", help="show one job in full")
    show.add_argument("job", type=int)
    show.set_defaults(func=cmd_show)

    logs = commands.add_parser("logs", help="print a job's output")
    logs.add_argument("job", type=int)
    logs.add_argument("-f", "--follow", action="store_true", help="keep printing until it ends")
    logs.add_argument("-n", "--lines", type=int, default=50)
    logs.set_defaults(func=cmd_logs)

    cancel = commands.add_parser("cancel", help="cancel queued or running jobs")
    cancel.add_argument("jobs", type=int, nargs="+")
    cancel.set_defaults(func=cmd_cancel)

    retry = commands.add_parser("retry", help="queue a finished job again")
    retry.add_argument("job", type=int)
    retry.set_defaults(func=cmd_retry)

    commands.add_parser("daemon", help="run the scheduler in the foreground").set_defaults(func=cmd_daemon)
    commands.add_parser("test-alert", help="send a test phone alert").set_defaults(func=cmd_test_alert)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        return 0
