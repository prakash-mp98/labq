"""The scheduling loop: observe hosts, update jobs, start new ones, raise alerts.

Two rules shape everything here:

* Fail safe on GPUs. If a GPU's state cannot be read, it counts as busy.
* Never guess about jobs. Only a complete answer from a host can change a job's
  state. An unreachable host leaves its jobs running in the database, which is
  usually the truth, because jobs do not depend on the connection.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor

from labq import alerts, db, remote
from labq.config import Config, Host
from labq.remote import GpuInfo, Probe
from labq.transport import Transport, transport_for

log = logging.getLogger("labq")

LAUNCH_GRACE_SECONDS = 120
STUCK_UTIL_PERCENT = 5
_GPU_OOM = re.compile(r"CUDA out of memory|OutOfMemoryError|CUDA_ERROR_OUT_OF_MEMORY")


def block_reason(gpu: GpuInfo, config: Config, occupied_by: int | None = None) -> str | None:
    """Why this GPU cannot take a job right now, or None if it can."""
    limits = config.scheduler
    if occupied_by is not None:
        return f"running labq job {occupied_by}"
    if gpu.mem_used is None or gpu.mem_total is None or gpu.util is None:
        return "metrics unreadable"
    if gpu.processes:
        return f"{gpu.processes} process(es) on GPU"
    if gpu.mem_used > limits.max_mem_used_mib:
        return f"{gpu.mem_used} MiB in use"
    if gpu.util > limits.max_util_percent:
        return f"{gpu.util}% utilisation"
    return None


def format_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def describe_failure(exit_code: int | None, log_tail: str) -> str:
    if _GPU_OOM.search(log_tail):
        return "GPU out of memory"
    if exit_code == 126:
        return "could not start (see log)"
    if exit_code == 137:
        return "killed with SIGKILL, possibly out of host memory"
    return f"exit code {exit_code}"


class Scheduler:
    def __init__(
        self,
        config: Config,
        conn: sqlite3.Connection,
        transports: dict[str, Transport] | None = None,
    ):
        self.config = config
        self.conn = conn
        self.transports = transports or {
            name: transport_for(host) for name, host in config.hosts.items()
        }
        self.free_streak: dict[tuple[str, int], int] = {}
        self.host_down_since: dict[str, float] = {}
        self.host_down_alerted: set[str] = set()
        self.stuck_since: dict[int, float] = {}
        self.stuck_alerted: set[int] = set()
        self.idle_since: dict[str, float] = {}
        self.idle_alerted: set[str] = set()

    # ---- one pass -------------------------------------------------------------

    def tick(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        probes = self._probe_all()
        for name, probe in probes.items():
            if probe.reachable:
                self._reconcile_host(name, probe, now)
        self._process_cancels(probes, now)
        occupied = self._occupied()
        self._update_streaks(probes, occupied)
        self._dispatch(probes, now)
        self._check_conditions(probes, self._occupied(), now)
        alerts.flush(self.conn, self.config.alerts, now)

    def _probe_all(self) -> dict[str, Probe]:
        active = db.jobs_in(self.conn, db.ACTIVE)

        def probe_host(name: str) -> Probe:
            dirs = [db.job_dir(j) for j in active if j["host"] == name]
            result = self.transports[name].run(remote.probe_script(dirs))
            if result.transport_failed:
                return Probe(False, error=result.stderr.strip()[:200])
            return remote.parse_probe(result.stdout)

        names = list(self.config.hosts)
        if not names:
            return {}
        with ThreadPoolExecutor(max_workers=len(names)) as pool:
            return dict(zip(names, pool.map(probe_host, names)))

    # ---- following running jobs -----------------------------------------------

    def _reconcile_host(self, name: str, probe: Probe, now: float) -> None:
        for job in db.jobs_in(self.conn, db.ACTIVE):
            if job["host"] != name or job["cancel_requested"]:
                continue
            seen = probe.jobs.get(db.job_dir(job))
            if seen is None:
                continue
            if seen.status == "alive":
                if job["state"] == db.STARTING:
                    db.mark_running(self.conn, job["id"])
            elif seen.status == "exited":
                self._job_exited(job, seen.exit_code, now)
            elif seen.status == "dead":
                self._job_lost(job, "process vanished without an exit code", now)
            elif seen.status == "launching":
                if now - job["started_at"] > LAUNCH_GRACE_SECONDS:
                    self._job_lost(job, "launch never completed", now)
            elif seen.status == "absent":
                if job["state"] == db.STARTING:
                    # The launch never reached the host, so this costs no retry.
                    log.info("job %s never reached %s, requeueing", job["id"], name)
                    db.requeue(self.conn, job["id"], "launch did not reach host", False)
                else:
                    self._job_lost(job, "job directory disappeared", now)

    def _log_tail(self, job: sqlite3.Row, lines: int = 40) -> str:
        result = self.transports[job["host"]].run(
            remote.tail_script(db.job_dir(job), lines), timeout=20
        )
        return "" if result.transport_failed else result.stdout

    def _job_exited(self, job: sqlite3.Row, exit_code: int | None, now: float) -> None:
        self._forget_job(job["id"])
        elapsed = format_duration(now - job["started_at"])
        label = f"{job['name']} on {job['host']}"
        if exit_code == 0:
            db.finish(self.conn, job["id"], db.SUCCEEDED, 0, "", now)
            db.add_alert(
                self.conn, f"Finished: {label}", f"Job {job['id']} took {elapsed}.",
                3, ["white_check_mark"], now,
            )
            return

        tail = self._log_tail(job)
        reason = describe_failure(exit_code, tail)
        body = f"Job {job['id']} failed after {elapsed}: {reason}."
        if self.config.alerts.include_log_tail and tail:
            body += "\n\n" + "\n".join(tail.splitlines()[-10:])
        if job["retries_left"] > 0:
            db.requeue(self.conn, job["id"], reason)
            db.add_alert(
                self.conn, f"Retrying: {label}",
                f"{body} Retrying now, with {job['retries_left'] - 1} more after this.",
                3, ["repeat"], now,
            )
            return
        db.finish(self.conn, job["id"], db.FAILED, exit_code, reason, now)
        db.add_alert(self.conn, f"Failed: {label}", body, 4, ["x"], now)

    def _job_lost(self, job: sqlite3.Row, reason: str, now: float) -> None:
        self._forget_job(job["id"])
        label = f"{job['name']} on {job['host']}"
        body = f"Job {job['id']}: {reason}. The host may have rebooted."
        if job["retries_left"] > 0:
            db.requeue(self.conn, job["id"], reason)
            db.add_alert(self.conn, f"Retrying: {label}", body, 3, ["repeat"], now)
            return
        db.finish(self.conn, job["id"], db.LOST, None, reason, now)
        db.add_alert(self.conn, f"Lost: {label}", body, 4, ["warning"], now)

    def _forget_job(self, job_id: int) -> None:
        self.stuck_since.pop(job_id, None)
        self.stuck_alerted.discard(job_id)

    def _process_cancels(self, probes: dict[str, Probe], now: float) -> None:
        for job in db.jobs_in(self.conn, db.ACTIVE):
            probe = probes.get(job["host"])
            if not job["cancel_requested"] or probe is None or not probe.reachable:
                continue
            result = self.transports[job["host"]].run(
                remote.cancel_script(db.job_dir(job)), timeout=60
            )
            outcome = result.stdout.strip().splitlines()[-1:]
            if result.transport_failed or outcome not in (["absent"], ["finished"], ["killed"]):
                # Still launching, still alive, or an unclear answer. Reporting a
                # job as cancelled while it holds the GPU would be worse than waiting.
                log.warning("job %s not stopped yet (%s), will try again",
                            job["id"], outcome or result.stderr.strip()[:200])
                continue
            self._forget_job(job["id"])
            db.finish(self.conn, job["id"], db.CANCELLED, None, "cancelled by user", now)

    # ---- starting jobs --------------------------------------------------------

    def _occupied(self) -> dict[tuple[str, int], int]:
        """GPUs that labq has given out. A new job may not have allocated memory
        yet, so nvidia-smi alone would wrongly report these as free."""
        taken: dict[tuple[str, int], int] = {}
        for job in db.jobs_in(self.conn, db.ACTIVE):
            for gpu in json.loads(job["gpu_ids"] or "[]"):
                taken[(job["host"], gpu)] = job["id"]
        return taken

    def _allowed_gpus(self, host: Host, probe: Probe) -> list[GpuInfo]:
        if not probe.reachable or probe.gpu_error:
            return []
        return [g for g in probe.gpus if host.gpus is None or g.index in host.gpus]

    def _update_streaks(
        self, probes: dict[str, Probe], occupied: dict[tuple[str, int], int]
    ) -> None:
        current: dict[tuple[str, int], int] = {}
        for name, host in self.config.hosts.items():
            for gpu in self._allowed_gpus(host, probes[name]):
                key = (name, gpu.index)
                if block_reason(gpu, self.config, occupied.get(key)) is None:
                    current[key] = self.free_streak.get(key, 0) + 1
        # Replacing the map resets every GPU that was busy or unreadable this pass.
        self.free_streak = current

    def _dispatch(self, probes: dict[str, Probe], now: float) -> None:
        for job in db.jobs_in(self.conn, (db.QUEUED,)):
            placement = self._place(job, probes)
            if placement is None:
                continue
            name, gpu_ids = placement
            if not db.claim(self.conn, job["id"], name, gpu_ids, now):
                continue
            for gpu in gpu_ids:
                self.free_streak.pop((name, gpu), None)
            self._launch(db.get_job(self.conn, job["id"]), now)

    def _place(self, job: sqlite3.Row, probes: dict[str, Probe]) -> tuple[str, list[int]] | None:
        wanted = json.loads(job["hosts"]) if job["hosts"] else None
        needed = self.config.scheduler.free_polls_required
        best: tuple[int, str, list[int]] | None = None
        for name, host in self.config.hosts.items():
            if wanted and name not in wanted:
                continue
            ready = [
                g
                for g in self._allowed_gpus(host, probes[name])
                if self.free_streak.get((name, g.index), 0) >= needed
                and g.mem_total - g.mem_used >= job["min_mem_mib"]
            ]
            if len(ready) < job["gpus"]:
                continue
            ready.sort(key=lambda g: (g.mem_used - g.mem_total, g.index))
            chosen = ready[: job["gpus"]]
            free_mem = sum(g.mem_total - g.mem_used for g in chosen)
            if best is None or free_mem > best[0]:
                best = (free_mem, name, sorted(g.index for g in chosen))
        return (best[1], best[2]) if best else None

    def _launch(self, job: sqlite3.Row, now: float) -> None:
        name = job["host"]
        script = remote.launch_script(
            db.job_dir(job),
            job["id"],
            job["command"],
            job["cwd"],
            json.loads(job["gpu_ids"]),
            self.config.hosts[name].setup,
        )
        result = self.transports[name].run(script)
        if result.transport_failed:
            # The launch may or may not have happened. The next probe will tell.
            log.warning("launch of job %s on %s unconfirmed: %s",
                        job["id"], name, result.stderr.strip()[:200])
            return
        if result.returncode == remote.EXIT_ALREADY_LAUNCHED:
            return
        if result.returncode != 0:
            reason = f"could not launch: {result.stderr.strip()[:200]}"
            db.finish(self.conn, job["id"], db.FAILED, None, reason, now)
            db.add_alert(
                self.conn, f"Failed: {job['name']} on {name}",
                f"Job {job['id']} could not be launched.", 4, ["x"], now,
            )
            return
        db.mark_running(self.conn, job["id"])
        log.info("started job %s on %s GPU %s", job["id"], name, job["gpu_ids"])

    # ---- alerts about conditions rather than events ---------------------------

    def _check_conditions(
        self, probes: dict[str, Probe], occupied: dict[tuple[str, int], int], now: float
    ) -> None:
        self._check_hosts(probes, now)
        self._check_stuck_jobs(probes, now)
        self._check_idle_gpus(probes, occupied, now)

    def _check_hosts(self, probes: dict[str, Probe], now: float) -> None:
        limit = self.config.alerts.host_down_minutes * 60
        active = db.jobs_in(self.conn, db.ACTIVE)
        for name, probe in probes.items():
            if probe.reachable:
                self.host_down_since.pop(name, None)
                if name in self.host_down_alerted:
                    self.host_down_alerted.discard(name)
                    db.add_alert(
                        self.conn, f"Back online: {name}",
                        "The server can be reached again.", 3, ["white_check_mark"], now,
                    )
                continue
            since = self.host_down_since.setdefault(name, now)
            count = sum(1 for j in active if j["host"] == name)
            # A server with no jobs of yours being offline is not worth a page.
            if limit and count and now - since >= limit and name not in self.host_down_alerted:
                self.host_down_alerted.add(name)
                db.add_alert(
                    self.conn, f"Cannot reach {name}",
                    f"No contact for {format_duration(now - since)}. {count} job(s) were "
                    "running there and may still be fine.",
                    4, ["warning"], now,
                )

    def _check_stuck_jobs(self, probes: dict[str, Probe], now: float) -> None:
        limit = self.config.alerts.stuck_job_minutes * 60
        if not limit:
            return
        for job in db.jobs_in(self.conn, (db.RUNNING,)):
            probe = probes.get(job["host"])
            if probe is None or not probe.reachable or probe.gpu_error:
                continue
            ids = set(json.loads(job["gpu_ids"]))
            gpus = [g for g in probe.gpus if g.index in ids]
            quiet = bool(gpus) and all(
                g.util is not None and g.util <= STUCK_UTIL_PERCENT for g in gpus
            )
            if not quiet:
                self._forget_job(job["id"])
                continue
            since = self.stuck_since.setdefault(job["id"], now)
            if now - since >= limit and job["id"] not in self.stuck_alerted:
                self.stuck_alerted.add(job["id"])
                db.add_alert(
                    self.conn, f"Possibly stuck: {job['name']} on {job['host']}",
                    f"Job {job['id']} is running, but its GPU has been near 0% for "
                    f"{format_duration(now - since)}.",
                    4, ["hourglass"], now,
                )

    def _check_idle_gpus(
        self, probes: dict[str, Probe], occupied: dict[tuple[str, int], int], now: float
    ) -> None:
        limit = self.config.alerts.idle_gpu_minutes * 60
        queue_empty = not db.jobs_in(self.conn, (db.QUEUED,))
        for name, host in self.config.hosts.items():
            free = [
                g
                for g in self._allowed_gpus(host, probes[name])
                if block_reason(g, self.config, occupied.get((name, g.index))) is None
            ]
            if not (limit and queue_empty and free):
                self.idle_since.pop(name, None)
                self.idle_alerted.discard(name)
                continue
            since = self.idle_since.setdefault(name, now)
            if now - since >= limit and name not in self.idle_alerted:
                self.idle_alerted.add(name)
                db.add_alert(
                    self.conn, f"GPU idle: {name}",
                    f"{len(free)} GPU(s) free for {format_duration(now - since)} "
                    "and your queue is empty.",
                    2, ["zzz"], now,
                )
