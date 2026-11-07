"""Tests run the real shell scripts against a fake nvidia-smi and a temp $HOME.

Each simulated server is a LocalTransport with its own HOME, so job wrappers,
process groups and exit codes are all real. Only the GPU readings are faked.
"""

from __future__ import annotations

import json
import stat
import time
from pathlib import Path

import pytest

from labq import db
from labq.config import AlertConfig, Config, Host, SchedulerConfig
from labq.scheduler import Scheduler
from labq.transport import LocalTransport, Result, Transport

FAKE_NVIDIA_SMI = """#!/bin/sh
dir=$(dirname "$0")
[ -f "$dir/smi_broken" ] && { echo "Failed to initialize NVML" >&2; exit 9; }
case "$*" in
  *query-gpu*) cat "$dir/gpus.csv" ;;
  *query-compute-apps*) cat "$dir/apps.csv" ;;
esac
"""


class FakeServer:
    def __init__(self, root: Path):
        self.home = root / "home"
        self.bin = root / "bin"
        self.home.mkdir(parents=True)
        self.bin.mkdir()
        smi = self.bin / "nvidia-smi"
        smi.write_text(FAKE_NVIDIA_SMI)
        smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
        self.set_gpus([(0, 10, 0)])
        self.set_apps([])
        self.down = False
        self.real = LocalTransport(
            {"HOME": str(self.home), "PATH": f"{self.bin}:/usr/bin:/bin"}
        )

    def set_gpus(self, gpus: list[tuple[int, int, int]], total: int = 16000) -> None:
        """Each GPU is (index, memory used in MiB, utilisation percent)."""
        lines = [f"{i}, GPU-{i}, {used}, {total}, {util}" for i, used, util in gpus]
        (self.bin / "gpus.csv").write_text("\n".join(lines) + "\n")

    def set_raw_gpus(self, text: str) -> None:
        (self.bin / "gpus.csv").write_text(text)

    def set_apps(self, uuids: list[str]) -> None:
        (self.bin / "apps.csv").write_text("".join(f"{u}, 4242\n" for u in uuids))

    def break_smi(self) -> None:
        (self.bin / "smi_broken").touch()

    def job_dir(self, job: object) -> Path:
        return self.home / ".labq" / "jobs" / db.job_dir(job)


class SwitchableTransport(Transport):
    """A real local transport that can pretend the network dropped."""

    def __init__(self, server: FakeServer):
        self.server = server
        self.lose_launch_reply = False

    def run(self, script: str, timeout: float = 45) -> Result:
        if self.server.down:
            return Result(255, "", "ssh: connect to host: timed out", True)
        result = self.server.real.run(script, timeout)
        if self.lose_launch_reply and "setsid" in script:
            # The job started on the host but the answer never came back.
            return Result(255, "", "connection reset", True)
        return result

    def stream(self, script: str) -> int:
        return self.server.real.stream(script)


class Rig:
    def __init__(self, tmp_path: Path, names: tuple[str, ...], **config):
        self.servers = {n: FakeServer(tmp_path / n) for n in names}
        self.transports = {n: SwitchableTransport(s) for n, s in self.servers.items()}
        scheduler = {"poll_seconds": 0.01, "free_polls_required": 1, **config.pop("scheduler", {})}
        self.config = Config(
            hosts=config.pop("hosts", {n: Host(n, "unused") for n in names}),
            scheduler=SchedulerConfig(**scheduler),
            alerts=config.pop("alerts", AlertConfig(idle_gpu_minutes=0)),
        )
        self.conn = db.connect(tmp_path / "labq.db")
        self.scheduler = self.new_scheduler()

    def new_scheduler(self) -> Scheduler:
        return Scheduler(self.config, self.conn, self.transports)

    def submit(self, command: str, **kwargs) -> int:
        kwargs.setdefault("cwd", "~")
        return db.submit(self.conn, command, **kwargs)

    def job(self, job_id: int):
        return db.get_job(self.conn, job_id)

    def tick_until(self, job_id: int, *states: str, timeout: float = 15) -> object:
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.scheduler.tick()
            job = self.job(job_id)
            if job["state"] in states:
                return job
            time.sleep(0.05)
        raise AssertionError(f"job {job_id} stayed {self.job(job_id)['state']}, wanted {states}")

    def alert_titles(self) -> list[str]:
        return [r["title"] for r in self.conn.execute("SELECT title FROM alerts ORDER BY id")]

    def gpu_ids(self, job_id: int) -> list[int]:
        return json.loads(self.job(job_id)["gpu_ids"])


@pytest.fixture
def rig(tmp_path):
    return Rig(tmp_path, ("a",))


@pytest.fixture
def make_rig(tmp_path):
    def build(names=("a",), **config):
        return Rig(tmp_path, names, **config)

    return build
