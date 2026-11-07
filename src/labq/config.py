"""Load and validate the TOML config. Unknown keys are errors, so typos surface."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

LOCAL = "local"
_HOST_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

SAMPLE_CONFIG = """\
# labq config. Every value below [hosts.*] except `ssh` is optional.

[scheduler]
poll_seconds = 30          # how often to check the servers
free_polls_required = 2    # a GPU must look free this many checks in a row
max_mem_used_mib = 1024    # above this, someone is using the GPU
max_util_percent = 20      # above this, someone is using the GPU

[alerts]
# Topics on ntfy.sh are public to anyone who knows the name, so pick a long
# random one. Leave empty to turn alerts off.
ntfy_topic = ""
ntfy_server = "https://ntfy.sh"
ntfy_token = ""            # only for a private or authenticated ntfy server
stuck_job_minutes = 30     # job is running but its GPU sits at about 0% (0 = off)
idle_gpu_minutes = 60      # queue is empty and a GPU is free (0 = off)
host_down_minutes = 15     # a server with running jobs cannot be reached (0 = off)
include_log_tail = false   # add the last log lines to failure alerts

# One section per server. `ssh` is anything you could pass to the ssh command,
# usually a Host alias from ~/.ssh/config. Key-based login is required.
[hosts.server1]
ssh = "server1"
# gpus = [0]               # only ever use these GPU indices
# setup = "source ~/miniconda3/etc/profile.d/conda.sh"  # runs before each job

# [hosts.server2]
# ssh = "user@server2.example.org"

# ssh = "local" runs jobs on this machine without SSH.
# [hosts.this-machine]
# ssh = "local"
"""


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Host:
    name: str
    ssh: str
    gpus: tuple[int, ...] | None = None
    setup: str = ""


@dataclass(frozen=True)
class SchedulerConfig:
    poll_seconds: float = 30
    free_polls_required: int = 2
    max_mem_used_mib: int = 1024
    max_util_percent: int = 20


@dataclass(frozen=True)
class AlertConfig:
    ntfy_topic: str = ""
    ntfy_server: str = "https://ntfy.sh"
    ntfy_token: str = ""
    stuck_job_minutes: float = 30
    idle_gpu_minutes: float = 60
    host_down_minutes: float = 15
    include_log_tail: bool = False


@dataclass(frozen=True)
class Config:
    hosts: dict[str, Host]
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)


def labq_home() -> Path:
    return Path(os.environ.get("LABQ_HOME", Path.home() / ".labq"))


def config_path() -> Path:
    return labq_home() / "config.toml"


def db_path() -> Path:
    return labq_home() / "labq.db"


def _build(cls, section: str, raw: dict):
    known = set(cls.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"[{section}] has unknown keys: {', '.join(sorted(unknown))}")
    try:
        return cls(**raw)
    except TypeError as exc:  # pragma: no cover
        raise ConfigError(f"[{section}]: {exc}") from exc


def _number(section: str, values: object, names: tuple[str, ...], minimum: float) -> None:
    for name in names:
        value = getattr(values, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < minimum:
            raise ConfigError(f"[{section}] {name} must be a number >= {minimum}")


def parse_config(raw: dict) -> Config:
    unknown = set(raw) - {"scheduler", "alerts", "hosts"}
    if unknown:
        raise ConfigError(f"unknown sections: {', '.join(sorted(unknown))}")

    scheduler = _build(SchedulerConfig, "scheduler", raw.get("scheduler", {}))
    _number("scheduler", scheduler, ("poll_seconds",), 0.01)
    _number("scheduler", scheduler, ("free_polls_required",), 1)
    _number("scheduler", scheduler, ("max_mem_used_mib", "max_util_percent"), 0)

    alerts = _build(AlertConfig, "alerts", raw.get("alerts", {}))
    _number(
        "alerts", alerts, ("stuck_job_minutes", "idle_gpu_minutes", "host_down_minutes"), 0
    )

    hosts: dict[str, Host] = {}
    for name, entry in raw.get("hosts", {}).items():
        if not _HOST_NAME.match(name):
            raise ConfigError(f"bad host name: {name!r}")
        host_unknown = set(entry) - {"ssh", "gpus", "setup"}
        if host_unknown:
            raise ConfigError(
                f"[hosts.{name}] has unknown keys: {', '.join(sorted(host_unknown))}"
            )
        target = entry.get("ssh")
        if not isinstance(target, str) or not target or target.startswith("-"):
            raise ConfigError(f"[hosts.{name}] needs a valid ssh target")
        gpus = entry.get("gpus")
        if gpus is not None and not (
            isinstance(gpus, list)
            and gpus
            and all(isinstance(g, int) and not isinstance(g, bool) and g >= 0 for g in gpus)
        ):
            raise ConfigError(f"[hosts.{name}] gpus must be a list of GPU indices")
        setup = entry.get("setup", "")
        if not isinstance(setup, str):
            raise ConfigError(f"[hosts.{name}] setup must be a string")
        hosts[name] = Host(name, target, tuple(gpus) if gpus else None, setup)

    return Config(hosts=hosts, scheduler=scheduler, alerts=alerts)


def load_config(path: Path | None = None) -> Config:
    path = path or config_path()
    try:
        with open(path, "rb") as handle:
            raw = tomllib.load(handle)
    except FileNotFoundError:
        raise ConfigError(f"no config at {path}. Run `labq init` first.") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    config = parse_config(raw)
    if not config.hosts:
        raise ConfigError(f"{path} defines no [hosts.*] sections")
    return config


def parse_mem(text: str) -> int:
    """Turn '8G', '512M' or '2048' (MiB) into MiB."""
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([GgMm]?)[iI]?[Bb]?", text.strip())
    if not match:
        raise ValueError(f"cannot read memory size {text!r}; try 8G or 2048M")
    amount = float(match.group(1))
    return int(amount * 1024) if match.group(2).lower() == "g" else int(amount)


def normalize_cwd(path: str, home: str | None = None) -> str:
    """Store paths under your home as ~/..., because usernames differ per server."""
    home = (home or str(Path.home())).rstrip("/")
    if path == home:
        return "~"
    if path.startswith(home + "/"):
        return "~/" + path[len(home) + 1 :]
    return path
