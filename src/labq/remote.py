"""The POSIX shell scripts labq runs on a host, and the parsing of their output.

Each job gets a directory, ~/.labq/jobs/<job_dir>/, holding:

    job.sh      the user's setup, cd and command
    run.sh      the wrapper that runs job.sh and records its exit code
    pid         the wrapper's pid, which is also its process group id
    log         stdout and stderr
    exit_code   written last, by rename, so it is never seen half-written

The wrapper runs in its own session, so nothing on the host depends on the SSH
connection or on the scheduler staying alive.
"""

from __future__ import annotations

import secrets
import shlex
from dataclasses import dataclass, field

MARK = "@@LABQ"
EXIT_ALREADY_LAUNCHED = 75


@dataclass(frozen=True)
class GpuInfo:
    index: int
    uuid: str
    mem_used: int | None
    mem_total: int | None
    util: int | None
    processes: int = 0


@dataclass(frozen=True)
class JobObservation:
    status: str  # exited | alive | dead | launching | absent
    exit_code: int | None = None


@dataclass
class Probe:
    reachable: bool
    error: str = ""
    gpu_error: str = ""
    gpus: list[GpuInfo] = field(default_factory=list)
    jobs: dict[str, JobObservation] = field(default_factory=dict)


def _job_path(job_dir: str) -> str:
    return f'"$HOME/.labq/jobs/"{shlex.quote(job_dir)}'


_ALIVE_CHECK = """\
is_alive() {
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null \\
    && tr '\\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "$1"
}
"""


def probe_script(job_dirs: list[str]) -> str:
    quoted = " ".join(shlex.quote(d) for d in job_dirs)
    return f"""\
{_ALIVE_CHECK}
echo "{MARK} GPUS"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu \\
    --format=csv,noheader,nounits 2>/dev/null || echo "!ERROR"
else
  echo "!MISSING"
fi
echo "{MARK} APPS"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null \\
    || echo "!ERROR"
fi
echo "{MARK} JOBS"
for d in {quoted}; do
  dir="$HOME/.labq/jobs/$d"
  if [ -f "$dir/exit_code" ]; then
    echo "$d exited $(cat "$dir/exit_code")"
  elif [ -f "$dir/pid" ]; then
    pid=$(cat "$dir/pid")
    if is_alive "$d"; then
      echo "$d alive"
    elif [ -f "$dir/exit_code" ]; then
      echo "$d exited $(cat "$dir/exit_code")"
    else
      echo "$d dead"
    fi
  elif [ -d "$dir" ]; then
    echo "$d launching"
  else
    echo "$d absent"
  fi
done
echo "{MARK} END"
"""


def _int_or_none(text: str) -> int | None:
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_probe(stdout: str) -> Probe:
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in stdout.splitlines():
        if line.startswith(MARK + " "):
            current = sections.setdefault(line.split()[1], [])
        elif current is not None and line.strip():
            current.append(line.strip())

    # Without the end marker the output was cut off, so nothing in it can be
    # trusted. In particular a missing job line must not read as a dead job.
    if "END" not in sections:
        return Probe(False, error="incomplete response from host")

    probe = Probe(True)

    process_counts: dict[str, int] = {}
    for line in sections.get("APPS", []):
        if line.startswith("!"):
            probe.gpu_error = "could not list GPU processes"
            continue
        uuid = line.split(",")[0].strip()
        process_counts[uuid] = process_counts.get(uuid, 0) + 1

    gpu_lines = sections.get("GPUS", [])
    if "!MISSING" in gpu_lines:
        probe.gpu_error = "nvidia-smi not found"
    elif "!ERROR" in gpu_lines:
        probe.gpu_error = "nvidia-smi failed"
    else:
        for line in gpu_lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 5 or not parts[0].isdigit():
                probe.gpu_error = f"could not read nvidia-smi line: {line!r}"
                continue
            probe.gpus.append(
                GpuInfo(
                    index=int(parts[0]),
                    uuid=parts[1],
                    mem_used=_int_or_none(parts[2]),
                    mem_total=_int_or_none(parts[3]),
                    util=_int_or_none(parts[4]),
                    processes=process_counts.get(parts[1], 0),
                )
            )

    for line in sections.get("JOBS", []):
        parts = line.split()
        if len(parts) >= 3 and parts[1] == "exited":
            probe.jobs[parts[0]] = JobObservation("exited", _int_or_none(parts[2]))
        elif len(parts) == 2 and parts[1] in {"alive", "dead", "launching", "absent"}:
            probe.jobs[parts[0]] = JobObservation(parts[1])
    return probe


def _cd_line(cwd: str) -> str:
    if cwd == "~":
        target = '"$HOME"'
    elif cwd.startswith("~/"):
        target = '"$HOME"/' + shlex.quote(cwd[2:])
    else:
        target = shlex.quote(cwd)
    failure = f"labq: cannot cd to {cwd}"
    return f"cd {target} || {{ echo {shlex.quote(failure)} >&2; exit 126; }}"


def launch_script(
    job_dir: str,
    job_id: int,
    command: str,
    cwd: str,
    gpu_ids: list[int],
    setup: str = "",
) -> str:
    """Script that starts a job. `mkdir` makes a repeated launch a harmless no-op."""
    path = _job_path(job_dir)
    delimiter = f"LABQ_{secrets.token_hex(8)}"
    visible = ",".join(str(g) for g in gpu_ids)
    return f"""\
set -e
mkdir -p "$HOME/.labq/jobs"
mkdir {path} 2>/dev/null || exit {EXIT_ALREADY_LAUNCHED}
dir={path}
cat > "$dir/job.sh" <<'{delimiter}'
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES={visible}
export PYTHONUNBUFFERED=1
export LABQ_JOB_ID={job_id}
{setup}
{_cd_line(cwd)}
{command}
{delimiter}
cat > "$dir/run.sh" <<'{delimiter}'
dir=$(dirname "$0")
echo $$ > "$dir/pid"
# Keep the wrapper alive through a group kill long enough to record the exit code.
trap : TERM INT HUP
if command -v bash >/dev/null 2>&1; then
  bash "$dir/job.sh" > "$dir/log" 2>&1 < /dev/null
else
  sh "$dir/job.sh" > "$dir/log" 2>&1 < /dev/null
fi
code=$?
# Stop anything the job left behind, such as DataLoader workers holding the GPU.
kill -s TERM -- "-$$" 2>/dev/null
echo "$code" > "$dir/exit_code.tmp"
mv "$dir/exit_code.tmp" "$dir/exit_code"
{delimiter}
setsid sh "$dir/run.sh" > /dev/null 2>&1 < /dev/null &
echo started
"""


def cancel_script(job_dir: str, grace_seconds: int = 10) -> str:
    return f"""\
{_ALIVE_CHECK}
d={shlex.quote(job_dir)}
dir="$HOME/.labq/jobs/$d"
if [ ! -d "$dir" ]; then echo absent; exit 0; fi
if [ -f "$dir/exit_code" ]; then echo finished; exit 0; fi
if [ ! -f "$dir/pid" ]; then echo launching; exit 0; fi
pid=$(cat "$dir/pid")
if ! is_alive "$d"; then echo finished; exit 0; fi
kill -s TERM -- "-$pid" 2>/dev/null
i=0
while [ "$i" -lt {grace_seconds} ] && is_alive "$d"; do
  sleep 1
  i=$((i + 1))
done
kill -s KILL -- "-$pid" 2>/dev/null
i=0
while [ "$i" -lt 3 ] && is_alive "$d"; do
  sleep 1
  i=$((i + 1))
done
# Only claim success once the job is verifiably gone.
if is_alive "$d"; then echo alive; else echo killed; fi
"""


def tail_script(job_dir: str, lines: int, follow: bool = False) -> str:
    path = _job_path(job_dir)
    if not follow:
        return f"tail -n {int(lines)} {path}/log 2>/dev/null\n"
    return f"""\
dir={path}
while [ ! -f "$dir/log" ] && [ ! -f "$dir/exit_code" ]; do sleep 1; done
if [ -f "$dir/pid" ] && [ ! -f "$dir/exit_code" ]; then
  exec tail -n {int(lines)} --pid="$(cat "$dir/pid")" -f "$dir/log"
fi
exec tail -n {int(lines)} "$dir/log"
"""
