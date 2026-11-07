"""Run shell scripts on a host, locally or through the system ssh binary.

Using the real ssh binary means ~/.ssh/config, keys, agents and ProxyJump all work
as they already do for the user. BatchMode stops ssh from ever prompting for a
password, so a misconfigured host fails fast rather than hanging the scheduler.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

from labq.config import LOCAL, Host

SSH_OPTIONS = [
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=2",
]


@dataclass(frozen=True)
class Result:
    returncode: int
    stdout: str
    stderr: str
    # True when we cannot tell whether the script ran (SSH failure or timeout).
    transport_failed: bool = False


class Transport:
    def argv(self) -> list[str]:
        raise NotImplementedError

    def env(self) -> dict[str, str] | None:
        return None

    def is_transport_failure(self, returncode: int) -> bool:
        return False

    def run(self, script: str, timeout: float = 45) -> Result:
        try:
            done = subprocess.run(
                self.argv(),
                input=script,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
                env=self.env(),
            )
        except subprocess.TimeoutExpired:
            return Result(255, "", f"timed out after {timeout:g}s", True)
        except OSError as exc:
            return Result(255, "", str(exc), True)
        return Result(
            done.returncode,
            done.stdout,
            done.stderr,
            self.is_transport_failure(done.returncode),
        )

    def stream(self, script: str) -> int:
        """Run a script with its output connected to this terminal."""
        process = subprocess.Popen(
            self.argv(), stdin=subprocess.PIPE, text=True, env=self.env()
        )
        try:
            assert process.stdin is not None
            process.stdin.write(script)
            process.stdin.close()
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            return 130


class LocalTransport(Transport):
    def __init__(self, env: dict[str, str] | None = None):
        self._env = {**os.environ, **env} if env else None

    def argv(self) -> list[str]:
        return ["sh", "-s"]

    def env(self) -> dict[str, str] | None:
        return self._env


class SshTransport(Transport):
    def __init__(self, target: str):
        self.target = target

    def argv(self) -> list[str]:
        return ["ssh", *SSH_OPTIONS, "--", self.target, "sh -s"]

    def is_transport_failure(self, returncode: int) -> bool:
        # ssh reserves 255 for its own errors.
        return returncode == 255


def transport_for(host: Host) -> Transport:
    return LocalTransport() if host.ssh == LOCAL else SshTransport(host.ssh)
