from __future__ import annotations

import json
import stat
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from labq import cli, db
from labq.config import AlertConfig, parse_config
from labq.remote import probe_script
from labq.transport import SshTransport


class FakeNtfy:
    def __init__(self):
        self.received: list[dict] = []
        self.headers: list = []
        self.fail = False
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                if outer.fail:
                    self.send_response(503)
                else:
                    outer.received.append(json.loads(body))
                    outer.headers.append(self.headers)
                    self.send_response(200)
                self.end_headers()

            def log_message(self, *_):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def ntfy():
    fake = FakeNtfy()
    yield fake
    fake.close()


def test_alert_reaches_ntfy_with_the_right_shape(make_rig, ntfy):
    rig = make_rig(alerts=AlertConfig(ntfy_topic="t", ntfy_server=ntfy.url, idle_gpu_minutes=0))
    job_id = rig.submit("true", name="train")
    rig.tick_until(job_id, db.SUCCEEDED)
    assert ntfy.received[0]["topic"] == "t"
    assert ntfy.received[0]["title"] == "Finished: train on a"
    assert ntfy.received[0]["priority"] == 3


def test_token_is_sent_only_when_configured(make_rig, ntfy):
    rig = make_rig(alerts=AlertConfig(
        ntfy_topic="t", ntfy_server=ntfy.url, ntfy_token="sekret", idle_gpu_minutes=0))
    job_id = rig.submit("true")
    rig.tick_until(job_id, db.SUCCEEDED)
    assert ntfy.headers[0]["Authorization"] == "Bearer sekret"


def test_failed_sends_retry_in_order_and_never_block_scheduling(make_rig, ntfy):
    rig = make_rig(alerts=AlertConfig(ntfy_topic="t", ntfy_server=ntfy.url, idle_gpu_minutes=0))
    ntfy.fail = True
    first = rig.submit("true", name="one")
    rig.tick_until(first, db.SUCCEEDED)
    second = rig.submit("true", name="two")
    rig.tick_until(second, db.SUCCEEDED)  # the queue keeps moving
    assert not ntfy.received
    ntfy.fail = False
    rig.scheduler.tick()
    assert [a["title"] for a in ntfy.received] == ["Finished: one on a", "Finished: two on a"]
    rig.scheduler.tick()
    assert len(ntfy.received) == 2  # delivered exactly once


def test_no_topic_means_nothing_is_sent(make_rig, ntfy):
    rig = make_rig(alerts=AlertConfig(ntfy_topic="", ntfy_server=ntfy.url, idle_gpu_minutes=0))
    job_id = rig.submit("true")
    rig.tick_until(job_id, db.SUCCEEDED)
    assert not ntfy.received
    assert not db.unsent_alerts(rig.conn)


def test_unreachable_ntfy_server_does_not_raise(make_rig):
    rig = make_rig(alerts=AlertConfig(
        ntfy_topic="t", ntfy_server="http://127.0.0.1:9", idle_gpu_minutes=0))
    job_id = rig.submit("true")
    rig.tick_until(job_id, db.SUCCEEDED)


def test_very_old_alert_is_dropped_rather_than_sent(make_rig, ntfy):
    rig = make_rig(alerts=AlertConfig(ntfy_topic="t", ntfy_server=ntfy.url, idle_gpu_minutes=0))
    db.add_alert(rig.conn, "stale", "x", 3, [], time.time() - 90000)
    rig.scheduler.tick()
    assert not ntfy.received
    assert not db.unsent_alerts(rig.conn)


def test_alerts_keep_commands_and_logs_private_by_default(make_rig, ntfy):
    rig = make_rig(alerts=AlertConfig(ntfy_topic="t", ntfy_server=ntfy.url, idle_gpu_minutes=0))
    job_id = rig.submit("echo API_KEY=hunter2; exit 1")
    rig.tick_until(job_id, db.FAILED)
    assert ntfy.received
    assert "hunter2" not in json.dumps(ntfy.received)
    assert ntfy.received[0]["title"] == f"Failed: job-{job_id} on a"


def test_log_tail_only_when_opted_in(make_rig, ntfy):
    rig = make_rig(alerts=AlertConfig(
        ntfy_topic="t", ntfy_server=ntfy.url, idle_gpu_minutes=0, include_log_tail=True))
    job_id = rig.submit("echo the-last-line; exit 1")
    rig.tick_until(job_id, db.FAILED)
    assert "the-last-line" in ntfy.received[0]["message"]


# ---- real ssh code path, with a stand-in ssh binary ---------------------------------


def test_ssh_transport_runs_through_the_ssh_binary(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "ssh"
    fake.write_text(f'#!/bin/sh\necho "$@" > {tmp_path}/args\nshift 9\nexec sh -c "$2"\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    result = SshTransport("server1").run("echo works\n")
    assert result.stdout.strip() == "works" and not result.transport_failed
    assert "BatchMode=yes" in (tmp_path / "args").read_text()


def test_ssh_failure_is_reported_as_unreachable(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "ssh"
    fake.write_text("#!/bin/sh\necho 'connect: timed out' >&2\nexit 255\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    result = SshTransport("server1").run(probe_script([]))
    assert result.transport_failed


# ---- command line ------------------------------------------------------------------


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LABQ_HOME", str(tmp_path / "state"))
    assert cli.main(["init"]) == 0
    config = tmp_path / "state" / "config.toml"
    config.write_text('[hosts.a]\nssh = "local"\n[hosts.b]\nssh = "local"\ngpus = [0]\n')
    return tmp_path / "state"


def test_sample_config_is_valid(tmp_path, monkeypatch):
    monkeypatch.setenv("LABQ_HOME", str(tmp_path))
    assert cli.main(["init"]) == 0
    from labq.config import load_config

    config = load_config()
    assert list(config.hosts) == ["server1"]
    assert cli.main(["init"]) == 1  # refuses to overwrite


def test_config_file_is_private(tmp_path, monkeypatch):
    monkeypatch.setenv("LABQ_HOME", str(tmp_path))
    cli.main(["init"])
    assert stat.S_IMODE((tmp_path / "config.toml").stat().st_mode) == 0o600


def test_submit_quotes_separate_arguments_and_keeps_shell_strings(home, capsys):
    assert cli.main(["submit", "--cwd", "/tmp", "--", "python", "a b.py", "--x=1"]) == 0
    assert cli.main(["submit", "--cwd", "/tmp", "--", "echo a | wc -l"]) == 0
    conn = db.connect(home / "labq.db")
    assert db.get_job(conn, 1)["command"] == "python 'a b.py' --x=1"
    assert db.get_job(conn, 2)["command"] == "echo a | wc -l"


@pytest.mark.parametrize(
    "args",
    [
        ["submit"],
        ["submit", "--"],
        ["submit", "--host", "nope", "--", "x"],
        ["submit", "--gpus", "0", "--", "x"],
        ["submit", "--retries", "-1", "--", "x"],
        ["submit", "--mem", "lots", "--", "x"],
        ["submit", "--host", "b", "--gpus", "2", "--", "x"],
    ],
)
def test_submit_rejects_bad_requests(home, args):
    assert cli.main(args) == 2


def test_list_show_cancel_and_retry(home, capsys):
    cli.main(["submit", "--name", "demo", "--cwd", "/tmp", "--", "true"])
    capsys.readouterr()
    assert cli.main(["list"]) == 0
    assert "demo" in capsys.readouterr().out
    assert cli.main(["show", "1"]) == 0
    assert "demo" in capsys.readouterr().out
    assert cli.main(["retry", "1"]) == 1  # still queued
    assert cli.main(["cancel", "1"]) == 0
    assert cli.main(["retry", "1"]) == 0
    conn = db.connect(home / "labq.db")
    job = db.get_job(conn, 1)
    assert (job["state"], job["attempt"]) == (db.QUEUED, 2)


def test_commands_on_missing_jobs_fail_cleanly(home):
    assert cli.main(["show", "99"]) == 1
    assert cli.main(["logs", "99"]) == 1
    assert cli.main(["cancel", "99"]) == 1
    assert cli.main(["retry", "99"]) == 1


def test_logs_before_start_explains_itself(home, capsys):
    cli.main(["submit", "--cwd", "/tmp", "--", "true"])
    capsys.readouterr()
    assert cli.main(["logs", "1"]) == 1
    assert "not started" in capsys.readouterr().err


def test_missing_config_gives_a_helpful_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LABQ_HOME", str(tmp_path / "empty"))
    assert cli.main(["hosts"]) == 2
    assert "labq init" in capsys.readouterr().err


def test_parse_config_accepts_full_example():
    config = parse_config({
        "scheduler": {"poll_seconds": 5},
        "alerts": {"ntfy_topic": "x"},
        "hosts": {"s1": {"ssh": "s1", "gpus": [0, 1], "setup": "true"}},
    })
    assert config.hosts["s1"].gpus == (0, 1)
