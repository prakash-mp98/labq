"""End-to-end scheduler behaviour using real wrappers and processes."""

from __future__ import annotations

import os
import signal
import time

from labq import db
from labq.config import AlertConfig, Host


def read(rig, job_id, name="log"):
    server = rig.servers[rig.job(job_id)["host"]]
    return (server.job_dir(rig.job(job_id)) / name).read_text()


def wrapper_pid(rig, job_id, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            return int(read(rig, job_id, "pid"))
        except (FileNotFoundError, ValueError):
            time.sleep(0.02)
    raise AssertionError("wrapper never wrote its pid")


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers signal 0 but is no longer running.
    with open(f"/proc/{pid}/stat") as handle:
        return handle.read().rsplit(")", 1)[1].split()[0] != "Z"


# ---- running jobs ---------------------------------------------------------------


def test_job_runs_and_succeeds(rig):
    job_id = rig.submit("echo hello from $LABQ_JOB_ID", name="demo")
    job = rig.tick_until(job_id, db.SUCCEEDED)
    assert job["exit_code"] == 0
    assert read(rig, job_id).strip() == f"hello from {job_id}"
    assert rig.alert_titles() == ["Finished: demo on a"]


def test_job_sees_only_its_gpu_and_matching_device_order(make_rig):
    rig = make_rig()
    rig.servers["a"].set_gpus([(0, 5000, 90), (1, 0, 0)])
    job_id = rig.submit('echo "$CUDA_VISIBLE_DEVICES $CUDA_DEVICE_ORDER"')
    rig.tick_until(job_id, db.SUCCEEDED)
    assert read(rig, job_id).strip() == "1 PCI_BUS_ID"


def test_cwd_expands_to_the_home_of_the_server_not_the_submitter(rig):
    (rig.servers["a"].home / "work dir").mkdir()
    job_id = rig.submit("pwd", cwd="~/work dir")
    rig.tick_until(job_id, db.SUCCEEDED)
    assert read(rig, job_id).strip() == str(rig.servers["a"].home / "work dir")


def test_missing_directory_fails_with_a_clear_message(rig):
    job_id = rig.submit("echo should not run", cwd="~/nope", name="x")
    job = rig.tick_until(job_id, db.FAILED)
    assert job["exit_code"] == 126
    assert job["reason"] == "could not start (see log)"
    log = read(rig, job_id)
    assert "cannot cd" in log and "should not run" not in log


def test_host_setup_runs_before_the_command(make_rig):
    rig = make_rig(hosts={"a": Host("a", "unused", setup="export GREETING=hi")})
    job_id = rig.submit('echo "$GREETING"')
    rig.tick_until(job_id, db.SUCCEEDED)
    assert read(rig, job_id).strip() == "hi"


def test_command_with_shell_metacharacters_is_preserved(rig):
    command = """echo "$((1 + 2))" 'single $HOME' | tr a-z A-Z; echo 'EOF'"""
    job_id = rig.submit(command)
    rig.tick_until(job_id, db.SUCCEEDED)
    assert read(rig, job_id).splitlines() == ["3 SINGLE $HOME", "EOF"]


# ---- when a GPU counts as free ------------------------------------------------


def assert_stays_queued(rig, job_id, ticks=3):
    for _ in range(ticks):
        rig.scheduler.tick()
    assert rig.job(job_id)["state"] == db.QUEUED


def test_busy_memory_blocks_dispatch(rig):
    rig.servers["a"].set_gpus([(0, 5000, 0)])
    assert_stays_queued(rig, rig.submit("true"))


def test_busy_utilisation_blocks_dispatch(rig):
    rig.servers["a"].set_gpus([(0, 0, 95)])
    assert_stays_queued(rig, rig.submit("true"))


def test_other_process_blocks_dispatch_even_with_low_memory(rig):
    rig.servers["a"].set_apps(["GPU-0"])
    assert_stays_queued(rig, rig.submit("true"))


def test_broken_nvidia_smi_blocks_dispatch(rig):
    rig.servers["a"].break_smi()
    assert_stays_queued(rig, rig.submit("true"))


def test_unreadable_metrics_block_dispatch(rig):
    rig.servers["a"].set_raw_gpus("0, GPU-0, 10, 16000, [N/A]\n")
    assert_stays_queued(rig, rig.submit("true"))


def test_garbled_gpu_output_blocks_every_gpu_on_that_host(rig):
    rig.servers["a"].set_raw_gpus("0, GPU-0, 10, 16000, 0\nthis is not csv\n")
    assert_stays_queued(rig, rig.submit("true"))


def test_gpu_must_stay_free_for_consecutive_polls(make_rig):
    rig = make_rig(scheduler={"free_polls_required": 3})
    job_id = rig.submit("true")
    rig.scheduler.tick()
    rig.scheduler.tick()
    assert rig.job(job_id)["state"] == db.QUEUED
    # A burst of use resets the count, so a brief lull does not start a job.
    rig.servers["a"].set_gpus([(0, 8000, 90)])
    rig.scheduler.tick()
    rig.servers["a"].set_gpus([(0, 10, 0)])
    rig.scheduler.tick()
    rig.scheduler.tick()
    assert rig.job(job_id)["state"] == db.QUEUED
    rig.tick_until(job_id, db.SUCCEEDED)


def test_new_job_holds_its_gpu_before_allocating_memory(rig):
    # nvidia-smi still reports the GPU as idle while the first job is loading.
    first = rig.submit("sleep 30")
    second = rig.submit("true")
    rig.tick_until(first, db.RUNNING)
    for _ in range(3):
        rig.scheduler.tick()
    assert rig.job(second)["state"] == db.QUEUED
    db.cancel(rig.conn, first, time.time())
    rig.tick_until(first, db.CANCELLED)
    rig.tick_until(second, db.SUCCEEDED)


def test_excluded_gpu_is_never_used(make_rig):
    rig = make_rig(hosts={"a": Host("a", "unused", gpus=(1,))})
    rig.servers["a"].set_gpus([(0, 0, 0), (1, 0, 0)])
    job_id = rig.submit("true")
    rig.tick_until(job_id, db.SUCCEEDED)
    assert rig.gpu_ids(job_id) == [1]


def test_memory_requirement_is_respected(rig):
    rig.servers["a"].set_gpus([(0, 0, 0)], total=8000)
    too_big = rig.submit("true", min_mem_mib=12000)
    fits = rig.submit("true", min_mem_mib=4000)
    rig.tick_until(fits, db.SUCCEEDED)
    assert rig.job(too_big)["state"] == db.QUEUED


# ---- one queue across servers ---------------------------------------------------


def test_jobs_spread_across_servers_from_one_queue(make_rig):
    rig = make_rig(("a", "b"))
    jobs = [rig.submit("sleep 30") for _ in range(3)]
    rig.scheduler.tick()
    placed = {rig.job(j)["host"] for j in jobs[:2]}
    assert placed == {"a", "b"}
    assert rig.job(jobs[2])["state"] == db.QUEUED
    for job_id in jobs:
        db.cancel(rig.conn, job_id, time.time())
    rig.tick_until(jobs[0], db.CANCELLED)
    rig.tick_until(jobs[1], db.CANCELLED)


def test_job_moves_to_whichever_server_frees_up(make_rig):
    rig = make_rig(("a", "b"))
    rig.servers["a"].set_gpus([(0, 8000, 90)])
    job_id = rig.submit("true")
    rig.tick_until(job_id, db.SUCCEEDED)
    assert rig.job(job_id)["host"] == "b"


def test_host_constraint_is_respected(make_rig):
    rig = make_rig(("a", "b"))
    rig.servers["b"].set_gpus([(0, 8000, 90)])
    job_id = rig.submit("true", hosts=["b"])
    assert_stays_queued(rig, job_id)
    rig.servers["b"].set_gpus([(0, 0, 0)])
    rig.tick_until(job_id, db.SUCCEEDED)
    assert rig.job(job_id)["host"] == "b"


def test_multi_gpu_job_stays_on_one_server(make_rig):
    rig = make_rig(("a", "b"))
    rig.servers["b"].set_gpus([(0, 0, 0), (1, 0, 0)])
    job_id = rig.submit('echo "$CUDA_VISIBLE_DEVICES"', gpus=2)
    rig.tick_until(job_id, db.SUCCEEDED)
    assert rig.job(job_id)["host"] == "b"
    assert read(rig, job_id).strip() == "0,1"


def test_priority_runs_first(rig):
    low = rig.submit("sleep 30")
    high = rig.submit("sleep 30", priority=5)
    rig.scheduler.tick()
    assert rig.job(high)["state"] == db.RUNNING
    assert rig.job(low)["state"] == db.QUEUED
    db.cancel(rig.conn, low, time.time())
    db.cancel(rig.conn, high, time.time())
    rig.tick_until(high, db.CANCELLED)


def test_dead_host_does_not_stop_the_others(make_rig):
    rig = make_rig(("a", "b"))
    rig.servers["a"].down = True
    job_id = rig.submit("true")
    rig.tick_until(job_id, db.SUCCEEDED)
    assert rig.job(job_id)["host"] == "b"


# ---- surviving disconnection and restarts --------------------------------------


def test_job_survives_lost_connection_and_is_collected_afterwards(rig):
    marker = rig.servers["a"].home / "marker"
    job_id = rig.submit(f"sleep 1; touch {marker}; exit 3", name="x")
    rig.tick_until(job_id, db.RUNNING)
    rig.servers["a"].down = True
    deadline = time.time() + 10
    while not marker.exists() and time.time() < deadline:
        rig.scheduler.tick()
        assert rig.job(job_id)["state"] == db.RUNNING  # never guessed to be dead
        time.sleep(0.05)
    assert marker.exists(), "job should finish while the host is unreachable"
    rig.servers["a"].down = False
    assert rig.tick_until(job_id, db.FAILED)["exit_code"] == 3


def test_job_survives_scheduler_restart(rig):
    job_id = rig.submit("sleep 1; echo done")
    rig.tick_until(job_id, db.RUNNING)
    rig.scheduler = rig.new_scheduler()  # all in-memory state is gone
    rig.tick_until(job_id, db.SUCCEEDED)
    assert read(rig, job_id).strip() == "done"


def test_unconfirmed_launch_is_not_started_twice(rig):
    counter = rig.servers["a"].home / "starts"
    job_id = rig.submit(f"echo start >> {counter}; sleep 1")
    transport = rig.transports["a"]
    transport.lose_launch_reply = True
    rig.scheduler.tick()  # the job launched, but the reply was lost
    assert rig.job(job_id)["state"] == db.STARTING
    transport.lose_launch_reply = False
    rig.tick_until(job_id, db.SUCCEEDED)
    assert counter.read_text().count("start") == 1
    assert rig.job(job_id)["attempt"] == 1


def test_launch_that_never_arrived_is_requeued_without_spending_a_retry(rig):
    job_id = rig.submit("true", retries=1)
    transport = rig.transports["a"]
    original = transport.run

    def lost_launch(script, timeout=45):
        if "setsid" in script:
            from labq.transport import Result

            return Result(255, "", "connection reset", True)
        return original(script, timeout)

    transport.run = lost_launch
    rig.scheduler.tick()
    assert rig.job(job_id)["state"] == db.STARTING
    transport.run = original
    rig.tick_until(job_id, db.SUCCEEDED)
    job = rig.job(job_id)
    assert (job["attempt"], job["retries_left"]) == (2, 1)


def test_hard_killed_job_is_lost_not_succeeded(rig):
    job_id = rig.submit("sleep 30", name="x")
    rig.tick_until(job_id, db.RUNNING)
    pid = wrapper_pid(rig, job_id)
    os.killpg(pid, signal.SIGKILL)
    job = rig.tick_until(job_id, db.LOST)
    assert job["exit_code"] is None
    assert rig.alert_titles()[-1] == "Lost: x on a"


def test_reused_pid_is_not_mistaken_for_the_job(rig):
    job_id = rig.submit("sleep 1")
    rig.tick_until(job_id, db.RUNNING)
    pid = wrapper_pid(rig, job_id)
    time.sleep(0.2)
    server = rig.servers["a"]
    directory = server.job_dir(rig.job(job_id))
    os.killpg(pid, signal.SIGKILL)
    # Simulates a reboot: the pid file now points at an unrelated live process.
    (directory / "pid").write_text(f"{os.getpid()}\n")
    assert rig.tick_until(job_id, db.LOST)


def test_vanished_job_directory_is_lost(rig):
    import shutil

    job_id = rig.submit("sleep 2")
    rig.tick_until(job_id, db.RUNNING)
    pid = wrapper_pid(rig, job_id)
    os.killpg(pid, signal.SIGKILL)
    shutil.rmtree(rig.servers["a"].job_dir(rig.job(job_id)))
    assert rig.tick_until(job_id, db.LOST)["reason"] == "job directory disappeared"


def test_broken_nvidia_smi_does_not_stop_following_running_jobs(rig):
    job_id = rig.submit("sleep 1")
    rig.tick_until(job_id, db.RUNNING)
    rig.servers["a"].break_smi()
    assert rig.tick_until(job_id, db.SUCCEEDED)["exit_code"] == 0


def test_removing_a_host_from_config_does_not_freeze_the_queue(make_rig):
    rig = make_rig(("a", "b"), alerts=AlertConfig(idle_gpu_minutes=0, stuck_job_minutes=1))
    orphan = rig.submit("sleep 30", hosts=["a"])
    rig.tick_until(orphan, db.RUNNING)
    db.cancel(rig.conn, orphan, time.time())  # a cancel is also pending for it
    pid = wrapper_pid(rig, orphan)

    remaining = {"b": Host("b", "unused")}
    transports = {"b": rig.transports["b"]}
    from dataclasses import replace

    from labq.scheduler import Scheduler

    rig.scheduler = Scheduler(replace(rig.config, hosts=remaining), rig.conn, transports)
    job_id = rig.submit("true")
    rig.tick_until(job_id, db.SUCCEEDED)
    assert rig.job(job_id)["host"] == "b"
    os.killpg(pid, signal.SIGKILL)


# ---- retries and failures -------------------------------------------------------


def test_retries_then_gives_up(rig):
    job_id = rig.submit("exit 7", retries=2, name="x")
    job = rig.tick_until(job_id, db.FAILED)
    assert (job["attempt"], job["retries_left"], job["exit_code"]) == (3, 0, 7)
    assert rig.alert_titles() == ["Retrying: x on a", "Retrying: x on a", "Failed: x on a"]


def test_retry_recovers_flaky_job(rig):
    flag = rig.servers["a"].home / "flag"
    job_id = rig.submit(f"[ -f {flag} ] || {{ touch {flag}; exit 1; }}", retries=1)
    job = rig.tick_until(job_id, db.SUCCEEDED)
    assert job["attempt"] == 2


def test_gpu_out_of_memory_is_named(rig):
    job_id = rig.submit("echo 'CUDA out of memory. Tried to allocate 2 GiB'; exit 1")
    assert rig.tick_until(job_id, db.FAILED)["reason"] == "GPU out of memory"


# ---- cancelling -----------------------------------------------------------------


def test_cancel_queued_job_never_runs(rig):
    rig.servers["a"].set_gpus([(0, 8000, 90)])
    marker = rig.servers["a"].home / "ran"
    job_id = rig.submit(f"touch {marker}")
    assert db.cancel(rig.conn, job_id, time.time()) == "cancelled"
    rig.servers["a"].set_gpus([(0, 0, 0)])
    rig.scheduler.tick()
    rig.scheduler.tick()
    assert rig.job(job_id)["state"] == db.CANCELLED
    assert not marker.exists()


def test_cancel_kills_the_whole_process_tree(rig):
    child_pid_file = rig.servers["a"].home / "child"
    # A background child, like a DataLoader worker, must die with the job.
    job_id = rig.submit(f"sleep 60 & echo $! > {child_pid_file}; wait")
    rig.tick_until(job_id, db.RUNNING)
    deadline = time.time() + 5
    while not child_pid_file.exists() and time.time() < deadline:
        time.sleep(0.02)
    child = int(child_pid_file.read_text())
    wrapper = wrapper_pid(rig, job_id)
    assert alive(child) and alive(wrapper)

    assert db.cancel(rig.conn, job_id, time.time()) == "requested"
    rig.tick_until(job_id, db.CANCELLED)
    time.sleep(0.2)
    assert not alive(child)
    assert not alive(wrapper)


def test_cancel_is_not_reported_until_the_job_is_really_gone(rig):
    from labq.transport import Result

    job_id = rig.submit("sleep 30")
    rig.tick_until(job_id, db.RUNNING)
    db.cancel(rig.conn, job_id, time.time())
    transport = rig.transports["a"]
    original = transport.run
    for unclear_answer in ("alive", "", "something unexpected"):
        transport.run = lambda script, timeout=45, answer=unclear_answer: (
            Result(0, answer + "\n", "") if "kill -s KILL" in script else original(script, timeout)
        )
        rig.scheduler.tick()
        assert rig.job(job_id)["state"] == db.RUNNING
    transport.run = original
    rig.tick_until(job_id, db.CANCELLED)
    assert not alive(wrapper_pid(rig, job_id))


def test_cancel_kills_a_job_that_ignores_sigterm(rig):
    pid_file = rig.servers["a"].home / "stubborn"
    job_id = rig.submit(f"trap '' TERM; echo $$ > {pid_file}; sleep 60 & wait")
    rig.tick_until(job_id, db.RUNNING)
    deadline = time.time() + 5
    while not pid_file.exists() and time.time() < deadline:
        time.sleep(0.02)
    stubborn = int(pid_file.read_text())
    db.cancel(rig.conn, job_id, time.time())
    rig.tick_until(job_id, db.CANCELLED, timeout=30)
    time.sleep(0.2)
    assert not alive(stubborn)


def test_finished_job_leaves_no_background_processes_on_the_gpu(rig):
    child_pid_file = rig.servers["a"].home / "worker"
    job_id = rig.submit(f"sleep 60 & echo $! > {child_pid_file}; exit 0")
    rig.tick_until(job_id, db.SUCCEEDED)
    time.sleep(0.2)
    assert not alive(int(child_pid_file.read_text()))


def test_cancel_does_not_retry_or_count_as_failure(rig):
    job_id = rig.submit("sleep 30", retries=3)
    rig.tick_until(job_id, db.RUNNING)
    db.cancel(rig.conn, job_id, time.time())
    job = rig.tick_until(job_id, db.CANCELLED)
    assert job["attempt"] == 1
    assert not [t for t in rig.alert_titles() if t.startswith(("Failed", "Retrying"))]


def test_cancel_waits_for_an_unreachable_host(rig):
    job_id = rig.submit("sleep 30")
    rig.tick_until(job_id, db.RUNNING)
    db.cancel(rig.conn, job_id, time.time())
    rig.servers["a"].down = True
    rig.scheduler.tick()
    assert rig.job(job_id)["state"] == db.RUNNING
    rig.servers["a"].down = False
    rig.tick_until(job_id, db.CANCELLED)


def test_cancel_finished_job_reports_it(rig):
    job_id = rig.submit("true")
    rig.tick_until(job_id, db.SUCCEEDED)
    assert db.cancel(rig.conn, job_id, time.time()) == "finished"
    assert db.cancel(rig.conn, 999, time.time()) == "missing"


# ---- condition alerts -----------------------------------------------------------


def test_stuck_job_alerts_once_and_rearms_when_active(make_rig):
    rig = make_rig(alerts=AlertConfig(stuck_job_minutes=30, idle_gpu_minutes=0))
    job_id = rig.submit("sleep 30", name="x")
    start = time.time()
    rig.scheduler.tick(start)
    server = rig.servers["a"]
    server.set_gpus([(0, 8000, 0)])
    rig.scheduler.tick(start + 60)
    rig.scheduler.tick(start + 29 * 60)
    assert not [t for t in rig.alert_titles() if "stuck" in t]
    rig.scheduler.tick(start + 31 * 60)
    rig.scheduler.tick(start + 32 * 60)
    assert [t for t in rig.alert_titles() if "stuck" in t] == ["Possibly stuck: x on a"]
    # It recovers, then stalls again: that is a new episode.
    server.set_gpus([(0, 8000, 90)])
    rig.scheduler.tick(start + 33 * 60)
    server.set_gpus([(0, 8000, 0)])
    rig.scheduler.tick(start + 34 * 60)
    rig.scheduler.tick(start + 65 * 60)
    assert len([t for t in rig.alert_titles() if "stuck" in t]) == 2
    db.cancel(rig.conn, job_id, time.time())
    rig.tick_until(job_id, db.CANCELLED)


def test_busy_gpu_never_raises_a_stuck_alert(make_rig):
    rig = make_rig(alerts=AlertConfig(stuck_job_minutes=1, idle_gpu_minutes=0))
    job_id = rig.submit("sleep 30")
    start = time.time()
    rig.scheduler.tick(start)
    rig.servers["a"].set_gpus([(0, 8000, 80)])
    rig.scheduler.tick(start + 10)
    rig.scheduler.tick(start + 600)
    assert not rig.alert_titles()
    db.cancel(rig.conn, job_id, time.time())
    rig.tick_until(job_id, db.CANCELLED)


def test_idle_gpu_alerts_once_only_when_queue_is_empty(make_rig):
    rig = make_rig(alerts=AlertConfig(idle_gpu_minutes=60))
    start = time.time()
    rig.scheduler.tick(start)
    rig.scheduler.tick(start + 59 * 60)
    assert not rig.alert_titles()
    rig.scheduler.tick(start + 61 * 60)
    rig.scheduler.tick(start + 120 * 60)
    assert rig.alert_titles() == ["GPU idle: a"]


def test_someone_elses_job_is_not_called_idle(make_rig):
    rig = make_rig(alerts=AlertConfig(idle_gpu_minutes=60))
    rig.servers["a"].set_gpus([(0, 8000, 90)])
    start = time.time()
    rig.scheduler.tick(start)
    rig.scheduler.tick(start + 120 * 60)
    assert not rig.alert_titles()


def test_host_down_alerts_only_with_jobs_there_then_recovers(make_rig):
    rig = make_rig(alerts=AlertConfig(host_down_minutes=15, idle_gpu_minutes=0))
    server = rig.servers["a"]
    start = time.time()
    server.down = True
    rig.scheduler.tick(start)
    rig.scheduler.tick(start + 3600)
    assert not rig.alert_titles()  # nothing of ours there, so nothing to say

    server.down = False
    job_id = rig.submit("sleep 30")
    rig.scheduler.tick(start + 3700)
    server.down = True
    rig.scheduler.tick(start + 3800)
    rig.scheduler.tick(start + 3800 + 16 * 60)
    rig.scheduler.tick(start + 3800 + 17 * 60)
    assert rig.alert_titles() == ["Cannot reach a"]
    server.down = False
    rig.scheduler.tick(start + 3800 + 18 * 60)
    assert rig.alert_titles() == ["Cannot reach a", "Back online: a"]
    db.cancel(rig.conn, job_id, time.time())
    rig.tick_until(job_id, db.CANCELLED)
