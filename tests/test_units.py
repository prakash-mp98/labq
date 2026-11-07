import pytest

from labq.config import ConfigError, normalize_cwd, parse_config, parse_mem
from labq.remote import MARK, parse_probe
from labq.scheduler import describe_failure
from labq.transport import LocalTransport, SshTransport


@pytest.mark.parametrize(
    "text,expected",
    [("8G", 8192), ("8g", 8192), ("512M", 512), ("2048", 2048), ("1.5G", 1536), ("8GiB", 8192)],
)
def test_parse_mem(text, expected):
    assert parse_mem(text) == expected


@pytest.mark.parametrize("text", ["lots", "", "-1G", "8T"])
def test_parse_mem_rejects(text):
    with pytest.raises(ValueError):
        parse_mem(text)


def test_cwd_under_home_becomes_portable():
    assert normalize_cwd("/home/me/work", "/home/me") == "~/work"
    assert normalize_cwd("/home/me", "/home/me") == "~"
    assert normalize_cwd("/data/x", "/home/me") == "/data/x"
    assert normalize_cwd("/home/mexico/x", "/home/me") == "/home/mexico/x"


def test_config_catches_typos():
    with pytest.raises(ConfigError, match="poll_secs"):
        parse_config({"scheduler": {"poll_secs": 5}})
    with pytest.raises(ConfigError, match="gpu"):
        parse_config({"hosts": {"a": {"ssh": "a", "gpu": [0]}}})


@pytest.mark.parametrize(
    "entry",
    [{}, {"ssh": ""}, {"ssh": "-oProxyCommand=evil"}, {"ssh": "a", "gpus": []},
     {"ssh": "a", "gpus": [-1]}, {"ssh": "a", "gpus": "0"}, {"ssh": "a", "setup": 3}],
)
def test_config_rejects_bad_hosts(entry):
    with pytest.raises(ConfigError):
        parse_config({"hosts": {"a": entry}})


def test_config_rejects_bad_numbers():
    with pytest.raises(ConfigError):
        parse_config({"scheduler": {"poll_seconds": 0}})
    with pytest.raises(ConfigError):
        parse_config({"alerts": {"stuck_job_minutes": -1}})


def probe_text(gpus="0, GPU-0, 10, 16000, 0", apps="", jobs="", end=True):
    text = f"{MARK} GPUS\n{gpus}\n{MARK} APPS\n{apps}\n{MARK} JOBS\n{jobs}\n"
    return text + (f"{MARK} END\n" if end else "")


def test_probe_parses_gpu_and_processes():
    probe = parse_probe(probe_text(apps="GPU-0, 11\nGPU-0, 12"))
    assert probe.reachable and not probe.gpu_error
    gpu = probe.gpus[0]
    assert (gpu.index, gpu.mem_used, gpu.mem_total, gpu.util, gpu.processes) == (0, 10, 16000, 0, 2)


def test_truncated_probe_is_never_trusted():
    probe = parse_probe(probe_text(jobs="1-a-1 alive", end=False))
    assert not probe.reachable and not probe.jobs


def test_noise_before_markers_is_ignored():
    probe = parse_probe("Welcome to the server!\n" + probe_text())
    assert probe.reachable and len(probe.gpus) == 1


def test_na_metrics_are_kept_as_unknown():
    probe = parse_probe(probe_text(gpus="0, GPU-0, 10, 16000, [N/A]"))
    assert probe.gpus[0].util is None


@pytest.mark.parametrize("marker", ["!ERROR", "!MISSING"])
def test_smi_problems_are_reported(marker):
    assert parse_probe(probe_text(gpus=marker)).gpu_error


def test_probe_reads_job_states():
    jobs = "1-a-1 exited 3\n2-b-1 alive\n3-c-1 dead\n4-d-1 launching\n5-e-1 absent"
    states = parse_probe(probe_text(jobs=jobs)).jobs
    assert states["1-a-1"].exit_code == 3
    assert [states[k].status for k in ("2-b-1", "3-c-1", "4-d-1", "5-e-1")] == [
        "alive", "dead", "launching", "absent",
    ]


def test_ssh_never_prompts_and_cannot_take_target_as_option():
    argv = SshTransport("server1").argv()
    assert "BatchMode=yes" in argv
    assert argv[0] == "ssh"
    assert argv[argv.index("--") + 1] == "server1"


def test_ssh_255_is_a_transport_failure_but_script_errors_are_not():
    transport = SshTransport("server1")
    assert transport.is_transport_failure(255)
    assert not transport.is_transport_failure(1)
    assert not LocalTransport().is_transport_failure(255)


@pytest.mark.parametrize(
    "code,tail,expected",
    [
        (1, "torch.cuda.OutOfMemoryError: CUDA out of memory", "GPU out of memory"),
        (137, "", "killed with SIGKILL, possibly out of host memory"),
        (126, "", "could not start (see log)"),
        (2, "boom", "exit code 2"),
    ],
)
def test_failure_descriptions(code, tail, expected):
    assert describe_failure(code, tail) == expected
