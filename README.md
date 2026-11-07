# labq

[![tests](https://github.com/prakash-mp98/labq/actions/workflows/tests.yml/badge.svg)](https://github.com/prakash-mp98/labq/actions/workflows/tests.yml)

**One GPU job queue across the SSH servers you already have.** No root, nothing to install on the servers, and alerts on your phone.

If you share lab GPUs, you probably know the routine: SSH into each server, run `nvidia-smi`, start your job in `tmux`, and check back later to find it crashed hours ago. labq replaces that with one queue:

```console
$ labq submit --name resnet-sweep -- python train.py --lr 1e-4
Queued job 1 (runs in ~/project).

$ labq hosts
HOST     GPU  MEMORY           UTIL  STATUS
server1  0    2259/16376 MiB   100%  running labq job 1
server2  0    11024/16376 MiB  97%   11024 MiB in use
```

Your job starts on the first GPU that is really free, on any server. It keeps running if your wifi drops or your laptop dies.

## Why use it

- **One queue, many servers.** A job waits in one place and takes whichever server frees up first.
- **Jobs don't depend on your connection.** Each one runs in its own session on the server. When the connection returns, labq picks up the result.
- **It stays out of other people's way.** A GPU counts as free only if it looks free several checks in a row. If labq can't read a GPU's state, it treats the GPU as busy.
- **Phone alerts through [ntfy](https://ntfy.sh).** You hear about finished jobs, crashes, GPU out-of-memory errors, jobs stuck at 0% GPU, idle GPUs, and lost servers.
- **Nothing to set up on the servers.** They need only `sh`, `nvidia-smi`, and your SSH login, so no admin is involved.

## Install

labq needs Python 3.10+ on the machine that runs the scheduler.

```bash
git clone https://github.com/prakash-mp98/labq.git
pip install ./labq
```

## Quick start

**1. Make sure SSH works without a password.** labq never prompts, so key-based login is required:

```bash
ssh-copy-id server1
```

**2. Add your servers:**

```bash
labq init     # writes ~/.labq/config.toml
```

```toml
[hosts.server1]
ssh = "server1"          # anything you could pass to the ssh command
setup = "source ~/miniconda3/etc/profile.d/conda.sh"   # runs before each job

[hosts.server2]
ssh = "me@server2.example.org"
gpus = [0]               # optional: only ever use these GPUs

[alerts]
ntfy_topic = "a-long-random-name"
```

**3. Check them, then start the scheduler in `tmux`:**

```bash
labq hosts
labq test-alert
labq daemon
```

**4. Submit work from another terminal:**

```bash
labq submit --name exp1 --mem 8G --retries 1 -- python train.py --seed 1
labq submit --host server2 --cwd '~/project' -- 'conda activate torch && python train.py'
```

The code must already be on the servers at that path. Paths under your home are stored as `~/...`, so they still work when your username differs between servers.

## Commands

| Command | What it does |
|---------|--------------|
| `labq submit [options] -- CMD` | Add a job to the queue |
| `labq list`, `labq list --all` | Show active jobs, or all jobs |
| `labq show ID` | Show one job in full |
| `labq logs ID [-f] [-n N]` | Print or follow a job's output |
| `labq cancel ID...` | Remove queued jobs, or stop running ones |
| `labq retry ID` | Queue a finished job again |
| `labq hosts` | Check each server and explain each GPU's status |
| `labq daemon` | Run the scheduler |
| `labq test-alert` | Send a test alert to your phone |

Options for `submit`:

| Option | Meaning |
|--------|---------|
| `--name` | Short name, shown in alerts |
| `--host` | Allowed server, repeatable (default: any) |
| `--gpus N` | GPUs needed, all on one server (default: 1) |
| `--mem 8G` | Minimum free GPU memory |
| `--cwd DIR` | Where to run (default: the current directory) |
| `--priority N` | Higher runs first |
| `--retries N` | Automatic retries after a failure |

Each job gets `CUDA_VISIBLE_DEVICES` set to its assigned GPUs, with the device order matched to `nvidia-smi`.

## When a GPU counts as free

All of these must hold, on `free_polls_required` checks in a row (default 2, 30 seconds apart):

| Check | Default |
|-------|---------|
| No labq job already assigned to it | |
| `nvidia-smi` lists no processes on it | |
| Memory used is at or below the limit | 1024 MiB |
| Utilisation is at or below the limit | 20% |
| Every reading could be parsed | |

The first check matters because a job that has just started may not have allocated memory yet, so the GPU would still look idle. `labq hosts` shows which check is blocking each GPU.

## Alerts

| Alert | When |
|-------|------|
| Finished, Failed, Retrying | A job ends, with the cause when known, such as `GPU out of memory` |
| Lost | A job vanished without an exit code, for example after a reboot |
| Possibly stuck | A job is running, but its GPU has been near 0% for `stuck_job_minutes` |
| GPU idle | Your queue is empty and a GPU has been free for `idle_gpu_minutes` |
| Cannot reach, Back online | A server with your jobs on it stopped answering, then returned |

Each condition alerts once per episode. If a send fails, labq tries again later.

**Privacy:** anyone who knows an ntfy.sh topic name can read it, so choose a long random one. Alerts include job names, server aliases, and exit codes. They don't include commands, and they include log lines only if you set `include_log_tail = true`. Jobs without `--name` appear as `job-12`.

## Where to run the scheduler

Running jobs never need the scheduler, but **queued jobs only start, and alerts only fire, while it is running.** If you close your laptop, running jobs continue, while the queue waits and alerts are delayed until it reopens.

For a queue that keeps moving, run the scheduler in `tmux` on a machine that stays on, such as one of the servers. List that server with `ssh = "local"`, and submit from there:

```bash
ssh server1 labq submit -- python train.py
```

The scheduler reads its config at startup, so restart it after editing the config.

## How it works

For each job, labq writes a small wrapper to `~/.labq/jobs/<id>/` on the server and starts it with `setsid`. The wrapper records the output, the process ID, and finally the exit code. To follow a job, labq only has to read those files, so the SSH connection can come and go.

- **A lost reply can't start a job twice.** The launch uses `mkdir`, which can succeed only once for each attempt.
- **Unclear answers change nothing.** Only a complete reply from a server can change a job's state, so a dropped connection is never mistaken for a dead job.
- **A reused process ID can't fool it.** labq checks that the process is really the job's wrapper, which matters after a reboot.
- **Cancelling covers the whole process tree.** This includes DataLoader workers. A job is marked cancelled only once its wrapper is confirmed gone.
- **Nothing is left behind.** When a job ends, its leftover background processes are stopped so they can't keep holding the GPU.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

The test suite runs the real shell scripts and real processes against a fake `nvidia-smi`, so it needs no GPU. Before release, labq was also run end to end over real SSH, against a local `sshd`, with PyTorch on a real GPU. That run included stopping `sshd` and force-killing the scheduler in the middle of training.

**Not yet tested:** multiple physical GPUs, which have only been simulated, and long-term use on a busy multi-user server. Bug reports from real labs are very welcome.

## Limitations

- **Linux and NVIDIA only.**
- **One job per GPU.** Sharing a GPU between jobs isn't supported.
- **No code sync.** Your code and data must already be on the server.
- **Queues are per user.** labq watches GPU use, not other people's queues, so two people can still start jobs at the same moment. It's a good neighbour, not an enforcer.
- **Cancelling can miss some processes.** A process that moves itself to a new session with `setsid` escapes the cancel.
- **Job folders aren't cleaned up.** Logs stay in `~/.labq/jobs/` until you delete them.
- **Lower-priority jobs can overtake.** A smaller job may start ahead of a larger job that's still waiting, which could delay the larger one for a long time.

## Similar tools

Based on each project's documentation in November 2026. Please open an issue if something here is out of date.

| Tool | Good for | Difference from labq |
|------|----------|----------------------|
| [Slurm](https://slurm.schedmd.com/) | Real clusters | Needs an admin to install it on the servers |
| [gpuc](https://github.com/brendanlong/gpu-coordinator) | Personal GPUs and rented RunPod machines, with per-run code copies and preemption | Each server has its own queue, and its write-up doesn't mention phone alerts |
| [SumsJob](https://github.com/lululxvi/sumsjob) | Sending a job to a free server | Retries in a loop instead of keeping a queue, and needs `gpustat` on each server |
| [qgpu](https://github.com/noamteyssier/qgpu) | Spreading a batch of jobs across nodes | Its documentation doesn't mention job alerts |
| [task-spooler (GPU fork)](https://github.com/yao-sh/task-spooler) | One machine | Doesn't span servers |
| [SkyPilot](https://docs.skypilot.co/) | Clouds and larger fleets | A much larger system, with an API server and a runtime on your machines |

If you need code sync or preemption, gpuc may suit you better. If your lab can run Slurm, use Slurm.

## License

[MIT](LICENSE)
