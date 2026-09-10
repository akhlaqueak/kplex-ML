#!/bin/bash
#SBATCH --job-name=Heur
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --partition=medium
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=%x.out
#SBATCH --error=%x.err
""":"
module load OpenMPI
module load mpi4py

set -euo pipefail
cp ../UMKPEX .

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
SCRIPT_PATH="${SUBMIT_DIR}/mpi_ctcp.py"
export MPI_DYNAMIC_BASEDIR="$SUBMIT_DIR"
cd "$SUBMIT_DIR"

collect_only=0
for arg in "$@"; do
    case "$arg" in
        --collect-only|--postprocess-only)
            collect_only=1
            ;;
    esac
done

if [[ "$collect_only" -eq 1 ]]; then
    exec python3 -u "$SCRIPT_PATH" "$@"
fi

SECONDS=0
echo "Job started at $(date)"
srun --mpi=pmix --cpu-bind=cores python3 -u "$SCRIPT_PATH" "$@"
echo "Job finished at $(date)"
echo "Job completed in $((SECONDS/3600))h $((SECONDS%3600/60))m"
exit $?
":"""


from mpi4py import MPI
import os
import re
import shlex
import subprocess
from pathlib import Path
import sys

TAG_READY = 1
TAG_TASK = 2
TAG_DONE = 3
TAG_STOP = 4

BR="S-Br".split()
ALGO="UMKPEX".split()
HEUR="kPEX-Heur".split()
CTCP="CTCP CF-CTCP".split()
# BR="Binary-Br".split()
# ALGO="kPEX".split()
RESULT_WIDTH = 6

TASK_TIMEOUT = 1800
TIMEOUT_GRACE = 30
DS_LOC = "/home/akhlaque.ak@gmail.com/graphs/large-collection"
OUTFILE = f"output-{os.environ.get('SLURM_JOB_NAME', 'local')}.txt"
LOG_DIR = Path("logs")
SCRIPT_DIR = Path(os.environ.get("MPI_DYNAMIC_BASEDIR", Path(__file__).resolve().parent)).resolve()
DS_FILE = (SCRIPT_DIR / "ds.txt").resolve()
RESUME_LOG = None
SKIP_TIMEOUT_LOGS = False


def parse_ntasks_per_node() -> int | None:
    raw = os.environ.get("SLURM_NTASKS_PER_NODE")
    if not raw:
        return None
    match = re.match(r"(\d+)", raw)
    return int(match.group(1)) if match else None


def get_local_rank() -> int:
    for key in ("SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID"):
        raw = os.environ.get(key)
        if raw is not None:
            try:
                return int(raw)
            except ValueError:
                pass
    return 0



def ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_outfile(cmd: str) -> str:
    args = shlex.split(cmd)

    if len(args) < 2:
        raise ValueError(f"Invalid command format: {cmd}")

    base = os.path.basename(args[1])
    outfile = os.path.splitext(base)[0]

    for arg in args[2:]:
        if not arg.startswith("-"):
            outfile += f"_{arg}"

    outfile = outfile.replace(" ", "_")
    return str(LOG_DIR / f"{outfile}.log")


def get_cmd(ds, k, br, heur, ct):
    switches={
        "UMKP": f"-algorithm UMKP+ -bounding None -branching {br}",
        "UMKP+": f"-algorithm UMKP+ -bounding AltRB -branching {br}",
        "kPEX": f"-branching {br}"
    }
    cmd = f"UMKPEX {DS_LOC}/{ds}.bin {k} -branching {br} -ctcp {ct}"
    return cmd.split()

def canonicalize_cmd(cmd: str) -> str:
    return " ".join(shlex.quote(x) for x in shlex.split(cmd))


def load_completed_tasks(log_path: str | None):
    if not log_path:
        return set()

    path = Path(log_path)
    if not path.is_file():
        print(f"[RESUME] Log file not found, ignoring: {path}", flush=True)
        return set()

    completed = set()
    pattern = re.compile(r"\b(SUCCESS|TIMEOUT)\b\s*::?\s*(.+)$")

    with path.open() as fh:
        for raw_line in fh:
            line = raw_line.strip()
            match = pattern.search(line)
            if not match:
                continue
            state, cmd = match.groups()
            if "UMKPEX " not in cmd:
                continue
            completed.add(canonicalize_cmd(cmd))

    print(f"[RESUME] Loaded {len(completed)} completed tasks from {path}", flush=True)
    return completed


def log_has_timeout(log_path: str) -> bool:
    path = Path(log_path)
    if not path.is_file():
        return False

    try:
        with path.open() as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if "TIMEOUT " in line or "TIMEOUT ::" in line:
                    return True
    except Exception:
        return False

    return False


def generate_tasks():
    tasks = []
    completed = load_completed_tasks(RESUME_LOG)

    with open(DS_FILE) as f:
        for line in f:
            if not line.strip():
                continue

            ds, k = line.strip().split()

            for heur in HEUR:
                for br in BR:
                    for ct in CTCP:
                        cmd = get_cmd(ds, k, br, heur, ct)
                        task = " ".join(shlex.quote(x) for x in cmd)
                        if task in completed:
                            continue
                        if SKIP_TIMEOUT_LOGS and log_has_timeout(get_outfile(task)):
                            continue
                        tasks.append(task)
    return tasks


def extract_result_values(path: str):
    try:
        with open(path) as fh:
            lines = fh.readlines()
    except Exception:
        return None

    result_line = ""
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith(">>"):
            result_line = stripped
            break

    if not result_line:
        return None

    parts = result_line.split()
    values = []
    idx = 0

    while idx < len(parts):
        token = parts[idx]
        if token.endswith(":") and idx + 1 < len(parts):
            values.append(parts[idx + 1])
            idx += 2
            continue
        idx += 1

    return values or None


def run_task(cmd: str):
    rank = MPI.COMM_WORLD.Get_rank()
    ensure_log_dir()

    outfile = get_outfile(cmd)

    args = shlex.split(cmd)
    if not args:
        print(f"[rank {rank}] EMPTY TASK", flush=True)
        return 0, outfile, "EMPTY"

    print(f"[rank {rank}] START :: {cmd}", flush=True)
    exe = args[0]

    if not os.path.isabs(exe) and os.path.exists(exe):
        exe = f"./{exe}"

    run_cmd = [
        "timeout",
        "--signal=TERM",
        f"--kill-after={TIMEOUT_GRACE}",
        str(TASK_TIMEOUT),
        "stdbuf",
        "-oL",
        "-eL",
        exe,
        *args[1:],
    ]

    print(f"[rank {rank}] EXEC: {run_cmd}", flush=True)

    try:
        with open(outfile, "w") as f:
            f.write(f"COMMAND {cmd}\n")
            f.flush()
            proc = subprocess.run(
                run_cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False,
            )
            rc = proc.returncode
    except Exception as e:
        print(f"[rank {rank}] EXCEPTION: {e}", flush=True)
        return -1, outfile, "EXCEPTION"

    if rc == 0:
        print(f"[rank {rank}] SUCCESS :: {cmd}", flush=True)
        return rc, outfile, "SUCCESS"
    elif rc == -7:
        print(f"[rank {rank}] SIGBUS :: {cmd}", flush=True)
        return rc, outfile, "SIGBUS"
    elif rc == 124:
        with open(outfile, "a") as f:
            f.write(f"\nTIMEOUT {cmd}\n")
        print(f"[rank {rank}] TIMEOUT :: {cmd}", flush=True)
        return rc, outfile, "TIMEOUT"
    elif rc == 137:
        with open(outfile, "a") as f:
            f.write(f"\nKILLED {cmd}\n")
        print(f"[rank {rank}] KILLED :: {cmd}", flush=True)
        return rc, outfile, "KILLED"

    with open(outfile, "a") as f:
        f.write(f"\nERROR rc={rc}\n")
    print(f"[rank {rank}] FAILED :: {cmd} (rc={rc})", flush=True)
    return rc, outfile, "FAILED"


def assign_available_workers(ready_workers, phase_name, phase_tasks, next_idx, comm, inflight, retry_half_local_limit):
    made_progress = True
    while made_progress:
        made_progress = False
        for src in list(ready_workers.keys()):
            if next_idx >= len(phase_tasks):
                break
            meta = ready_workers[src]
            eligible = True
            if phase_name == "retry" and retry_half_local_limit is not None:
                eligible = meta["local_rank"] < retry_half_local_limit
            if not eligible:
                continue
            task = phase_tasks[next_idx]
            print(f"[MASTER] Assigning {phase_name} task {next_idx} to rank {src}", flush=True)
            comm.send(task, dest=src, tag=TAG_TASK)
            del ready_workers[src]
            next_idx += 1
            inflight += 1
            made_progress = True
    return next_idx, inflight


def master(comm, size):
    initial_tasks = generate_tasks()
    retry_tasks = []
    retry_seen = set()
    phase_name = "initial"
    phase_tasks = initial_tasks
    next_idx = 0
    stopped = 0
    nworkers = size - 1
    inflight = 0
    ready_workers = {}
    retry_half_local_limit = None
    ntasks_per_node = parse_ntasks_per_node()
    if ntasks_per_node:
        retry_half_local_limit = max(1, ntasks_per_node // 2)

    print(f"[MASTER] Loaded {len(initial_tasks)} tasks", flush=True)

    status = MPI.Status()

    while stopped < nworkers:
        msg = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        src = status.Get_source()
        tag = status.Get_tag()

        if tag == TAG_READY:
            ready_workers[src] = msg or {"local_rank": 0, "node": "unknown"}
            next_idx, inflight = assign_available_workers(
                ready_workers, phase_name, phase_tasks, next_idx, comm, inflight, retry_half_local_limit
            )
            phase_complete = next_idx >= len(phase_tasks) and inflight == 0
            if phase_complete and phase_name == "initial":
                phase_name = "retry"
                phase_tasks = retry_tasks
                next_idx = 0
                print(f"[MASTER] Now retrying the -7 tasks ({len(retry_tasks)} total)", flush=True)
                next_idx, inflight = assign_available_workers(
                    ready_workers, phase_name, phase_tasks, next_idx, comm, inflight, retry_half_local_limit
                )
                phase_complete = next_idx >= len(phase_tasks) and inflight == 0
            if phase_complete:
                if src in ready_workers:
                    comm.send(None, dest=src, tag=TAG_STOP)
                    del ready_workers[src]
                    stopped += 1

        elif tag == TAG_DONE:
            task, rc, outfile, state = msg
            inflight -= 1
            print(f"[MASTER] [rank {src}] {state}: {task}", flush=True)
            if phase_name == "initial" and rc == -7:
                canon = canonicalize_cmd(task)
                if canon not in retry_seen:
                    retry_seen.add(canon)
                    retry_tasks.append(task)
            next_idx, inflight = assign_available_workers(
                ready_workers, phase_name, phase_tasks, next_idx, comm, inflight, retry_half_local_limit
            )
            phase_complete = next_idx >= len(phase_tasks) and inflight == 0
            if phase_complete and phase_name == "initial":
                phase_name = "retry"
                phase_tasks = retry_tasks
                next_idx = 0
                print(f"[MASTER] Now retrying the -7 tasks ({len(retry_tasks)} total)", flush=True)
                next_idx, inflight = assign_available_workers(
                    ready_workers, phase_name, phase_tasks, next_idx, comm, inflight, retry_half_local_limit
                )
                phase_complete = next_idx >= len(phase_tasks) and inflight == 0
            if phase_complete:
                for worker_src in list(ready_workers.keys()):
                    comm.send(None, dest=worker_src, tag=TAG_STOP)
                    del ready_workers[worker_src]
                    stopped += 1

    print("[MASTER] All workers stopped", flush=True)


def worker(comm, rank):
    local_rank = get_local_rank()
    node = os.environ.get("SLURMD_NODENAME") or os.environ.get("HOSTNAME") or "unknown"
    while True:
        comm.send({"local_rank": local_rank, "node": node}, dest=0, tag=TAG_READY)

        status = MPI.Status()
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == TAG_STOP or task is None:
            print(f"[rank {rank}] STOP received", flush=True)
            comm.send(None, dest=0, tag=TAG_STOP)
            break

        print(f"[rank {rank}] Received task: {task}", flush=True)
        rc, outfile, state = run_task(task)
        comm.send((task, rc, outfile, state), dest=0, tag=TAG_DONE)


def collect_results():
    print("[POST] Collecting results...", flush=True)

    with open(OUTFILE, "w") as out:
        with open(DS_FILE) as f:
            for line in f:
                if not line.strip():
                    continue

                ds, k = line.strip().split()

                for heur in HEUR:
                    for br in BR:
                        for ct in CTCP:
                            cmd = get_cmd(ds, k, br, heur, ct)

                            fname = get_outfile(" ".join(shlex.quote(x) for x in cmd))
                            values=None
                            if os.path.isfile(fname):
                                values = extract_result_values(fname)
                            if not values:
                                values = ["x"] * RESULT_WIDTH
                                
                            out.write(" ".join(values) + " ")

                out.write("\n")

    print(f"[POST] Wrote {OUTFILE}", flush=True)

def parse_args(argv):
    collect_only = any(arg in {"--collect-only", "--postprocess-only"} for arg in argv[1:])
    resume_log = None
    skip_timeout_logs = False

    idx = 1
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--resume-log" and idx + 1 < len(argv):
            resume_log = argv[idx + 1]
            idx += 2
            continue
        if arg.startswith("--resume-log="):
            resume_log = arg.split("=", 1)[1]
        if arg == "--skip-timeout-logs":
            skip_timeout_logs = True
        idx += 1

    return collect_only, resume_log, skip_timeout_logs


def main():
    global RESUME_LOG, SKIP_TIMEOUT_LOGS
    collect_only, resume_log, skip_timeout_logs = parse_args(sys.argv)
    RESUME_LOG = resume_log or os.environ.get("RESUME_LOG")
    SKIP_TIMEOUT_LOGS = skip_timeout_logs or os.environ.get("SKIP_TIMEOUT_LOGS") == "1"

    if collect_only:
        collect_results()
        return

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        master(comm, size)
    else:
        worker(comm, rank)

    if rank == 0:
        collect_results()


if __name__ == "__main__":
    main()
