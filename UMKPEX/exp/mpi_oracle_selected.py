#!/bin/bash
#SBATCH --job-name=kPEX-oracle-selected-oriented
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

export HEUR="${HEUR:-Both-Heur}"
export BR="${BR:-Pivot-Br}"
export CT="${CT:-CTCP}"
export ALGO="${ALGO:-kPEX}"
export BOUND="${BOUND:-None}"

resolve_script_path() {
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        local cmd
        cmd="$(scontrol show job "${SLURM_JOB_ID}" | awk -F= '/ Command=/{print $2; exit}')"
        if [[ -n "$cmd" && -f "$cmd" ]]; then
            realpath "$cmd"
            return
        fi
    fi
    realpath "$0"
}

SCRIPT_PATH="$(resolve_script_path)"
SUBMIT_DIR="$(dirname "$SCRIPT_PATH")"
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
":"""

"""Run selected Oracle graphs at a fixed set of k values with dynamic MPI scheduling."""

from mpi4py import MPI
import os
import re
import shlex
import subprocess
from pathlib import Path
import sys


TAG_READY, TAG_TASK, TAG_DONE, TAG_STOP = 1, 2, 3, 4
ROW_IDS = [
    195, 963, 1573, 2021, 2189, 2519, 2633, 2740, 3689,
    4185, 21963, 22670, 24475, 24532, 25141, 25647, 25812, 26295,
]
K_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
BR = "S-Br R-Br Pivot-Br Binary-Br".split()
ALGO = "kPEX UMKP UMKP+".split()
TASK_TIMEOUT = 1800
TIMEOUT_GRACE = 30
RESULT_WIDTH = 6
OUTFILE = f"output-{os.environ.get('SLURM_JOB_NAME', 'local')}.txt"
LOG_DIR = Path("logs")
SCRIPT_DIR = Path(os.environ.get("MPI_DYNAMIC_BASEDIR", Path(__file__).resolve().parent)).resolve()
ORACLE_DIR = (SCRIPT_DIR / "/home/akhlaque.ak@gmail.com/graphs/oracle/oracle-bin-all").resolve()
RESUME_LOG = None
SKIP_TIMEOUT_LOGS = False


def get_local_rank():
    for key in ("SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID"):
        try:
            return int(os.environ[key])
        except (KeyError, ValueError):
            pass
    return 0


def parse_ntasks_per_node():
    raw = os.environ.get("SLURM_NTASKS_PER_NODE")
    match = re.match(r"(\d+)", raw or "")
    return int(match.group(1)) if match else None


def ensure_log_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_outfile(cmd):
    args = shlex.split(cmd)
    if len(args) < 2:
        raise ValueError(f"Invalid command format: {cmd}")
    name = os.path.splitext(os.path.basename(args[1]))[0]
    for arg in args[2:]:
        if not arg.startswith("-"):
            name += f"_{arg}"
    return str(LOG_DIR / f"{name.replace(' ', '_')}.log")


def get_cmd(row_id, k, br, algo):
    switches = {
        "UMKP": f"-algorithm UMKP+ -bounding None -branching {br}",
        "UMKP+": f"-algorithm UMKP+ -bounding AltRB -branching {br}",
        "kPEX": f"-branching {br}",
    }
    graph_path = ORACLE_DIR / f"oracle_{row_id}.bin"
    return f"UMKPEX {graph_path} {k} {switches[algo]} -framework oriented".split()


def canonicalize_cmd(cmd):
    return " ".join(shlex.quote(x) for x in shlex.split(cmd))


def load_completed_tasks(log_path):
    if not log_path:
        return set()
    path = Path(log_path)
    if not path.is_file():
        print(f"[RESUME] Log file not found, ignoring: {path}", flush=True)
        return set()
    completed = set()
    pattern = re.compile(r"\b(SUCCESS|TIMEOUT)\b\s*::?\s*(.+)$")
    with path.open() as fh:
        for line in fh:
            match = pattern.search(line.strip())
            if match and "UMKPEX " in match.group(2):
                completed.add(canonicalize_cmd(match.group(2)))
    print(f"[RESUME] Loaded {len(completed)} completed tasks from {path}", flush=True)
    return completed


def log_has_timeout(log_path):
    try:
        with Path(log_path).open() as fh:
            return any("TIMEOUT " in line or "TIMEOUT ::" in line for line in fh)
    except Exception:
        return False


def generate_tasks():
    missing = [row_id for row_id in ROW_IDS if not (ORACLE_DIR / f"oracle_{row_id}.bin").is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} Oracle binary file(s) in {ORACLE_DIR}: {missing}"
        )
    tasks, completed = [], load_completed_tasks(RESUME_LOG)
    for row_id in ROW_IDS:
        for k in K_VALUES:
            for algo in ALGO:
                for br in BR:
                    task = " ".join(shlex.quote(x) for x in get_cmd(row_id, k, br, algo))
                    if task not in completed and not (
                        SKIP_TIMEOUT_LOGS and log_has_timeout(get_outfile(task))
                    ):
                        tasks.append(task)
    return tasks


def extract_result_values(path):
    try:
        lines = Path(path).read_text().splitlines()
    except Exception:
        return None
    result_line = next((line.strip() for line in reversed(lines) if line.strip().startswith(">>")), "")
    if not result_line:
        return None
    parts, values, index = result_line.split(), [], 0
    while index < len(parts):
        if parts[index].endswith(":") and index + 1 < len(parts):
            values.append(parts[index + 1])
            index += 2
        else:
            index += 1
    return values or None


def run_task(cmd):
    rank = MPI.COMM_WORLD.Get_rank()
    ensure_log_dir()
    outfile, args = get_outfile(cmd), shlex.split(cmd)
    if not args:
        return 0, outfile, "EMPTY"
    print(f"[rank {rank}] START :: {cmd}", flush=True)
    exe = f"./{args[0]}" if not os.path.isabs(args[0]) and os.path.exists(args[0]) else args[0]
    run_cmd = ["timeout", "--signal=TERM", f"--kill-after={TIMEOUT_GRACE}", str(TASK_TIMEOUT),
               "stdbuf", "-oL", "-eL", exe, *args[1:]]
    print(f"[rank {rank}] EXEC: {run_cmd}", flush=True)
    try:
        with open(outfile, "w") as fh:
            fh.write(f"COMMAND {cmd}\n")
            fh.flush()
            rc = subprocess.run(run_cmd, stdout=fh, stderr=subprocess.STDOUT, check=False).returncode
    except Exception as exc:
        print(f"[rank {rank}] EXCEPTION: {exc}", flush=True)
        return -1, outfile, "EXCEPTION"
    if rc == 0:
        state = "SUCCESS"
    elif rc == -7:
        state = "SIGBUS"
    elif rc == 124:
        state = "TIMEOUT"
        with open(outfile, "a") as fh:
            fh.write(f"\nTIMEOUT {cmd}\n")
    elif rc == 137:
        state = "KILLED"
        with open(outfile, "a") as fh:
            fh.write(f"\nKILLED {cmd}\n")
    else:
        state = "FAILED"
        with open(outfile, "a") as fh:
            fh.write(f"\nERROR rc={rc}\n")
    suffix = f" (rc={rc})" if state in {"SIGBUS", "FAILED"} else ""
    print(f"[rank {rank}] {state} :: {cmd}{suffix}", flush=True)
    return rc, outfile, state


def assign_available_workers(ready_workers, phase_name, phase_tasks, next_idx, comm, inflight, retry_limit):
    while True:
        assigned = False
        for rank in list(ready_workers):
            if next_idx >= len(phase_tasks):
                return next_idx, inflight
            if phase_name == "retry" and retry_limit is not None and ready_workers[rank]["local_rank"] >= retry_limit:
                continue
            print(f"[MASTER] Assigning {phase_name} task {next_idx} to rank {rank}", flush=True)
            comm.send(phase_tasks[next_idx], dest=rank, tag=TAG_TASK)
            del ready_workers[rank]
            next_idx, inflight, assigned = next_idx + 1, inflight + 1, True
        if not assigned:
            return next_idx, inflight


def master(comm, size):
    initial_tasks, retry_tasks, retry_seen = generate_tasks(), [], set()
    phase_name, phase_tasks, next_idx = "initial", initial_tasks, 0
    stopped, inflight, ready_workers = 0, 0, {}
    nworkers = size - 1
    ntasks = parse_ntasks_per_node()
    retry_limit = max(1, ntasks // 2) if ntasks else None
    print(f"[MASTER] Loaded {len(initial_tasks)} tasks", flush=True)
    status = MPI.Status()
    while stopped < nworkers:
        msg = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        src, tag = status.Get_source(), status.Get_tag()
        if tag == TAG_READY:
            ready_workers[src] = msg or {"local_rank": 0, "node": "unknown"}
        elif tag == TAG_DONE:
            task, rc, _, state = msg
            inflight -= 1
            print(f"[MASTER] [rank {src}] {state}: {task}", flush=True)
            if phase_name == "initial" and rc == -7 and canonicalize_cmd(task) not in retry_seen:
                retry_seen.add(canonicalize_cmd(task))
                retry_tasks.append(task)
        else:
            continue
        next_idx, inflight = assign_available_workers(
            ready_workers, phase_name, phase_tasks, next_idx, comm, inflight, retry_limit
        )
        if next_idx >= len(phase_tasks) and inflight == 0 and phase_name == "initial":
            phase_name, phase_tasks, next_idx = "retry", retry_tasks, 0
            print(f"[MASTER] Now retrying the -7 tasks ({len(retry_tasks)} total)", flush=True)
            next_idx, inflight = assign_available_workers(
                ready_workers, phase_name, phase_tasks, next_idx, comm, inflight, retry_limit
            )
        if next_idx >= len(phase_tasks) and inflight == 0:
            for rank in list(ready_workers):
                comm.send(None, dest=rank, tag=TAG_STOP)
                del ready_workers[rank]
                stopped += 1
    print("[MASTER] All workers stopped", flush=True)


def worker(comm, rank):
    details = {"local_rank": get_local_rank(), "node": os.environ.get("SLURMD_NODENAME", "unknown")}
    while True:
        comm.send(details, dest=0, tag=TAG_READY)
        status = MPI.Status()
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == TAG_STOP or task is None:
            print(f"[rank {rank}] STOP received", flush=True)
            comm.send(None, dest=0, tag=TAG_STOP)
            return
        print(f"[rank {rank}] Received task: {task}", flush=True)
        comm.send((task, *run_task(task)), dest=0, tag=TAG_DONE)


def collect_results():
    print("[POST] Collecting results...", flush=True)
    with open(OUTFILE, "w") as out:
        for row_id in ROW_IDS:
            for k in K_VALUES:
                for algo in ALGO:
                    for br in BR:
                        task = " ".join(shlex.quote(x) for x in get_cmd(row_id, k, br, algo))
                        values = extract_result_values(get_outfile(task))
                        out.write(" ".join(values or ["x"] * RESULT_WIDTH) + " ")
                out.write("\n")
    print(f"[POST] Wrote {OUTFILE}", flush=True)


def parse_args(argv):
    collect_only = any(arg in {"--collect-only", "--postprocess-only"} for arg in argv[1:])
    resume_log, skip_timeout_logs, index = None, False, 1
    while index < len(argv):
        arg = argv[index]
        if arg == "--resume-log" and index + 1 < len(argv):
            resume_log, index = argv[index + 1], index + 2
            continue
        if arg.startswith("--resume-log="):
            resume_log = arg.split("=", 1)[1]
        if arg == "--skip-timeout-logs":
            skip_timeout_logs = True
        index += 1
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
    if comm.Get_rank() == 0:
        master(comm, comm.Get_size())
        collect_results()
    else:
        worker(comm, comm.Get_rank())


if __name__ == "__main__":
    main()
