#!/bin/bash
#SBATCH --job-name=kPEX-baselines
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

export HEUR="${HEUR:-Both-Heur}"
export BR="${BR:-Pivot-Br}"
export CT="${CT:-CTCP}"
export ALGO="${ALGO:-kPEX}"
export BOUND="${BOUND:-None}"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
SCRIPT_PATH="${SUBMIT_DIR}/$(basename "$0")"
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

BR="S-Br R-Br Pivot-Br Binary-Br".split()
ALGO="UMKP UMKP+".split()
# BR="Binary-Br".split()
ALGO="kPEX".split()
RESULT_WIDTH = 6

TASK_TIMEOUT = 1800
TIMEOUT_GRACE = 30
DS_LOC = "/home/akhlaque.ak@gmail.com/graphs/large-collection"
OUTFILE = f"output-{os.environ.get('SLURM_JOB_NAME', 'local')}.txt"
LOG_DIR = Path("logs")
SCRIPT_DIR = Path(os.environ.get("MPI_DYNAMIC_BASEDIR", Path(__file__).resolve().parent)).resolve()
DS_FILE = (SCRIPT_DIR / "../ds.txt").resolve()
RESUME_LOG = None



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


def get_cmd(ds, k, br, algo):
    switches={
        "UMKP": f"-algorithm UMKP+ -bounding None -branching {br}",
        "UMKP+": f"-algorithm UMKP+ -bounding AltRB -branching {br}",
        "kPEX": f"-branching {br}"
    }
    cmd = f"UMKPEX {DS_LOC}/{ds}.bin {k} "+switches[algo]
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


def generate_tasks():
    tasks = []
    completed = load_completed_tasks(RESUME_LOG)

    with open(DS_FILE) as f:
        for line in f:
            if not line.strip():
                continue

            ds, k = line.strip().split()

            for algo in ALGO:
                for br in BR:
                    cmd = get_cmd(ds, k, br, algo)
                    task = " ".join(shlex.quote(x) for x in cmd)
                    if task in completed:
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
        exe,
        *args[1:],
    ]

    print(f"[rank {rank}] EXEC: {run_cmd}", flush=True)

    try:
        with open(outfile, "w") as f:
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


def master(comm, size):
    tasks = generate_tasks()
    next_idx = 0
    stopped = 0
    nworkers = size - 1

    print(f"[MASTER] Loaded {len(tasks)} tasks", flush=True)

    status = MPI.Status()

    while stopped < nworkers:
        msg = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        src = status.Get_source()
        tag = status.Get_tag()

        if tag == TAG_READY:
            if next_idx < len(tasks):
                task = tasks[next_idx]
                print(f"[MASTER] Assigning task {next_idx} to rank {src}", flush=True)
                comm.send(task, dest=src, tag=TAG_TASK)
                next_idx += 1
            else:
                comm.send(None, dest=src, tag=TAG_STOP)

        elif tag == TAG_DONE:
            task, rc, outfile, state = msg
            print(f"[MASTER] [rank {src}] {state}: {task}", flush=True)

        elif tag == TAG_STOP:
            stopped += 1

    print("[MASTER] All workers stopped", flush=True)


def worker(comm, rank):
    while True:
        comm.send(None, dest=0, tag=TAG_READY)

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

                for algo in ALGO:
                    for br in BR:
                        cmd = get_cmd(ds, k, br, algo)

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

    idx = 1
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--resume-log" and idx + 1 < len(argv):
            resume_log = argv[idx + 1]
            idx += 2
            continue
        if arg.startswith("--resume-log="):
            resume_log = arg.split("=", 1)[1]
        idx += 1

    return collect_only, resume_log


def main():
    global RESUME_LOG
    collect_only, resume_log = parse_args(sys.argv)
    RESUME_LOG = resume_log or os.environ.get("RESUME_LOG")

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
