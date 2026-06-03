#!/bin/bash
#SBATCH --job-name=kPEX
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --partition=short
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
SCRIPT_PATH="${SUBMIT_DIR}/baselines_exp.py"
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

cp -f ../UMKPEX .

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

import os
import shlex
import subprocess
import sys
from pathlib import Path

MPI = None

TAG_READY = 1
TAG_TASK = 2
TAG_DONE = 3
TAG_STOP = 4

TASK_TIMEOUT = 1800
TIMEOUT_GRACE = 30
DS_LOC = "/home/akhlaque.ak@gmail.com/graphs/large-collection"
OUTFILE = f"output-{os.environ.get('SLURM_JOB_NAME', 'local')}.txt"
LOG_DIR = Path("logs")
SCRIPT_DIR = Path(os.environ.get("MPI_DYNAMIC_BASEDIR", Path(__file__).resolve().parent)).resolve()
DS_FILE = (SCRIPT_DIR / "ds.txt").resolve()

ALGOS="kPEX".split()

switches={
    "UMKP-Imp": "-algorithm UMKP+ -bounding None -branching Adaptive-Br",
    "UMKP": "-algorithm UMKP+ -heuristic EGO-Heur -bounding None -branching Adaptive-Br",
    "kPEX-Imp": "-heuristic Both-Heur -branching Binary-Br",
    "kPEX": "-heuristic kPEX-Heur -ctcp CF-CTCP -branching Binary-Br"
}

def ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

def get_outfile(cmd: str) -> str:
    args = shlex.split(cmd)

    if not args:
        raise ValueError(f"Invalid command format: {cmd}")

    file_arg_idx = None
    for i, arg in enumerate(args[1:], start=1):
        if arg.startswith("-"):
            continue
        if arg.endswith(".bin") or "/" in arg:
            file_arg_idx = i
            break

    if file_arg_idx is None:
        raise ValueError(f"Could not detect dataset file in command: {cmd}")

    solver = os.path.basename(args[0])
    base = os.path.basename(args[file_arg_idx])
    outfile = f"{solver}_{os.path.splitext(base)[0]}"

    for arg in args[file_arg_idx + 1 :]:
        if not arg.startswith("-"):
            outfile += f"_{arg}"

    outfile = outfile.replace(" ", "_")
    return str(LOG_DIR / f"{outfile}.log")


def get_t_total(filepath):
    try:
        with open(filepath, "r") as f:
            for line in f:
                if not line.startswith(">>"):
                    if " kPEX time:" in line:
                        tail = line.split(" kPEX time:", 1)[1].strip()
                        if tail.endswith(" s"):
                            tail = tail[:-2].strip()
                        return float(tail)
                    continue
                return float(line.split()[-1])
    except (OSError, ValueError, IndexError):
        return None

    return None

def get_task(alg, ds, k):
    task=""
    if alg == "kPlexT" or alg=="kPlexT-forward":
        task = f"{alg} -g {DS_LOC}/{ds}.bin -k {k} -b"
    elif alg=="UMKPEX":
        task = f"{alg} {DS_LOC}/{ds}.bin {k} -algorithm UMKP+ -bounding None"
    elif alg=="UMKP-":
        task = f"{alg} {DS_LOC}/{ds}.bin {k} -algorithm UMKP+ -bounding None"
    else:
        task = f"{alg} {DS_LOC}/{ds}.bin {k}"
    return task
    
    # task=f"{alg} {DS_LOC}/{ds}.bin {k} "+switches[alg]
    # return task

def generate_tasks():
    tasks = []

    with open(DS_FILE) as f:
        for line in f:
            if not line.strip():
                continue

            ds, k = line.strip().split()

            for alg in ALGOS:
                task = get_task(alg, ds, k)
                tasks.append(task)
    return tasks



def run_task(cmd: str, rank: int):
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
        rc, outfile, state = run_task(task, rank)
        comm.send((task, rc, outfile, state), dest=0, tag=TAG_DONE)


def collect_results():
    print("[POST] Collecting results...", flush=True)

    with open(OUTFILE, "w") as out:
        with open(DS_FILE) as f:
            for line in f:
                if not line.strip():
                    continue

                ds, k = line.strip().split()
                for alg in ALGOS:
                    task = get_task(alg, ds, k)
                    fname = get_outfile(task)
                    total_t = get_t_total(fname)
                    out.write((str(total_t) if total_t is not None else "x") + " ")
                out.write("\n")

    print(f"[POST] Wrote {OUTFILE}", flush=True)


def parse_args(argv):
    collect_only = any(arg in {"--collect-only", "--postprocess-only"} for arg in argv[1:])
    return collect_only


def main():
    global MPI
    collect_only = parse_args(sys.argv)

    if collect_only:
        collect_results()
        return

    from mpi4py import MPI

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
