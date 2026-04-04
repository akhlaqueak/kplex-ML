#!/usr/bin/env python3
from mpi4py import MPI
import os
import shlex
import subprocess
from pathlib import Path

TAG_READY = 1
TAG_TASK = 2
TAG_DONE = 3
TAG_STOP = 4

HEUR = os.environ.get("HEUR", "EGO-Heur").split()
BR = os.environ.get("BR", "S-Br").split()
CT = os.environ.get("CT", "CTCP").split()
ALGO = os.environ.get("ALGO", "kPEX")
BOUND = os.environ.get("BOUND", "AltRB")

TASK_TIMEOUT = 1800
TIMEOUT_GRACE = 30
DS_LOC = "/home/akhlaque.ak@gmail.com/graphs/large-collection"
OUTFILE = f"output-{os.environ.get('SLURM_JOB_NAME', 'local')}.txt"
LOG_DIR = Path("logs")
SCRIPT_DIR = Path(__file__).resolve().parent
DS_FILE = (SCRIPT_DIR / "../ds.txt").resolve()


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


def get_cmd(ds, k, br, heur, algo, bound, ct):
    return [
        "UMKPEX",
        f"{DS_LOC}/{ds}.bin",
        k,
        "-branching", br,
        "-heuristic", heur,
        "-algorithm", algo,
        "-bounding", bound,
        "-ctcp", ct,
    ]


def generate_tasks():
    tasks = []

    with open(DS_FILE) as f:
        for line in f:
            if not line.strip():
                continue

            ds, k = line.strip().split()

            for heur in HEUR:
                for br in BR:
                    for ct in CT:
                        cmd = get_cmd(ds, k, br, heur, ALGO, BOUND, ct)
                        tasks.append(" ".join(shlex.quote(x) for x in cmd))

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

                for heur in HEUR:
                    for br in BR:
                        for ct in CT:
                            cmd_str = get_cmd(ds, k, br, heur, ALGO, BOUND, ct)
                            fname = get_outfile(" ".join(shlex.quote(x) for x in cmd_str))

                            if os.path.isfile(fname):
                                values = extract_result_values(fname)
                            else:
                                values = None

                            if values:
                                out.write(" ".join(values) + " ")
                            else:
                                out.write("x x x x x x")

                out.write("\n")

    print(f"[POST] Wrote {OUTFILE}", flush=True)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # if rank == 0:
    #     master(comm, size)
    # else:
    #     worker(comm, rank)

    if rank == 0:
        collect_results()


if __name__ == "__main__":
    main()
