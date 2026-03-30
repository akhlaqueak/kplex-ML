#!/usr/bin/env python3
# command to run:
# srun -n 8 python -u mpi_dynamic.py "$TASKFILE"
from mpi4py import MPI
import os
import shlex
import subprocess
import sys

TAG_READY = 1
TAG_TASK = 2
TAG_DONE = 3
TAG_STOP = 4

TASK_TIMEOUT = int(os.environ.get("TASK_TIMEOUT", "3600"))
TIMEOUT_GRACE = int(os.environ.get("TIMEOUT_GRACE", "30"))
SKIP_TIMEOUT = os.environ.get("SKIP_TIMEOUT", "false").lower() == "true"

def get_outfile(cmd: str) -> str:
    # Replace this with your existing outfile naming rule if needed.
    safe = cmd.strip().replace("/", "_").replace(" ", "_")
    os.makedirs("logs", exist_ok=True)
    return os.path.join("logs", safe + ".out")

def last_line_starts_with_timeout(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size == 0:
                return False
            block = min(4096, size)
            f.seek(-block, os.SEEK_END)
            data = f.read().decode("utf-8", errors="ignore").splitlines()
            return bool(data) and data[-1].startswith("TIMEOUT")
    except OSError:
        return False

def run_task(cmd: str):
    outfile = get_outfile(cmd)

    if SKIP_TIMEOUT and last_line_starts_with_timeout(outfile):
        print(f"SKIP (already TIMEOUT) :: {cmd}", flush=True)
        return 0, outfile, "SKIP"

    print(f"START :: {cmd}", flush=True)

    args = shlex.split(cmd)
    if not args:
        return 0, outfile, "EMPTY"

    # Your shell script runs "./${args[@]}".
    run_cmd = [
        "timeout",
        "--signal=TERM",
        f"--kill-after={TIMEOUT_GRACE}",
        str(TASK_TIMEOUT),
        f"./{args[0]}",
        *args[1:],
    ]

    with open(outfile, "w") as f:
        rc = subprocess.run(
            run_cmd,
            stdout=f,
            stderr=subprocess.STDOUT
        ).returncode

    if rc == 0:
        print(f"SUCCESS  {cmd}", flush=True)
        return rc, outfile, "SUCCESS"
    elif rc == 124:
        with open(outfile, "a") as f:
            f.write(f"TIMEOUT  {cmd}\n")
        print(f"TIMEOUT  {cmd}", flush=True)
        return rc, outfile, "TIMEOUT"
    elif rc == 137:
        with open(outfile, "a") as f:
            f.write(f"KILLED  {cmd}\n")
        print(f"KILLED  {cmd}", flush=True)
        return rc, outfile, "KILLED"
    else:
        with open(outfile, "a") as f:
            f.write(f"FAILED  {cmd} (exit={rc})\n")
        print(f"FAILED  {cmd} (exit={rc})", flush=True)
        return rc, outfile, "FAILED"

def load_tasks(taskfile: str):
    tasks = []
    with open(taskfile, "r") as f:
        for line in f:
            cmd = line.rstrip("\n")
            if not cmd.strip():
                continue
            if cmd.lstrip().startswith("#"):
                continue
            tasks.append(cmd)
    return tasks

def master(comm, size, taskfile: str):
    tasks = load_tasks(taskfile)
    next_idx = 0
    stopped = 0
    nworkers = size - 1

    print(f"Loaded {len(tasks)} tasks", flush=True)

    status = MPI.Status()
    while stopped < nworkers:
        msg = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        src = status.Get_source()
        tag = status.Get_tag()

        if tag == TAG_READY:
            if next_idx < len(tasks):
                comm.send(tasks[next_idx], dest=src, tag=TAG_TASK)
                next_idx += 1
            else:
                comm.send(None, dest=src, tag=TAG_STOP)

        elif tag == TAG_DONE:
            cmd, rc, outfile, state = msg
            print(f"[rank {src}] {state}: {cmd}", flush=True)

        elif tag == TAG_STOP:
            stopped += 1

def worker(comm, rank: int):
    while True:
        comm.send(None, dest=0, tag=TAG_READY)

        status = MPI.Status()
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == TAG_STOP or task is None:
            comm.send(None, dest=0, tag=TAG_STOP)
            break

        rc, outfile, state = run_task(task)
        comm.send((task, rc, outfile, state), dest=0, tag=TAG_DONE)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if len(sys.argv) < 2:
        if rank == 0:
            print(f"Usage: {sys.argv[0]} TASKFILE", file=sys.stderr)
        sys.exit(1)

    taskfile = sys.argv[1]

    if rank == 0:
        master(comm, size, taskfile)
    else:
        worker(comm, rank)

    comm.Barrier()

if __name__ == "__main__":
    main()