#!/usr/bin/env python3
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
    args = shlex.split(cmd)

    # Skip program name
    args = args[1:]

    parts = []
    skip_next = False

    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue

        # Skip flags like -x
        if arg.startswith("-"):
            # If flag has a value, include only the value
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                val = args[i + 1]
                parts.append(os.path.basename(val))
                skip_next = True
            continue
        else:
            # Non-flag argument → include basename
            parts.append(os.path.basename(arg))

    # Sanitize
    safe_parts = []
    for p in parts:
        p = p.replace("/", "_").replace(" ", "_")
        safe_parts.append(p)

    filename = "_".join(safe_parts)

    os.makedirs("logs", exist_ok=True)
    return os.path.join("logs", filename + ".out")

def last_line_starts_with_failed(path: str) -> bool:
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
            return bool(data) and data[-1].startswith("FAILED")
    except OSError:
        return False


def run_task(cmd: str):
    rank = MPI.COMM_WORLD.Get_rank()
    outfile = get_outfile(cmd)

    if not last_line_starts_with_failed(outfile):
        print(f"[rank {rank}] SKIP (last line is not FAILED) :: {cmd}", flush=True)
        return 0, outfile, "SKIP_NOT_FAILED"

    print(f"[rank {rank}] START :: {cmd}", flush=True)

    args = shlex.split(cmd)
    if not args:
        return 0, outfile, "EMPTY"

    exe = args[0]

    # If executable is not absolute path and exists in current dir → prefix ./
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
            rc = subprocess.run(
                run_cmd,
                stdout=f,
                stderr=subprocess.STDOUT
            ).returncode
    except Exception as e:
        print(f"[rank {rank}] EXCEPTION: {e}", flush=True)
        return -1, outfile, "EXCEPTION"

    if rc == 0:
        print(f"[rank {rank}] SUCCESS {cmd}", flush=True)
        return rc, outfile, "SUCCESS"
    elif rc == 124:
        with open(outfile, "a") as f:
            f.write(f"TIMEOUT  {cmd}\n")
        print(f"[rank {rank}] TIMEOUT {cmd}", flush=True)
        return rc, outfile, "TIMEOUT"
    elif rc == 137:
        with open(outfile, "a") as f:
            f.write(f"KILLED  {cmd}\n")
        print(f"[rank {rank}] KILLED {cmd}", flush=True)
        return rc, outfile, "KILLED"
    else:
        with open(outfile, "a") as f:
            f.write(f"FAILED  {cmd} (exit={rc})\n")
        print(f"[rank {rank}] FAILED {cmd} (exit={rc})", flush=True)
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
            cmd, rc, outfile, state = msg
            print(f"[MASTER] [rank {src}] {state}: {cmd}", flush=True)

        elif tag == TAG_STOP:
            stopped += 1

    print("[MASTER] All workers stopped.", flush=True)


def worker(comm, rank: int):
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


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print(f"[INIT] rank={rank} size={size} host={os.uname()[1]}", flush=True)

    if len(sys.argv) < 2:
        if rank == 0:
            print(f"Usage: {sys.argv[0]} TASKFILE", file=sys.stderr)
        sys.exit(1)

    taskfile = sys.argv[1]

    if rank == 0:
        master(comm, size, taskfile)
    else:
        worker(comm, rank)


if __name__ == "__main__":
    main()
