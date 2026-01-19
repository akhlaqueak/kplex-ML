from mpi4py import MPI
import sys
import os
import subprocess

TASKFILE = sys.argv[1]
LOGDIR = "test-1m"
TIMEOUT = 1800  # seconds

os.makedirs(LOGDIR, exist_ok=True)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def run_task(line, rank):
    parts = line.split()
    prog, ds, k = parts[:3]
    args = parts[3:]

    outfile = f"{LOGDIR}/{prog} {ds}.bin {k} {' '.join(args)}.txt"
    outfile = outfile.replace(" ", "_")

    try:
        with open(outfile, "w") as f:
            subprocess.run(
                [
                    f"./{prog}",
                    f"{os.environ['HOME']}/graphs/large-collection/{ds}.bin",
                    k,
                    *args,
                ],
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=TIMEOUT,
            )
    except subprocess.TimeoutExpired:
        with open(outfile, "a") as f:
            f.write(f"\nOOT {TIMEOUT}\n")

# ------------------------
# Rank 0 = task manager
# ------------------------
if rank == 0:
    with open(TASKFILE) as f:
        tasks = [line.strip() for line in f if line.strip()]

    next_task = 0
    active = size - 1

    # initial dispatch
    for r in range(1, size):
        if next_task < len(tasks):
            comm.send(tasks[next_task], dest=r)
            next_task += 1
        else:
            comm.send(None, dest=r)
            active -= 1

    # dynamic scheduling
    while active > 0:
        r = comm.recv(source=MPI.ANY_SOURCE)
        if next_task < len(tasks):
            comm.send(tasks[next_task], dest=r)
            next_task += 1
        else:
            comm.send(None, dest=r)
            active -= 1

# ------------------------
# Workers
# ------------------------
else:
    while True:
        task = comm.recv(source=0)
        if task is None:
            break
        run_task(task, rank)
        comm.send(rank, dest=0)
