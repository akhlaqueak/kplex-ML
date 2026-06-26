#!/bin/bash
#SBATCH --job-name=UMKPEX-all
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --partition=medium
#SBATCH --time=48:00:00
#SBATCH --no-requeue
#SBATCH --output=%x.out
#SBATCH --error=%x.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

echo "Wrapper job started at $(date)"
echo "Running all three experiment scripts inside one allocation"
echo "Node list: ${SLURM_JOB_NODELIST:-unknown}"
echo "Tasks: ${SLURM_NTASKS:-unknown}, ntasks-per-node: ${SLURM_NTASKS_PER_NODE:-unknown}"

run_one() {
    local script="$1"
    shift || true
    local base
    base="$(basename "${script}" .py)"
    local framework_tag="default"
    local prev=""
    for arg in "$@"; do
        if [[ "${prev}" == "--framework" ]]; then
            framework_tag="${arg}"
        fi
        prev="${arg}"
    done
    export JOB_OUTPUT_TAG="${base}-${framework_tag}"

    echo
    echo "============================================================"
    echo "Starting ${script} at $(date)"
    echo "Command: bash ${script} $*"
    echo "Output tag: ${JOB_OUTPUT_TAG}"
    echo "============================================================"
    bash "${script}" "$@"
    echo "Finished ${script} at $(date)"
    unset JOB_OUTPUT_TAG
}

if (( $# > 0 )); then
    COMMON_ARGS=("$@")
else
    COMMON_ARGS=()
fi

SCRIPTS=(
    "./mpi_UMKP+.py"
    "./mpi_UMKP.py"
    "./mpi_kPEX.py"
)

echo
echo "=== Pass 1: default framework ==="
for script in "${SCRIPTS[@]}"; do
    if (( ${#COMMON_ARGS[@]} > 0 )); then
        run_one "${script}" "${COMMON_ARGS[@]}"
    else
        run_one "${script}"
    fi
done

echo
echo "=== Pass 2: oriented framework ==="
for script in "${SCRIPTS[@]}"; do
    if (( ${#COMMON_ARGS[@]} > 0 )); then
        run_one "${script}" "${COMMON_ARGS[@]}" --framework oriented
    else
        run_one "${script}" --framework oriented
    fi
done

echo
echo "Wrapper job finished at $(date)"
