#!/bin/bash

set -euo pipefail

home_dir=$HOME
benchmark_dir="${home_dir}/benchmark/ECP/bert-large/"
venv_activate="${home_dir}/ML/env/ml/bin/activate"
nproc_per_node=2

usage() {
    echo "Usage: $0 [-n NPROC_PER_NODE] [NPROC_PER_NODE] [nproc_per_node=N]"
}

if [[ $# -ge 1 ]]; then
    case "${1}" in
        -n|--nproc-per-node)
            if [[ $# -lt 2 ]]; then
                usage
                exit 1
            fi
            nproc_per_node="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        nproc_per_node=*|NPROC_PER_NODE=*)
            nproc_per_node="${1#*=}"
            shift 1
            ;;
        *)
            nproc_per_node="$1"
            shift 1
            ;;
    esac
fi

if ! [[ "${nproc_per_node}" =~ ^[0-9]+$ ]] || [[ "${nproc_per_node}" -lt 1 ]]; then
    echo "Invalid NPROC_PER_NODE: ${nproc_per_node}"
    usage
    exit 1
fi

cd "${benchmark_dir}"
source "${venv_activate}"


# training.py expects torch.distributed env vars (LOCAL_RANK/RANK/WORLD_SIZE),
# so launch via torchrun instead of plain python.
torchrun --standalone --nproc_per_node="${nproc_per_node}" training.py


deactivate
