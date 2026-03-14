#!/bin/bash

set -euo pipefail

USER_HOME="${HOME:-/home/xmyf}"
SLIME_DIR="${SLIME_DIR:-${USER_HOME}/slime}"
MODEL_DIR="${MODEL_DIR:-${USER_HOME}/slime_assets/models}"
MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-${USER_HOME}/slime_deps/Megatron-LM}"
CUDA_HOME="${CUDA_HOME:-${USER_HOME}/slime_deps/cuda-12.9}"
export SLIME_DIR MODEL_DIR MEGATRON_LM_PATH CUDA_HOME

INPUT_DIR="${MODEL_DIR}/Qwen3-4B"
OUTPUT_DIR="${MODEL_DIR}/Qwen3-4B_torch_dist"

for required_path in "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" "${INPUT_DIR}" "${MEGATRON_LM_PATH}" "${CUDA_HOME}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path not found: ${required_path}" >&2
        exit 1
    fi
done

export PATH="${CUDA_HOME}/bin:${PATH}"
export CPATH="${CUDA_HOME}/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="${CUDA_HOME}/include:${CPLUS_INCLUDE_PATH:-}"

NVIDIA_SITE_PACKAGES="${VIRTUAL_ENV:-${USER_HOME}/.venvs/slime}/lib/python3.12/site-packages/nvidia"
if [[ -d "${NVIDIA_SITE_PACKAGES}" ]]; then
    NVIDIA_LIB_DIRS="$(find "${NVIDIA_SITE_PACKAGES}" -maxdepth 2 \( -type d -name lib -o -type d -name lib64 \) | paste -sd: -)"
else
    NVIDIA_LIB_DIRS=""
fi

if [[ -n "${NVIDIA_LIB_DIRS}" ]]; then
    export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}:${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
else
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

mkdir -p "${OUTPUT_DIR}"

cd "${SLIME_DIR}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-4B.sh"

PYTHONPATH="${MEGATRON_LM_PATH}" torchrun \
    --nproc-per-node 8 \
    tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${INPUT_DIR}" \
    --save "${OUTPUT_DIR}"
