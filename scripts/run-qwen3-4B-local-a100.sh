#!/bin/bash

set -euo pipefail

# Local bare-metal launcher for the official Qwen3-4B example on 8xA100.

USER_HOME="${HOME:-/home/xmyf}"
SLIME_DIR="${SLIME_DIR:-${USER_HOME}/slime}"
MODEL_DIR="${MODEL_DIR:-${USER_HOME}/slime_assets/models}"
DATA_DIR="${DATA_DIR:-${USER_HOME}/slime_assets/data}"
MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-${USER_HOME}/slime_deps/Megatron-LM}"
CUDA_HOME="${CUDA_HOME:-${USER_HOME}/slime_deps/cuda-12.9}"
export SLIME_DIR MODEL_DIR DATA_DIR MEGATRON_LM_PATH CUDA_HOME

if [[ ! -d "${MEGATRON_LM_PATH}" ]]; then
    echo "Megatron-LM not found: ${MEGATRON_LM_PATH}" >&2
    exit 1
fi

if [[ ! -d "${CUDA_HOME}" ]]; then
    echo "CUDA_HOME not found: ${CUDA_HOME}" >&2
    exit 1
fi

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

# Reduce the chance of cleaning up unrelated local work.
pkill -f sglang || true
ray stop --force || true
sleep 2

export PYTHONBUFFERED=16

NUM_ROLLOUT="${NUM_ROLLOUT:-3000}"
NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
ENABLE_EVAL="${ENABLE_EVAL:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-16}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-16384}"
EVAL_TOP_P="${EVAL_TOP_P:-1}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-9216}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.7}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: ${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-4B.sh"

REQUIRED_PATHS=(
    "${SLIME_DIR}/train.py"
    "${MODEL_DIR}/Qwen3-4B"
    "${MODEL_DIR}/Qwen3-4B_torch_dist"
    "${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl"
    "${DATA_DIR}/aime-2024/aime-2024.jsonl"
)

for required_path in "${REQUIRED_PATHS[@]}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path not found: ${required_path}" >&2
        exit 1
    fi
done

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/Qwen3-4B"
   --ref-load "${MODEL_DIR}/Qwen3-4B_torch_dist"
   --load "${MODEL_DIR}/Qwen3-4B_slime/"
   --save "${MODEL_DIR}/Qwen3-4B_slime/"
   --save-interval "${SAVE_INTERVAL}"
)

ROLLOUT_ARGS=(
   --prompt-data "${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-temperature "${ROLLOUT_TEMPERATURE}"
   --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}"

   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --balance-data
)

EVAL_ARGS=()
if [[ "${ENABLE_EVAL}" == "1" ]]; then
   EVAL_ARGS=(
      --eval-interval "${EVAL_INTERVAL}"
      --eval-prompt-data aime "${DATA_DIR}/aime-2024/aime-2024.jsonl"
      --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
      --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
      --eval-top-p "${EVAL_TOP_P}"
   )
fi

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export no_proxy="127.0.0.1,localhost,${MASTER_ADDR}"
export NO_PROXY="${no_proxy}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

for _ in $(seq 1 60); do
    if python3 - <<'PY'
import sys
import urllib.request

try:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open("http://127.0.0.1:8265/api/version", timeout=2) as resp:
        sys.exit(0 if resp.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
    then
        break
    fi
    sleep 1
done

RUNTIME_ENV_JSON="$(cat <<EOF
{
  "env_vars": {
    "PYTHONPATH": "${MEGATRON_LM_PATH}",
    "CUDA_HOME": "${CUDA_HOME}",
    "PATH": "${CUDA_HOME}/bin:${PATH}",
    "LD_LIBRARY_PATH": "${LD_LIBRARY_PATH}",
    "CPATH": "${CPATH}",
    "CPLUS_INCLUDE_PATH": "${CPLUS_INCLUDE_PATH}",
    "no_proxy": "${no_proxy}",
    "NO_PROXY": "${NO_PROXY}",
    "http_proxy": "",
    "https_proxy": "",
    "HTTP_PROXY": "",
    "HTTPS_PROXY": "",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}"
  }
}
EOF
)"

cd "${SLIME_DIR}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
