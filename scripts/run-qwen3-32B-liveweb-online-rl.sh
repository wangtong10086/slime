#!/bin/bash

set -euo pipefail

USER_HOME="${HOME:-/home/xmyf}"
SLIME_DIR="${SLIME_DIR:-${USER_HOME}/slime}"
SLIME_VENV="${SLIME_VENV:-${USER_HOME}/.venvs/slime}"
MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-${USER_HOME}/slime_deps/Megatron-LM}"
CUDA_HOME="${CUDA_HOME:-${USER_HOME}/slime_deps/cuda-12.9}"
LIVEWEB_ARENA_DIR="${LIVEWEB_ARENA_DIR:-${USER_HOME}/liveweb-arena}"
HF_MODEL_DIR="${HF_MODEL_DIR:-${USER_HOME}/Qwen3-32B}"
REF_LOAD_DIR="${REF_LOAD_DIR:-${USER_HOME}/Qwen3-32B_torch_dist}"

TRAIN_PHASE="${TRAIN_PHASE:-preflight}"
TASK_MIX_PHASE="${TASK_MIX_PHASE:-$([[ "${TRAIN_PHASE}" == "main" ]] && echo main || echo warmup)}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/data/slime_runs/liveweb_online_rl_qwen3_32b_${TIMESTAMP}}"
RUN_CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
LIVEWEB_CACHE_DIR="${LIVEWEB_CACHE_DIR:-/data/liveweb_cache/persistent}"
LIVEWEB_SERVICE_ROOT="${LIVEWEB_SERVICE_ROOT:-/data/liveweb_sglang_services/liveweb_online_rl_${TIMESTAMP}}"
WANDB_SETTINGS_FILE="${WANDB_SETTINGS_FILE:-${USER_HOME}/.config/wandb/settings}"
WANDB_PROJECT="${WANDB_PROJECT:-slime-liveweb-online-rl}"
WANDB_GROUP="${WANDB_GROUP:-qwen3-32b-${TRAIN_PHASE}}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"

RAY_HEAD_PORT="${RAY_HEAD_PORT:-6390}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8270}"
RAY_JOB_PORT="${RAY_JOB_PORT:-${RAY_DASHBOARD_PORT}}"
RAY_TMPDIR="${RAY_TMPDIR:-/data/ray/liveweb_online_rl}"

ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-32768}"
LIVEWEB_MAX_COMPLETION_TOKENS="${LIVEWEB_MAX_COMPLETION_TOKENS:-1024}"
MAX_STEPS="${MAX_STEPS:-30}"

LR="${LR:-5e-7}"
MIN_LR="${MIN_LR:-5e-8}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-1536}"
SAVE_INTERVAL_OVERRIDE="${SAVE_INTERVAL_OVERRIDE:-}"
EVAL_INTERVAL_OVERRIDE="${EVAL_INTERVAL_OVERRIDE:-}"
NUM_ROLLOUT_OVERRIDE="${NUM_ROLLOUT_OVERRIDE:-}"
LIVEWEB_SKIP_SAVE_PRELIGHT="${LIVEWEB_SKIP_SAVE_PRELIGHT:-1}"
LIVEWEB_SKIP_SAVE_PREFLIGHT="${LIVEWEB_SKIP_SAVE_PREFLIGHT:-${LIVEWEB_SKIP_SAVE_PRELIGHT}}"
LIVEWEB_SKIP_SAVE_WARMUP="${LIVEWEB_SKIP_SAVE_WARMUP:-1}"
LIVEWEB_SKIP_SAVE_MAIN="${LIVEWEB_SKIP_SAVE_MAIN:-0}"
TRAIN_SAVE_BATCH_SIZE="${TRAIN_SAVE_BATCH_SIZE:-1}"
TRAIN_UPDATE_WEIGHTS_BATCH_SIZE="${TRAIN_UPDATE_WEIGHTS_BATCH_SIZE:-1}"
RAY_MEMORY_USAGE_THRESHOLD="${RAY_MEMORY_USAGE_THRESHOLD:-0.99}"

if [[ -n "${SLIME_VENV}" ]]; then
  # shellcheck disable=SC1090
  source "${SLIME_VENV}/bin/activate"
fi

if [[ ! -d "${LIVEWEB_ARENA_DIR}" ]]; then
  echo "liveweb-arena not found: ${LIVEWEB_ARENA_DIR}" >&2
  exit 1
fi
if [[ "$(git -C "${LIVEWEB_ARENA_DIR}" branch --show-current)" != "codex/liveweb-arena-stability-20260314" ]]; then
  echo "liveweb-arena must be on branch codex/liveweb-arena-stability-20260314" >&2
  exit 1
fi
if [[ ! -d "${HF_MODEL_DIR}" ]]; then
  echo "HF model dir not found: ${HF_MODEL_DIR}" >&2
  exit 1
fi
if [[ ! -d "${REF_LOAD_DIR}" ]]; then
  echo "Megatron checkpoint dir not found: ${REF_LOAD_DIR}" >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}" "${RUN_CHECKPOINT_DIR}" "${LIVEWEB_CACHE_DIR}" "${LIVEWEB_SERVICE_ROOT}" "${RAY_TMPDIR}"

export PATH="${CUDA_HOME}/bin:${PATH}"
export PYTHONBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NO_PROXY="127.0.0.1,localhost,214.2.15.1"
export no_proxy="${NO_PROXY}"
export SLIME_DIR SLIME_VENV MEGATRON_LM_PATH CUDA_HOME LIVEWEB_ARENA_DIR HF_MODEL_DIR REF_LOAD_DIR
export RUN_ROOT RUN_CHECKPOINT_DIR LIVEWEB_CACHE_DIR LIVEWEB_SERVICE_ROOT
export LIVEWEB_TASK_MIX_PHASE="${TASK_MIX_PHASE}"
export TASK_REGISTRY_VERSION="${TASK_REGISTRY_VERSION:-v2}"
export LIVEWEB_ENABLE_THINKING=0
export LIVEWEB_SEPARATE_REASONING=1
export LIVEWEB_MAX_COMPLETION_TOKENS
export LIVEWEB_MAX_STEPS="${MAX_STEPS}"
export LIVEWEB_TIMEOUT_SECONDS="${LIVEWEB_TIMEOUT_SECONDS:-1800}"
export LIVEWEB_MAX_BROWSER_SESSIONS="${LIVEWEB_MAX_BROWSER_SESSIONS:-32}"
export LIVEWEB_MAX_LLM_REQUESTS="${LIVEWEB_MAX_LLM_REQUESTS:-16}"
export LIVEWEB_PARALLEL_GROUPS="${LIVEWEB_PARALLEL_GROUPS:-8}"
export LIVEWEB_ROUTE_POLICY="${LIVEWEB_ROUTE_POLICY:-sticky_steal}"
export LIVEWEB_TASK_MIX_CONFIG="${LIVEWEB_TASK_MIX_CONFIG:-${SLIME_DIR}/scripts/configs/liveweb_online_task_mix.json}"
export LIVEWEB_ALLOW_ZERO_STD_FALLBACK="${LIVEWEB_ALLOW_ZERO_STD_FALLBACK:-1}"
if [[ "${TRAIN_PHASE}" == "main" ]]; then
  DEFAULT_MIN_GROUP_SIZE=2
  DEFAULT_ALLOW_PARTIAL_GROUP_FALLBACK=0
  DEFAULT_ALLOW_ENV_FALLBACK_GROUPS=0
  DEFAULT_ALLOW_LAST_RESORT_GROUP_FALLBACK=0
else
  DEFAULT_MIN_GROUP_SIZE=1
  DEFAULT_ALLOW_PARTIAL_GROUP_FALLBACK=1
  DEFAULT_ALLOW_ENV_FALLBACK_GROUPS=1
  DEFAULT_ALLOW_LAST_RESORT_GROUP_FALLBACK=1
fi
export LIVEWEB_MIN_GROUP_SIZE="${LIVEWEB_MIN_GROUP_SIZE:-${DEFAULT_MIN_GROUP_SIZE}}"
export LIVEWEB_ALLOW_PARTIAL_GROUP_FALLBACK="${LIVEWEB_ALLOW_PARTIAL_GROUP_FALLBACK:-${DEFAULT_ALLOW_PARTIAL_GROUP_FALLBACK}}"
export LIVEWEB_ALLOW_ENV_FALLBACK_GROUPS="${LIVEWEB_ALLOW_ENV_FALLBACK_GROUPS:-${DEFAULT_ALLOW_ENV_FALLBACK_GROUPS}}"
export LIVEWEB_ALLOW_LAST_RESORT_GROUP_FALLBACK="${LIVEWEB_ALLOW_LAST_RESORT_GROUP_FALLBACK:-${DEFAULT_ALLOW_LAST_RESORT_GROUP_FALLBACK}}"
export LIVEWEB_MODEL_NAME="${LIVEWEB_MODEL_NAME:-qwen3-32b-liveweb-online-rl}"
export LIVEWEB_API_KEY="${LIVEWEB_API_KEY:-local-liveweb}"
export LIVEWEB_QUICK_EVAL_PROMPTS="${LIVEWEB_QUICK_EVAL_PROMPTS:-32}"
export LIVEWEB_FORMAL_EVAL_PROMPTS="${LIVEWEB_FORMAL_EVAL_PROMPTS:-200}"
export LIVEWEB_FORMAL_EVAL_EVERY="${LIVEWEB_FORMAL_EVAL_EVERY:-50}"
export LIVEWEB_SGLANG_WORKER_PORTS="${LIVEWEB_SGLANG_WORKER_PORTS:-15000,15002,15004,15006,15008,15010,15012,15014}"
export TRAIN_SAVE_BATCH_SIZE
export TRAIN_UPDATE_WEIGHTS_BATCH_SIZE
export RAY_MEMORY_USAGE_THRESHOLD

NUM_ROLLOUT=1
SAVE_INTERVAL=1
EVAL_INTERVAL=10
DEBUG_ROLLOUT_ONLY=0
LIVEWEB_SKIP_SAVE=0
case "${TRAIN_PHASE}" in
  preflight)
    NUM_ROLLOUT=1
    SAVE_INTERVAL=1
    EVAL_INTERVAL=0
    DEBUG_ROLLOUT_ONLY=1
    LIVEWEB_SKIP_SAVE="${LIVEWEB_SKIP_SAVE_PREFLIGHT}"
    ;;
  warmup)
    NUM_ROLLOUT=120
    SAVE_INTERVAL=20
    EVAL_INTERVAL=10
    export LIVEWEB_TASK_MIX_PHASE="warmup"
    LIVEWEB_SKIP_SAVE="${LIVEWEB_SKIP_SAVE_WARMUP}"
    ;;
  main)
    NUM_ROLLOUT=300
    SAVE_INTERVAL=25
    EVAL_INTERVAL=10
    export LIVEWEB_TASK_MIX_PHASE="main"
    LIVEWEB_SKIP_SAVE="${LIVEWEB_SKIP_SAVE_MAIN}"
    ;;
  *)
    echo "Unsupported TRAIN_PHASE=${TRAIN_PHASE}" >&2
    exit 1
    ;;
esac

if [[ -n "${SAVE_INTERVAL_OVERRIDE}" ]]; then
  SAVE_INTERVAL="${SAVE_INTERVAL_OVERRIDE}"
fi
if [[ -n "${EVAL_INTERVAL_OVERRIDE}" ]]; then
  EVAL_INTERVAL="${EVAL_INTERVAL_OVERRIDE}"
fi
if [[ -n "${NUM_ROLLOUT_OVERRIDE}" ]]; then
  NUM_ROLLOUT="${NUM_ROLLOUT_OVERRIDE}"
fi
export LIVEWEB_SKIP_SAVE

WANDB_KEY=""
export WANDB_SETTINGS_FILE
if [[ -f "${WANDB_SETTINGS_FILE}" ]]; then
  WANDB_KEY="$(python - <<'PY'
import os
from pathlib import Path
path = Path(os.environ["WANDB_SETTINGS_FILE"])
for line in path.read_text().splitlines():
    if line.startswith("api_key = "):
        print(line.split(" = ", 1)[1].strip())
        break
PY
)"
fi
if [[ -z "${WANDB_RUN_ID}" ]]; then
  WANDB_RUN_ID="$(python - <<'PY'
import uuid
print(uuid.uuid4().hex[:8])
PY
)"
fi

source "${SLIME_DIR}/scripts/models/qwen3-32B.sh"

COMMON_ARGS=(
  --actor-num-nodes 1
  --actor-num-gpus-per-node 8
  --colocate
  --rollout-num-gpus 8
  --rollout-num-gpus-per-engine 1
  --num-gpus-per-node 8

  "${MODEL_ARGS[@]}"
  --hf-checkpoint "${HF_MODEL_DIR}"
  --ref-load "${REF_LOAD_DIR}"
  --load "${RUN_CHECKPOINT_DIR}"
  --save "${RUN_CHECKPOINT_DIR}"
  --save-interval "${SAVE_INTERVAL}"

  --data-source-path slime.rollout.liveweb_online.data_source.LiveWebOnlineDataSource
  --rollout-function-path slime.rollout.liveweb_online.rollout.generate_rollout
  --eval-function-path slime.rollout.liveweb_online.rollout.generate_rollout
  --custom-rm-path slime.rollout.liveweb_online.reward.reward_func
  --custom-reward-post-process-path slime.rollout.liveweb_online.reward.post_process_rewards

  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 0.01
  --kl-loss-type low_var_kl
  --entropy-coef 0.0
  --eps-clip 0.2
  --eps-clip-high 0.28

  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}"
  --rollout-max-response-len "${LIVEWEB_MAX_COMPLETION_TOKENS}"
  --rollout-temperature 0.7
  --rollout-top-p 1.0

  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"

  --optimizer adam
  --lr "${LR}"
  --min-lr "${MIN_LR}"
  --lr-decay-style cosine
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer

  --tensor-model-parallel-size 8
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --sequence-parallel
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1

  --sglang-api-key "${LIVEWEB_API_KEY}"
  --sglang-tool-call-parser qwen
  --sglang-reasoning-parser qwen3
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.80}"

  --attention-dropout 0.0
  --hidden-dropout 0.0
  --attention-backend flash
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
)

if [[ "${DEBUG_ROLLOUT_ONLY}" == "1" ]]; then
  COMMON_ARGS+=(--debug-rollout-only)
fi

EVAL_ARGS=()
if (( EVAL_INTERVAL > 0 )); then
  DUMMY_EVAL_FILE="${RUN_ROOT}/dummy_eval.jsonl"
  printf '{"input":"liveweb-eval-placeholder"}\n' > "${DUMMY_EVAL_FILE}"
  EVAL_ARGS=(
    --eval-interval "${EVAL_INTERVAL}"
    --eval-prompt-data liveweb_dummy "${DUMMY_EVAL_FILE}"
    --n-samples-per-eval-prompt 1
    --eval-max-response-len "${LIVEWEB_MAX_COMPLETION_TOKENS}"
  )
fi

WANDB_ARGS=()
if [[ -n "${WANDB_KEY}" ]]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-key "${WANDB_KEY}"
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
    --wandb-run-id "${WANDB_RUN_ID}"
  )
fi

cat > "${RUN_ROOT}/run_config.json" <<JSON
{
  "train_phase": "${TRAIN_PHASE}",
  "task_mix_phase": "${TASK_MIX_PHASE}",
  "num_rollout": ${NUM_ROLLOUT},
  "rollout_batch_size": ${ROLLOUT_BATCH_SIZE},
  "n_samples_per_prompt": ${N_SAMPLES_PER_PROMPT},
  "global_batch_size": ${GLOBAL_BATCH_SIZE},
  "rollout_max_prompt_len": ${ROLLOUT_MAX_PROMPT_LEN},
  "liveweb_max_completion_tokens": ${LIVEWEB_MAX_COMPLETION_TOKENS},
  "max_steps": ${MAX_STEPS},
  "skip_save": ${LIVEWEB_SKIP_SAVE},
  "save_batch_size": ${TRAIN_SAVE_BATCH_SIZE},
  "update_weights_batch_size": ${TRAIN_UPDATE_WEIGHTS_BATCH_SIZE},
  "ray_memory_usage_threshold": ${RAY_MEMORY_USAGE_THRESHOLD},
  "effective_save_interval": ${SAVE_INTERVAL},
  "hf_model_dir": "${HF_MODEL_DIR}",
  "ref_load_dir": "${REF_LOAD_DIR}",
  "liveweb_arena_dir": "${LIVEWEB_ARENA_DIR}",
  "liveweb_branch": "$(git -C "${LIVEWEB_ARENA_DIR}" branch --show-current)"
}
JSON

ray stop --force >/dev/null 2>&1 || true
sleep 2
ray stop --force >/dev/null 2>&1 || true
export RAY_memory_usage_threshold="${RAY_MEMORY_USAGE_THRESHOLD}"
ray start \
  --head \
  --node-ip-address 127.0.0.1 \
  --port "${RAY_HEAD_PORT}" \
  --dashboard-host 0.0.0.0 \
  --dashboard-port "${RAY_DASHBOARD_PORT}" \
  --num-gpus 8 \
  --disable-usage-stats \
  --temp-dir "${RAY_TMPDIR}"

RUNTIME_ENV_JSON="$(python - <<'PY'
import json
import os
env_vars = {
    "PYTHONPATH": ":".join([
        os.environ["SLIME_DIR"],
        os.environ["LIVEWEB_ARENA_DIR"],
        os.environ["MEGATRON_LM_PATH"],
    ]),
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NO_PROXY": os.environ["NO_PROXY"],
    "no_proxy": os.environ["no_proxy"],
    "LIVEWEB_ARENA_DIR": os.environ["LIVEWEB_ARENA_DIR"],
    "LIVEWEB_CACHE_DIR": os.environ["LIVEWEB_CACHE_DIR"],
    "LIVEWEB_ENABLE_THINKING": os.environ["LIVEWEB_ENABLE_THINKING"],
    "LIVEWEB_SEPARATE_REASONING": os.environ["LIVEWEB_SEPARATE_REASONING"],
    "LIVEWEB_MAX_COMPLETION_TOKENS": os.environ["LIVEWEB_MAX_COMPLETION_TOKENS"],
    "LIVEWEB_MAX_STEPS": os.environ["LIVEWEB_MAX_STEPS"],
    "LIVEWEB_TIMEOUT_SECONDS": os.environ["LIVEWEB_TIMEOUT_SECONDS"],
    "LIVEWEB_MAX_BROWSER_SESSIONS": os.environ["LIVEWEB_MAX_BROWSER_SESSIONS"],
    "LIVEWEB_MAX_LLM_REQUESTS": os.environ["LIVEWEB_MAX_LLM_REQUESTS"],
    "LIVEWEB_PARALLEL_GROUPS": os.environ["LIVEWEB_PARALLEL_GROUPS"],
    "LIVEWEB_ROUTE_POLICY": os.environ["LIVEWEB_ROUTE_POLICY"],
    "LIVEWEB_TASK_MIX_PHASE": os.environ["LIVEWEB_TASK_MIX_PHASE"],
    "LIVEWEB_TASK_MIX_CONFIG": os.environ["LIVEWEB_TASK_MIX_CONFIG"],
    "LIVEWEB_MIN_GROUP_SIZE": os.environ["LIVEWEB_MIN_GROUP_SIZE"],
    "LIVEWEB_ALLOW_PARTIAL_GROUP_FALLBACK": os.environ["LIVEWEB_ALLOW_PARTIAL_GROUP_FALLBACK"],
    "LIVEWEB_ALLOW_ENV_FALLBACK_GROUPS": os.environ["LIVEWEB_ALLOW_ENV_FALLBACK_GROUPS"],
    "LIVEWEB_ALLOW_ZERO_STD_FALLBACK": os.environ["LIVEWEB_ALLOW_ZERO_STD_FALLBACK"],
    "LIVEWEB_ALLOW_LAST_RESORT_GROUP_FALLBACK": os.environ["LIVEWEB_ALLOW_LAST_RESORT_GROUP_FALLBACK"],
    "LIVEWEB_USE_CURATED_WARMUP_POOL": os.environ.get("LIVEWEB_USE_CURATED_WARMUP_POOL", "1"),
    "LIVEWEB_WARMUP_POOL_SIZE": os.environ.get("LIVEWEB_WARMUP_POOL_SIZE", os.environ["LIVEWEB_QUICK_EVAL_PROMPTS"]),
    "LIVEWEB_WARMUP_POOL_BASE_SEED": os.environ.get("LIVEWEB_WARMUP_POOL_BASE_SEED", "900000"),
    "LIVEWEB_MODEL_NAME": os.environ["LIVEWEB_MODEL_NAME"],
    "LIVEWEB_API_KEY": os.environ["LIVEWEB_API_KEY"],
    "LIVEWEB_SGLANG_WORKER_PORTS": os.environ["LIVEWEB_SGLANG_WORKER_PORTS"],
    "LIVEWEB_QUICK_EVAL_PROMPTS": os.environ["LIVEWEB_QUICK_EVAL_PROMPTS"],
    "LIVEWEB_FORMAL_EVAL_PROMPTS": os.environ["LIVEWEB_FORMAL_EVAL_PROMPTS"],
    "LIVEWEB_FORMAL_EVAL_EVERY": os.environ["LIVEWEB_FORMAL_EVAL_EVERY"],
    "LIVEWEB_SKIP_SAVE": os.environ["LIVEWEB_SKIP_SAVE"],
    "TRAIN_SAVE_BATCH_SIZE": os.environ["TRAIN_SAVE_BATCH_SIZE"],
    "TRAIN_UPDATE_WEIGHTS_BATCH_SIZE": os.environ["TRAIN_UPDATE_WEIGHTS_BATCH_SIZE"]
}
print(json.dumps({"env_vars": env_vars}))
PY
)"

cd "${SLIME_DIR}"
ray job submit --address "http://127.0.0.1:${RAY_JOB_PORT}" \
  --runtime-env-json "${RUNTIME_ENV_JSON}" \
  -- python3 train.py "${COMMON_ARGS[@]}" "${EVAL_ARGS[@]}" "${WANDB_ARGS[@]}"
