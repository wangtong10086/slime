#!/usr/bin/env bash
set -euo pipefail

USER_HOME="${USER_HOME:-/home/xmyf}"
SLIME_HOME="${SLIME_HOME:-${USER_HOME}/slime}"
LIVEWEB_ARENA_DIR="${LIVEWEB_ARENA_DIR:-${USER_HOME}/liveweb-arena}"
RUN_DIR="${RUN_DIR:-${USER_HOME}/slime_runs/slime_liveweb_sft_debug_20260312_133321}"

BASE_MODEL_LABEL="${BASE_MODEL_LABEL:-base_qwen3_32b}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-${USER_HOME}/Qwen3-32B}"
FT_MODEL_LABEL="${FT_MODEL_LABEL:-sft_iter_0000055}"
FT_MODEL_DIR="${FT_MODEL_DIR:-${RUN_DIR}/arena_hf_compare_short/iter_0000055}"

NUM_EVAL_RUNS="${NUM_EVAL_RUNS:-25}"
NUM_TASKS_PER_RUN="${NUM_TASKS_PER_RUN:-4}"
MAX_STEPS="${MAX_STEPS:-30}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1800}"
START_SEED="${START_SEED:-1001}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
TP_SIZE="${TP_SIZE:-8}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.82}"
API_KEY_LOCAL="${API_KEY_LOCAL:-local-liveweb}"
LIVEWEB_MAX_COMPLETION_TOKENS="${LIVEWEB_MAX_COMPLETION_TOKENS:-32768}"
LIVEWEB_IGNORE_HTTPS_ERRORS="${LIVEWEB_IGNORE_HTTPS_ERRORS:-1}"

OUT_ROOT="${OUT_ROOT:-${RUN_DIR}/arena_eval_ablation_25x4_ctx32k_steps30}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

source "${USER_HOME}/.venvs/slime/bin/activate"

wait_for_server() {
  local port="$1"
  python - <<PY
import sys, time, httpx
port = ${port}
headers = {"Authorization": "Bearer ${API_KEY_LOCAL}"}
url = f"http://127.0.0.1:{port}/v1/models"
deadline = time.time() + 1800
last = None
while time.time() < deadline:
    try:
        r = httpx.get(url, headers=headers, timeout=15.0)
        if r.status_code == 200:
            print("SGLang ready", flush=True)
            sys.exit(0)
        last = f"status={r.status_code}"
    except Exception as e:
        last = repr(e)
    time.sleep(5)
print(f"SGLang not ready: {last}", file=sys.stderr)
sys.exit(1)
PY
}

run_model_batch() {
  local label="$1"
  local model_dir="$2"
  local port="$3"
  local out_dir="${OUT_ROOT}/${label}"
  local cache_root="${OUT_ROOT}/cache_${label}"
  local sglang_log="${LOG_DIR}/${label}_sglang.log"

  mkdir -p "${out_dir}" "${cache_root}"

  echo "========== ${label} =========="
  echo "model_dir=${model_dir}"
  echo "port=${port}"

  CUDA_VISIBLE_DEVICES="${GPU_IDS}" python -m sglang.launch_server \
    --model-path "${model_dir}" \
    --trust-remote-code \
    --host 127.0.0.1 \
    --port "${port}" \
    --api-key "${API_KEY_LOCAL}" \
    --served-model-name qwen3-32b-liveweb-eval \
    --tp-size "${TP_SIZE}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --context-length "${CONTEXT_LENGTH}" \
    --dtype bfloat16 \
    --disable-cuda-graph \
    > "${sglang_log}" 2>&1 &
  local server_pid=$!

  cleanup_server() {
    if kill -0 "${server_pid}" 2>/dev/null; then
      kill "${server_pid}" || true
      wait "${server_pid}" || true
    fi
  }

  wait_for_server "${port}"

  cd "${LIVEWEB_ARENA_DIR}"
  export LIVEWEB_MAX_COMPLETION_TOKENS
  export LIVEWEB_IGNORE_HTTPS_ERRORS

  for ((offset=0; offset<NUM_EVAL_RUNS; offset++)); do
    local seed=$((START_SEED + offset))
    export LIVEWEB_CACHE_DIR="${cache_root}/seed_${seed}"
    mkdir -p "${LIVEWEB_CACHE_DIR}"
    echo "===== ${label} seed ${seed} ====="
    python eval.py \
      --model qwen3-32b-liveweb-eval \
      --base-url "http://127.0.0.1:${port}/v1" \
      --api-key "${API_KEY_LOCAL}" \
      --seed "${seed}" \
      --num-tasks "${NUM_TASKS_PER_RUN}" \
      --max-steps "${MAX_STEPS}" \
      --timeout "${TIMEOUT_SECONDS}" \
      --temperature "${TEMPERATURE}" \
      --output "${out_dir}/seed_${seed}.json" || true
  done

  cleanup_server
}

BASE_PORT="${BASE_PORT:-31040}"
FT_PORT="${FT_PORT:-31041}"

run_model_batch "${BASE_MODEL_LABEL}" "${BASE_MODEL_DIR}" "${BASE_PORT}"
run_model_batch "${FT_MODEL_LABEL}" "${FT_MODEL_DIR}" "${FT_PORT}"

python "${SLIME_HOME}/scripts/summarize_liveweb_ablation.py" \
  "${OUT_ROOT}" \
  "${BASE_MODEL_LABEL}" \
  "${FT_MODEL_LABEL}"
