#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
LIVEWEB_ARENA_DIR=${LIVEWEB_ARENA_DIR:-/home/xmyf/liveweb-arena}
VENV_DIR=${VENV_DIR:-/home/xmyf/.venvs/slime}

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "missing python at ${VENV_DIR}/bin/python" >&2
  exit 1
fi

SERVICE_ROOT=${SERVICE_ROOT:-/data/liveweb_eval_service_tp2_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-/data/liveweb_batch_eval}
MODEL_DIR=${MODEL_DIR:-/data/slime_runs/liveweb_online_rl_qwen3_32b_20260316_003116/hf_exports/iter_0000029}
MODEL_NAME=${MODEL_NAME:-qwen3-32b-liveweb-iter-0000029}
API_KEY=${API_KEY:-local-liveweb}

SERVER_SPECS=${SERVER_SPECS:-"0,1:17000 2,3:17002 4,5:17004 6,7:17006"}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-32768}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.80}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-32}
NUM_PROMPTS=${NUM_PROMPTS:-200}
SEED=${SEED:-42}
TIMEOUT=${TIMEOUT:-1800}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-batch_eval_tp2}
READY_TIMEOUT=${READY_TIMEOUT:-1800}
TEMPERATURE=${TEMPERATURE:-0.0}
EXCLUDE_PLUGINS=${EXCLUDE_PLUGINS:-openlibrary,weather}
MIN_UNIQUE_PLUGINS=${MIN_UNIQUE_PLUGINS:-2}
FLUSH_EVERY=${FLUSH_EVERY:-10}
LIVEWEB_ENABLE_THINKING=${LIVEWEB_ENABLE_THINKING:-0}
LIVEWEB_SEPARATE_REASONING=${LIVEWEB_SEPARATE_REASONING:-1}
LIVEWEB_MAX_COMPLETION_TOKENS=${LIVEWEB_MAX_COMPLETION_TOKENS:-1024}

mkdir -p "${OUTPUT_DIR}"

echo "service_root=${SERVICE_ROOT}"
echo "model_dir=${MODEL_DIR}"
echo "server_specs=${SERVER_SPECS}"
echo "num_prompts=${NUM_PROMPTS} concurrency=${MAX_CONCURRENCY}"
echo "temperature=${TEMPERATURE} exclude_plugins=${EXCLUDE_PLUGINS}"

"${VENV_DIR}/bin/python" "${ROOT_DIR}/scripts/liveweb_sglang_service_manager.py" start \
  --service-root "${SERVICE_ROOT}" \
  --server-specs "${SERVER_SPECS}" \
  --model-dir "${MODEL_DIR}" \
  --served-model-name "${MODEL_NAME}" \
  --api-key "${API_KEY}" \
  --context-length "${CONTEXT_LENGTH}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --tool-call-parser qwen \
  --reasoning-parser qwen3 \
  --ready-timeout "${READY_TIMEOUT}"

SERVER_POOL_FILE="${SERVICE_ROOT}/server_pool.json"
if [[ ! -f "${SERVER_POOL_FILE}" ]]; then
  echo "server_pool_file not found: ${SERVER_POOL_FILE}" >&2
  exit 1
fi

export LIVEWEB_ENABLE_THINKING
export LIVEWEB_SEPARATE_REASONING
export LIVEWEB_MAX_COMPLETION_TOKENS

"${VENV_DIR}/bin/python" "${LIVEWEB_ARENA_DIR}/scripts/batch_eval.py" \
  --model "${MODEL_NAME}" \
  --server-pool-file "${SERVER_POOL_FILE}" \
  --api-key "${API_KEY}" \
  --num-prompts "${NUM_PROMPTS}" \
  --seed "${SEED}" \
  --max-concurrency "${MAX_CONCURRENCY}" \
  --temperature "${TEMPERATURE}" \
  --timeout "${TIMEOUT}" \
  --exclude-plugins "${EXCLUDE_PLUGINS}" \
  --min-unique-plugins "${MIN_UNIQUE_PLUGINS}" \
  --flush-every "${FLUSH_EVERY}" \
  --output-dir "${OUTPUT_DIR}" \
  --output-prefix "${OUTPUT_PREFIX}"
