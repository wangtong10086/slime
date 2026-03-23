#!/usr/bin/env bash
set -euo pipefail

USER_HOME="${HOME:-/home/xmyf}"
SLIME_DIR="${SLIME_DIR:-${USER_HOME}/slime}"
BENCH_DIR="${BENCH_DIR:-${USER_HOME}/liveweb-capability-bench}"
SLIME_VENV_PYTHON="${SLIME_VENV_PYTHON:-${USER_HOME}/.venvs/slime/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-${SLIME_VENV_PYTHON}}"
MODEL_PATH="${MODEL_PATH:-${USER_HOME}/Qwen3-32B}"
OUTPUT_DIR="${OUTPUT_DIR:-${USER_HOME}/slime_runs/qwen3_formatcheck}"
REGRESSION_DATASET_PATH="${REGRESSION_DATASET_PATH:?REGRESSION_DATASET_PATH is required}"
MODEL_NAME="${MODEL_NAME:-qwen3-32b-formatcheck}"
API_KEY="${API_KEY:-local-liveweb-bench}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-31200}"
SERVER_GPU_IDS="${SERVER_GPU_IDS:-0,1}"
SERVER_TP_SIZE="${SERVER_TP_SIZE:-2}"
SERVER_MEM_FRACTION="${SERVER_MEM_FRACTION:-0.88}"
SERVER_CONTEXT_LENGTH="${SERVER_CONTEXT_LENGTH:-32768}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-120}"
PARSER_HEALTH_GATE="${PARSER_HEALTH_GATE:-0.99}"
FORMATCHECK_LIMIT="${FORMATCHECK_LIMIT:-64}"

mkdir -p "${OUTPUT_DIR}"
SERVER_LOG="${OUTPUT_DIR}/server.log"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill -TERM -- "-${SERVER_PID}" >/dev/null 2>&1 || true
    sleep 2
    kill -KILL -- "-${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

setsid env \
CUDA_VISIBLE_DEVICES="${SERVER_GPU_IDS}" \
MODEL_PATH="${MODEL_PATH}" \
PORT="${PORT}" \
API_KEY="${API_KEY}" \
TP_SIZE="${SERVER_TP_SIZE}" \
MEM_FRACTION_STATIC="${SERVER_MEM_FRACTION}" \
CONTEXT_LENGTH="${SERVER_CONTEXT_LENGTH}" \
PYTHON_BIN="${PYTHON_BIN}" \
PYTHONPATH="${USER_HOME}/slime_deps/sglang/python${PYTHONPATH:+:${PYTHONPATH}}" \
bash "${BENCH_DIR}/scripts/start_local_qwen3_32b.sh" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

AUTH_HEADER="Authorization: Bearer ${API_KEY}"

for _ in $(seq 1 120); do
  if curl -sf -H "${AUTH_HEADER}" "http://${HOST}:${PORT}/v1/models" >/dev/null; then
    break
  fi
  sleep 2
done

if ! curl -sf -H "${AUTH_HEADER}" "http://${HOST}:${PORT}/v1/models" >/dev/null; then
  echo "formatcheck server failed to become ready" >&2
  exit 1
fi

"${PYTHON_BIN}" "${SLIME_DIR}/scripts/eval_toolcall_parser_health.py" \
  --dataset "${REGRESSION_DATASET_PATH}" \
  --base-url "http://${HOST}:${PORT}/v1" \
  --api-key "${API_KEY}" \
  --model "${MODEL_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --limit "${FORMATCHECK_LIMIT}" \
  --timeout "${TIMEOUT_SECONDS}"

python3 - <<PY
import json
from pathlib import Path

summary = json.loads(Path("${OUTPUT_DIR}/format_eval.json").read_text())
gate = float("${PARSER_HEALTH_GATE}")
if float(summary["parser_success_rate"]) < gate:
    raise SystemExit(
        f"parser_success_rate {summary['parser_success_rate']:.4f} below gate {gate:.4f}"
    )
if float(summary["dangling_tool_call_rate"]) > 0.0:
    raise SystemExit(
        f"dangling_tool_call_rate {summary['dangling_tool_call_rate']:.4f} is non-zero"
    )
PY
