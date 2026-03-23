#!/bin/bash

set -euo pipefail

USER_HOME="${HOME:-/home/xmyf}"
SLIME_DIR="${SLIME_DIR:-${USER_HOME}/slime}"
SLIME_VENV="${SLIME_VENV:-${USER_HOME}/.venvs/slime}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SESSION_NAME="${SESSION_NAME:-slime_liveweb_sft_${RUN_STAMP}}"
RUN_DIR="${RUN_DIR:-${USER_HOME}/slime_runs/${SESSION_NAME}}"
TRAIN_GPU_IDS="${TRAIN_GPU_IDS:-0,1,2,3,4,5,6,7}"
TRAIN_TP_SIZE="${TRAIN_TP_SIZE:-2}"
TRAIN_PP_SIZE="${TRAIN_PP_SIZE:-2}"
TRAIN_CP_SIZE="${TRAIN_CP_SIZE:-2}"
EVAL_GPU_IDS="${EVAL_GPU_IDS:-0,1,2,3,4,5,6,7}"
ARENA_EVAL_MODE="${ARENA_EVAL_MODE:-watch}"
SKIP_ARENA_EVAL="${SKIP_ARENA_EVAL:-0}"
SKIP_INITIAL_EVAL="${SKIP_INITIAL_EVAL:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-slime-liveweb}"
WANDB_GROUP="${WANDB_GROUP:-${SESSION_NAME}}"
NUM_EPOCH="${NUM_EPOCH:-1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-512}"
LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-2048}"
RECOMPUTE_LOSS_FUNCTION="${RECOMPUTE_LOSS_FUNCTION:-1}"
ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-32768}"
FILTER_COUNT_SAMPLE_SIZE="${FILTER_COUNT_SAMPLE_SIZE:-128}"
LR="${LR:-1e-5}"
MIN_LR="${MIN_LR:-1e-6}"
SAVE_INTERVAL_OVERRIDE="${SAVE_INTERVAL_OVERRIDE:-}"
FILTERED_DATASET_ROWS_OVERRIDE="${FILTERED_DATASET_ROWS_OVERRIDE:-}"
MIN_TRAIN_GPU_FREE_MEM_MB="${MIN_TRAIN_GPU_FREE_MEM_MB:-70000}"
TRAIN_STEP_TOKEN_BUDGET="${TRAIN_STEP_TOKEN_BUDGET:-48000}"
TRAIN_STEP_LOGIT_BUDGET="${TRAIN_STEP_LOGIT_BUDGET:-24000}"
TRAIN_MIN_DYNAMIC_GLOBAL_BATCH_SIZE="${TRAIN_MIN_DYNAMIC_GLOBAL_BATCH_SIZE:-$(( GLOBAL_BATCH_SIZE > 1 ? GLOBAL_BATCH_SIZE / 2 : 1 ))}"
TRAIN_UNDERFILLED_STEP_MIN_SAMPLES="${TRAIN_UNDERFILLED_STEP_MIN_SAMPLES:-4}"
TRAIN_STEP_PACKING_STRATEGY="${TRAIN_STEP_PACKING_STRATEGY:-greedy_desc}"
TRAIN_STEP_LONG_SAMPLE_THRESHOLD="${TRAIN_STEP_LONG_SAMPLE_THRESHOLD:-16800}"
TRAIN_MAX_LONG_SAMPLES_PER_STEP="${TRAIN_MAX_LONG_SAMPLES_PER_STEP:-2}"
TRAIN_MAX_SINGLE_SAMPLE_TOKENS="${TRAIN_MAX_SINGLE_SAMPLE_TOKENS:-12000}"
TRAIN_MAX_TOTAL_TOKENS_PER_SAMPLE="${TRAIN_MAX_TOTAL_TOKENS_PER_SAMPLE:-4096}"
TRAIN_MAX_RESPONSE_TOKENS_PER_SAMPLE="${TRAIN_MAX_RESPONSE_TOKENS_PER_SAMPLE:-2048}"
SERVER_PORT="${SERVER_PORT:-31000}"
RAY_HEAD_PORT="${RAY_HEAD_PORT:-6382}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8266}"
TASK_DATASET_PATH="${TASK_DATASET_PATH:-/data/liveweb_sft/liveweb_success_sft_20260321/train_32k.jsonl}"
DATASET_PATH="${DATASET_PATH:-${TASK_DATASET_PATH}}"
HF_MODEL_DIR="${HF_MODEL_DIR:-${USER_HOME}/Qwen3-32B}"
REF_LOAD_DIR="${REF_LOAD_DIR:-${USER_HOME}/Qwen3-32B_torch_dist_tp${TRAIN_TP_SIZE}_pp${TRAIN_PP_SIZE}}"
LIVEWEB_ARENA_DIR="${LIVEWEB_ARENA_DIR:-${USER_HOME}/liveweb-arena}"
LIVEWEB_CACHE_DIR="${LIVEWEB_CACHE_DIR:-${RUN_DIR}/liveweb_cache}"
WANDB_RUN_ID="${WANDB_RUN_ID:-$(python3 - <<'PY'
import uuid
print(uuid.uuid4().hex[:8])
PY
)}"

mkdir -p "${RUN_DIR}/logs" "${LIVEWEB_CACHE_DIR}"

STEPS_PER_EPOCH="$("${SLIME_VENV}/bin/python" - <<PY
from pathlib import Path

path = Path("${DATASET_PATH}")
if path.suffix == ".parquet":
    import pyarrow.parquet as pq
    rows = pq.read_metadata(path).num_rows
else:
    rows = sum(1 for line in path.open("r", encoding="utf-8") if line.strip())
batch = int("${ROLLOUT_BATCH_SIZE}")
print((rows + batch - 1) // batch)
PY
)"
FINAL_ITERATION="$(( STEPS_PER_EPOCH * NUM_EPOCH ))"

ENV_FILE="${RUN_DIR}/session.env"
python3 - <<PY
from pathlib import Path

env_path = Path("${ENV_FILE}")
env_path.write_text(
    "\n".join(
        [
            "export SLIME_DIR='${SLIME_DIR}'",
            "export SLIME_VENV='${SLIME_VENV}'",
            "export RUN_DIR='${RUN_DIR}'",
            "export TRAIN_GPU_IDS='${TRAIN_GPU_IDS}'",
            "export TRAIN_TP_SIZE='${TRAIN_TP_SIZE}'",
            "export TRAIN_PP_SIZE='${TRAIN_PP_SIZE}'",
            "export TRAIN_CP_SIZE='${TRAIN_CP_SIZE}'",
            "export EVAL_GPU_IDS='${EVAL_GPU_IDS}'",
            "export ARENA_EVAL_MODE='${ARENA_EVAL_MODE}'",
            "export SKIP_ARENA_EVAL='${SKIP_ARENA_EVAL}'",
            "export SKIP_INITIAL_EVAL='${SKIP_INITIAL_EVAL}'",
            "export WANDB_PROJECT='${WANDB_PROJECT}'",
            "export WANDB_GROUP='${WANDB_GROUP}'",
            "export WANDB_RUN_ID='${WANDB_RUN_ID}'",
            "export NUM_EPOCH='${NUM_EPOCH}'",
            "export ROLLOUT_BATCH_SIZE='${ROLLOUT_BATCH_SIZE}'",
            "export GLOBAL_BATCH_SIZE='${GLOBAL_BATCH_SIZE}'",
            "export MAX_TOKENS_PER_GPU='${MAX_TOKENS_PER_GPU}'",
            "export LOG_PROBS_CHUNK_SIZE='${LOG_PROBS_CHUNK_SIZE}'",
            "export RECOMPUTE_LOSS_FUNCTION='${RECOMPUTE_LOSS_FUNCTION}'",
            "export ROLLOUT_MAX_PROMPT_LEN='${ROLLOUT_MAX_PROMPT_LEN}'",
            "export FILTER_COUNT_SAMPLE_SIZE='${FILTER_COUNT_SAMPLE_SIZE}'",
            "export LR='${LR}'",
            "export MIN_LR='${MIN_LR}'",
            "export SAVE_INTERVAL_OVERRIDE='${SAVE_INTERVAL_OVERRIDE}'",
            "export FILTERED_DATASET_ROWS_OVERRIDE='${FILTERED_DATASET_ROWS_OVERRIDE}'",
            "export MIN_TRAIN_GPU_FREE_MEM_MB='${MIN_TRAIN_GPU_FREE_MEM_MB}'",
            "export TRAIN_STEP_TOKEN_BUDGET='${TRAIN_STEP_TOKEN_BUDGET}'",
            "export TRAIN_STEP_LOGIT_BUDGET='${TRAIN_STEP_LOGIT_BUDGET}'",
            "export TRAIN_MIN_DYNAMIC_GLOBAL_BATCH_SIZE='${TRAIN_MIN_DYNAMIC_GLOBAL_BATCH_SIZE}'",
            "export TRAIN_UNDERFILLED_STEP_MIN_SAMPLES='${TRAIN_UNDERFILLED_STEP_MIN_SAMPLES}'",
            "export TRAIN_STEP_PACKING_STRATEGY='${TRAIN_STEP_PACKING_STRATEGY}'",
            "export TRAIN_STEP_LONG_SAMPLE_THRESHOLD='${TRAIN_STEP_LONG_SAMPLE_THRESHOLD}'",
            "export TRAIN_MAX_LONG_SAMPLES_PER_STEP='${TRAIN_MAX_LONG_SAMPLES_PER_STEP}'",
            "export TRAIN_MAX_SINGLE_SAMPLE_TOKENS='${TRAIN_MAX_SINGLE_SAMPLE_TOKENS}'",
            "export TRAIN_MAX_TOTAL_TOKENS_PER_SAMPLE='${TRAIN_MAX_TOTAL_TOKENS_PER_SAMPLE}'",
            "export TRAIN_MAX_RESPONSE_TOKENS_PER_SAMPLE='${TRAIN_MAX_RESPONSE_TOKENS_PER_SAMPLE}'",
            "export RAY_HEAD_PORT='${RAY_HEAD_PORT}'",
            "export RAY_DASHBOARD_PORT='${RAY_DASHBOARD_PORT}'",
            "export DATASET_PATH='${DATASET_PATH}'",
            "export TASK_DATASET_PATH='${TASK_DATASET_PATH}'",
            "export HF_MODEL_DIR='${HF_MODEL_DIR}'",
            "export REF_LOAD_DIR='${REF_LOAD_DIR}'",
            "export LIVEWEB_ARENA_DIR='${LIVEWEB_ARENA_DIR}'",
            "export LIVEWEB_CACHE_DIR='${LIVEWEB_CACHE_DIR}'",
            "export SERVER_PORT='${SERVER_PORT}'",
            "export STEPS_PER_EPOCH='${STEPS_PER_EPOCH}'",
            "export FINAL_ITERATION='${FINAL_ITERATION}'",
        ]
    )
    + "\n"
)
PY

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION_NAME}" >&2
    exit 1
fi

TRAIN_SCRIPT="scripts/run-qwen3-32B-liveweb-sft.sh"
if [[ "${SKIP_ARENA_EVAL}" != "1" && "${ARENA_EVAL_MODE}" == "start_end_serial" ]]; then
    TRAIN_SCRIPT="scripts/run-liveweb-arena-start-end.sh"
fi

TRAIN_CMD="bash -lc 'cd \"${SLIME_DIR}\" && source \"${ENV_FILE}\" && bash ${TRAIN_SCRIPT} 2>&1 | tee \"${RUN_DIR}/logs/train.log\"'"
EVAL_CMD="bash -lc 'cd \"${SLIME_DIR}\" && source \"${ENV_FILE}\" && bash scripts/run-liveweb-arena-eval-watch.sh 2>&1 | tee \"${RUN_DIR}/logs/arena_eval.log\"'"
WANDB_WATCH_CMD="bash -lc 'cd \"${SLIME_DIR}\" && source \"${ENV_FILE}\" && source \"${SLIME_VENV}/bin/activate\" && python scripts/wandb_train_log_watch.py --log-glob \"${RUN_DIR}/wandb/wandb/run-*/files/output_*.log\" --state-file \"${RUN_DIR}/logs/wandb_train_watch_state.json\" --wandb-project \"${WANDB_PROJECT}\" --wandb-run-id \"${WANDB_RUN_ID}\" --wandb-dir \"${RUN_DIR}/wandb\" 2>&1 | tee \"${RUN_DIR}/logs/wandb_train_watch.log\"'"

tmux new-session -d -s "${SESSION_NAME}" -n train "${TRAIN_CMD}"
if [[ "${SKIP_ARENA_EVAL}" != "1" && "${ARENA_EVAL_MODE}" != "start_end_serial" ]]; then
    tmux new-window -t "${SESSION_NAME}" -n arena_eval "${EVAL_CMD}"
fi
tmux new-window -t "${SESSION_NAME}" -n wandb_watch "${WANDB_WATCH_CMD}"
tmux select-window -t "${SESSION_NAME}:train"

echo "tmux session started: ${SESSION_NAME}"
echo "attach: tmux attach -t ${SESSION_NAME}"
echo "run dir: ${RUN_DIR}"
