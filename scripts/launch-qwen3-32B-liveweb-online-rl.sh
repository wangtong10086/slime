#!/bin/bash

set -euo pipefail

USER_HOME="${HOME:-/home/xmyf}"
SLIME_DIR="${SLIME_DIR:-${USER_HOME}/slime}"
TRAIN_PHASE="${TRAIN_PHASE:-preflight}"
NUM_ROLLOUT_OVERRIDE_VALUE="${NUM_ROLLOUT_OVERRIDE:-}"
SAVE_INTERVAL_OVERRIDE_VALUE="${SAVE_INTERVAL_OVERRIDE:-}"
EVAL_INTERVAL_OVERRIDE_VALUE="${EVAL_INTERVAL_OVERRIDE:-}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SESSION_NAME="${SESSION_NAME:-slime_liveweb_online_rl_${TRAIN_PHASE}_${TIMESTAMP}}"
RUN_ROOT="${RUN_ROOT:-/data/slime_runs/liveweb_online_rl_qwen3_32b_${TIMESTAMP}}"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${LOG_DIR}"

tmux kill-session -t "${SESSION_NAME}" >/dev/null 2>&1 || true
tmux new-session -d -s "${SESSION_NAME}" -n train
tmux send-keys -t "${SESSION_NAME}:train" "cd ${SLIME_DIR}" C-m
tmux send-keys -t "${SESSION_NAME}:train" "RUN_ROOT='${RUN_ROOT}' TRAIN_PHASE='${TRAIN_PHASE}' NUM_ROLLOUT_OVERRIDE='${NUM_ROLLOUT_OVERRIDE_VALUE}' SAVE_INTERVAL_OVERRIDE='${SAVE_INTERVAL_OVERRIDE_VALUE}' EVAL_INTERVAL_OVERRIDE='${EVAL_INTERVAL_OVERRIDE_VALUE}' bash ${SLIME_DIR}/scripts/run-qwen3-32B-liveweb-online-rl.sh | tee ${LOG_DIR}/train.log" C-m

echo "session=${SESSION_NAME}"
echo "run_root=${RUN_ROOT}"
