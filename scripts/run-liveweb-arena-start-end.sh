#!/bin/bash

set -euo pipefail

if [[ -z "${SLIME_DIR:-}" || -z "${RUN_DIR:-}" || -z "${LIVEWEB_ARENA_DIR:-}" ]]; then
    echo "Required environment variables are missing. Source the session env first." >&2
    exit 1
fi

SLIME_VENV="${SLIME_VENV:-${HOME}/.venvs/slime}"
EVAL_GPU_IDS="${EVAL_GPU_IDS:-${TRAIN_GPU_IDS:-0,1,2,3,4,5,6,7}}"
IFS=',' read -r -a EVAL_GPU_ARRAY <<< "${EVAL_GPU_IDS}"
EVAL_TP_SIZE="${EVAL_TP_SIZE:-${#EVAL_GPU_ARRAY[@]}}"
SKIP_INITIAL_EVAL="${SKIP_INITIAL_EVAL:-0}"
RUN_CONFIG_PATH="${RUN_DIR}/run_config.json"

if [[ -n "${SLIME_VENV}" ]]; then
    # shellcheck disable=SC1090
    source "${SLIME_VENV}/bin/activate"
fi

ensure_playwright_chromium() {
    local install_location
    local shell_path
    local attempt

    install_location="$(
        python -m playwright install --dry-run chromium-headless-shell 2>/dev/null \
            | awk '/Install location:/ {print $3; exit}'
    )"
    shell_path="${install_location}/chrome-headless-shell-linux64/chrome-headless-shell"
    if [[ -n "${install_location}" && -x "${shell_path}" ]]; then
        return 0
    fi

    for attempt in 1 2 3; do
        if python -m playwright install chromium-headless-shell; then
            return 0
        fi
        echo "playwright chromium-headless-shell install failed on attempt ${attempt}, retrying..." >&2
        sleep 5
    done
    echo "playwright chromium-headless-shell install failed after retries" >&2
    return 1
}

run_initial_eval() {
    python scripts/liveweb_arena_eval_watch.py \
        --slime-dir "${SLIME_DIR}" \
        --watch-dir "${RUN_DIR}/checkpoints" \
        --origin-hf-dir "${HF_MODEL_DIR}" \
        --converted-root "${RUN_DIR}/arena_hf" \
        --arena-dir "${LIVEWEB_ARENA_DIR}" \
        --arena-cache-dir "${LIVEWEB_CACHE_DIR}" \
        --arena-output-dir "${RUN_DIR}/arena_eval" \
        --python-bin "${SLIME_VENV}/bin/python" \
        --eval-gpus "${EVAL_GPU_IDS}" \
        --tp-size "${EVAL_TP_SIZE}" \
        --server-port "${SERVER_PORT}" \
        --served-model-name "qwen3-32b-liveweb-sft" \
        --disable-cuda-graph \
        --steps-per-epoch "${STEPS_PER_EPOCH:-1}" \
        --num-epoch "${NUM_EPOCH}" \
        --stop-after-iteration 0 \
        --only-initial-eval \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-group "${WANDB_GROUP}" \
        --wandb-run-id "${WANDB_RUN_ID}"
}

run_final_eval() {
    if [[ ! -f "${RUN_CONFIG_PATH}" ]]; then
        echo "run_config.json not found, skipping final LiveWeb Arena evaluation." >&2
        return 1
    fi

    readarray -t RUN_CONFIG_VALUES < <(python - <<PY
import json
from pathlib import Path
cfg = json.loads(Path("${RUN_CONFIG_PATH}").read_text())
print(cfg["steps_per_epoch"])
print(cfg["final_iteration"])
print(cfg["num_epoch"])
PY
)

    local steps_per_epoch="${RUN_CONFIG_VALUES[0]}"
    local final_iteration="${RUN_CONFIG_VALUES[1]}"
    local num_epoch="${RUN_CONFIG_VALUES[2]}"
    local final_ckpt_dir="${RUN_DIR}/checkpoints/iter_$(printf '%07d' "${final_iteration}")"

    if [[ ! -d "${final_ckpt_dir}" ]]; then
        echo "Final checkpoint ${final_ckpt_dir} not found, skipping final LiveWeb Arena evaluation." >&2
        return 1
    fi

    python scripts/liveweb_arena_eval_watch.py \
        --slime-dir "${SLIME_DIR}" \
        --watch-dir "${RUN_DIR}/checkpoints" \
        --origin-hf-dir "${HF_MODEL_DIR}" \
        --converted-root "${RUN_DIR}/arena_hf" \
        --arena-dir "${LIVEWEB_ARENA_DIR}" \
        --arena-cache-dir "${LIVEWEB_CACHE_DIR}" \
        --arena-output-dir "${RUN_DIR}/arena_eval" \
        --python-bin "${SLIME_VENV}/bin/python" \
        --eval-gpus "${EVAL_GPU_IDS}" \
        --tp-size "${EVAL_TP_SIZE}" \
        --server-port "${SERVER_PORT}" \
        --served-model-name "qwen3-32b-liveweb-sft" \
        --disable-cuda-graph \
        --steps-per-epoch "${steps_per_epoch}" \
        --num-epoch "${num_epoch}" \
        --stop-after-iteration "${final_iteration}" \
        --skip-initial-eval \
        --only-iteration "${final_iteration}" \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-group "${WANDB_GROUP}" \
        --wandb-run-id "${WANDB_RUN_ID}"
}

cd "${SLIME_DIR}"
mkdir -p "${RUN_DIR}/arena_eval" "${RUN_DIR}/arena_hf" "${LIVEWEB_CACHE_DIR}"
ensure_playwright_chromium

echo "Running initial LiveWeb Arena evaluation before training."
if [[ "${SKIP_INITIAL_EVAL}" == "1" ]]; then
    echo "Skipping initial LiveWeb Arena evaluation."
else
    run_initial_eval
fi

echo "Starting training after initial LiveWeb Arena evaluation."
if bash scripts/run-qwen3-32B-liveweb-sft.sh; then
    train_exit_code=0
else
    train_exit_code=$?
fi

if [[ "${train_exit_code}" -eq 0 ]]; then
    echo "Training finished successfully. Running final LiveWeb Arena evaluation."
    run_final_eval
fi

exit "${train_exit_code}"
