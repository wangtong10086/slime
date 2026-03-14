#!/bin/bash

set -euo pipefail

if [[ -z "${SLIME_DIR:-}" || -z "${RUN_DIR:-}" || -z "${LIVEWEB_ARENA_DIR:-}" ]]; then
    echo "Required environment variables are missing. Source the session env first." >&2
    exit 1
fi

SLIME_VENV="${SLIME_VENV:-${HOME}/.venvs/slime}"
EVAL_GPU_IDS="${EVAL_GPU_IDS:-4,5,6,7}"
IFS=',' read -r -a EVAL_GPU_ARRAY <<< "${EVAL_GPU_IDS}"
EVAL_TP_SIZE="${#EVAL_GPU_ARRAY[@]}"

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

cd "${SLIME_DIR}"

RUN_CONFIG_PATH="${RUN_DIR}/run_config.json"
for _ in $(seq 1 300); do
    if [[ -f "${RUN_CONFIG_PATH}" ]]; then
        break
    fi
    sleep 2
done

if [[ -f "${RUN_CONFIG_PATH}" ]]; then
    readarray -t RUN_CONFIG_VALUES < <(python - <<PY
import json
from pathlib import Path
cfg = json.loads(Path("${RUN_CONFIG_PATH}").read_text())
print(cfg["steps_per_epoch"])
print(cfg["final_iteration"])
print(cfg["num_epoch"])
PY
)
    STEPS_PER_EPOCH="${RUN_CONFIG_VALUES[0]}"
    FINAL_ITERATION="${RUN_CONFIG_VALUES[1]}"
    NUM_EPOCH="${RUN_CONFIG_VALUES[2]}"
fi

ensure_playwright_chromium

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
    --steps-per-epoch "${STEPS_PER_EPOCH}" \
    --num-epoch "${NUM_EPOCH}" \
    --stop-after-iteration "${FINAL_ITERATION}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-group "${WANDB_GROUP}" \
    --wandb-run-id "${WANDB_RUN_ID}"
