#!/bin/bash

set -euo pipefail

USER_HOME="${HOME:-/home/xmyf}"
SLIME_DIR="${SLIME_DIR:-${USER_HOME}/slime}"
SLIME_VENV="${SLIME_VENV:-${USER_HOME}/.venvs/slime}"
MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-${USER_HOME}/slime_deps/Megatron-LM}"
CUDA_HOME="${CUDA_HOME:-${USER_HOME}/slime_deps/cuda-12.9}"
HF_MODEL_DIR="${HF_MODEL_DIR:-${USER_HOME}/Qwen3-32B}"
REF_LOAD_DIR="${REF_LOAD_DIR:-}"
TASK_DATASET_PATH="${TASK_DATASET_PATH:-/data/liveweb_sft/liveweb_success_sft_20260321/train_32k.jsonl}"
DATASET_PATH="${DATASET_PATH:-${TASK_DATASET_PATH}}"
RUN_DIR="${RUN_DIR:-${USER_HOME}/slime_runs/qwen3-32b-liveweb-sft}"
WANDB_SETTINGS_FILE="${WANDB_SETTINGS_FILE:-${USER_HOME}/.config/wandb/settings}"
TRAIN_GPU_IDS="${TRAIN_GPU_IDS:-4,5,6,7}"
WANDB_PROJECT="${WANDB_PROJECT:-slime-liveweb}"
WANDB_GROUP="${WANDB_GROUP:-qwen3-32b-liveweb-sft}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
RAY_HEAD_PORT="${RAY_HEAD_PORT:-6382}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8266}"
RAY_JOB_PORT="${RAY_JOB_PORT:-${RAY_DASHBOARD_PORT}}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
NUM_EPOCH="${NUM_EPOCH:-1}"
LR="${LR:-1e-5}"
MIN_LR="${MIN_LR:-1e-6}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-512}"
LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-2048}"
RECOMPUTE_LOSS_FUNCTION="${RECOMPUTE_LOSS_FUNCTION:-1}"
ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-32768}"
RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-4}"
FILTER_COUNT_SAMPLE_SIZE="${FILTER_COUNT_SAMPLE_SIZE:-128}"
FILTERED_DATASET_ROWS_OVERRIDE="${FILTERED_DATASET_ROWS_OVERRIDE:-}"
SAVE_INTERVAL_OVERRIDE="${SAVE_INTERVAL_OVERRIDE:-}"
MIN_TRAIN_GPU_FREE_MEM_MB="${MIN_TRAIN_GPU_FREE_MEM_MB:-70000}"
NO_SAVE_OPTIM="${NO_SAVE_OPTIM:-1}"
TOOLCALL_REGRESSION_RATIO="${TOOLCALL_REGRESSION_RATIO:-0.2}"
TRAIN_STEP_TOKEN_BUDGET="${TRAIN_STEP_TOKEN_BUDGET:-48000}"
TRAIN_STEP_LOGIT_BUDGET="${TRAIN_STEP_LOGIT_BUDGET:-24000}"
TRAIN_UNDERFILLED_STEP_MIN_SAMPLES="${TRAIN_UNDERFILLED_STEP_MIN_SAMPLES:-4}"
TRAIN_STEP_PACKING_STRATEGY="${TRAIN_STEP_PACKING_STRATEGY:-greedy_desc}"
TRAIN_STEP_LONG_SAMPLE_THRESHOLD="${TRAIN_STEP_LONG_SAMPLE_THRESHOLD:-16800}"
TRAIN_MAX_LONG_SAMPLES_PER_STEP="${TRAIN_MAX_LONG_SAMPLES_PER_STEP:-2}"
TRAIN_MAX_SINGLE_SAMPLE_TOKENS="${TRAIN_MAX_SINGLE_SAMPLE_TOKENS:-12000}"
TRAIN_MAX_TOTAL_TOKENS_PER_SAMPLE="${TRAIN_MAX_TOTAL_TOKENS_PER_SAMPLE:-4096}"
TRAIN_MAX_RESPONSE_TOKENS_PER_SAMPLE="${TRAIN_MAX_RESPONSE_TOKENS_PER_SAMPLE:-2048}"
REGRESSION_DATASET_PATH="${REGRESSION_DATASET_PATH:-${RUN_DIR}/datasets/toolcall_regression.jsonl}"
MIXED_DATASET_PATH="${MIXED_DATASET_PATH:-${RUN_DIR}/datasets/train_mixed.jsonl}"
FORMATCHECK_OUTPUT_DIR="${FORMATCHECK_OUTPUT_DIR:-${RUN_DIR}/format_checks}"
ENABLE_FORMAT_PREFLIGHT="${ENABLE_FORMAT_PREFLIGHT:-1}"
ENABLE_FORMAT_POSTCHECK="${ENABLE_FORMAT_POSTCHECK:-1}"
PARSER_HEALTH_GATE="${PARSER_HEALTH_GATE:-0.99}"
FORMATCHECK_SERVER_PORT="${FORMATCHECK_SERVER_PORT:-31200}"
FORMATCHECK_SERVER_GPU_IDS="${FORMATCHECK_SERVER_GPU_IDS:-0,1}"
FORMATCHECK_SERVER_TP_SIZE="${FORMATCHECK_SERVER_TP_SIZE:-2}"
FORMATCHECK_SERVER_MEM_FRACTION="${FORMATCHECK_SERVER_MEM_FRACTION:-0.88}"
FORMATCHECK_LIMIT="${FORMATCHECK_LIMIT:-64}"

if [[ -n "${SLIME_VENV}" ]]; then
    # shellcheck disable=SC1090
    source "${SLIME_VENV}/bin/activate"
fi

export SLIME_DIR MEGATRON_LM_PATH CUDA_HOME HF_MODEL_DIR REF_LOAD_DIR DATASET_PATH TASK_DATASET_PATH RUN_DIR

if [[ ! -d "${SLIME_DIR}" ]]; then
    echo "slime repo not found: ${SLIME_DIR}" >&2
    exit 1
fi
if [[ ! -d "${MEGATRON_LM_PATH}" ]]; then
    echo "Megatron-LM not found: ${MEGATRON_LM_PATH}" >&2
    exit 1
fi
if [[ ! -d "${CUDA_HOME}" ]]; then
    echo "CUDA_HOME not found: ${CUDA_HOME}" >&2
    exit 1
fi
if [[ ! -d "${HF_MODEL_DIR}" ]]; then
    echo "HF model dir not found: ${HF_MODEL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${TASK_DATASET_PATH}" ]]; then
    echo "Task dataset not found: ${TASK_DATASET_PATH}" >&2
    exit 1
fi

export PATH="${CUDA_HOME}/bin:${PATH}"
export CPATH="${CUDA_HOME}/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="${CUDA_HOME}/include:${CPLUS_INCLUDE_PATH:-}"

NVIDIA_SITE_PACKAGES="${VIRTUAL_ENV:-${SLIME_VENV}}/lib/python3.12/site-packages/nvidia"
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

IFS=',' read -r -a TRAIN_GPU_ARRAY <<< "${TRAIN_GPU_IDS}"
N_TRAIN_GPUS="${#TRAIN_GPU_ARRAY[@]}"
if [[ "${N_TRAIN_GPUS}" -lt 1 ]]; then
    echo "TRAIN_GPU_IDS is empty." >&2
    exit 1
fi

TRAIN_TP_SIZE="${TRAIN_TP_SIZE:-${N_TRAIN_GPUS}}"
TRAIN_PP_SIZE="${TRAIN_PP_SIZE:-1}"
TRAIN_CP_SIZE="${TRAIN_CP_SIZE:-1}"
CONVERT_MODEL_PARALLEL_SIZE=$(( TRAIN_TP_SIZE * TRAIN_PP_SIZE ))
TRAIN_PARALLEL_SIZE=$(( TRAIN_TP_SIZE * TRAIN_PP_SIZE * TRAIN_CP_SIZE ))
if (( TRAIN_PARALLEL_SIZE < 1 )); then
    echo "Invalid train parallelism: TRAIN_TP_SIZE (${TRAIN_TP_SIZE}) * TRAIN_PP_SIZE (${TRAIN_PP_SIZE}) * TRAIN_CP_SIZE (${TRAIN_CP_SIZE}) must be >= 1." >&2
    exit 1
fi
if (( N_TRAIN_GPUS % TRAIN_PARALLEL_SIZE != 0 )); then
    echo "Invalid train parallelism: number of training GPUs (${N_TRAIN_GPUS}) must be divisible by TRAIN_TP_SIZE (${TRAIN_TP_SIZE}) * TRAIN_PP_SIZE (${TRAIN_PP_SIZE}) * TRAIN_CP_SIZE (${TRAIN_CP_SIZE})." >&2
    exit 1
fi
TRAIN_DP_SIZE=$(( N_TRAIN_GPUS / TRAIN_PARALLEL_SIZE ))
LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.03}"
if [[ -z "${TRAIN_MIN_DYNAMIC_GLOBAL_BATCH_SIZE}" ]]; then
    if (( GLOBAL_BATCH_SIZE > 1 )); then
        TRAIN_MIN_DYNAMIC_GLOBAL_BATCH_SIZE=$(( GLOBAL_BATCH_SIZE / 2 ))
    else
        TRAIN_MIN_DYNAMIC_GLOBAL_BATCH_SIZE=1
    fi
fi

if [[ -z "${REF_LOAD_DIR}" ]]; then
    REF_LOAD_DIR="${USER_HOME}/Qwen3-32B_torch_dist_tp${TRAIN_TP_SIZE}_pp${TRAIN_PP_SIZE}"
fi
export REF_LOAD_DIR TRAIN_TP_SIZE TRAIN_PP_SIZE TRAIN_CP_SIZE TRAIN_DP_SIZE CONVERT_MODEL_PARALLEL_SIZE TRAIN_PARALLEL_SIZE LR_WARMUP_FRACTION
export LOG_PROBS_CHUNK_SIZE RECOMPUTE_LOSS_FUNCTION
export TRAIN_STEP_TOKEN_BUDGET TRAIN_STEP_LOGIT_BUDGET TRAIN_MIN_DYNAMIC_GLOBAL_BATCH_SIZE
export TRAIN_UNDERFILLED_STEP_MIN_SAMPLES TRAIN_STEP_PACKING_STRATEGY TRAIN_STEP_LONG_SAMPLE_THRESHOLD
export TRAIN_MAX_LONG_SAMPLES_PER_STEP TRAIN_MAX_SINGLE_SAMPLE_TOKENS
export TRAIN_MAX_TOTAL_TOKENS_PER_SAMPLE TRAIN_MAX_RESPONSE_TOKENS_PER_SAMPLE

check_train_gpu_memory() {
    local gpu_list_csv="$1"
    local min_free_mem_mb="$2"
    local query_output

    query_output="$(nvidia-smi --query-gpu=index,memory.free,memory.used --format=csv,noheader,nounits)"

    python - <<PY
import sys

requested = [int(x) for x in "${gpu_list_csv}".split(",") if x.strip()]
min_free = int("${min_free_mem_mb}")
stats = {}
for line in """${query_output}""".strip().splitlines():
    if not line.strip():
        continue
    idx_str, free_str, used_str = [part.strip() for part in line.split(",")]
    stats[int(idx_str)] = (int(free_str), int(used_str))

missing = [idx for idx in requested if idx not in stats]
if missing:
    raise SystemExit(f"Missing GPU stats for indices: {missing}")

insufficient = []
for idx in requested:
    free_mem, used_mem = stats[idx]
    if free_mem < min_free:
        insufficient.append((idx, free_mem, used_mem))

if insufficient:
    print("Selected training GPUs do not have enough free memory:", file=sys.stderr)
    for idx, free_mem, used_mem in insufficient:
        print(
            f"  GPU {idx}: free={free_mem} MiB, used={used_mem} MiB, required_free>={min_free} MiB",
            file=sys.stderr,
        )
    raise SystemExit(1)

print(
    "GPU preflight passed: "
    + ", ".join(
        f"GPU {idx} free={stats[idx][0]} MiB used={stats[idx][1]} MiB"
        for idx in requested
    )
)
PY
}

check_train_gpu_memory "${TRAIN_GPU_IDS}" "${MIN_TRAIN_GPU_FREE_MEM_MB}"

RUN_DATASET_DIR="${RUN_DIR}/datasets"
mkdir -p "${RUN_DATASET_DIR}" "${FORMATCHECK_OUTPUT_DIR}"

python "${SLIME_DIR}/scripts/build_liveweb_toolcall_regression_dataset.py" \
    --input "${TASK_DATASET_PATH}" \
    --output "${REGRESSION_DATASET_PATH}" \
    --hf-model-dir "${HF_MODEL_DIR}"

python "${SLIME_DIR}/scripts/build_liveweb_mixed_sft_dataset.py" \
    --task-dataset "${TASK_DATASET_PATH}" \
    --regression-dataset "${REGRESSION_DATASET_PATH}" \
    --output "${MIXED_DATASET_PATH}" \
    --regression-ratio "${TOOLCALL_REGRESSION_RATIO}"

DATASET_PATH="${MIXED_DATASET_PATH}"
export DATASET_PATH WANDB_SETTINGS_FILE RUN_DIR ROLLOUT_MAX_PROMPT_LEN FILTER_COUNT_SAMPLE_SIZE FILTERED_DATASET_ROWS_OVERRIDE DATASET_ROWS="" STEPS_PER_EPOCH="" FINAL_ITERATION=""

DATASET_ROWS="$(python - <<PY
import os
import json
from transformers import AutoTokenizer

dataset_path = os.environ["DATASET_PATH"]
max_prompt_len = int(os.environ["ROLLOUT_MAX_PROMPT_LEN"])
override = os.environ.get("FILTERED_DATASET_ROWS_OVERRIDE", "").strip()
if override:
    print(int(override))
    raise SystemExit

sample_size = max(1, int(os.environ["FILTER_COUNT_SAMPLE_SIZE"]))
def read_rows(path):
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    elif path.endswith(".parquet"):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches():
            yield from batch.to_pylist()
    else:
        raise SystemExit(f"Unsupported dataset format: {path}")

raw_rows = 0
cached_rows = []
for row in read_rows(dataset_path):
    raw_rows += 1
    cached_rows.append(row)
target_sample_size = min(raw_rows, sample_size)
stride = max(raw_rows // max(target_sample_size, 1), 1)

tokenizer = AutoTokenizer.from_pretrained(os.environ["HF_MODEL_DIR"], trust_remote_code=True)

count = 0
sampled = 0
row_idx = 0
for row in cached_rows:
    should_sample = (row_idx % stride == 0) and (sampled < target_sample_size)
    row_idx += 1
    if not should_sample:
        continue
    tools = row.get("tools")
    try:
        if tools is not None:
            input_ids = tokenizer.apply_chat_template(
                row["messages"],
                tools=tools,
                tokenize=True,
                add_generation_prompt=False,
            )
        else:
            input_ids = tokenizer.apply_chat_template(
                row["messages"],
                tokenize=True,
                add_generation_prompt=False,
            )
    except TypeError:
        input_ids = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=True,
            add_generation_prompt=False,
        )
    sampled += 1
    if len(input_ids) <= max_prompt_len:
        count += 1

estimated_rows = max(1, round(raw_rows * count / max(sampled, 1)))
print(estimated_rows)
PY
)"
STEPS_PER_EPOCH="$(( (DATASET_ROWS + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE ))"
FINAL_ITERATION="$(( STEPS_PER_EPOCH * NUM_EPOCH ))"
SAVE_INTERVAL="${SAVE_INTERVAL_OVERRIDE:-${FINAL_ITERATION}}"

WANDB_KEY=""
if [[ -f "${WANDB_SETTINGS_FILE}" ]]; then
    WANDB_KEY="$(python - <<PY
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

RUN_CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
RUN_WANDB_DIR="${RUN_DIR}/wandb"
RUN_LOG_DIR="${RUN_DIR}/logs"
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/slime_ray_qwen3_32b_liveweb}"
mkdir -p "${RUN_CHECKPOINT_DIR}" "${RUN_WANDB_DIR}" "${RUN_LOG_DIR}" "${RAY_TMPDIR}"

export DATASET_ROWS STEPS_PER_EPOCH FINAL_ITERATION ROLLOUT_BATCH_SIZE GLOBAL_BATCH_SIZE TRAIN_GPU_IDS WANDB_PROJECT WANDB_GROUP WANDB_RUN_ID ROLLOUT_MAX_PROMPT_LEN MAX_TOKENS_PER_GPU RECOMPUTE_NUM_LAYERS
export LOG_PROBS_CHUNK_SIZE RECOMPUTE_LOSS_FUNCTION
export TRAIN_STEP_TOKEN_BUDGET TRAIN_STEP_LOGIT_BUDGET TRAIN_MIN_DYNAMIC_GLOBAL_BATCH_SIZE
export TRAIN_UNDERFILLED_STEP_MIN_SAMPLES TRAIN_STEP_PACKING_STRATEGY TRAIN_STEP_LONG_SAMPLE_THRESHOLD
export TRAIN_MAX_LONG_SAMPLES_PER_STEP TRAIN_MAX_SINGLE_SAMPLE_TOKENS
export TRAIN_MAX_TOTAL_TOKENS_PER_SAMPLE TRAIN_MAX_RESPONSE_TOKENS_PER_SAMPLE

python - <<PY
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "run_config.json").write_text(
    json.dumps(
        {
            "dataset_rows": int(os.environ["DATASET_ROWS"]),
            "steps_per_epoch": int(os.environ["STEPS_PER_EPOCH"]),
            "num_epoch": int("${NUM_EPOCH}"),
            "final_iteration": int(os.environ["FINAL_ITERATION"]),
            "rollout_batch_size": int(os.environ["ROLLOUT_BATCH_SIZE"]),
            "global_batch_size": int(os.environ["GLOBAL_BATCH_SIZE"]),
            "rollout_max_prompt_len": int(os.environ["ROLLOUT_MAX_PROMPT_LEN"]),
            "max_tokens_per_gpu": int(os.environ["MAX_TOKENS_PER_GPU"]),
            "log_probs_chunk_size": int(os.environ["LOG_PROBS_CHUNK_SIZE"]),
            "recompute_loss_function": os.environ["RECOMPUTE_LOSS_FUNCTION"] == "1",
            "recompute_num_layers": int(os.environ["RECOMPUTE_NUM_LAYERS"]),
            "use_dynamic_batch_size": True,
            "train_step_token_budget": int(os.environ["TRAIN_STEP_TOKEN_BUDGET"]),
            "train_step_logit_budget": int(os.environ["TRAIN_STEP_LOGIT_BUDGET"]),
            "train_min_dynamic_global_batch_size": int(os.environ["TRAIN_MIN_DYNAMIC_GLOBAL_BATCH_SIZE"]),
            "train_underfilled_step_min_samples": int(os.environ["TRAIN_UNDERFILLED_STEP_MIN_SAMPLES"]),
            "train_step_packing_strategy": os.environ["TRAIN_STEP_PACKING_STRATEGY"],
            "train_step_long_sample_threshold": int(os.environ["TRAIN_STEP_LONG_SAMPLE_THRESHOLD"]),
            "train_max_long_samples_per_step": int(os.environ["TRAIN_MAX_LONG_SAMPLES_PER_STEP"]),
            "train_max_single_sample_tokens": int(os.environ["TRAIN_MAX_SINGLE_SAMPLE_TOKENS"]),
            "train_max_total_tokens_per_sample": int(os.environ["TRAIN_MAX_TOTAL_TOKENS_PER_SAMPLE"]),
            "train_max_response_tokens_per_sample": int(os.environ["TRAIN_MAX_RESPONSE_TOKENS_PER_SAMPLE"]),
            "train_gpu_ids": os.environ["TRAIN_GPU_IDS"],
            "train_tp_size": int(os.environ["TRAIN_TP_SIZE"]),
            "train_pp_size": int(os.environ["TRAIN_PP_SIZE"]),
            "train_cp_size": int(os.environ["TRAIN_CP_SIZE"]),
            "train_dp_size": int(os.environ["TRAIN_DP_SIZE"]),
            "ref_load_dir": os.environ["REF_LOAD_DIR"],
            "wandb_project": os.environ["WANDB_PROJECT"],
            "wandb_group": os.environ["WANDB_GROUP"],
            "wandb_run_id": os.environ["WANDB_RUN_ID"],
            "task_dataset_path": os.environ["TASK_DATASET_PATH"],
            "regression_dataset_path": "${REGRESSION_DATASET_PATH}",
            "mixed_dataset_path": os.environ["DATASET_PATH"],
            "toolcall_regression_ratio": float("${TOOLCALL_REGRESSION_RATIO}"),
            "parser_health_gate": float("${PARSER_HEALTH_GATE}"),
            "enable_format_preflight": "${ENABLE_FORMAT_PREFLIGHT}" == "1",
            "enable_format_postcheck": "${ENABLE_FORMAT_POSTCHECK}" == "1",
        },
        indent=2,
    )
)
PY

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-32B.sh"

ensure_ref_load_dir() {
    if [[ -f "${REF_LOAD_DIR}/latest_checkpointed_iteration.txt" \
        && -d "${REF_LOAD_DIR}/release" \
        && -f "${REF_LOAD_DIR}/release/common.pt" \
        && -f "${REF_LOAD_DIR}/release/metadata.json" \
        && -f "${REF_LOAD_DIR}/release/.metadata" ]]; then
        return
    fi

    echo "Converting ${HF_MODEL_DIR} to torch_dist at ${REF_LOAD_DIR}"
    mkdir -p "${REF_LOAD_DIR}"
    cd "${SLIME_DIR}"
    CUDA_VISIBLE_DEVICES="${TRAIN_GPU_IDS}" CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH="${MEGATRON_LM_PATH}" torchrun \
        --nproc-per-node "${CONVERT_MODEL_PARALLEL_SIZE}" \
        tools/convert_hf_to_torch_dist.py \
        "${MODEL_ARGS[@]}" \
        --tensor-model-parallel-size "${TRAIN_TP_SIZE}" \
        --pipeline-model-parallel-size "${TRAIN_PP_SIZE}" \
        --hf-checkpoint "${HF_MODEL_DIR}" \
        --save "${REF_LOAD_DIR}"
}

ensure_ref_load_dir

run_formatcheck() {
    local model_path="$1"
    local output_dir="$2"
    local model_name="$3"
    MODEL_PATH="${model_path}" \
    OUTPUT_DIR="${output_dir}" \
    REGRESSION_DATASET_PATH="${REGRESSION_DATASET_PATH}" \
    MODEL_NAME="${model_name}" \
    PORT="${FORMATCHECK_SERVER_PORT}" \
    SERVER_GPU_IDS="${FORMATCHECK_SERVER_GPU_IDS}" \
    SERVER_TP_SIZE="${FORMATCHECK_SERVER_TP_SIZE}" \
    SERVER_MEM_FRACTION="${FORMATCHECK_SERVER_MEM_FRACTION}" \
    FORMATCHECK_LIMIT="${FORMATCHECK_LIMIT}" \
    PARSER_HEALTH_GATE="${PARSER_HEALTH_GATE}" \
        bash "${SLIME_DIR}/scripts/run-qwen3-32B-liveweb-sft-formatcheck.sh"
}

latest_iteration_dir() {
    local checkpoint_root="$1"
    local tracker="${checkpoint_root}/latest_checkpointed_iteration.txt"
    if [[ ! -f "${tracker}" ]]; then
        return 1
    fi
    local iteration
    iteration="$(cat "${tracker}")"
    [[ -n "${iteration}" ]] || return 1
    printf "%s/iter_%07d" "${checkpoint_root}" "${iteration}"
}

convert_latest_checkpoint_to_hf() {
    local checkpoint_root="$1"
    local output_dir="$2"
    local iteration_dir
    iteration_dir="$(latest_iteration_dir "${checkpoint_root}")" || return 1
    rm -rf "${output_dir}"
    mkdir -p "$(dirname "${output_dir}")"
    (
        cd "${SLIME_DIR}"
        python3 tools/convert_torch_dist_to_hf.py \
            --input-dir "${iteration_dir}" \
            --output-dir "${output_dir}" \
            --origin-hf-dir "${HF_MODEL_DIR}" \
            --force
    )
}

if [[ "${ENABLE_FORMAT_PREFLIGHT}" == "1" ]]; then
    echo "Running tool-call parser preflight on base model"
    run_formatcheck "${HF_MODEL_DIR}" "${FORMATCHECK_OUTPUT_DIR}/preflight_base" "qwen3-32b-base"
fi

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export no_proxy="127.0.0.1,${MASTER_ADDR}"
export PYTHONBUFFERED=16
export RAY_TMPDIR

ray stop --force >/dev/null 2>&1 || true
CUDA_VISIBLE_DEVICES="${TRAIN_GPU_IDS}" ray start \
    --head \
    --port "${RAY_HEAD_PORT}" \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "${N_TRAIN_GPUS}" \
    --disable-usage-stats \
    --dashboard-host 0.0.0.0 \
    --dashboard-port "${RAY_DASHBOARD_PORT}"

CKPT_ARGS=(
    --hf-checkpoint "${HF_MODEL_DIR}"
    --ref-load "${REF_LOAD_DIR}"
    --load "${RUN_CHECKPOINT_DIR}"
    --save "${RUN_CHECKPOINT_DIR}"
    --save-interval "${SAVE_INTERVAL}"
)

SFT_ARGS=(
    --rollout-function-path slime.rollout.sft_rollout.generate_rollout
    --prompt-data "${DATASET_PATH}"
    --input-key messages
    --rollout-shuffle
    --num-epoch "${NUM_EPOCH}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}"
    --loss-type sft_loss
    --calculate-per-token-loss
    --disable-compute-advantages-and-returns
    --debug-train-only
)

PERF_ARGS=(
    --tensor-model-parallel-size "${TRAIN_TP_SIZE}"
    --sequence-parallel
    --pipeline-model-parallel-size "${TRAIN_PP_SIZE}"
    --context-parallel-size "${TRAIN_CP_SIZE}"
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers "${RECOMPUTE_NUM_LAYERS}"
    --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}"
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)
if [[ "${RECOMPUTE_LOSS_FUNCTION}" == "1" ]]; then
    PERF_ARGS+=(--recompute-loss-function)
fi

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR}"
    --lr-decay-style cosine
    --min-lr "${MIN_LR}"
    --lr-warmup-fraction "${LR_WARMUP_FRACTION}"
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
)
if [[ "${NO_SAVE_OPTIM}" == "1" ]]; then
    OPTIMIZER_ARGS+=(--no-save-optim)
fi

WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
    --wandb-dir "${RUN_WANDB_DIR}"
    --wandb-run-id "${WANDB_RUN_ID}"
    --disable-wandb-random-suffix
    --wandb-always-use-train-step
)
if [[ -n "${WANDB_KEY}" ]]; then
    WANDB_ARGS+=(--wandb-key "${WANDB_KEY}")
fi

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

export MEGATRON_LM_PATH HAS_NVLINK
RUNTIME_ENV_JSON="$(python - <<PY
import json
import os
print(
    json.dumps(
        {
            "env_vars": {
                "PYTHONPATH": os.environ["MEGATRON_LM_PATH"],
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "NCCL_NVLS_ENABLE": os.environ["HAS_NVLINK"],
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "LOG_PROBS_CHUNK_SIZE": os.environ["LOG_PROBS_CHUNK_SIZE"],
                "RECOMPUTE_LOSS_FUNCTION": os.environ["RECOMPUTE_LOSS_FUNCTION"],
                "TRAIN_STEP_TOKEN_BUDGET": os.environ["TRAIN_STEP_TOKEN_BUDGET"],
                "TRAIN_STEP_LOGIT_BUDGET": os.environ["TRAIN_STEP_LOGIT_BUDGET"],
                "TRAIN_MIN_DYNAMIC_GLOBAL_BATCH_SIZE": os.environ["TRAIN_MIN_DYNAMIC_GLOBAL_BATCH_SIZE"],
                "TRAIN_UNDERFILLED_STEP_MIN_SAMPLES": os.environ["TRAIN_UNDERFILLED_STEP_MIN_SAMPLES"],
                "TRAIN_STEP_PACKING_STRATEGY": os.environ["TRAIN_STEP_PACKING_STRATEGY"],
                "TRAIN_STEP_LONG_SAMPLE_THRESHOLD": os.environ["TRAIN_STEP_LONG_SAMPLE_THRESHOLD"],
                "TRAIN_MAX_LONG_SAMPLES_PER_STEP": os.environ["TRAIN_MAX_LONG_SAMPLES_PER_STEP"],
                "TRAIN_MAX_SINGLE_SAMPLE_TOKENS": os.environ["TRAIN_MAX_SINGLE_SAMPLE_TOKENS"],
                "TRAIN_MAX_TOTAL_TOKENS_PER_SAMPLE": os.environ["TRAIN_MAX_TOTAL_TOKENS_PER_SAMPLE"],
                "TRAIN_MAX_RESPONSE_TOKENS_PER_SAMPLE": os.environ["TRAIN_MAX_RESPONSE_TOKENS_PER_SAMPLE"],
            }
        }
    )
)
PY
)"

cd "${SLIME_DIR}"

SUBMIT_LOG="${RUN_LOG_DIR}/ray_job_submit.log"
CUDA_VISIBLE_DEVICES="${TRAIN_GPU_IDS}" ray job submit \
    --no-wait \
    --address "http://127.0.0.1:${RAY_JOB_PORT}" \
    --runtime-env-json "${RUNTIME_ENV_JSON}" \
    -- python3 train_async.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${N_TRAIN_GPUS}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${SFT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    2>&1 | tee "${SUBMIT_LOG}"

job_id="$(sed -n "s/.*Job '\\([^']*\\)'.*/\\1/p" "${SUBMIT_LOG}" | tail -n 1)"
if [[ -z "${job_id}" ]]; then
    echo "Failed to parse Ray job id from submit output." >&2
    exit 1
fi

echo "Following Ray job logs for ${job_id}"
ray job logs "${job_id}" \
    --address "http://127.0.0.1:${RAY_JOB_PORT}" \
    -f

job_status="$(ray job status "${job_id}" --address "http://127.0.0.1:${RAY_JOB_PORT}" | tr -d '\r')"
echo "${job_status}"
if [[ "${job_status}" != *"SUCCEEDED"* ]]; then
    echo "Ray job ${job_id} did not succeed; skipping post-training format check." >&2
    exit 1
fi

if [[ "${ENABLE_FORMAT_POSTCHECK}" == "1" ]]; then
    POSTCHECK_HF_DIR="${RUN_DIR}/hf_exports/posttrain_latest"
    echo "Converting latest checkpoint to HF for post-training format check"
    convert_latest_checkpoint_to_hf "${RUN_CHECKPOINT_DIR}" "${POSTCHECK_HF_DIR}"
    echo "Running tool-call parser postcheck on trained checkpoint"
    run_formatcheck "${POSTCHECK_HF_DIR}" "${FORMATCHECK_OUTPUT_DIR}/posttrain" "qwen3-32b-sft"
fi
