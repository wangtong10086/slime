#!/usr/bin/env bash
set -euo pipefail

USER_HOME="${USER_HOME:-/home/xmyf}"
SLIME_HOME="${SLIME_HOME:-${USER_HOME}/slime}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
NUM_EVAL_RUNS="${NUM_EVAL_RUNS:-50}"
NUM_TASKS_PER_RUN="${NUM_TASKS_PER_RUN:-4}"
TOTAL_TASKS=$(( NUM_EVAL_RUNS * NUM_TASKS_PER_RUN ))
START_SEED="${START_SEED:-1001}"

COMPARE_ROOT="${COMPARE_ROOT:-${USER_HOME}/slime_runs/liveweb_qwen32b_thinking_compare_${TOTAL_TASKS}_${RUN_STAMP}}"
mkdir -p "${COMPARE_ROOT}"
export COMPARE_ROOT
export NUM_EVAL_RUNS
export NUM_TASKS_PER_RUN
export START_SEED

run_variant() {
  local variant="$1"
  local enable_thinking="$2"
  local out_root="${COMPARE_ROOT}/${variant}"
  local archive_dir="${out_root}/archive"
  mkdir -p "${archive_dir}"

  export RUN_STAMP="${RUN_STAMP}_${variant}"
  export OUT_ROOT="${out_root}"
  export ARCHIVE_DIR="${archive_dir}"
  export ARCHIVE_PATH="${archive_dir}/$(basename "${out_root}").tar.gz"
  export MANIFEST_PATH="${archive_dir}/manifest.json"
  export NUM_EVAL_RUNS
  export NUM_TASKS_PER_RUN
  export START_SEED
  export SCHEDULE_UNIT="prompt"
  export SERVICE_ROOT="${USER_HOME}/slime_runs/liveweb_sglang_services/qwen3_32b_tp1_${variant}_${RUN_STAMP}"
  export SERVICE_GENERATION_ID="${variant}-${RUN_STAMP}"
  export SERVICE_REQUIRE_CLEAN=1
  export TOOL_CALL_PARSER="qwen"
  export REASONING_PARSER="qwen3"
  export LIVEWEB_ENABLE_THINKING="${enable_thinking}"
  export LIVEWEB_SEPARATE_REASONING="1"
  export LIVEWEB_MAX_COMPLETION_TOKENS="${LIVEWEB_MAX_COMPLETION_TOKENS:-1024}"
  export MAX_STEPS="${MAX_STEPS:-30}"
  export TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1800}"
  export CACHE_PROFILE="${CACHE_PROFILE:-hot}"
  export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.95}"
  export EVAL_CONCURRENCY_PER_SERVER="${EVAL_CONCURRENCY_PER_SERVER:-6}"
  export MAX_BROWSER_SESSIONS="${MAX_BROWSER_SESSIONS:-48}"
  export MAX_LLM_REQUESTS="${MAX_LLM_REQUESTS:-32}"
  export BASE_MODEL_LABEL="base_qwen3_32b_mcp"

  bash "${SLIME_HOME}/scripts/run-liveweb-arena-formal-archive.sh"
}

run_variant "think_off" "0"
run_variant "think_on" "1"

python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["COMPARE_ROOT"])
variants = {}
for variant in ["think_off", "think_on"]:
    summary_path = root / variant / "summary.json"
    manifest_path = root / variant / "archive" / "manifest.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    variants[variant] = {
        "summary_path": str(summary_path),
        "manifest_path": str(manifest_path),
        "mean_run_score": summary.get("mean_run_score"),
        "success_count": summary.get("success_count"),
        "num_runs": summary.get("num_runs"),
        "failure_reasons": summary.get("failure_reasons"),
        "environment_failures": summary.get("environment_failures"),
        "archive_path": manifest.get("archive_path"),
        "archive_sha256": manifest.get("archive_sha256"),
    }

comparison = {
    "compare_root": str(root),
    "num_eval_runs": int(os.environ["NUM_EVAL_RUNS"]),
    "num_tasks_per_run": int(os.environ["NUM_TASKS_PER_RUN"]),
    "start_seed": int(os.environ["START_SEED"]),
    "variants": variants,
    "delta_mean_run_score_think_on_minus_off": (
        (variants["think_on"]["mean_run_score"] or 0.0) - (variants["think_off"]["mean_run_score"] or 0.0)
    ),
    "delta_success_count_think_on_minus_off": (
        (variants["think_on"]["success_count"] or 0) - (variants["think_off"]["success_count"] or 0)
    ),
}
out = root / "comparison_summary.json"
out.write_text(json.dumps(comparison, ensure_ascii=True, indent=2))
print(json.dumps(comparison, ensure_ascii=True, indent=2))
PY
