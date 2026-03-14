#!/usr/bin/env bash
set -euo pipefail

USER_HOME="${USER_HOME:-/home/xmyf}"
SLIME_HOME="${SLIME_HOME:-${USER_HOME}/slime}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
NUM_EVAL_RUNS="${NUM_EVAL_RUNS:-100}"
NUM_TASKS_PER_RUN="${NUM_TASKS_PER_RUN:-4}"
TOTAL_TASKS=$(( NUM_EVAL_RUNS * NUM_TASKS_PER_RUN ))

OUT_ROOT="${OUT_ROOT:-${USER_HOME}/slime_runs/liveweb_qwen32b_formal_eval_${TOTAL_TASKS}_${RUN_STAMP}}"
ARCHIVE_DIR="${ARCHIVE_DIR:-${OUT_ROOT}/archive}"
ARCHIVE_BASENAME="${ARCHIVE_BASENAME:-$(basename "${OUT_ROOT}")}"
ARCHIVE_PATH="${ARCHIVE_PATH:-${ARCHIVE_DIR}/${ARCHIVE_BASENAME}.tar.gz}"
MANIFEST_PATH="${MANIFEST_PATH:-${ARCHIVE_DIR}/manifest.json}"

mkdir -p "${ARCHIVE_DIR}"

export RUN_STAMP
export OUT_ROOT
export ARCHIVE_DIR
export ARCHIVE_PATH
export MANIFEST_PATH
export NUM_EVAL_RUNS
export NUM_TASKS_PER_RUN
export SCHEDULE_UNIT="${SCHEDULE_UNIT:-prompt}"
export SERVICE_ROOT="${SERVICE_ROOT:-${USER_HOME}/slime_runs/liveweb_sglang_services/qwen3_32b_tp1_formal_${RUN_STAMP}}"
export SERVICE_GENERATION_ID="${SERVICE_GENERATION_ID:-formal-${RUN_STAMP}}"
export SERVICE_REQUIRE_CLEAN="${SERVICE_REQUIRE_CLEAN:-1}"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen}"
export REASONING_PARSER="${REASONING_PARSER:-}"
export LIVEWEB_MAX_COMPLETION_TOKENS="${LIVEWEB_MAX_COMPLETION_TOKENS:-1024}"
export LIVEWEB_ENABLE_THINKING="${LIVEWEB_ENABLE_THINKING:-}"
export LIVEWEB_SEPARATE_REASONING="${LIVEWEB_SEPARATE_REASONING:-}"
export EVAL_CONCURRENCY_PER_SERVER="${EVAL_CONCURRENCY_PER_SERVER:-6}"
export MAX_BROWSER_SESSIONS="${MAX_BROWSER_SESSIONS:-48}"
export MAX_LLM_REQUESTS="${MAX_LLM_REQUESTS:-32}"
export CACHE_PROFILE="${CACHE_PROFILE:-hot}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.95}"
export BASE_MODEL_LABEL="${BASE_MODEL_LABEL:-base_qwen3_32b_mcp}"
export START_SEED="${START_SEED:-1001}"
export MAX_STEPS="${MAX_STEPS:-30}"
export TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1800}"
export ROUTE_POLICY="${ROUTE_POLICY:-sticky_steal}"
export STICKY_SLACK="${STICKY_SLACK:-0}"
export STICKY_LATENCY_SLACK_S="${STICKY_LATENCY_SLACK_S:-10}"

bash "${SLIME_HOME}/scripts/run-liveweb-arena-base-concurrent.sh"

python3 - <<'PY'
import hashlib
import json
import os
import tarfile
from pathlib import Path

out_root = Path(os.environ["OUT_ROOT"])
archive_dir = Path(os.environ["ARCHIVE_DIR"])
archive_path = Path(os.environ["ARCHIVE_PATH"])
manifest_path = Path(os.environ["MANIFEST_PATH"])
archive_dir.mkdir(parents=True, exist_ok=True)

result_dir = out_root / os.environ.get("BASE_MODEL_LABEL", "base_qwen3_32b_mcp")
summary_path = out_root / "summary.json"
progress_path = out_root / "logs" / "progress.log"

with tarfile.open(archive_path, "w:gz") as tar:
    tar.add(out_root, arcname=out_root.name)

sha256 = hashlib.sha256()
with archive_path.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
        sha256.update(chunk)

manifest = {
    "run_stamp": os.environ["RUN_STAMP"],
    "out_root": str(out_root),
    "archive_path": str(archive_path),
    "archive_sha256": sha256.hexdigest(),
    "num_eval_runs": int(os.environ["NUM_EVAL_RUNS"]),
    "num_tasks_per_run": int(os.environ["NUM_TASKS_PER_RUN"]),
    "schedule_unit": os.environ.get("SCHEDULE_UNIT", "prompt"),
    "reasoning_parser": os.environ.get("REASONING_PARSER", ""),
    "enable_thinking": os.environ.get("LIVEWEB_ENABLE_THINKING", ""),
    "separate_reasoning": os.environ.get("LIVEWEB_SEPARATE_REASONING", ""),
    "result_dir": str(result_dir),
    "result_file_count": len(list(result_dir.glob("*.json"))) if result_dir.exists() else 0,
    "summary_exists": summary_path.exists(),
    "progress_log_exists": progress_path.exists(),
}
manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2))
print(json.dumps(manifest, ensure_ascii=True, indent=2))
PY
