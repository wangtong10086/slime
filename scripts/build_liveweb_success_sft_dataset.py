#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deduplicated SFT dataset from successful liveweb trajectories."
    )
    parser.add_argument(
        "--input-glob",
        default="/data/liveweb_runs/liveweb_capability*/samples/*/tasks_*.jsonl",
        help="Glob for successful sample jsonl files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--summary-output",
        default="",
        help="Optional summary path. Defaults to <output>.summary.json.",
    )
    parser.add_argument(
        "--dedup-key",
        choices=("conversation", "trajectory"),
        default="conversation",
        help="Deduplicate by normalized conversation or by full trajectory object.",
    )
    return parser.parse_args()


def _normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key in ("role", "content", "tool_calls", "tool_call_id", "name", "step_loss_mask"):
        if key in message:
            normalized[key] = message[key]
    return normalized


def _normalize_conversation(trajectory: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    conversation = trajectory.get("conversation") or []
    if not isinstance(conversation, list) or not conversation:
        raise ValueError("trajectory conversation is empty or invalid")

    tools: list[dict[str, Any]] = []
    normalized_messages: list[dict[str, Any]] = []

    for idx, message in enumerate(conversation):
        if not isinstance(message, dict):
            continue
        if idx == 0 and isinstance(message.get("tools"), list):
            tools = message["tools"]
        normalized_messages.append(_normalize_message(message))

    if not normalized_messages:
        raise ValueError("normalized conversation is empty")

    return normalized_messages, tools


def _stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = (
        Path(args.summary_output).resolve()
        if args.summary_output
        else output_path.with_suffix(output_path.suffix + ".summary.json")
    )

    sample_files = sorted(glob.glob(args.input_glob))
    if not sample_files:
        raise SystemExit(f"no sample files matched: {args.input_glob}")

    total_success_rows = 0
    written_rows = 0
    seen_hashes: set[str] = set()
    model_counter: Counter[str] = Counter()
    task_counter: Counter[str] = Counter()
    score_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    dropped_counter: Counter[str] = Counter()

    with output_path.open("w", encoding="utf-8") as out_f:
        for sample_file in sample_files:
            sample_path = Path(sample_file)
            with sample_path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    sample = json.loads(line)
                    if not sample.get("success", False):
                        continue

                    trajectory_path = sample.get("trajectory_path")
                    if not trajectory_path:
                        dropped_counter["missing_trajectory_path"] += 1
                        continue

                    traj_path = Path(trajectory_path)
                    if not traj_path.is_file():
                        dropped_counter["trajectory_missing_on_disk"] += 1
                        continue

                    trajectory = json.loads(traj_path.read_text(encoding="utf-8"))
                    try:
                        messages, tools = _normalize_conversation(trajectory)
                    except Exception:
                        dropped_counter["invalid_conversation"] += 1
                        continue

                    total_success_rows += 1

                    hash_obj: Any
                    if args.dedup_key == "trajectory":
                        hash_obj = trajectory
                    else:
                        hash_obj = {"messages": messages, "tools": tools}
                    row_hash = _stable_hash(hash_obj)
                    if row_hash in seen_hashes:
                        dropped_counter["duplicate"] += 1
                        continue
                    seen_hashes.add(row_hash)

                    row = {
                        "id": f"{sample_path.name}:{line_no}:{row_hash[:12]}",
                        "messages": messages,
                        "tools": tools,
                        "metadata": {
                            "task_id": sample.get("task_id"),
                            "prompt_id": sample.get("prompt_id"),
                            "num_tasks": sample.get("num_tasks"),
                            "model_id": sample.get("model_id"),
                            "provider": sample.get("provider"),
                            "run_id": sample.get("run_id"),
                            "sample_index": sample.get("sample_index"),
                            "score": sample.get("score"),
                            "trajectory_hash": row_hash,
                            "trajectory_path": str(traj_path),
                            "sample_file": str(sample_path),
                            "answer_details": trajectory.get("answer_details"),
                        },
                    }
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written_rows += 1

                    model_counter[str(sample.get("model_id") or "unknown")] += 1
                    task_counter[f"tasks_{sample.get('num_tasks')}"] += 1
                    score_counter[str(sample.get("score"))] += 1
                    source_counter[sample_path.parent.name] += 1

    summary = {
        "input_glob": args.input_glob,
        "output_path": str(output_path),
        "dedup_key": args.dedup_key,
        "sample_files": len(sample_files),
        "total_success_rows_seen": total_success_rows,
        "rows_written": written_rows,
        "duplicate_rows_dropped": dropped_counter.get("duplicate", 0),
        "dropped_reasons": dict(sorted(dropped_counter.items())),
        "model_counts": dict(sorted(model_counter.items())),
        "task_counts": dict(sorted(task_counter.items())),
        "score_counts": dict(sorted(score_counter.items())),
        "source_counts": dict(sorted(source_counter.items())),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
