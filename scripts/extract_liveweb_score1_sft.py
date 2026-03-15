#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract score==1 liveweb rollout trajectories into an SFT dataset."
    )
    parser.add_argument("--run-root", required=True, help="Source run root containing rollout_dumps/")
    parser.add_argument(
        "--output",
        default="",
        help="Output JSONL path. Defaults to <run-root>/derived_datasets/liveweb_score1_sft.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    dump_root = run_root / "rollout_dumps"
    if not dump_root.is_dir():
        raise SystemExit(f"rollout_dumps not found: {dump_root}")

    output_path = (
        Path(args.output).resolve()
        if args.output
        else run_root / "derived_datasets" / "liveweb_score1_sft.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_path.with_suffix(".summary.json")

    total_records = 0
    score1_records = 0
    plugin_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()

    with output_path.open("w", encoding="utf-8") as out_f:
        for dump_file in sorted(dump_root.glob("*.jsonl")):
            with dump_file.open("r", encoding="utf-8") as in_f:
                for line_no, line in enumerate(in_f, start=1):
                    if not line.strip():
                        continue
                    total_records += 1
                    record = json.loads(line)
                    score = float(record.get("score", 0.0))
                    if score != 1.0:
                        continue

                    extra = record.get("extra") or {}
                    conversation = extra.get("conversation") or []
                    plugin_name = str(extra.get("plugin_name") or "unknown")
                    answer_details = extra.get("answer_details") or []

                    dataset_row = {
                        "id": f"{dump_file.name}:{line_no}",
                        "source_run": str(run_root),
                        "source_file": dump_file.name,
                        "task_name": record.get("task_name"),
                        "plugin_name": plugin_name,
                        "score": score,
                        "success": bool(record.get("success", False)),
                        "time_taken": record.get("time_taken"),
                        "messages": conversation,
                        "answer_details": answer_details,
                        "final_url": extra.get("final_url"),
                        "steps_used": extra.get("steps_used"),
                        "usage": extra.get("usage"),
                        "metadata": {
                            "seed": extra.get("seed"),
                            "task_seed": extra.get("task_seed"),
                            "llm_seed": extra.get("llm_seed"),
                            "parent_seed": extra.get("parent_seed"),
                            "subtask_index": extra.get("subtask_index"),
                            "num_subtasks": extra.get("num_subtasks"),
                            "output_format": extra.get("output_format"),
                        },
                    }
                    out_f.write(json.dumps(dataset_row, ensure_ascii=False) + "\n")
                    score1_records += 1
                    plugin_counter[plugin_name] += 1
                    source_counter[dump_file.name] += 1

    summary = {
        "run_root": str(run_root),
        "dump_root": str(dump_root),
        "output_path": str(output_path),
        "total_records": total_records,
        "score1_records": score1_records,
        "plugin_counts": dict(sorted(plugin_counter.items())),
        "source_file_counts": dict(sorted(source_counter.items())),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
