#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mix task SFT data with tool-call regression rows.")
    parser.add_argument("--task-dataset", required=True)
    parser.add_argument("--regression-dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--regression-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    task_rows = _read_jsonl(Path(args.task_dataset))
    regression_rows = _read_jsonl(Path(args.regression_dataset))
    if not task_rows:
        raise SystemExit("task dataset is empty")
    if not regression_rows:
        raise SystemExit("regression dataset is empty")

    rng = random.Random(args.seed)
    regression_target = max(1, round(len(task_rows) * args.regression_ratio / max(1e-9, 1.0 - args.regression_ratio)))
    mixed = list(task_rows)
    for idx in range(regression_target):
        mixed.append(regression_rows[idx % len(regression_rows)])
    rng.shuffle(mixed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in mixed:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "task_rows": len(task_rows),
        "regression_rows": len(regression_rows),
        "regression_target": regression_target,
        "mixed_rows": len(mixed),
        "regression_ratio": args.regression_ratio,
        "seed": args.seed,
    }
    output_path.with_suffix(output_path.suffix + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
