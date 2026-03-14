#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def build_bundle(root: Path, model_label: str, model_summary: dict):
    model_dir = root / model_label
    run_files = sorted(model_dir.glob("seed_*.json"))

    runs = []
    for path in run_files:
        data = load_json(path)
        runs.append(
            {
                "seed": path.stem.split("_", 1)[1],
                "relative_file": str(path.relative_to(root)),
                "task_name": data.get("task_name"),
                "score": data.get("score"),
                "success": data.get("success"),
                "time_taken": data.get("time_taken"),
                "extra": data.get("extra", {}),
            }
        )

    return {
        "model_label": model_label,
        "source_dir": str(model_dir.relative_to(root)),
        "summary": model_summary,
        "run_count": len(runs),
        "runs": runs,
    }


def main():
    if len(sys.argv) != 2:
        print("Usage: export_liveweb_trajectory_bundle.py <experiment_root>", file=sys.stderr)
        sys.exit(1)

    root = Path(sys.argv[1]).resolve()
    summary_path = root / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json: {summary_path}")

    summary = load_json(summary_path)

    bundles = [
        ("base_qwen3_32b", summary["base"]),
        ("sft_iter_0000055", summary["finetuned"]),
    ]

    for model_label, model_summary in bundles:
        bundle = build_bundle(root, model_label, model_summary)
        out_path = root / f"{model_label}_summary_with_trajectories.json"
        out_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2))
        print(out_path)


if __name__ == "__main__":
    main()
