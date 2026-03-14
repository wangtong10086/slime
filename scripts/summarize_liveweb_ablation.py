#!/usr/bin/env python3
import json
import statistics
import sys
from collections import Counter
from pathlib import Path


def load_results(model_dir: Path):
    results = {}
    for path in sorted(model_dir.glob("seed_*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        seed = path.stem.split("_", 1)[1]
        results[seed] = data
    return results


def categorize_reason(reason: str) -> str:
    reason_lower = reason.lower()
    if "data not collected" in reason_lower:
        return "data_not_collected"
    if "no answer provided" in reason_lower:
        return "no_answer"
    if "within ±2" in reason or "within +-" in reason_lower:
        return "partial_credit_close_count"
    if any(
        phrase in reason_lower
        for phrase in (
            "incorrectly identified",
            "not within",
            "no overlap",
            "missed the correct count",
            "completely missed",
        )
    ):
        return "wrong_reasoning_or_choice"
    return "other"


def summarize_model(name: str, results: dict):
    run_scores = []
    subtask_scores = []
    failure_categories = Counter()
    exact_correct = 0
    positive_score = 0

    for seed, data in sorted(results.items()):
        run_scores.append(
            {
                "seed": seed,
                "score": float(data.get("score", 0.0)),
                "success": bool(data.get("success", False)),
                "time_taken": float(data.get("time_taken", 0.0)),
            }
        )
        for detail in data.get("extra", {}).get("answer_details", []):
            score = float(detail.get("score", 0.0))
            subtask_scores.append(
                {
                    "seed": seed,
                    "answer_tag": detail.get("answer_tag"),
                    "score": score,
                    "reasoning": detail.get("reasoning", ""),
                }
            )
            if score == 1.0:
                exact_correct += 1
            if score > 0:
                positive_score += 1
            failure_categories[categorize_reason(detail.get("reasoning", ""))] += 1

    mean_run_score = statistics.fmean(r["score"] for r in run_scores) if run_scores else 0.0
    mean_subtask_score = statistics.fmean(s["score"] for s in subtask_scores) if subtask_scores else 0.0

    return {
        "model": name,
        "num_runs": len(run_scores),
        "num_subtasks": len(subtask_scores),
        "mean_run_score": mean_run_score,
        "mean_subtask_score": mean_subtask_score,
        "exact_correct": exact_correct,
        "positive_score_count": positive_score,
        "run_scores": run_scores,
        "failure_categories": dict(failure_categories),
    }


def paired_compare(base_results: dict, ft_results: dict):
    shared_seeds = sorted(set(base_results) & set(ft_results))
    paired = []
    for seed in shared_seeds:
        base_score = float(base_results[seed].get("score", 0.0))
        ft_score = float(ft_results[seed].get("score", 0.0))
        paired.append(
            {
                "seed": seed,
                "base_score": base_score,
                "ft_score": ft_score,
                "delta": ft_score - base_score,
            }
        )

    mean_delta = statistics.fmean(item["delta"] for item in paired) if paired else 0.0
    ft_better = sum(item["delta"] > 0 for item in paired)
    tied = sum(item["delta"] == 0 for item in paired)
    base_better = sum(item["delta"] < 0 for item in paired)

    return {
        "paired_runs": len(paired),
        "mean_delta_ft_minus_base": mean_delta,
        "ft_better_count": ft_better,
        "base_better_count": base_better,
        "tied_count": tied,
        "per_seed": paired,
    }


def main():
    if len(sys.argv) != 4:
        print("Usage: summarize_liveweb_ablation.py <root_dir> <base_label> <ft_label>", file=sys.stderr)
        sys.exit(1)

    root = Path(sys.argv[1])
    base_label = sys.argv[2]
    ft_label = sys.argv[3]

    base_results = load_results(root / base_label)
    ft_results = load_results(root / ft_label)

    summary = {
        "base": summarize_model(base_label, base_results),
        "finetuned": summarize_model(ft_label, ft_results),
        "paired_comparison": paired_compare(base_results, ft_results),
    }

    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True))
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
