#!/usr/bin/env python3
import argparse
import csv
import json
import math
import shutil
import tarfile
from collections import Counter
from pathlib import Path
from statistics import mean, median


def percentile(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * p
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    frac = rank - low
    return float(ordered[low] * (1 - frac) + ordered[high] * frac)


def classify_answer_detail(detail):
    reasoning = (detail.get("reasoning") or "").lower()
    actual = (detail.get("actual") or "").lower()
    if "data not collected" in reasoning or "data not collected" in actual:
        return "data_not_collected"
    if "site unreachable" in reasoning or "site unreachable" in actual:
        return "site_unreachable"
    if "prefetch" in reasoning or "cache" in reasoning:
        return "cache_related"
    if "parse" in reasoning:
        return "parse_failed"
    if detail.get("score", 0) > 0:
        return "partial_credit"
    return "wrong_answer_or_reasoning"


def analyze_result_file(path):
    data = json.loads(path.read_text())
    extra = data.get("extra", {})
    conversation = extra.get("conversation", [])
    usage = extra.get("usage", {})
    answer_details = extra.get("answer_details", [])

    role_counts = Counter(msg.get("role") for msg in conversation)
    tool_calls = 0
    stop_calls = 0
    assistant_visible_chars = 0
    think_tag_count = 0

    for msg in conversation:
        content = msg.get("content")
        if isinstance(content, str):
            assistant_visible_chars += len(content) if msg.get("role") == "assistant" else 0
            think_tag_count += content.count("<think>")

        for tool_call in msg.get("tool_calls", []) or []:
            tool_calls += 1
            function_name = ((tool_call.get("function") or {}).get("name")) or ""
            if function_name == "stop":
                stop_calls += 1

    answer_category_counts = Counter(classify_answer_detail(detail) for detail in answer_details)

    return {
        "job_id": path.stem,
        "path": str(path),
        "task_name": data.get("task_name"),
        "score": data.get("score", 0.0),
        "success": bool(data.get("success", False)),
        "time_taken": data.get("time_taken", 0.0),
        "failure_reason": extra.get("failure_reason"),
        "conversation_length": len(conversation),
        "assistant_messages": role_counts.get("assistant", 0),
        "tool_messages": role_counts.get("tool", 0),
        "user_messages": role_counts.get("user", 0),
        "tool_calls": tool_calls,
        "stop_calls": stop_calls,
        "assistant_visible_chars": assistant_visible_chars,
        "think_tag_count": think_tag_count,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "answer_detail_count": len(answer_details),
        "answer_category_counts": dict(answer_category_counts),
        "parent_seed": extra.get("parent_seed"),
        "subtask_index": extra.get("subtask_index"),
        "schedule_unit": extra.get("schedule_unit"),
    }


def summarize_variant(name, variant_dir):
    result_dir = variant_dir / "base_qwen3_32b_mcp"
    result_files = sorted(result_dir.glob("*.json"))
    rows = [analyze_result_file(path) for path in result_files]

    def values(key):
        return [row[key] for row in rows]

    failure_reason_dist = Counter((row["failure_reason"] or "None") for row in rows)
    answer_failure_dist = Counter()
    for row in rows:
        answer_failure_dist.update(row["answer_category_counts"])

    summary = json.loads((variant_dir / "summary.json").read_text())

    out = {
        "variant": name,
        "summary_path": str(variant_dir / "summary.json"),
        "result_count": len(rows),
        "mean_score": mean(values("score")) if rows else 0.0,
        "success_count": sum(1 for row in rows if row["success"]),
        "success_rate": (sum(1 for row in rows if row["success"]) / len(rows)) if rows else 0.0,
        "mean_time_s": mean(values("time_taken")) if rows else 0.0,
        "median_time_s": median(values("time_taken")) if rows else 0.0,
        "p90_time_s": percentile(values("time_taken"), 0.9),
        "mean_conversation_length": mean(values("conversation_length")) if rows else 0.0,
        "mean_assistant_messages": mean(values("assistant_messages")) if rows else 0.0,
        "mean_tool_messages": mean(values("tool_messages")) if rows else 0.0,
        "mean_tool_calls": mean(values("tool_calls")) if rows else 0.0,
        "mean_stop_calls": mean(values("stop_calls")) if rows else 0.0,
        "mean_prompt_tokens": mean(values("prompt_tokens")) if rows else 0.0,
        "mean_completion_tokens": mean(values("completion_tokens")) if rows else 0.0,
        "mean_total_tokens": mean(values("total_tokens")) if rows else 0.0,
        "median_prompt_tokens": median(values("prompt_tokens")) if rows else 0.0,
        "median_completion_tokens": median(values("completion_tokens")) if rows else 0.0,
        "median_total_tokens": median(values("total_tokens")) if rows else 0.0,
        "mean_visible_assistant_chars": mean(values("assistant_visible_chars")) if rows else 0.0,
        "mean_think_tag_count": mean(values("think_tag_count")) if rows else 0.0,
        "failure_reason_dist": dict(failure_reason_dist),
        "answer_failure_dist": dict(answer_failure_dist),
        "environment_failures": summary.get("environment_failures", {}),
        "top_level_failure_reasons": summary.get("failure_reasons", {}),
        "rows": rows,
    }
    return out


def paired_analysis(think_off, think_on):
    off_rows = {row["job_id"]: row for row in think_off["rows"]}
    on_rows = {row["job_id"]: row for row in think_on["rows"]}
    shared = sorted(set(off_rows) & set(on_rows))
    paired_rows = []
    for job_id in shared:
        off = off_rows[job_id]
        on = on_rows[job_id]
        paired_rows.append(
            {
                "job_id": job_id,
                "parent_seed": off["parent_seed"],
                "subtask_index": off["subtask_index"],
                "score_off": off["score"],
                "score_on": on["score"],
                "score_delta_on_minus_off": on["score"] - off["score"],
                "success_off": off["success"],
                "success_on": on["success"],
                "time_off_s": off["time_taken"],
                "time_on_s": on["time_taken"],
                "time_delta_s_on_minus_off": on["time_taken"] - off["time_taken"],
                "conversation_length_off": off["conversation_length"],
                "conversation_length_on": on["conversation_length"],
                "conversation_length_delta": on["conversation_length"] - off["conversation_length"],
                "tool_calls_off": off["tool_calls"],
                "tool_calls_on": on["tool_calls"],
                "tool_calls_delta": on["tool_calls"] - off["tool_calls"],
                "prompt_tokens_off": off["prompt_tokens"],
                "prompt_tokens_on": on["prompt_tokens"],
                "prompt_tokens_delta": on["prompt_tokens"] - off["prompt_tokens"],
                "completion_tokens_off": off["completion_tokens"],
                "completion_tokens_on": on["completion_tokens"],
                "completion_tokens_delta": on["completion_tokens"] - off["completion_tokens"],
                "failure_reason_off": off["failure_reason"] or "None",
                "failure_reason_on": on["failure_reason"] or "None",
            }
        )

    return {
        "shared_job_count": len(shared),
        "mean_score_delta_on_minus_off": mean(row["score_delta_on_minus_off"] for row in paired_rows) if paired_rows else 0.0,
        "mean_time_delta_s_on_minus_off": mean(row["time_delta_s_on_minus_off"] for row in paired_rows) if paired_rows else 0.0,
        "mean_conversation_length_delta": mean(row["conversation_length_delta"] for row in paired_rows) if paired_rows else 0.0,
        "mean_tool_calls_delta": mean(row["tool_calls_delta"] for row in paired_rows) if paired_rows else 0.0,
        "mean_completion_tokens_delta": mean(row["completion_tokens_delta"] for row in paired_rows) if paired_rows else 0.0,
        "think_on_better_count": sum(1 for row in paired_rows if row["score_delta_on_minus_off"] > 0),
        "think_off_better_count": sum(1 for row in paired_rows if row["score_delta_on_minus_off"] < 0),
        "score_tie_count": sum(1 for row in paired_rows if row["score_delta_on_minus_off"] == 0),
        "think_on_slower_count": sum(1 for row in paired_rows if row["time_delta_s_on_minus_off"] > 0),
        "rows": paired_rows,
    }


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def copy_json(src, dst):
    dst.write_text(json.dumps(json.loads(src.read_text()), ensure_ascii=False, indent=2))


def build_report(bundle_dir, compare_root, think_off, think_on, paired):
    report = f"""# Qwen3-32B LiveWeb Arena Thinking 开关对照报告

## 1. 实验目的

本报告对比 `think_off` 与 `think_on` 两种运行模式在 `liveweb-arena` 上的表现差异，重点关注：

- 得分与成功率
- 轨迹长度与交互次数
- token 消耗
- 失败原因分布

实验原始目录：[`../source`](../source)  
打包目录中的关键文件：

- 总对比：[`./comparison_summary.json`](./comparison_summary.json)
- `think_off` 汇总：[`./think_off_summary.json`](./think_off_summary.json)
- `think_on` 汇总：[`./think_on_summary.json`](./think_on_summary.json)
- 附件说明：[`./attachments/README.md`](./attachments/README.md)

## 2. 主要结论

在本次 `200 vs 200` 个 prompt 级任务的正式对照中，`think_on` 相比 `think_off` 有**小幅正向提升**：

- 平均分从 `{think_off["mean_score"]:.4f}` 提升到 `{think_on["mean_score"]:.4f}`
- 成功数从 `{think_off["success_count"]}` 提升到 `{think_on["success_count"]}`
- 平均分增量为 `{paired["mean_score_delta_on_minus_off"]:.4f}`

但这个提升不是“纯赚”的，伴随一些额外代价：

- 平均耗时变化：`{paired["mean_time_delta_s_on_minus_off"]:+.2f}s`
- 平均轨迹长度变化：`{paired["mean_conversation_length_delta"]:+.2f}`
- 平均工具调用次数变化：`{paired["mean_tool_calls_delta"]:+.2f}`
- 平均 completion tokens 变化：`{paired["mean_completion_tokens_delta"]:+.2f}`

## 3. 汇总指标

### think_off

- 任务数：`{think_off["result_count"]}`
- 平均分：`{think_off["mean_score"]:.4f}`
- 成功率：`{think_off["success_rate"]:.2%}`
- 平均耗时：`{think_off["mean_time_s"]:.2f}s`
- 中位耗时：`{think_off["median_time_s"]:.2f}s`
- P90 耗时：`{think_off["p90_time_s"]:.2f}s`
- 平均轨迹长度：`{think_off["mean_conversation_length"]:.2f}`
- 平均 assistant 消息数：`{think_off["mean_assistant_messages"]:.2f}`
- 平均工具调用数：`{think_off["mean_tool_calls"]:.2f}`
- 平均 prompt/completion/total tokens：`{think_off["mean_prompt_tokens"]:.1f}` / `{think_off["mean_completion_tokens"]:.1f}` / `{think_off["mean_total_tokens"]:.1f}`

### think_on

- 任务数：`{think_on["result_count"]}`
- 平均分：`{think_on["mean_score"]:.4f}`
- 成功率：`{think_on["success_rate"]:.2%}`
- 平均耗时：`{think_on["mean_time_s"]:.2f}s`
- 中位耗时：`{think_on["median_time_s"]:.2f}s`
- P90 耗时：`{think_on["p90_time_s"]:.2f}s`
- 平均轨迹长度：`{think_on["mean_conversation_length"]:.2f}`
- 平均 assistant 消息数：`{think_on["mean_assistant_messages"]:.2f}`
- 平均工具调用数：`{think_on["mean_tool_calls"]:.2f}`
- 平均 prompt/completion/total tokens：`{think_on["mean_prompt_tokens"]:.1f}` / `{think_on["mean_completion_tokens"]:.1f}` / `{think_on["mean_total_tokens"]:.1f}`

## 4. 失败原因差异

顶层失败原因：

- `think_off`：`{json.dumps(think_off["top_level_failure_reasons"], ensure_ascii=False)}`
- `think_on`：`{json.dumps(think_on["top_level_failure_reasons"], ensure_ascii=False)}`

环境侧失败：

- `think_off`：`{json.dumps(think_off["environment_failures"], ensure_ascii=False)}`
- `think_on`：`{json.dumps(think_on["environment_failures"], ensure_ascii=False)}`

按 answer_details 归类后的失败：

- `think_off`：`{json.dumps(think_off["answer_failure_dist"], ensure_ascii=False)}`
- `think_on`：`{json.dumps(think_on["answer_failure_dist"], ensure_ascii=False)}`

从这轮结果看：

- `data_not_collected` 仍然是两边共同的主失败源
- `think_on` 分数更高，但 `parse_failed` 和 `site_unreachable` 没有同步下降
- 因此当前收益更像是“部分任务上更有帮助”，而不是整体更稳定

## 5. 对结果的解释

从更细的轨迹指标看，`think_on` 的收益并不大，但确实存在：

- 如果后续目标是**追求更高分数**，当前证据更偏向 `think_on`
- 如果后续目标是**追求更稳定、噪声更少的评测**，还需要继续压环境侧和协议侧残余失败

这意味着：

- `think_on` 目前值得保留为一个候选配置
- 但还不应该直接下结论说 “开启 think 明显更优”
- 更合理的说法是：在本次 `200` 个任务规模下，`think_on` 呈现出**小幅优势**

## 6. 附件

- 指标总表：[`./attachments/variant_metrics.csv`](./attachments/variant_metrics.csv)
- 配对逐任务差异：[`./attachments/paired_job_metrics.csv`](./attachments/paired_job_metrics.csv)
- 顶层失败原因对比：[`./attachments/failure_reason_comparison.json`](./attachments/failure_reason_comparison.json)
- 轨迹与 token 统计：[`./attachments/trajectory_metrics.json`](./attachments/trajectory_metrics.json)
- 代表性长尾/差异样本：[`./attachments/notable_outliers.csv`](./attachments/notable_outliers.csv)
"""
    (bundle_dir / "report.md").write_text(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    compare_root = Path(args.compare_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    bundle_dir = output_dir / compare_root.name
    attachments_dir = bundle_dir / "attachments"
    source_dir = bundle_dir / "source"

    if bundle_dir.exists():
        for child in bundle_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        bundle_dir.mkdir(parents=True, exist_ok=True)
    attachments_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    think_off = summarize_variant("think_off", compare_root / "think_off")
    think_on = summarize_variant("think_on", compare_root / "think_on")
    paired = paired_analysis(think_off, think_on)

    copy_json(compare_root / "comparison_summary.json", bundle_dir / "comparison_summary.json")
    copy_json(compare_root / "think_off" / "summary.json", bundle_dir / "think_off_summary.json")
    copy_json(compare_root / "think_on" / "summary.json", bundle_dir / "think_on_summary.json")

    variant_rows = [
        {
            "variant": variant["variant"],
            "result_count": variant["result_count"],
            "mean_score": variant["mean_score"],
            "success_count": variant["success_count"],
            "success_rate": variant["success_rate"],
            "mean_time_s": variant["mean_time_s"],
            "median_time_s": variant["median_time_s"],
            "p90_time_s": variant["p90_time_s"],
            "mean_conversation_length": variant["mean_conversation_length"],
            "mean_assistant_messages": variant["mean_assistant_messages"],
            "mean_tool_messages": variant["mean_tool_messages"],
            "mean_tool_calls": variant["mean_tool_calls"],
            "mean_stop_calls": variant["mean_stop_calls"],
            "mean_prompt_tokens": variant["mean_prompt_tokens"],
            "mean_completion_tokens": variant["mean_completion_tokens"],
            "mean_total_tokens": variant["mean_total_tokens"],
        }
        for variant in [think_off, think_on]
    ]
    write_csv(
        attachments_dir / "variant_metrics.csv",
        variant_rows,
        list(variant_rows[0].keys()),
    )

    write_csv(
        attachments_dir / "paired_job_metrics.csv",
        paired["rows"],
        list(paired["rows"][0].keys()) if paired["rows"] else ["job_id"],
    )

    outliers = sorted(
        paired["rows"],
        key=lambda row: abs(row["time_delta_s_on_minus_off"]),
        reverse=True,
    )[:30]
    write_csv(
        attachments_dir / "notable_outliers.csv",
        outliers,
        list(outliers[0].keys()) if outliers else ["job_id"],
    )

    (attachments_dir / "failure_reason_comparison.json").write_text(
        json.dumps(
            {
                "think_off": {
                    "top_level_failure_reasons": think_off["top_level_failure_reasons"],
                    "environment_failures": think_off["environment_failures"],
                    "answer_failure_dist": think_off["answer_failure_dist"],
                },
                "think_on": {
                    "top_level_failure_reasons": think_on["top_level_failure_reasons"],
                    "environment_failures": think_on["environment_failures"],
                    "answer_failure_dist": think_on["answer_failure_dist"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    (attachments_dir / "trajectory_metrics.json").write_text(
        json.dumps(
            {
                "think_off": {
                    k: v
                    for k, v in think_off.items()
                    if k not in {"rows"}
                },
                "think_on": {
                    k: v
                    for k, v in think_on.items()
                    if k not in {"rows"}
                },
                "paired": {
                    k: v
                    for k, v in paired.items()
                    if k not in {"rows"}
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    (attachments_dir / "README.md").write_text(
        "# 附件说明\n\n"
        "- `variant_metrics.csv`: 两个配置的核心汇总指标\n"
        "- `paired_job_metrics.csv`: 同一个 prompt 任务在 think_off / think_on 下的逐条对照\n"
        "- `failure_reason_comparison.json`: 失败原因和环境失败对比\n"
        "- `trajectory_metrics.json`: 轨迹长度、交互次数、token 等更详细的汇总\n"
        "- `notable_outliers.csv`: 按耗时差异排序的代表性样本\n"
    )

    (source_dir / "README.md").write_text(
        f"原始结果目录：`{compare_root}`\n"
        "本 bundle 只放报告、摘要和附件，不复制全部原始轨迹文件。\n"
    )

    build_report(bundle_dir, compare_root, think_off, think_on, paired)

    archive_path = bundle_dir / f"{compare_root.name}.tar.gz"
    if archive_path.exists():
        archive_path.unlink()
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(bundle_dir, arcname=bundle_dir.name)

    manifest = {
        "compare_root": str(compare_root),
        "bundle_dir": str(bundle_dir),
        "archive_path": str(archive_path),
        "think_off_mean_score": think_off["mean_score"],
        "think_on_mean_score": think_on["mean_score"],
        "delta_mean_score": paired["mean_score_delta_on_minus_off"],
        "think_off_success_count": think_off["success_count"],
        "think_on_success_count": think_on["success_count"],
    }
    (bundle_dir / "bundle_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
