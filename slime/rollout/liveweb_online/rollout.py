from __future__ import annotations

import asyncio
import copy
import os
import statistics
import time
from typing import Any

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.utils.types import Sample

from .common import (
    LiveWebRolloutState,
    PromptJob,
    build_eval_jobs,
    compute_reward_from_result,
    evaluate_prompt_job,
    get_eval_profile,
    make_sample_from_result,
    should_allow_environment_fallback,
    summarize_cache_stats,
)


def _result_metrics(results: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    if not results:
        return {}
    scores = [float(item.get("score", 0.0)) for item in results]
    times = [float(item.get("time_taken", 0.0)) for item in results]
    failure_reasons = [(item.get("extra") or {}).get("failure_reason") for item in results]
    metrics: dict[str, float] = {
        f"{prefix}/mean_score": statistics.fmean(scores),
        f"{prefix}/mean_time": statistics.fmean(times),
        f"{prefix}/success_rate": statistics.fmean([1.0 if item.get("success") else 0.0 for item in results]),
    }
    for reason in sorted({reason for reason in failure_reasons if reason}):
        metrics[f"{prefix}/failure/{reason}"] = statistics.fmean(
            [1.0 if current == reason else 0.0 for current in failure_reasons]
        )
    metrics |= summarize_cache_stats([(item.get("extra") or {}).get("cache_stats") or {} for item in results])
    return metrics


def _effective_sample_count(groups: list[list[Sample]]) -> int:
    return sum(len(group) for group in groups)


def _dynamic_batch_enabled(args) -> bool:
    return bool(
        getattr(args, "use_dynamic_batch_size", False)
        or getattr(args, "use_dynamic_global_batch_size", False)
    )


def _minimum_train_samples(args) -> int:
    if not _dynamic_batch_enabled(args):
        return int(args.global_batch_size)
    configured = os.getenv("LIVEWEB_MIN_TRAIN_SAMPLES", "").strip()
    if configured:
        return max(1, int(configured))
    return max(1, min(int(args.global_batch_size), int(args.rollout_batch_size)))


async def _run_prompt_job(
    state: LiveWebRolloutState,
    sample: Sample,
    *,
    evaluation: bool,
) -> tuple[Sample | None, dict[str, Any]]:
    metadata = sample.metadata or {}
    job = PromptJob(
        parent_seed=int(metadata["parent_seed"]),
        task_seed=int(metadata["task_seed"]),
        llm_seed=int(metadata["llm_seed"]),
        subtask_index=int(metadata["subtask_index"]),
        num_subtasks=int(metadata["num_subtasks"]),
        templates=[tuple(item) for item in metadata["templates"]],
        task_name=str(metadata["task_name"]),
        plugin_name=str(metadata["plugin_name"]),
        phase=str(metadata["phase"]),
        route_key=str(metadata["route_key"]),
    )
    result = await evaluate_prompt_job(state.args, state, job)

    reward, reward_meta = compute_reward_from_result(result)
    sample.metadata["liveweb_result"] = result
    if reward is None:
        if not evaluation and should_allow_environment_fallback():
            conversation = ((result.get("extra") or {}).get("conversation")) or []
            if conversation:
                fallback_meta = {
                    **reward_meta,
                    "drop_reason": None,
                    "environment_failure_type": reward_meta.get("environment_failure_type"),
                    "environment_fallback": True,
                }
                built = make_sample_from_result(
                    sample=sample,
                    result=result,
                    tokenizer=state.tokenizer,
                    reward=0.0,
                    reward_meta=fallback_meta,
                )
                return built, result
        sample.metadata |= reward_meta
        if evaluation:
            sample.reward = 0.0
            sample.response = ""
            sample.response_length = 0
            sample.loss_mask = []
            sample.status = Sample.Status.FAILED
            sample.metadata.update(
                {
                    "task_name": result.get("task_name"),
                    "score": result.get("score", 0.0),
                    "success": result.get("success", False),
                    "time_taken": result.get("time_taken", 0.0),
                    "failure_reason": (result.get("extra") or {}).get("failure_reason"),
                    "cache_stats": (result.get("extra") or {}).get("cache_stats") or {},
                    "usage": (result.get("extra") or {}).get("usage") or {},
                    "answer_details": (result.get("extra") or {}).get("answer_details") or [],
                    "prompt_tokens": ((result.get("extra") or {}).get("usage") or {}).get("prompt_tokens", 0),
                    "completion_tokens": ((result.get("extra") or {}).get("usage") or {}).get("completion_tokens", 0),
                    "total_tokens": ((result.get("extra") or {}).get("usage") or {}).get("total_tokens", 0),
                }
            )
            return sample, result
        return None, result

    built = make_sample_from_result(
        sample=sample,
        result=result,
        tokenizer=state.tokenizer,
        reward=reward,
        reward_meta=reward_meta,
    )
    return built, result


async def _run_groups(args, groups: list[list[Sample]], evaluation: bool) -> tuple[list[list[Sample]], list[dict[str, Any]], dict[str, float]]:
    state = LiveWebRolloutState(args)
    if evaluation:
        await state.prepare_for_eval()
    else:
        await state.prepare_for_train_rollout()
    browser_metrics_before = state.snapshot_browser_metrics()
    max_parallel_groups = int(os.getenv("LIVEWEB_PARALLEL_GROUPS", "8"))
    allow_zero_std_fallback = os.getenv("LIVEWEB_ALLOW_ZERO_STD_FALLBACK", "1") == "1"
    min_group_size = int(os.getenv("LIVEWEB_MIN_GROUP_SIZE", "2"))
    allow_partial_group_fallback = os.getenv("LIVEWEB_ALLOW_PARTIAL_GROUP_FALLBACK", "0") == "1"
    allow_last_resort_group_fallback = os.getenv("LIVEWEB_ALLOW_LAST_RESORT_GROUP_FALLBACK", "1") == "1"
    semaphore = asyncio.Semaphore(max_parallel_groups)
    accepted_groups: list[list[Sample]] = []
    zero_std_fallback_groups: list[list[Sample]] = []
    partial_fallback_groups: list[list[Sample]] = []
    nonempty_fallback_groups: list[list[Sample]] = []
    all_results: list[dict[str, Any]] = []
    zero_response_groups = 0
    zero_response_samples_dropped = 0

    async def run_single(sample: Sample) -> tuple[Sample | None, dict[str, Any]]:
        async with semaphore:
            return await _run_prompt_job(state, copy.deepcopy(sample), evaluation=evaluation)

    async def run_group(group: list[Sample]) -> tuple[list[Sample], list[dict[str, Any]]]:
        local_results = await asyncio.gather(*[run_single(sample) for sample in group])
        samples: list[Sample] = []
        results: list[dict[str, Any]] = []
        for maybe_sample, result in local_results:
            results.append(result)
            if maybe_sample is not None:
                samples.append(maybe_sample)
        return samples, results

    executed = await asyncio.gather(*[run_group(group) for group in groups])
    dropped_groups = 0
    zero_std_groups = 0
    partial_groups = 0
    for group_samples, group_results in executed:
        all_results.extend(group_results)
        if evaluation:
            accepted_groups.append(group_samples)
            continue
        valid_group_samples = [sample for sample in group_samples if sample.response_length > 0]
        dropped_for_zero_response = len(group_samples) - len(valid_group_samples)
        if dropped_for_zero_response:
            zero_response_groups += 1
            zero_response_samples_dropped += dropped_for_zero_response
            group_samples = valid_group_samples
        if group_samples:
            nonempty_fallback_groups.append(group_samples)
        if len(group_samples) < min_group_size:
            if allow_partial_group_fallback and group_samples:
                partial_fallback_groups.append(group_samples)
            dropped_groups += 1
            continue
        if len(group_samples) != len(groups[0]):
            partial_groups += 1
        rewards = [float(sample.reward) for sample in group_samples if sample.reward is not None]
        if len(set(round(reward, 8) for reward in rewards)) <= 1:
            zero_std_groups += 1
            if allow_zero_std_fallback:
                zero_std_fallback_groups.append(group_samples)
            continue
        accepted_groups.append(group_samples)

    zero_std_fallback_used = 0
    partial_fallback_used = 0
    last_resort_fallback_used = 0
    if not evaluation and not accepted_groups and zero_std_fallback_groups:
        accepted_groups = list(zero_std_fallback_groups)
        zero_std_fallback_used = len(accepted_groups)
    if not evaluation and not accepted_groups and partial_fallback_groups:
        accepted_groups = list(partial_fallback_groups)
        partial_fallback_used = len(accepted_groups)
    if not evaluation and not accepted_groups and allow_last_resort_group_fallback and nonempty_fallback_groups:
        accepted_groups = list(nonempty_fallback_groups)
        last_resort_fallback_used = len(accepted_groups)

    metrics = {
        "liveweb/accepted_groups": float(len(accepted_groups)),
        "liveweb/accepted_samples": float(_effective_sample_count(accepted_groups)),
        "liveweb/dropped_groups": float(dropped_groups),
        "liveweb/partial_groups": float(partial_groups),
        "liveweb/zero_std_groups": float(zero_std_groups),
        "liveweb/zero_response_groups": float(zero_response_groups),
        "liveweb/zero_response_samples_dropped": float(zero_response_samples_dropped),
        "liveweb/zero_std_fallback_groups": float(zero_std_fallback_used),
        "liveweb/partial_group_fallback_groups": float(partial_fallback_used),
        "liveweb/last_resort_fallback_groups": float(last_resort_fallback_used),
    }
    browser_metrics_after = state.snapshot_browser_metrics()
    for key, value in browser_metrics_after.items():
        metrics[f"liveweb/{key}"] = float(value - browser_metrics_before.get(key, 0))
    metrics |= _result_metrics(all_results, "liveweb")
    return accepted_groups, all_results, metrics


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    if evaluation:
        dataset_name, num_prompts, phase = get_eval_profile(rollout_id)
        base_seed = int(os.getenv("LIVEWEB_EVAL_BASE_SEED", "900000"))
        if dataset_name == "formal_eval":
            base_seed += 10000
        jobs = build_eval_jobs(
            rollout_id=rollout_id,
            dataset_name=dataset_name,
            num_prompts=num_prompts,
            phase=phase,
            base_seed=base_seed,
        )
        groups = []
        sample_index = 0
        for group_index, job in enumerate(jobs):
            sample = Sample(
                group_index=group_index,
                index=sample_index,
                prompt=f"{job.task_name}:{job.parent_seed}",
                metadata=job.to_metadata(),
                session_id=job.route_key,
            )
            sample_index += 1
            groups.append([sample])
        started_at = time.time()
        accepted_groups, _results, metrics = asyncio.run(_run_groups(args, groups, evaluation=True))
        samples = [sample for group in accepted_groups for sample in group]
        rewards = [float(sample.reward or 0.0) for sample in samples]
        metrics["liveweb/eval_wall_time"] = time.time() - started_at
        return RolloutFnEvalOutput(
            data={
                dataset_name: {
                    "rewards": rewards,
                    "samples": samples,
                    "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
                }
            },
            metrics=metrics,
        )

    target_groups = args.rollout_batch_size
    target_samples = _minimum_train_samples(args)
    collected_groups: list[list[Sample]] = []
    collected_results: list[dict[str, Any]] = []
    aggregate_metrics: dict[str, float] = {}
    attempts = 0
    max_attempts = max(4, target_groups * 4)
    started_at = time.time()

    while _effective_sample_count(collected_groups) < target_samples and attempts < max_attempts:
        attempts += 1
        missing_groups = max(1, target_groups - len(collected_groups))
        requested = min(max(missing_groups, target_groups), target_groups * 2)
        groups = data_source.get_samples(requested)
        accepted_groups, results, metrics = asyncio.run(_run_groups(args, groups, evaluation=False))
        collected_groups.extend(accepted_groups)
        collected_results.extend(results)
        for key, value in metrics.items():
            aggregate_metrics[key] = aggregate_metrics.get(key, 0.0) + value

    if not collected_groups:
        error_examples = [
            result.get("error")
            for result in collected_results
            if (result.get("extra") or {}).get("failure_reason") == "rollout_exception" and result.get("error")
        ][:3]
        raise RuntimeError(
            "LiveWeb online rollout produced no valid groups; "
            f"metrics={aggregate_metrics}; "
            f"error_examples={error_examples}"
        )
    if _effective_sample_count(collected_groups) < target_samples:
        error_examples = [
            result.get("error")
            for result in collected_results
            if (result.get("extra") or {}).get("failure_reason") == "rollout_exception" and result.get("error")
        ][:3]
        raise RuntimeError(
            "LiveWeb online rollout did not collect enough trainable samples; "
            f"effective_samples={_effective_sample_count(collected_groups)}; "
            f"target_samples={target_samples}; "
            f"metrics={aggregate_metrics}; "
            f"error_examples={error_examples}"
        )

    aggregate_metrics["liveweb/train_wall_time"] = time.time() - started_at
    aggregate_metrics["liveweb/attempts"] = float(attempts)
    aggregate_metrics["liveweb/final_groups"] = float(len(collected_groups))
    aggregate_metrics["liveweb/final_samples"] = float(_effective_sample_count(collected_groups))
    aggregate_metrics["liveweb/target_samples"] = float(target_samples)
    aggregate_metrics |= _result_metrics(collected_results, "liveweb/train")
    return RolloutFnTrainOutput(samples=collected_groups, metrics=aggregate_metrics)
