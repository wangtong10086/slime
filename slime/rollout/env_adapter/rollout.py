from __future__ import annotations

import copy
import json
import math
import os
from typing import Any

from slime.env_adapters.base import JobSpec, RolloutResult, RuntimeErrorInfo
from slime.env_adapters.registry import load_environment_adapter
from slime.env_adapters.runtime_worker import RuntimeJobRequest
from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

_SUM_METRIC_KEYS = {
    "env/accepted_groups",
    "env/accepted_samples",
    "env/dropped_groups",
    "env/partial_groups",
    "env/zero_std_groups",
    "env/zero_response_groups",
    "env/zero_response_samples_dropped",
    "env/zero_std_fallback_groups",
    "env/partial_group_fallback_groups",
    "env/last_resort_fallback_groups",
    "env/browser_rebuild_count",
    "env/browser_reuse_failures",
    "env/browser_recovery_success_count",
    "env/runtime_reset_count",
    "env/runtime_pool_hits",
    "env/worker_reuse_count",
    "env/worker_prepare_reset_count",
    "env/runtime_instance_reuse_count",
    "env/runtime_instance_reset_count",
    "scheduler/runtime_completed_jobs",
}

_MAX_METRIC_KEYS = {
    "scheduler/runtime_active_job_cap",
    "scheduler/runtime_queued_jobs",
    "env/jit_kernel_enabled",
    "env/kernel_fallback",
}

_CONST_METRIC_KEYS = {
    "scheduler/config_max_parallel_env_jobs",
    "scheduler/config_max_parallel_llm_jobs",
}


class AdapterRolloutState(metaclass=SingletonMeta):
    def __init__(self, args):
        self.args = args
        self.adapter = load_environment_adapter(
            getattr(args, "environment_adapter_path", None),
            env_name=getattr(args, "environment_name", None),
            args=args,
        )
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.workers: dict[str, Any] = {}
        self.active_scope: str | None = None

    def prepare(self, *, evaluation: bool, rollout_id: int | None = None):
        scope = self.adapter.get_runtime_scope(evaluation=evaluation, rollout_id=rollout_id)
        reset_scope = self.adapter.should_reset_scope(
            self.args,
            evaluation=evaluation,
            scope=scope,
            rollout_id=rollout_id,
        )
        self.active_scope = scope
        worker = self.workers.get(scope)
        if worker is None:
            worker = self.adapter.build_runtime_worker(self.args, scope=scope)
            self.workers[scope] = worker
        worker.prepare(evaluation=evaluation, rollout_id=rollout_id, reset_scope=reset_scope)
        return scope, worker

    def recover_runtime(self, *, scope: str, error: BaseException | None = None) -> bool:
        worker = self.workers[scope]
        return worker.recover(
            RuntimeErrorInfo(
                scope=scope,
                message=str(error) if error is not None else None,
                error_type=type(error).__name__ if error is not None else None,
            )
        )

    def snapshot_runtime_metrics(self, scope: str | None = None):
        active_scope = scope or self.active_scope
        if active_scope is None:
            return {}
        worker = self.workers.get(active_scope)
        if worker is None:
            return {}
        return worker.snapshot_metrics()

    def close(self) -> None:
        for worker in self.workers.values():
            worker.stop()
        self.workers = {}


def _effective_sample_count(groups: list[list[Sample]]) -> int:
    return sum(len(group) for group in groups)


def _maybe_dump_raw_results(
    *,
    args: Any,
    raw_results: list[dict[str, Any]],
    rollout_id: int,
    evaluation: bool,
) -> None:
    if os.getenv("SLIME_DUMP_ROLLOUT_TRAJECTORIES", "0") != "1":
        return
    run_root = os.getenv("RUN_ROOT") or getattr(args, "save", None)
    if not run_root:
        return
    dump_dir = os.path.join(run_root, "rollout_dumps")
    os.makedirs(dump_dir, exist_ok=True)
    prefix = "eval" if evaluation else "train"
    path = os.path.join(dump_dir, f"{prefix}_rollout_{rollout_id:05d}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for item in raw_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _materialize_sample(
    state: AdapterRolloutState,
    sample: Sample,
    result: RolloutResult,
    *,
    evaluation: bool,
) -> tuple[Sample | None, RolloutResult]:
    reward, reward_meta = state.adapter.compute_reward(result)
    sample.metadata["adapter_rollout_result"] = result.raw_result
    if reward is None:
        if evaluation:
            sample.reward = 0.0
            sample.response = ""
            sample.response_length = 0
            sample.loss_mask = []
            sample.status = Sample.Status.FAILED
            sample.metadata.update(
                {
                    "task_name": result.task_name,
                    "score": result.reward,
                    "success": result.success,
                    "time_taken": result.time_taken,
                    "failure_kind": result.failure_kind.value if result.failure_kind else None,
                    "raw_reward": result.reward,
                    "learning_bucket": ((result.raw_result.get("extra") or {}).get("learning_bucket")),
                    "progress_summary": ((result.raw_result.get("extra") or {}).get("progress_summary") or {}),
                }
            )
            return sample, result
        return None, result
    built = state.adapter.build_training_sample(
        sample=sample,
        result=result,
        tokenizer=state.tokenizer,
        reward=reward,
        reward_meta=reward_meta,
    )
    return built, result


def _merge_rollout_metrics(aggregate: dict[str, float], metrics: dict[str, float]) -> None:
    for key in _SUM_METRIC_KEYS:
        if key in metrics:
            aggregate[key] = aggregate.get(key, 0.0) + float(metrics[key])

    for key in _MAX_METRIC_KEYS:
        if key in metrics:
            aggregate[key] = max(aggregate.get(key, float("-inf")), float(metrics[key]))

    for key in _CONST_METRIC_KEYS:
        if key in metrics and key not in aggregate:
            aggregate[key] = float(metrics[key])


def _finalize_rollout_metrics(
    *,
    args: Any,
    adapter: Any,
    accepted_groups: list[list[Sample]],
    all_results: list[RolloutResult],
    aggregate_metrics: dict[str, float],
    total_requested_groups: int,
    total_requested_jobs: int,
    base_group_target: int,
) -> dict[str, float]:
    metrics = dict(aggregate_metrics)
    metrics["scheduler/group_fill_rate"] = float(len(accepted_groups) / max(1, total_requested_groups))
    metrics["scheduler/oversample_ratio"] = float(total_requested_groups / max(1, base_group_target))
    metrics["scheduler/requested_groups"] = float(total_requested_groups)
    metrics["scheduler/requested_jobs"] = float(total_requested_jobs)

    if "scheduler/runtime_active_job_cap" in metrics:
        metrics["scheduler/active_jobs"] = metrics["scheduler/runtime_active_job_cap"]
    if "scheduler/runtime_queued_jobs" in metrics:
        metrics["scheduler/queued_jobs"] = metrics["scheduler/runtime_queued_jobs"]
    if "scheduler/runtime_completed_jobs" in metrics:
        metrics["scheduler/completed_jobs"] = metrics["scheduler/runtime_completed_jobs"]
    if "scheduler/config_max_parallel_env_jobs" in metrics:
        metrics["scheduler/max_parallel_env_jobs"] = metrics["scheduler/config_max_parallel_env_jobs"]
    if "scheduler/config_max_parallel_llm_jobs" in metrics:
        metrics["scheduler/max_parallel_llm_jobs"] = metrics["scheduler/config_max_parallel_llm_jobs"]

    if all_results:
        metrics |= adapter.summarize_results(all_results, "env")
        runtime_reset_count = metrics.get("env/runtime_reset_count", 0.0)
        runtime_pool_hits = metrics.get("env/runtime_pool_hits", 0.0)
        metrics["env/runtime_reset_rate"] = runtime_reset_count / len(all_results)
        metrics["env/runtime_pool_hit_rate"] = runtime_pool_hits / max(1.0, runtime_pool_hits + runtime_reset_count)
        worker_reuse_count = metrics.get("env/worker_reuse_count", 0.0)
        worker_prepare_reset_count = metrics.get("env/worker_prepare_reset_count", 0.0)
        runtime_instance_reuse_count = metrics.get("env/runtime_instance_reuse_count", 0.0)
        runtime_instance_reset_count = metrics.get("env/runtime_instance_reset_count", 0.0)
        metrics["env/runtime_worker_reuse_rate"] = worker_reuse_count / len(all_results)
        metrics["env/runtime_instance_reuse_rate"] = runtime_instance_reuse_count / len(all_results)
        metrics["env/runtime_prepare_reset_rate"] = worker_prepare_reset_count / len(all_results)
        metrics["env/runtime_instance_reset_rate"] = runtime_instance_reset_count / len(all_results)

    return metrics


def _run_single_job(
    state: AdapterRolloutState,
    worker: Any,
    scope: str,
    sample: Sample,
    *,
    evaluation: bool,
) -> tuple[Sample | None, RolloutResult]:
    request = RuntimeJobRequest(
        group_position=0,
        sample_position=0,
        job=JobSpec.from_dict(sample.metadata["job_spec"]),
    )
    outcome = worker.run_jobs(
        [request],
        evaluation=evaluation,
        max_parallel_env_jobs=1,
    )[0]
    built, result = _materialize_sample(
        state,
        copy.deepcopy(sample),
        outcome.result,
        evaluation=evaluation,
    )
    return built, result


def _run_groups(args, groups: list[list[Sample]], evaluation: bool, rollout_id: int | None = None):
    state = AdapterRolloutState(args)
    scope, worker = state.prepare(evaluation=evaluation, rollout_id=rollout_id)
    metrics_before = state.snapshot_runtime_metrics(scope=scope)
    max_parallel_env_jobs = int(
        os.getenv(
            "SLIME_ENV_MAX_PARALLEL_ENV_JOBS",
            os.getenv("LIVEWEB_MAX_BROWSER_SESSIONS", "32"),
        )
    )
    max_parallel_llm_jobs = int(
        os.getenv(
            "SLIME_ENV_MAX_PARALLEL_LLM_JOBS",
            os.getenv("LIVEWEB_MAX_LLM_REQUESTS", "16"),
        )
    )
    allow_zero_std_fallback = os.getenv("SLIME_ENV_ALLOW_ZERO_STD_FALLBACK", "1") == "1"
    min_group_size = int(os.getenv("SLIME_ENV_MIN_GROUP_SIZE", "2"))
    allow_partial_group_fallback = os.getenv("SLIME_ENV_ALLOW_PARTIAL_GROUP_FALLBACK", "0") == "1"
    allow_last_resort_group_fallback = os.getenv("SLIME_ENV_ALLOW_LAST_RESORT_GROUP_FALLBACK", "1") == "1"
    accepted_groups: list[list[Sample]] = []
    zero_std_fallback_groups: list[list[Sample]] = []
    partial_fallback_groups: list[list[Sample]] = []
    nonempty_fallback_groups: list[list[Sample]] = []
    all_results: list[RolloutResult] = []
    zero_response_groups = 0
    zero_response_samples_dropped = 0
    total_jobs = sum(len(group) for group in groups)
    completed_jobs = 0
    requests = [
        RuntimeJobRequest(
            group_position=group_position,
            sample_position=sample_position,
            job=JobSpec.from_dict(sample.metadata["job_spec"]),
        )
        for group_position, group in enumerate(groups)
        for sample_position, sample in enumerate(group)
    ]
    grouped_results: dict[int, list[tuple[int, Sample | None, RolloutResult]]] = {idx: [] for idx in range(len(groups))}
    executed = worker.run_jobs(
        requests,
        evaluation=evaluation,
        max_parallel_env_jobs=max_parallel_env_jobs,
    )
    max_active_jobs = min(max_parallel_env_jobs, len(requests))
    completed_jobs = len(executed)
    request_to_sample = {
        (group_position, sample_position): groups[group_position][sample_position]
        for group_position, group in enumerate(groups)
        for sample_position, _sample in enumerate(group)
    }
    for outcome in executed:
        original_sample = copy.deepcopy(
            request_to_sample[(outcome.group_position, outcome.sample_position)]
        )
        maybe_sample, result = _materialize_sample(
            state,
            original_sample,
            outcome.result,
            evaluation=evaluation,
        )
        grouped_results[outcome.group_position].append((outcome.sample_position, maybe_sample, result))
    dropped_groups = 0
    zero_std_groups = 0
    partial_groups = 0
    group_feedback: list[dict[str, Any]] = []

    for group_position in range(len(groups)):
        entries = sorted(grouped_results[group_position], key=lambda item: item[0])
        group_samples: list[Sample] = []
        group_results: list[RolloutResult] = []
        for _, maybe_sample, result in entries:
            group_results.append(result)
            if maybe_sample is not None:
                group_samples.append(maybe_sample)
        all_results.extend(group_results)
        if evaluation:
            accepted_groups.append(group_samples)
            continue
        combo_key = ""
        if groups[group_position]:
            job_spec = groups[group_position][0].metadata.get("job_spec", {})
            combo_key = (
                ((job_spec.get("metadata") or {}).get("combo_key"))
                or ((job_spec.get("task") or {}).get("metadata") or {}).get("combo_key")
                or ""
            )
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
        zero_std = len(set(round(reward, 8) for reward in rewards)) <= 1
        group_feedback.append(
            {
                "combo_key": combo_key,
                "mean_score": (sum(item.reward for item in group_results) / len(group_results)) if group_results else 0.0,
                "success_rate": (
                    sum(1.0 if item.success else 0.0 for item in group_results) / len(group_results)
                ) if group_results else 0.0,
                "env_error_rate": (
                    sum(1.0 if item.environment_pollution else 0.0 for item in group_results) / len(group_results)
                ) if group_results else 0.0,
                "mean_progress_score": (
                    sum(
                        float(((item.raw_result.get("extra") or {}).get("progress_summary") or {}).get("progress_score", 0.0))
                        for item in group_results
                    )
                    / len(group_results)
                ) if group_results else 0.0,
                "near_miss_rate": (
                    sum(
                        1.0 if ((item.raw_result.get("extra") or {}).get("learning_bucket") == "near_miss") else 0.0
                        for item in group_results
                    )
                    / len(group_results)
                ) if group_results else 0.0,
                "format_failure_rate": (
                    sum(
                        1.0 if ((item.raw_result.get("extra") or {}).get("learning_bucket") == "format_failure") else 0.0
                        for item in group_results
                    )
                    / len(group_results)
                ) if group_results else 0.0,
                "accepted": False,
                "zero_std": zero_std,
            }
        )
        if zero_std:
            zero_std_groups += 1
            if allow_zero_std_fallback:
                zero_std_fallback_groups.append(group_samples)
            continue
        group_feedback[-1]["accepted"] = True
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
    state.adapter.record_group_feedback(group_feedback, evaluation=evaluation)

    metrics = {
        "env/accepted_groups": float(len(accepted_groups)),
        "env/accepted_samples": float(_effective_sample_count(accepted_groups)),
        "env/dropped_groups": float(dropped_groups),
        "env/partial_groups": float(partial_groups),
        "env/zero_std_groups": float(zero_std_groups),
        "env/zero_response_groups": float(zero_response_groups),
        "env/zero_response_samples_dropped": float(zero_response_samples_dropped),
        "env/zero_std_fallback_groups": float(zero_std_fallback_used),
        "env/partial_group_fallback_groups": float(partial_fallback_used),
        "env/last_resort_fallback_groups": float(last_resort_fallback_used),
        "scheduler/runtime_active_job_cap": float(max_active_jobs),
        "scheduler/runtime_queued_jobs": float(max(0, total_jobs - max_active_jobs)),
        "scheduler/runtime_completed_jobs": float(completed_jobs),
        "scheduler/group_fill_rate": float(len(accepted_groups) / max(1, len(groups))),
        "scheduler/config_max_parallel_env_jobs": float(max_parallel_env_jobs),
        "scheduler/config_max_parallel_llm_jobs": float(max_parallel_llm_jobs),
    }
    metrics["scheduler/active_jobs"] = metrics["scheduler/runtime_active_job_cap"]
    metrics["scheduler/queued_jobs"] = metrics["scheduler/runtime_queued_jobs"]
    metrics["scheduler/completed_jobs"] = metrics["scheduler/runtime_completed_jobs"]
    metrics["scheduler/max_parallel_env_jobs"] = metrics["scheduler/config_max_parallel_env_jobs"]
    metrics["scheduler/max_parallel_llm_jobs"] = metrics["scheduler/config_max_parallel_llm_jobs"]
    metrics_after = state.snapshot_runtime_metrics(scope=scope)
    for key, value in metrics_after.items():
        metrics[f"env/{key}"] = float(value - metrics_before.get(key, 0))
    if all_results:
        runtime_reset_count = metrics.get("env/runtime_reset_count", 0.0)
        runtime_pool_hits = metrics.get("env/runtime_pool_hits", 0.0)
        metrics["env/runtime_reset_rate"] = runtime_reset_count / len(all_results)
        metrics["env/runtime_pool_hit_rate"] = runtime_pool_hits / max(1.0, runtime_pool_hits + runtime_reset_count)
        worker_reuse_count = metrics.get("env/worker_reuse_count", 0.0)
        worker_prepare_reset_count = metrics.get("env/worker_prepare_reset_count", 0.0)
        runtime_instance_reuse_count = metrics.get("env/runtime_instance_reuse_count", 0.0)
        runtime_instance_reset_count = metrics.get("env/runtime_instance_reset_count", 0.0)
        metrics["env/runtime_worker_reuse_rate"] = worker_reuse_count / len(all_results)
        metrics["env/runtime_instance_reuse_rate"] = runtime_instance_reuse_count / len(all_results)
        metrics["env/runtime_prepare_reset_rate"] = worker_prepare_reset_count / len(all_results)
        metrics["env/runtime_instance_reset_rate"] = runtime_instance_reset_count / len(all_results)
    metrics |= state.adapter.summarize_results(all_results, "env")
    return accepted_groups, all_results, metrics


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    state = AdapterRolloutState(args)
    adapter = state.adapter
    if evaluation:
        plan = adapter.get_evaluation_plan(args, rollout_id)
        tasks = adapter.sample_tasks(split="eval", count=plan.num_tasks, phase=plan.phase, rollout_id=rollout_id)
        jobs = adapter.expand_jobs(tasks=tasks, n_samples_per_task=1, mode="eval")
        groups = []
        sample_index = 0
        for group_index, job in enumerate(jobs):
            groups.append(
                [
                    Sample(
                        group_index=group_index,
                        index=sample_index,
                        prompt=job.prompt_hint or job.job_id,
                        metadata={"job_spec": job.to_dict(), "env_name": adapter.name},
                        session_id=job.affinity_key,
                    )
                ]
            )
            sample_index += 1
        accepted_groups, results, metrics = _run_groups(args, groups, evaluation=True, rollout_id=rollout_id)
        samples = [sample for group in accepted_groups for sample in group]
        rewards = [float(sample.reward or 0.0) for sample in samples]
        data = {
            plan.dataset_name: {
                "rewards": rewards,
                "samples": samples,
                "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
                "raw_results": [result.raw_result for result in results],
            }
        }
        _maybe_dump_raw_results(
            args=args,
            raw_results=data[plan.dataset_name]["raw_results"],
            rollout_id=rollout_id,
            evaluation=True,
        )
        return RolloutFnEvalOutput(data=data, metrics=metrics)

    base_group_target = int(args.rollout_batch_size)
    group_size = max(1, int(getattr(args, "n_samples_per_prompt", 1)))
    target_active_jobs = int(os.getenv("SLIME_ENV_TARGET_ACTIVE_JOBS", "64"))
    target_ready_groups = int(os.getenv("SLIME_ENV_TARGET_READY_GROUPS", "8"))
    oversample_factor = float(os.getenv("SLIME_ENV_OVERSAMPLE_FACTOR", "2.0"))
    initial_group_target = max(
        base_group_target,
        target_ready_groups,
        int(math.ceil(target_active_jobs / group_size)),
        int(math.ceil(base_group_target * oversample_factor)),
    )
    groups = data_source.get_samples(initial_group_target)
    accepted_groups, initial_results, metrics = _run_groups(args, groups, evaluation=False, rollout_id=rollout_id)
    aggregate_metrics: dict[str, float] = {}
    _merge_rollout_metrics(aggregate_metrics, metrics)
    all_results = list(initial_results)
    all_raw_results = [result.raw_result for result in initial_results]
    min_samples = max(1, min(int(args.global_batch_size), int(args.rollout_batch_size)))
    total_requested_groups = len(groups)
    total_requested_jobs = sum(len(group) for group in groups)
    max_total_requested_groups = max(initial_group_target, int(math.ceil(base_group_target * max(oversample_factor, 1.0) * 2.0)))
    while _effective_sample_count(accepted_groups) < min_samples or len(accepted_groups) < target_ready_groups:
        if total_requested_groups >= max_total_requested_groups:
            break
        effective_samples = _effective_sample_count(accepted_groups)
        missing_samples = max(0, min_samples - effective_samples)
        missing_groups = max(0, target_ready_groups - len(accepted_groups))
        extra_group_target = max(
            int(args.over_sampling_batch_size or args.rollout_batch_size),
            missing_groups,
            int(math.ceil(max(missing_samples, group_size) / group_size)),
        )
        extra_group_target = min(extra_group_target, max_total_requested_groups - total_requested_groups)
        extra_groups = data_source.get_samples(extra_group_target)
        if not extra_groups:
            break
        extra_accepted, extra_results, extra_metrics = _run_groups(
            args,
            extra_groups,
            evaluation=False,
            rollout_id=rollout_id,
        )
        accepted_groups.extend(extra_accepted)
        all_results.extend(extra_results)
        all_raw_results.extend(result.raw_result for result in extra_results)
        total_requested_groups += len(extra_groups)
        total_requested_jobs += sum(len(group) for group in extra_groups)
        _merge_rollout_metrics(aggregate_metrics, extra_metrics)

    if not accepted_groups:
        error_examples = [
            result.get("error")
            for result in all_raw_results
            if (result.get("extra") or {}).get("failure_reason") in {"rollout_exception", "site_unreachable", "cache_error"}
            and result.get("error")
        ][:3]
        raise RuntimeError(
            "Environment-adapter rollout produced no trainable groups; "
            f"metrics={aggregate_metrics}; "
            f"error_examples={error_examples}"
        )

    metrics = _finalize_rollout_metrics(
        args=args,
        adapter=adapter,
        accepted_groups=accepted_groups,
        all_results=all_results,
        aggregate_metrics=aggregate_metrics,
        total_requested_groups=total_requested_groups,
        total_requested_jobs=total_requested_jobs,
        base_group_target=base_group_target,
    )
    _maybe_dump_raw_results(
        args=args,
        raw_results=all_raw_results,
        rollout_id=rollout_id,
        evaluation=False,
    )
    return RolloutFnTrainOutput(samples=accepted_groups, metrics=metrics)
