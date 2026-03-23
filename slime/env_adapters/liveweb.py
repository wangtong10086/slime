from __future__ import annotations

import os
import re
from collections import deque
from typing import Any
from urllib.parse import urlparse

from slime.rollout.liveweb_online.common import (
    LiveWebRolloutState,
    PromptJob,
    classify_overflow_failure,
    compute_reward_from_result,
    evaluate_prompt_job,
    get_eval_profile,
    make_sample_from_result,
    should_allow_environment_fallback,
    summarize_cache_stats,
)
from slime.rollout.liveweb_online.task_sampling import (
    LiveWebDynamicSampler,
    canonicalize_phase_name,
    load_task_mix_config,
    parse_plugin_csv_env,
)
from slime.utils.metric_utils import summarize_task_count_metrics
from slime.utils.types import Sample

from .base import (
    EnvironmentAdapter,
    EvaluationPlan,
    FailureKind,
    JobSpec,
    ResourceProfile,
    RolloutResult,
    RuntimeCapabilities,
    TaskSpec,
)


class LiveWebEnvironmentAdapter(EnvironmentAdapter):
    name = "liveweb"

    def __init__(self, args=None):
        self.args = args
        self._scoped_states: dict[str, LiveWebRolloutState] = {}
        self._phase = canonicalize_phase_name(os.getenv("LIVEWEB_TASK_MIX_PHASE", "bootstrap"))
        self._phase_metrics_window = int(os.getenv("LIVEWEB_CURRICULUM_METRICS_WINDOW", "32"))
        self._phase_feedback: deque[dict[str, float]] = deque(maxlen=self._phase_metrics_window)
        self._parent_seed_cursor = int(os.getenv("LIVEWEB_TRAIN_BASE_SEED", "100000"))
        self._excluded_plugins = parse_plugin_csv_env("LIVEWEB_EXCLUDE_PLUGINS", "weather,openlibrary")
        self._sampler = LiveWebDynamicSampler(
            excluded_plugins=self._excluded_plugins,
            min_unique_plugins=int(os.getenv("LIVEWEB_MIN_UNIQUE_PLUGINS", "2")),
            window_size=int(os.getenv("LIVEWEB_DYNAMIC_SAMPLER_WINDOW", "64")),
            dynamic_mix=(
                float(os.getenv("LIVEWEB_DYNAMIC_SAMPLING_DYNAMIC_RATIO", "0.5")),
                float(os.getenv("LIVEWEB_DYNAMIC_SAMPLING_BASE_RATIO", "0.3")),
                float(os.getenv("LIVEWEB_DYNAMIC_SAMPLING_EXPLORE_RATIO", "0.2")),
            ),
        )

    def get_runtime_capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            supports_evaluate=True,
            supports_openenv=True,
            supports_trajectory_export=True,
            supports_offline_dataset_export=True,
        )

    def _phase_config(self) -> dict[str, Any]:
        phase = canonicalize_phase_name(os.getenv("LIVEWEB_TASK_MIX_PHASE", self._phase))
        config = load_task_mix_config()
        return dict(config.get("phases", {}).get(phase) or {})

    def _curriculum_config(self) -> dict[str, Any]:
        config = load_task_mix_config()
        curriculum = config.get("curriculum") or {}
        return dict(curriculum.get(canonicalize_phase_name(self._phase)) or {})

    def _maybe_promote_phase(self) -> None:
        if os.getenv("LIVEWEB_ENABLE_AUTO_CURRICULUM", "1") != "1":
            return
        curriculum = self._curriculum_config()
        next_phase = curriculum.get("next_phase")
        if not next_phase or not self._phase_feedback:
            return

        window = list(self._phase_feedback)
        count = len(window)
        if count <= 0:
            return

        def _mean(key: str) -> float:
            return sum(float(item.get(key, 0.0)) for item in window) / count

        metrics = {
            "wrong_domain_loop_rate": _mean("wrong_domain_loop_rate"),
            "unsupported_stop_rate": _mean("unsupported_stop_rate"),
            "mean_score": _mean("mean_score"),
            "near_miss_rate": _mean("near_miss_rate"),
            "max_steps_reached_rate": _mean("max_steps_reached_rate"),
        }
        if count >= 2:
            split = max(1, count // 2)
            first = window[:split]
            second = window[split:]
            if second:
                first_mean = sum(float(item.get("max_steps_reached_rate", 0.0)) for item in first) / len(first)
                second_mean = sum(float(item.get("max_steps_reached_rate", 0.0)) for item in second) / len(second)
                metrics["max_steps_reached_rate_delta"] = second_mean - first_mean
            else:
                metrics["max_steps_reached_rate_delta"] = 0.0
        else:
            metrics["max_steps_reached_rate_delta"] = 0.0

        checks: list[bool] = []
        if "wrong_domain_loop_rate_lte" in curriculum:
            checks.append(metrics["wrong_domain_loop_rate"] <= float(curriculum["wrong_domain_loop_rate_lte"]))
        if "unsupported_stop_rate_lte" in curriculum:
            checks.append(metrics["unsupported_stop_rate"] <= float(curriculum["unsupported_stop_rate_lte"]))
        if "mean_score_gte" in curriculum:
            checks.append(metrics["mean_score"] >= float(curriculum["mean_score_gte"]))
        if "near_miss_rate_gte" in curriculum:
            checks.append(metrics["near_miss_rate"] >= float(curriculum["near_miss_rate_gte"]))
        if "max_steps_reached_rate_delta_lte" in curriculum:
            checks.append(
                metrics["max_steps_reached_rate_delta"] <= float(curriculum["max_steps_reached_rate_delta_lte"])
            )
        if checks and all(checks):
            self._phase = canonicalize_phase_name(str(next_phase))
            os.environ["LIVEWEB_TASK_MIX_PHASE"] = self._phase
            self._phase_feedback.clear()

    def get_train_phase(self, args) -> str:
        self._phase = canonicalize_phase_name(os.getenv("LIVEWEB_TASK_MIX_PHASE", self._phase))
        return self._phase

    def get_evaluation_plan(self, args, rollout_id: int) -> EvaluationPlan:
        dataset_name, num_prompts, phase = get_eval_profile(rollout_id)
        return EvaluationPlan(dataset_name=dataset_name, num_tasks=num_prompts, phase=phase)

    def should_reset_scope(self, args, *, evaluation: bool, scope: str, rollout_id: int | None = None) -> bool:
        if evaluation:
            return True
        return os.getenv("LIVEWEB_FORCE_RESET_TRAIN_SCOPE", "0") == "1"

    def sample_tasks(self, *, split: str, count: int, phase: str, rollout_id: int | None = None) -> list[TaskSpec]:
        phase = canonicalize_phase_name(phase)
        tasks: list[TaskSpec] = []
        if split == "eval":
            base_seed = int(os.getenv("LIVEWEB_EVAL_BASE_SEED", "900000"))
            if phase in {"main_warm", "online_align"}:
                base_seed += 10000
            for idx in range(count):
                parent_seed = base_seed + idx
                selection = self._sampler.sample(seed=parent_seed, phase=phase, evaluation=True)
                tasks.append(
                    TaskSpec(
                        env_name=self.name,
                        task_family="composite_prompt",
                        task_id=selection["task_id"],
                        seed=parent_seed,
                        metadata={"phase": phase, "split": split, **selection},
                    )
                )
            return tasks

        for _ in range(count):
            parent_seed = self._parent_seed_cursor
            self._parent_seed_cursor += 1
            selection = self._sampler.sample(seed=parent_seed, phase=phase, evaluation=False)
            tasks.append(
                TaskSpec(
                    env_name=self.name,
                    task_family="composite_prompt",
                    task_id=selection["task_id"],
                    seed=parent_seed,
                    metadata={"phase": phase, "split": split, **selection},
                )
            )
        return tasks

    def expand_jobs(self, *, tasks: list[TaskSpec], n_samples_per_task: int, mode: str) -> list[JobSpec]:
        jobs: list[JobSpec] = []
        for task in tasks:
            parent_seed = int(task.seed)
            sample_count = 1 if mode == "eval" else n_samples_per_task
            for sample_index in range(sample_count):
                task_id = int(task.metadata["task_id"])
                combo_key = str(task.metadata["combo_key"])
                prompt_job = PromptJob(
                    parent_seed=parent_seed,
                    task_id=task_id,
                    task_seed=int(task.metadata["task_seed"]),
                    llm_seed=abs(hash((parent_seed, sample_index, "liveweb_online_rl"))) % (2**31 - 1),
                    subtask_index=1,
                    num_subtasks=int(task.metadata["num_subtasks"]),
                    templates=[tuple(item) for item in task.metadata["templates"]],
                    task_name=f"{mode}:{combo_key}",
                    plugin_name=combo_key,
                    plugin_names=list(task.metadata.get("plugin_names") or []),
                    combo_index=int(task.metadata["combo_index"]),
                    combo_key=combo_key,
                    phase=canonicalize_phase_name(str(task.metadata.get("phase", self._phase))),
                    failure_bucket=str(task.metadata.get("failure_bucket") or ""),
                    route_key=f"{mode}:task:{task_id}",
                )
                jobs.append(
                    JobSpec(
                        env_name=self.name,
                        job_id=f"{mode}:{parent_seed}:sample:{sample_index}",
                        group_id=f"{mode}:{parent_seed}",
                        index_in_group=sample_index,
                        mode=mode,
                        task=task,
                        seed=prompt_job.llm_seed,
                        metadata={
                            "prompt_job": prompt_job.to_metadata(),
                            "plugin_name": prompt_job.plugin_name,
                            "combo_key": combo_key,
                            "sampling_strategy": task.metadata.get("sampling_strategy"),
                        },
                        affinity_key=prompt_job.route_key,
                        resource_profile=ResourceProfile({"llm": 1, "env": 1}),
                        prompt_hint=f"{prompt_job.task_name}:{task_id}",
                    )
                )
        return jobs

    async def ensure_runtime(self, args, *, scope: str) -> LiveWebRolloutState:
        state = self._scoped_states.get(scope)
        if state is None:
            state = LiveWebRolloutState(args, scope=scope)
            self._scoped_states[scope] = state
        return state

    async def prepare_phase(self, args, *, evaluation: bool, scope: str, reset_scope: bool) -> None:
        state = await self.ensure_runtime(args, scope=scope)
        if reset_scope:
            await self.cleanup_scope(scope)
            state = await self.ensure_runtime(args, scope=scope)
        if evaluation:
            await state.prepare_for_eval()
        else:
            await state.prepare_for_train_rollout()

    async def recover_runtime(self, args, *, scope: str, error: BaseException | None = None) -> bool:
        state = await self.ensure_runtime(args, scope=scope)
        await state.recover_browser(force_refresh=True)
        return True

    async def cleanup_scope(self, scope: str) -> None:
        state = self._scoped_states.pop(scope, None)
        if state is not None:
            await state.actor.shutdown()

    async def run_job(self, args, runtime: LiveWebRolloutState, job: JobSpec, *, evaluation: bool) -> RolloutResult:
        prompt_metadata = dict(job.metadata["prompt_job"])
        prompt_metadata.setdefault("task_id", int(job.task.task_id))
        prompt_metadata.setdefault("plugin_names", list(job.task.metadata.get("plugin_names") or []))
        prompt_metadata.setdefault("combo_index", int(job.task.metadata.get("combo_index", 0)))
        prompt_metadata.setdefault("combo_key", str(job.task.metadata.get("combo_key") or job.metadata.get("combo_key") or ""))
        prompt_job = PromptJob(**prompt_metadata)
        result = await evaluate_prompt_job(args, runtime, prompt_job)
        self._normalize_failure_reason(result)
        runtime.record_job_outcome(result)
        failure_reason = (result.get("extra") or {}).get("failure_reason")
        environment_failure_type = self._classify_environment_failure_type(result)
        env_drop_reasons = {
            "site_unreachable",
            "cache_error",
            "llm_error",
            "rollout_exception",
            "prefetch_failed",
            "cache_fill_failed",
            "env_nav_timeout",
            "env_cdn_blocked",
            "challenge_page",
            "control_plane_auth_failure",
            "sample_wall_timeout",
            "group_wall_timeout",
        }
        is_env_pollution = (
            environment_failure_type in env_drop_reasons
            or failure_reason in env_drop_reasons
        )
        failure_kind = None
        if failure_reason == "rollout_exception":
            browser_closed = bool((result.get("extra") or {}).get("browser_transport_closed"))
            failure_kind = FailureKind.RECOVERABLE_RUNTIME_FAILURE if browser_closed else FailureKind.ENV_RUNTIME_FAILURE
        elif failure_reason in {"site_unreachable", "cache_error"}:
            failure_kind = FailureKind.ENV_RUNTIME_FAILURE
        elif failure_reason == "parse_failed":
            failure_kind = FailureKind.INVALID_MODEL_OUTPUT
        elif failure_reason in {"llm_error"}:
            failure_kind = FailureKind.MODEL_REQUEST_FAILURE
        elif failure_reason:
            failure_kind = FailureKind.TASK_FAILURE
        metrics = {
            **((result.get("extra") or {}).get("cache_stats") or {}),
            **({"usage": ((result.get("extra") or {}).get("usage") or {})}),
            "failure_reason": failure_reason,
            "environment_failure_type": environment_failure_type,
        }
        return RolloutResult(
            env_name=self.name,
            task_name=str(result.get("task_name", "liveweb")),
            reward=float(result.get("score", 0.0)),
            success=bool(result.get("success", False)),
            time_taken=float(result.get("time_taken", 0.0)),
            metrics=metrics,
            failure_kind=failure_kind,
            failure_stage=(result.get("extra") or {}).get("exception_stage"),
            recoverable=(failure_kind == FailureKind.RECOVERABLE_RUNTIME_FAILURE),
            runtime_scope=runtime.scope,
            environment_pollution=is_env_pollution,
            drop_from_training=is_env_pollution,
            raw_result=result,
            error=result.get("error"),
        )

    @staticmethod
    def _normalize_failure_reason(result: dict[str, Any]) -> None:
        extra = result.setdefault("extra", {})
        current = str(extra.get("failure_reason") or "")
        if current in {"control_plane_auth_failure", "format_recovery_overflow", "llm_context_overflow"}:
            return
        overflow_failure = classify_overflow_failure(result)
        if overflow_failure is not None:
            extra["failure_reason"] = overflow_failure
            return
        error_lower = str(result.get("error") or "").lower()
        if "401 unauthorized" in error_lower or "403 forbidden" in error_lower:
            extra["failure_reason"] = "control_plane_auth_failure"

    def compute_reward(self, result: RolloutResult) -> tuple[float | None, dict[str, Any]]:
        reward, reward_meta = compute_reward_from_result(result.raw_result)
        if reward is None and result.raw_result and should_allow_environment_fallback():
            conversation = ((result.raw_result.get("extra") or {}).get("conversation")) or []
            if conversation:
                reward_meta = {
                    **reward_meta,
                    "drop_reason": None,
                    "environment_failure_type": reward_meta.get("environment_failure_type"),
                    "environment_fallback": True,
                }
                reward = 0.0
        return reward, reward_meta

    def build_training_sample(
        self,
        *,
        sample: Sample,
        result: RolloutResult,
        tokenizer,
        reward: float,
        reward_meta: dict[str, Any],
    ) -> Sample | None:
        return make_sample_from_result(
            sample=sample,
            result=result.raw_result,
            tokenizer=tokenizer,
            reward=reward,
            reward_meta=reward_meta,
        )

    def summarize_results(self, results: list[RolloutResult], prefix: str) -> dict[str, float]:
        raw_results = [result.raw_result for result in results]
        metrics = super().summarize_results(results, prefix)
        metrics |= summarize_cache_stats([(item.get("extra") or {}).get("cache_stats") or {} for item in raw_results])
        if not results:
            return metrics

        def _rate(predicate) -> float:
            return sum(1.0 if predicate(item) else 0.0 for item in results) / len(results)

        metrics["env/site_unreachable_rate"] = _rate(
            lambda item: ((item.raw_result.get("extra") or {}).get("failure_reason") == "site_unreachable")
        )
        metrics["env/cache_error_rate"] = _rate(
            lambda item: ((item.raw_result.get("extra") or {}).get("failure_reason") == "cache_error")
        )
        metrics["env/parse_failed_rate"] = _rate(
            lambda item: ((item.raw_result.get("extra") or {}).get("failure_reason") == "parse_failed")
        )
        metrics["env/llm_error_rate"] = _rate(
            lambda item: ((item.raw_result.get("extra") or {}).get("failure_reason") == "llm_error")
        )
        metrics["env/llm_context_overflow_rate"] = _rate(
            lambda item: ((item.raw_result.get("extra") or {}).get("failure_reason") == "llm_context_overflow")
        )
        metrics["env/format_recovery_overflow_rate"] = _rate(
            lambda item: ((item.raw_result.get("extra") or {}).get("failure_reason") == "format_recovery_overflow")
        )
        metrics["env/control_plane_auth_failure_count"] = sum(
            1.0
            for item in results
            if ((item.raw_result.get("extra") or {}).get("failure_reason") == "control_plane_auth_failure")
        )
        metrics["env/sample_wall_timeout_count"] = sum(
            1.0
            for item in results
            if ((item.raw_result.get("extra") or {}).get("failure_reason") == "sample_wall_timeout")
        )
        metrics["env/abort_failed_count"] = max(
            float(metrics.get("env/abort_failed_count", 0.0)),
            sum(float((item.raw_result.get("extra") or {}).get("abort_failed_count", 0.0)) for item in results),
        )
        metrics["env/pending_samples"] = max(
            float(metrics.get("env/pending_samples", 0.0)),
            max(float((item.raw_result.get("extra") or {}).get("pending_samples", 0.0)) for item in results),
        )
        metrics["env/pending_groups"] = max(
            float(metrics.get("env/pending_groups", 0.0)),
            max(float((item.raw_result.get("extra") or {}).get("pending_groups", 0.0)) for item in results),
        )
        metrics["env/oldest_pending_age_seconds"] = max(
            float(metrics.get("env/oldest_pending_age_seconds", 0.0)),
            max(float((item.raw_result.get("extra") or {}).get("oldest_pending_age_seconds", 0.0)) for item in results),
        )
        metrics["env/active_decode_requests"] = max(
            float(metrics.get("env/active_decode_requests", 0.0)),
            max(float((item.raw_result.get("extra") or {}).get("active_decode_requests", 0.0)) for item in results),
        )
        metrics["env/domain_unreachable_rate"] = _rate(
            lambda item: self._classify_environment_failure_type(item.raw_result) == "domain_unreachable"
        )
        metrics["env/prefetch_failure_rate"] = _rate(
            lambda item: self._classify_environment_failure_type(item.raw_result) == "prefetch_failed"
        )
        metrics["env/cache_fill_failure_rate"] = _rate(
            lambda item: self._classify_environment_failure_type(item.raw_result) == "cache_fill_failed"
        )
        metrics["env/invalid_tool_format_rate"] = _rate(
            lambda item: self._classify_environment_failure_type(item.raw_result) == "invalid_tool_format"
        )
        attempts = sum(float((item.raw_result.get("extra") or {}).get("format_recovery_attempts", 0)) for item in results)
        successes = sum(float((item.raw_result.get("extra") or {}).get("format_recovery_successes", 0)) for item in results)
        exhausted = sum(float((item.raw_result.get("extra") or {}).get("format_recovery_exhausted", 0)) for item in results)
        metrics["env/format_recovery_attempts"] = attempts
        metrics["env/format_recovery_successes"] = successes
        metrics["env/format_recovery_exhausted"] = exhausted
        metrics["env/format_recovery_success_rate"] = (successes / attempts) if attempts else 0.0
        recoverable_rates = [
            float((item.raw_result.get("extra") or {}).get("format_failure_recoverable_rate", 0.0))
            for item in results
        ]
        terminal_rates = [
            float((item.raw_result.get("extra") or {}).get("format_failure_terminal_rate", 0.0))
            for item in results
        ]
        metrics["env/format_failure_recoverable_rate"] = (
            sum(recoverable_rates) / len(recoverable_rates) if recoverable_rates else 0.0
        )
        metrics["env/format_failure_terminal_rate"] = (
            sum(terminal_rates) / len(terminal_rates) if terminal_rates else 0.0
        )
        progress_scores = [
            float(((item.raw_result.get("extra") or {}).get("progress_summary") or {}).get("progress_score", 0.0))
            for item in results
        ]
        metrics["env/mean_progress_score"] = (
            sum(progress_scores) / len(progress_scores) if progress_scores else 0.0
        )
        num_tasks = [int((item.raw_result.get("extra") or {}).get("num_subtasks", 0)) for item in results]
        task_count_metrics = summarize_task_count_metrics(num_tasks)
        metrics["env/mean_num_tasks"] = task_count_metrics["mean_num_tasks"]
        metrics["env/num_tasks_unknown_rate"] = task_count_metrics["num_tasks_unknown_rate"]
        for task_count in (1, 2, 3, 4):
            metrics[f"env/num_tasks_{task_count}_rate"] = task_count_metrics[f"num_tasks_{task_count}_rate"]
        metrics["env/near_miss_rate"] = _rate(
            lambda item: ((item.raw_result.get("extra") or {}).get("learning_bucket") == "near_miss")
        )
        metrics["env/wrong_domain_loop_rate"] = _rate(
            lambda item: ((item.raw_result.get("extra") or {}).get("rl_failure_bucket") == "wrong_domain_loop")
        )
        metrics["env/premature_stop_rate"] = _rate(
            lambda item: ((item.raw_result.get("extra") or {}).get("rl_failure_bucket") == "premature_stop")
        )
        metrics["env/wrong_path_rate"] = _rate(
            lambda item: ((item.raw_result.get("extra") or {}).get("learning_bucket") == "wrong_path")
        )
        metrics["env/max_steps_reached_rate"] = _rate(
            lambda item: ((item.raw_result.get("extra") or {}).get("failure_reason") == "max_steps_reached")
        )
        metrics["env/unsupported_stop_rate"] = _rate(
            lambda item: bool((item.raw_result.get("extra") or {}).get("unsupported_stop"))
        )
        hallucinated_plugin_counts = [
            float((item.raw_result.get("extra") or {}).get("hallucinated_plugin_count", 0.0))
            for item in results
        ]
        metrics["env/hallucinated_plugin_mean"] = (
            sum(hallucinated_plugin_counts) / len(hallucinated_plugin_counts)
            if hallucinated_plugin_counts
            else 0.0
        )
        google_family_counts = [
            float(((item.raw_result.get("extra") or {}).get("trajectory_diagnostics") or {}).get("google_family_offdomain_count", 0.0))
            for item in results
        ]
        repeated_url_counts = [
            float(((item.raw_result.get("extra") or {}).get("trajectory_diagnostics") or {}).get("repeated_url_count", 0.0))
            for item in results
        ]
        same_page_counts = [
            float(((item.raw_result.get("extra") or {}).get("trajectory_diagnostics") or {}).get("same_page_loop_count", 0.0))
            for item in results
        ]
        metrics["env/google_family_offdomain_mean"] = (
            sum(google_family_counts) / len(google_family_counts) if google_family_counts else 0.0
        )
        metrics["env/repeated_url_mean"] = (
            sum(repeated_url_counts) / len(repeated_url_counts) if repeated_url_counts else 0.0
        )
        metrics["env/same_page_loop_mean"] = (
            sum(same_page_counts) / len(same_page_counts) if same_page_counts else 0.0
        )
        def _coerce_aggregate_number(value: object) -> float:
            if isinstance(value, dict):
                total = 0.0
                for item in value.values():
                    try:
                        total += float(item or 0.0)
                    except (TypeError, ValueError):
                        continue
                return total
            try:
                return float(value or 0.0)
            except (TypeError, ValueError):
                return 0.0

        browser_step_times = []
        browser_nav_times = []
        challenge_hits = 0
        for item in results:
            raw = item.raw_result
            extra = raw.get("extra") or {}
            steps_used = max(1, int(extra.get("steps_used", 0) or 0))
            browser_step_times.append(float(item.time_taken) / steps_used)
            cache_stats = extra.get("cache_stats") or {}
            miss_count = _coerce_aggregate_number(cache_stats.get("per_domain_miss_count", 0.0))
            miss_latency = _coerce_aggregate_number(cache_stats.get("per_domain_miss_latency_s", 0.0))
            if miss_count > 0:
                browser_nav_times.append(miss_latency / miss_count)
            env_kind = self._classify_environment_failure_type(raw)
            error_lower = str(raw.get("error") or "").lower()
            if env_kind in {"env_cdn_blocked", "challenge_page"} or any(
                marker in error_lower for marker in ("challenge", "captcha", "just a moment")
            ):
                challenge_hits += 1
        metrics["env/browser_step_time_mean"] = (
            sum(browser_step_times) / len(browser_step_times) if browser_step_times else 0.0
        )
        metrics["env/browser_nav_time_mean"] = (
            sum(browser_nav_times) / len(browser_nav_times) if browser_nav_times else 0.0
        )
        metrics["env/cache_hit_rate"] = float(metrics.get("cache/hit_rate", 0.0))
        metrics["env/challenge_page_rate"] = challenge_hits / len(results)
        metrics["scheduler/pending_samples"] = float(metrics.get("env/pending_samples", 0.0))
        metrics["scheduler/pending_groups"] = float(metrics.get("env/pending_groups", 0.0))
        metrics["scheduler/oldest_pending_age_seconds"] = float(metrics.get("env/oldest_pending_age_seconds", 0.0))
        metrics["scheduler/active_decode_requests"] = float(metrics.get("env/active_decode_requests", 0.0))
        metrics["env/learning_bucket/environment_failure_rate"] = _rate(
            lambda item: ((item.raw_result.get("extra") or {}).get("learning_bucket") == "environment_failure")
        )
        metrics["env/learning_bucket/format_failure_rate"] = _rate(
            lambda item: ((item.raw_result.get("extra") or {}).get("learning_bucket") == "format_failure")
        )
        audits = [(item.raw_result.get("extra") or {}).get("reachability_audit") or {} for item in results]
        nonempty_audits = [audit for audit in audits if audit]
        metrics["env/reachability_audit_count"] = float(len(nonempty_audits))
        if nonempty_audits:
            metrics["env/reachability_env_failure_rate"] = sum(
                1.0 if audit.get("is_environment_failure") else 0.0 for audit in nonempty_audits
            ) / len(nonempty_audits)
            metrics["env/reachability_model_hallucination_rate"] = sum(
                1.0 if audit.get("is_model_hallucination") else 0.0 for audit in nonempty_audits
            ) / len(nonempty_audits)
            for classification in (
                "env_nav_aborted",
                "env_target_closed",
                "env_nav_timeout",
                "env_tls_error",
                "env_cdn_blocked",
                "env_api_rate_limited",
                "env_api_empty",
            ):
                metric_name = classification.replace("env_", "env/reachability_") + "_rate"
                metrics[metric_name] = sum(
                    1.0 if audit.get("classification") == classification else 0.0 for audit in nonempty_audits
                ) / len(nonempty_audits)
        return metrics

    @staticmethod
    def _classify_environment_failure_type(result: dict[str, Any]) -> str | None:
        extra = result.get("extra") or {}
        failure_reason = extra.get("failure_reason")
        error = result.get("error") or ""
        cache_stats = extra.get("cache_stats") or {}
        reachability_audit = extra.get("reachability_audit") or {}
        audit_classification = reachability_audit.get("classification")

        if audit_classification:
            return audit_classification

        if failure_reason == "site_unreachable":
            lowered = error.lower()
            if any(marker in lowered for marker in ("challenge page", "captcha", "just a moment")):
                return "challenge_page"
            if "prefetch" in error.lower():
                return "prefetch_failed"
            if re.search(r"https?://", error):
                parsed = urlparse(re.search(r"https?://[^\s)]+", error).group(0))
                if parsed.netloc:
                    return "domain_unreachable"
            return "domain_unreachable"

        if failure_reason == "cache_error":
            if cache_stats.get("prefetch_timeouts", 0):
                return "prefetch_failed"
            return "cache_fill_failed"

        if failure_reason in {"control_plane_auth_failure", "sample_wall_timeout", "group_wall_timeout"}:
            return str(failure_reason)

        if failure_reason == "llm_error":
            return "model_request_failure"

        if failure_reason == "parse_failed":
            return "invalid_tool_format"

        if failure_reason in {"llm_context_overflow", "format_recovery_overflow"}:
            return failure_reason

        if failure_reason == "rollout_exception":
            browser_closed = bool(extra.get("browser_transport_closed"))
            return "browser_transport_closed" if browser_closed else "runtime_exception"

        return None

    def snapshot_runtime_metrics(self, scope: str | None = None) -> dict[str, int]:
        state = self._scoped_states.get(scope) if scope is not None else None
        if state is None and self._scoped_states:
            state = next(iter(self._scoped_states.values()))
        if state is None:
            return {}
        return state.snapshot_browser_metrics()

    def export_state(self) -> dict[str, Any]:
        return {
            "phase": canonicalize_phase_name(self._phase),
            "parent_seed_cursor": self._parent_seed_cursor,
            "phase_feedback": list(self._phase_feedback),
            "sampler_state": self._sampler.export_state(),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self._phase = canonicalize_phase_name(state.get("phase", self._phase))
        self._parent_seed_cursor = int(state.get("parent_seed_cursor", self._parent_seed_cursor))
        self._phase_feedback.clear()
        for item in state.get("phase_feedback") or []:
            if isinstance(item, dict):
                self._phase_feedback.append({str(k): float(v) for k, v in item.items() if isinstance(v, (int, float))})
        self._sampler.load_state(state.get("sampler_state") or {})

    def record_group_feedback(self, feedback: list[dict[str, Any]], *, evaluation: bool) -> None:
        if evaluation or not feedback:
            return
        self._sampler.record_group_feedback(feedback)
        for item in feedback:
            task_records = item.get("task_records") or []
            total = len(task_records) or 1
            wrong_domain_loop_rate = sum(
                1.0 for task in task_records if task.get("rl_failure_bucket") == "wrong_domain_loop"
            ) / total
            unsupported_stop_rate = sum(1.0 for task in task_records if task.get("unsupported_stop")) / total
            max_steps_reached_rate = sum(
                1.0 for task in task_records if task.get("failure_reason") == "max_steps_reached"
            ) / total
            self._phase_feedback.append(
                {
                    "mean_score": float(item.get("mean_score", 0.0)),
                    "near_miss_rate": float(item.get("near_miss_rate", 0.0)),
                    "wrong_domain_loop_rate": wrong_domain_loop_rate,
                    "unsupported_stop_rate": unsupported_stop_rate,
                    "max_steps_reached_rate": max_steps_reached_rate,
                }
            )
        self._maybe_promote_phase()

    async def shutdown(self) -> None:
        for state in self._scoped_states.values():
            await state.actor.shutdown()
        self._scoped_states.clear()
