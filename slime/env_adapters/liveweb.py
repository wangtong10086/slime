from __future__ import annotations

import os
import re
from typing import Any
from urllib.parse import urlparse

from slime.rollout.liveweb_online.common import (
    LiveWebRolloutState,
    PromptJob,
    build_eval_jobs,
    build_prompt_job,
    compute_reward_from_result,
    evaluate_prompt_job,
    get_eval_profile,
    make_sample_from_result,
    should_allow_environment_fallback,
    summarize_cache_stats,
)
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
        self._phase = os.getenv("LIVEWEB_TASK_MIX_PHASE", "warmup")
        self._parent_seed_cursor = int(os.getenv("LIVEWEB_TRAIN_BASE_SEED", "100000"))
        self._curated_warmup_pool = []
        self._curated_warmup_cursor = 0
        use_curated_warmup_pool = os.getenv("LIVEWEB_USE_CURATED_WARMUP_POOL", "1") == "1"
        if self._phase == "warmup" and use_curated_warmup_pool:
            num_prompts = int(os.getenv("LIVEWEB_WARMUP_POOL_SIZE", os.getenv("LIVEWEB_QUICK_EVAL_PROMPTS", "32")))
            base_seed = int(os.getenv("LIVEWEB_WARMUP_POOL_BASE_SEED", "900000"))
            self._curated_warmup_pool = build_eval_jobs(
                rollout_id=0,
                dataset_name="warmup_pool",
                num_prompts=num_prompts,
                phase="warmup",
                base_seed=base_seed,
            )

    def get_runtime_capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            supports_evaluate=True,
            supports_openenv=True,
            supports_trajectory_export=True,
            supports_offline_dataset_export=True,
        )

    def get_train_phase(self, args) -> str:
        return os.getenv("LIVEWEB_TASK_MIX_PHASE", self._phase)

    def get_evaluation_plan(self, args, rollout_id: int) -> EvaluationPlan:
        dataset_name, num_prompts, phase = get_eval_profile(rollout_id)
        return EvaluationPlan(dataset_name=dataset_name, num_tasks=num_prompts, phase=phase)

    def should_reset_scope(self, args, *, evaluation: bool, scope: str, rollout_id: int | None = None) -> bool:
        if evaluation:
            return True
        return os.getenv("LIVEWEB_FORCE_RESET_TRAIN_SCOPE", "0") == "1"

    def sample_tasks(self, *, split: str, count: int, phase: str, rollout_id: int | None = None) -> list[TaskSpec]:
        tasks: list[TaskSpec] = []
        if split == "eval":
            base_seed = int(os.getenv("LIVEWEB_EVAL_BASE_SEED", "900000"))
            if phase == "main":
                base_seed += 10000
            start_seed = base_seed + (0 if rollout_id is None else 0)
            for idx in range(count):
                parent_seed = start_seed + idx
                tasks.append(
                    TaskSpec(
                        env_name=self.name,
                        task_family="prompt",
                        task_id=parent_seed,
                        seed=parent_seed,
                        metadata={"phase": phase, "split": split},
                    )
                )
            return tasks

        for _ in range(count):
            if self._curated_warmup_pool:
                base_job = self._curated_warmup_pool[self._curated_warmup_cursor % len(self._curated_warmup_pool)]
                self._curated_warmup_cursor += 1
                parent_seed = base_job.parent_seed
                plugin_name = base_job.plugin_name
            else:
                parent_seed = self._parent_seed_cursor
                self._parent_seed_cursor += 1
                plugin_name = None
            tasks.append(
                TaskSpec(
                    env_name=self.name,
                    task_family="prompt",
                    task_id=parent_seed,
                    seed=parent_seed,
                    metadata={"phase": phase, "split": split, "plugin_name": plugin_name},
                )
            )
        return tasks

    def expand_jobs(self, *, tasks: list[TaskSpec], n_samples_per_task: int, mode: str) -> list[JobSpec]:
        jobs: list[JobSpec] = []
        for task_idx, task in enumerate(tasks):
            parent_seed = int(task.seed)
            if mode == "eval":
                plugin_name = str(task.metadata.get("plugin_name") or "")
                eval_jobs = build_eval_jobs(
                    rollout_id=0,
                    dataset_name=mode,
                    num_prompts=1,
                    phase=str(task.metadata.get("phase", "warmup")),
                    base_seed=parent_seed,
                )
                prompt_job = eval_jobs[0]
                affinity_key = prompt_job.route_key
                jobs.append(
                    JobSpec(
                        env_name=self.name,
                        job_id=f"{mode}:{parent_seed}",
                        group_id=f"{mode}:{parent_seed}",
                        index_in_group=0,
                        mode=mode,
                        task=task,
                        seed=prompt_job.llm_seed,
                        metadata={"prompt_job": prompt_job.to_metadata(), "plugin_name": plugin_name},
                        affinity_key=affinity_key,
                        resource_profile=ResourceProfile({"llm": 1, "env": 1}),
                        prompt_hint=f"{prompt_job.task_name}:{prompt_job.parent_seed}",
                    )
                )
                continue

            for sample_index in range(n_samples_per_task):
                if self._curated_warmup_pool and task.metadata.get("plugin_name"):
                    base_job = next((job for job in self._curated_warmup_pool if job.parent_seed == parent_seed), None)
                    if base_job is None:
                        prompt_job = build_prompt_job(
                            parent_seed=parent_seed,
                            group_index=task_idx,
                            sample_index=sample_index,
                            phase=str(task.metadata.get("phase", "warmup")),
                        )
                    else:
                        llm_seed = abs(hash((parent_seed + task_idx * 1000, sample_index, "liveweb_online_rl"))) % (2**31 - 1)
                        prompt_job = PromptJob(
                            parent_seed=base_job.parent_seed,
                            task_seed=base_job.task_seed,
                            llm_seed=llm_seed,
                            subtask_index=base_job.subtask_index,
                            num_subtasks=base_job.num_subtasks,
                            templates=list(base_job.templates),
                            task_name=base_job.task_name,
                            plugin_name=base_job.plugin_name,
                            phase=base_job.phase,
                            route_key=f"{task.metadata.get('phase', 'warmup')}:curated:seed:{parent_seed}:subtask:1",
                        )
                else:
                    prompt_job = build_prompt_job(
                        parent_seed=parent_seed,
                        group_index=task_idx,
                        sample_index=sample_index,
                        phase=str(task.metadata.get("phase", "warmup")),
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
                        metadata={"prompt_job": prompt_job.to_metadata(), "plugin_name": prompt_job.plugin_name},
                        affinity_key=prompt_job.route_key,
                        resource_profile=ResourceProfile({"llm": 1, "env": 1}),
                        prompt_hint=f"{prompt_job.task_name}:{prompt_job.parent_seed}",
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
        prompt_job = PromptJob(**job.metadata["prompt_job"])
        result = await evaluate_prompt_job(args, runtime, prompt_job)
        runtime.record_job_outcome(result)
        failure_reason = (result.get("extra") or {}).get("failure_reason")
        environment_failure_type = self._classify_environment_failure_type(result)
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
            environment_pollution=failure_reason in {"site_unreachable", "cache_error", "llm_error", "rollout_exception"},
            drop_from_training=failure_reason in {"site_unreachable", "cache_error", "llm_error", "rollout_exception"},
            raw_result=result,
            error=result.get("error"),
        )

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
        return metrics

    @staticmethod
    def _classify_environment_failure_type(result: dict[str, Any]) -> str | None:
        extra = result.get("extra") or {}
        failure_reason = extra.get("failure_reason")
        error = result.get("error") or ""
        cache_stats = extra.get("cache_stats") or {}

        if failure_reason == "site_unreachable":
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

        if failure_reason == "llm_error":
            return "model_request_failure"

        if failure_reason == "parse_failed":
            return "invalid_tool_format"

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
            "phase": self._phase,
            "parent_seed_cursor": self._parent_seed_cursor,
            "curated_warmup_cursor": self._curated_warmup_cursor,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self._phase = state.get("phase", self._phase)
        self._parent_seed_cursor = int(state.get("parent_seed_cursor", self._parent_seed_cursor))
        self._curated_warmup_cursor = int(state.get("curated_warmup_cursor", self._curated_warmup_cursor))

    async def shutdown(self) -> None:
        for state in self._scoped_states.values():
            await state.actor.shutdown()
        self._scoped_states.clear()
