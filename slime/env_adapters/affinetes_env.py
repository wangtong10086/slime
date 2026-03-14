from __future__ import annotations

import os
from typing import Any

from slime.utils.types import Sample

from .affinetes_bridge import AffinetesRuntimeConfig, AffinetesRuntimeHandle
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
from .training_utils import build_sample_from_conversation


class AffinetesEnvironmentAdapter(EnvironmentAdapter):
    name = "affinetes"
    env_var_prefix = "AFFINE_ENV"
    default_task_family = "default"
    default_mode = "evaluate"

    def __init__(self, args=None):
        self.args = args
        self._runtimes: dict[str, AffinetesRuntimeHandle] = {}
        self._task_seed_cursor = int(os.getenv(f"{self.env_var_prefix}_TRAIN_BASE_SEED", "100000"))

    def get_runtime_capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            supports_evaluate=True,
            supports_openenv=False,
            supports_trajectory_export=True,
            supports_offline_dataset_export=False,
        )

    def get_train_phase(self, args) -> str:
        return os.getenv(f"{self.env_var_prefix}_TRAIN_PHASE", "train")

    def get_evaluation_plan(self, args, rollout_id: int) -> EvaluationPlan:
        formal_every = int(os.getenv(f"{self.env_var_prefix}_FORMAL_EVAL_EVERY", "50"))
        if (rollout_id + 1) % formal_every == 0:
            return EvaluationPlan(
                dataset_name="formal_eval",
                num_tasks=int(os.getenv(f"{self.env_var_prefix}_FORMAL_EVAL_TASKS", "200")),
                phase="main",
            )
        return EvaluationPlan(
            dataset_name="quick_eval",
            num_tasks=int(os.getenv(f"{self.env_var_prefix}_QUICK_EVAL_TASKS", "32")),
            phase="warmup",
        )

    def sample_tasks(self, *, split: str, count: int, phase: str, rollout_id: int | None = None) -> list[TaskSpec]:
        base_seed_env = f"{self.env_var_prefix}_{'EVAL' if split == 'eval' else 'TRAIN'}_BASE_SEED"
        if split == "eval":
            base_seed = int(os.getenv(base_seed_env, "900000"))
            start = base_seed + (0 if rollout_id is None else rollout_id * 10000)
            return [
                TaskSpec(
                    env_name=self.name,
                    task_family=self.default_task_family,
                    task_id=start + idx,
                    seed=start + idx,
                    metadata={"phase": phase, "split": split},
                )
                for idx in range(count)
            ]
        tasks = []
        for _ in range(count):
            seed = self._task_seed_cursor
            self._task_seed_cursor += 1
            tasks.append(
                TaskSpec(
                    env_name=self.name,
                    task_family=self.default_task_family,
                    task_id=seed,
                    seed=seed,
                    metadata={"phase": phase, "split": split},
                )
            )
        return tasks

    def _derive_job_seed(self, task: TaskSpec, sample_index: int) -> int:
        return abs(hash((self.name, task.seed, sample_index))) % (2**31 - 1)

    def expand_jobs(self, *, tasks: list[TaskSpec], n_samples_per_task: int, mode: str) -> list[JobSpec]:
        jobs: list[JobSpec] = []
        for task_idx, task in enumerate(tasks):
            group_id = f"{mode}:{task.task_id}"
            affinity_key = self.affinity_key_from_task(task)
            for sample_index in range(n_samples_per_task):
                seed = self._derive_job_seed(task, sample_index)
                jobs.append(
                    JobSpec(
                        env_name=self.name,
                        job_id=f"{group_id}:sample:{sample_index}",
                        group_id=group_id,
                        index_in_group=sample_index,
                        mode=mode,
                        task=task,
                        seed=seed,
                        metadata={"task_index": task_idx, "phase": task.metadata.get("phase")},
                        affinity_key=affinity_key,
                        resource_profile=self.resource_profile_from_task(task),
                        prompt_hint=f"{self.name}:{task.task_family}:{task.task_id}",
                    )
                )
        return jobs

    async def ensure_runtime(self, args, *, scope: str) -> AffinetesRuntimeHandle:
        runtime = self._runtimes.get(scope)
        if runtime is None:
            config = AffinetesRuntimeConfig.from_env(self.env_var_prefix)
            runtime = AffinetesRuntimeHandle(config, scope=scope)
            self._runtimes[scope] = runtime
        await runtime.ensure_loaded()
        return runtime

    async def prepare_phase(self, args, *, evaluation: bool, scope: str, reset_scope: bool) -> None:
        runtime = await self.ensure_runtime(args, scope=scope)
        if reset_scope:
            await runtime.reset_scope()

    async def recover_runtime(self, args, *, scope: str, error: BaseException | None = None) -> bool:
        runtime = await self.ensure_runtime(args, scope=scope)
        await runtime.reset_scope()
        return True

    async def cleanup_scope(self, scope: str) -> None:
        runtime = self._runtimes.pop(scope, None)
        if runtime is not None:
            await runtime.cleanup_scope()

    def _build_evaluate_kwargs(self, args, job: JobSpec) -> dict[str, Any]:
        kwargs = {
            "task_id": job.task.task_id,
            "seed": job.seed,
            "model": os.getenv("AFFINE_MODEL_NAME", getattr(args, "hf_checkpoint", "")),
        }
        if os.getenv(f"{self.env_var_prefix}_BASE_URL"):
            kwargs["base_url"] = os.getenv(f"{self.env_var_prefix}_BASE_URL")
        task_type = os.getenv(f"{self.env_var_prefix}_TASK_TYPE")
        if task_type:
            kwargs["task_type"] = task_type
        timeout = os.getenv(f"{self.env_var_prefix}_TIMEOUT")
        if timeout:
            kwargs["timeout"] = int(timeout)
        temperature = os.getenv(f"{self.env_var_prefix}_TEMPERATURE")
        if temperature:
            kwargs["temperature"] = float(temperature)
        return kwargs

    async def run_job(self, args, runtime: AffinetesRuntimeHandle, job: JobSpec, *, evaluation: bool) -> RolloutResult:
        result = await runtime.evaluate(**self._build_evaluate_kwargs(args, job))
        failure_kind = None
        if result.get("error"):
            failure_kind = FailureKind.MODEL_REQUEST_FAILURE
        elif not result.get("success", False):
            failure_kind = FailureKind.TASK_FAILURE
        return RolloutResult(
            env_name=self.name,
            task_name=str(result.get("task_name", f"{self.name}:{job.task.task_family}")),
            reward=float(result.get("score", 0.0)),
            success=bool(result.get("success", False)),
            time_taken=float(result.get("time_taken", 0.0)),
            metrics={"usage": ((result.get("extra") or {}).get("usage") or {})},
            failure_kind=failure_kind,
            runtime_scope=runtime.scope,
            drop_from_training=False,
            raw_result=result,
            error=result.get("error"),
        )

    def compute_reward(self, result: RolloutResult) -> tuple[float | None, dict[str, Any]]:
        raw = result.raw_result
        if raw.get("error"):
            return None, {"drop_reason": raw.get("error_type", "error"), "environment_failure_type": raw.get("error_type")}
        reward = float(raw.get("score", 0.0))
        return reward, {
            "drop_reason": None,
            "environment_failure_type": None,
            "raw_reward": reward,
        }

    def _fallback_payload(self, result: RolloutResult) -> str:
        return result.raw_result.get("response") or result.error or "NO_ANSWER"

    def _conversation_from_result(self, result: RolloutResult, sample: Sample) -> list[dict[str, Any]]:
        raw = result.raw_result
        extra = raw.get("extra") or {}
        conversation = extra.get("conversation")
        if isinstance(conversation, list) and conversation:
            return conversation
        prompt_text = extra.get("prompt") or sample.prompt or (sample.metadata or {}).get("prompt_hint") or result.task_name
        answer_text = raw.get("response") or extra.get("response") or self._fallback_payload(result)
        return [
            {"role": "user", "content": str(prompt_text)},
            {"role": "assistant", "content": str(answer_text)},
        ]

    def build_training_sample(
        self,
        *,
        sample: Sample,
        result: RolloutResult,
        tokenizer,
        reward: float,
        reward_meta: dict[str, Any],
    ) -> Sample | None:
        conversation = self._conversation_from_result(result, sample)
        usage = ((result.raw_result.get("extra") or {}).get("usage") or {})
        metadata = {
            **reward_meta,
            "task_name": result.task_name,
            "score": result.raw_result.get("score", result.reward),
            "success": result.success,
            "time_taken": result.time_taken,
            "failure_kind": result.failure_kind.value if result.failure_kind else None,
            "failure_reason": result.raw_result.get("error_type"),
            "usage": usage,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "conversation_length": len(conversation),
        }
        return build_sample_from_conversation(
            sample=sample,
            tokenizer=tokenizer,
            conversation=conversation,
            reward=reward,
            metadata=metadata,
            fallback_payload=self._fallback_payload(result),
        )

    def affinity_key_from_task(self, task: TaskSpec) -> str | None:
        return f"{self.name}:{task.task_family}"

    def resource_profile_from_task(self, task: TaskSpec) -> ResourceProfile:
        return ResourceProfile({"llm": 1, "env": 1})

    def snapshot_runtime_metrics(self, scope: str | None = None) -> dict[str, int]:
        return {}

    def export_state(self) -> dict[str, Any]:
        return {"task_seed_cursor": self._task_seed_cursor}

    def load_state(self, state: dict[str, Any]) -> None:
        self._task_seed_cursor = int(state.get("task_seed_cursor", self._task_seed_cursor))

    async def shutdown(self) -> None:
        for runtime in self._runtimes.values():
            await runtime.cleanup()
        self._runtimes.clear()
