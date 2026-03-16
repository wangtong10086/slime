from __future__ import annotations

import abc
import dataclasses
from enum import Enum
from typing import Any

from slime.utils.types import Sample

from .runtime_worker import EnvironmentRuntimeWorker, RuntimeJobOutcome, RuntimeJobRequest


class FailureKind(str, Enum):
    ENV_RUNTIME_FAILURE = "env_runtime_failure"
    INVALID_MODEL_OUTPUT = "invalid_model_output"
    MODEL_REQUEST_FAILURE = "model_request_failure"
    TASK_FAILURE = "task_failure"
    REWARD_DROP = "reward_drop"
    RECOVERABLE_RUNTIME_FAILURE = "recoverable_runtime_failure"


@dataclasses.dataclass
class ResourceProfile:
    resources: dict[str, int] = dataclasses.field(default_factory=lambda: {"llm": 1, "env": 1})

    def to_dict(self) -> dict[str, Any]:
        return {"resources": dict(self.resources)}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ResourceProfile":
        if not data:
            return cls()
        return cls(resources=dict(data.get("resources") or {}))


@dataclasses.dataclass
class TaskSpec:
    env_name: str
    task_family: str
    task_id: str | int
    seed: int
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "env_name": self.env_name,
            "task_family": self.task_family,
            "task_id": self.task_id,
            "seed": self.seed,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskSpec":
        return cls(
            env_name=str(data["env_name"]),
            task_family=str(data["task_family"]),
            task_id=data["task_id"],
            seed=int(data["seed"]),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclasses.dataclass
class JobSpec:
    env_name: str
    job_id: str
    group_id: str
    index_in_group: int
    mode: str
    task: TaskSpec
    seed: int
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    affinity_key: str | None = None
    resource_profile: ResourceProfile = dataclasses.field(default_factory=ResourceProfile)
    prompt_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "env_name": self.env_name,
            "job_id": self.job_id,
            "group_id": self.group_id,
            "index_in_group": self.index_in_group,
            "mode": self.mode,
            "task": self.task.to_dict(),
            "seed": self.seed,
            "metadata": dict(self.metadata),
            "affinity_key": self.affinity_key,
            "resource_profile": self.resource_profile.to_dict(),
            "prompt_hint": self.prompt_hint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobSpec":
        return cls(
            env_name=str(data["env_name"]),
            job_id=str(data["job_id"]),
            group_id=str(data["group_id"]),
            index_in_group=int(data["index_in_group"]),
            mode=str(data["mode"]),
            task=TaskSpec.from_dict(data["task"]),
            seed=int(data["seed"]),
            metadata=dict(data.get("metadata") or {}),
            affinity_key=data.get("affinity_key"),
            resource_profile=ResourceProfile.from_dict(data.get("resource_profile")),
            prompt_hint=data.get("prompt_hint"),
        )


@dataclasses.dataclass
class EvaluationPlan:
    dataset_name: str
    num_tasks: int
    phase: str


@dataclasses.dataclass
class RuntimeCapabilities:
    supports_evaluate: bool = True
    supports_openenv: bool = False
    supports_trajectory_export: bool = False
    supports_offline_dataset_export: bool = False


@dataclasses.dataclass
class TrajectoryRecord:
    task_spec: TaskSpec
    messages_or_steps: list[dict[str, Any]]
    final_score: float
    success: bool
    failure_kind: FailureKind | None = None
    actions: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    observations: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    usage_metrics: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class RolloutResult:
    env_name: str
    task_name: str
    reward: float
    success: bool
    time_taken: float
    trajectory: TrajectoryRecord | None = None
    metrics: dict[str, Any] = dataclasses.field(default_factory=dict)
    failure_kind: FailureKind | None = None
    failure_stage: str | None = None
    recoverable: bool = False
    runtime_scope: str | None = None
    environment_pollution: bool = False
    drop_from_training: bool = False
    raw_result: dict[str, Any] = dataclasses.field(default_factory=dict)
    error: str | None = None


@dataclasses.dataclass
class RuntimeErrorInfo:
    scope: str
    stage: str | None = None
    message: str | None = None
    error_type: str | None = None


class EnvironmentAdapter(abc.ABC):
    name: str

    @abc.abstractmethod
    def get_runtime_capabilities(self) -> RuntimeCapabilities:
        raise NotImplementedError

    @abc.abstractmethod
    def get_train_phase(self, args) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def get_evaluation_plan(self, args, rollout_id: int) -> EvaluationPlan:
        raise NotImplementedError

    def get_runtime_scope(self, *, evaluation: bool, rollout_id: int | None = None) -> str:
        return "eval" if evaluation else "train_rollout"

    def build_runtime_worker(self, args, *, scope: str) -> EnvironmentRuntimeWorker:
        return EnvironmentRuntimeWorker(self, args, scope=scope)

    def should_reset_scope(self, args, *, evaluation: bool, scope: str, rollout_id: int | None = None) -> bool:
        return evaluation or scope == "train_rollout"

    @abc.abstractmethod
    def sample_tasks(
        self,
        *,
        split: str,
        count: int,
        phase: str,
        rollout_id: int | None = None,
    ) -> list[TaskSpec]:
        raise NotImplementedError

    @abc.abstractmethod
    def expand_jobs(
        self,
        *,
        tasks: list[TaskSpec],
        n_samples_per_task: int,
        mode: str,
    ) -> list[JobSpec]:
        raise NotImplementedError

    @abc.abstractmethod
    async def ensure_runtime(self, args, *, scope: str) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    async def prepare_phase(self, args, *, evaluation: bool, scope: str, reset_scope: bool) -> None:
        raise NotImplementedError

    async def recover_runtime(self, args, *, scope: str, error: BaseException | None = None) -> bool:
        return False

    async def cleanup_scope(self, scope: str) -> None:
        return None

    @abc.abstractmethod
    async def run_job(self, args, runtime: Any, job: JobSpec, *, evaluation: bool) -> RolloutResult:
        raise NotImplementedError

    @abc.abstractmethod
    def compute_reward(self, result: RolloutResult) -> tuple[float | None, dict[str, Any]]:
        raise NotImplementedError

    @abc.abstractmethod
    def build_training_sample(
        self,
        *,
        sample: Sample,
        result: RolloutResult,
        tokenizer,
        reward: float,
        reward_meta: dict[str, Any],
    ) -> Sample | None:
        raise NotImplementedError

    def summarize_results(self, results: list[RolloutResult], prefix: str) -> dict[str, float]:
        if not results:
            return {}
        scores = [float(item.reward) for item in results]
        times = [float(item.time_taken) for item in results]
        metrics: dict[str, float] = {
            f"{prefix}/mean_score": sum(scores) / len(scores),
            f"{prefix}/mean_time": sum(times) / len(times),
            f"{prefix}/success_rate": sum(1.0 if item.success else 0.0 for item in results) / len(results),
        }
        failure_kinds = {item.failure_kind for item in results if item.failure_kind is not None}
        for failure_kind in sorted(failure_kinds, key=lambda item: item.value):
            metrics[f"{prefix}/failure/{failure_kind.value}"] = (
                sum(1.0 if item.failure_kind == failure_kind else 0.0 for item in results) / len(results)
            )
        metrics["runtime/recoverable_failure_rate"] = (
            sum(1.0 if item.recoverable else 0.0 for item in results) / len(results)
        )
        metrics["env/pollution_rate"] = (
            sum(1.0 if item.environment_pollution else 0.0 for item in results) / len(results)
        )
        metrics["env/permanent_failure_rate"] = (
            sum(1.0 if item.failure_kind == FailureKind.ENV_RUNTIME_FAILURE else 0.0 for item in results) / len(results)
        )
        metrics["env/invalid_output_rate"] = (
            sum(1.0 if item.failure_kind == FailureKind.INVALID_MODEL_OUTPUT else 0.0 for item in results) / len(results)
        )
        return metrics

    def snapshot_runtime_metrics(self, scope: str | None = None) -> dict[str, int]:
        return {}

    def record_group_feedback(self, feedback: list[dict[str, Any]], *, evaluation: bool) -> None:
        return None

    def export_state(self) -> dict[str, Any]:
        return {}

    def load_state(self, state: dict[str, Any]) -> None:
        return None

    async def shutdown(self) -> None:
        return None
