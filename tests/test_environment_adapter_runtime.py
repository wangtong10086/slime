from types import SimpleNamespace

from slime.env_adapters.base import (
    EnvironmentAdapter,
    EvaluationPlan,
    FailureKind,
    JobSpec,
    ResourceProfile,
    RolloutResult,
    RuntimeCapabilities,
    TaskSpec,
)
from slime.rollout.env_adapter.rollout import AdapterRolloutState, _run_single_job
from slime.utils.types import Sample


class FakeRuntimeAdapter(EnvironmentAdapter):
    name = "fake-runtime"

    def __init__(self, args=None):
        self.args = args
        self.ensure_calls = []
        self.prepare_calls = []
        self.cleanup_calls = []
        self.recover_calls = []
        self.run_calls = 0
        self.runtime_counter = 0

    def get_runtime_capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities()

    def get_train_phase(self, args) -> str:
        return "train"

    def get_evaluation_plan(self, args, rollout_id: int) -> EvaluationPlan:
        return EvaluationPlan(dataset_name="quick_eval", num_tasks=1, phase="warmup")

    def should_reset_scope(self, args, *, evaluation: bool, scope: str, rollout_id: int | None = None) -> bool:
        return evaluation

    def sample_tasks(self, *, split: str, count: int, phase: str, rollout_id: int | None = None) -> list[TaskSpec]:
        return []

    def expand_jobs(self, *, tasks: list[TaskSpec], n_samples_per_task: int, mode: str) -> list[JobSpec]:
        return []

    async def ensure_runtime(self, args, *, scope: str):
        self.ensure_calls.append(scope)
        runtime = getattr(self, f"runtime_{scope}", None)
        if runtime is None:
            runtime = {"scope": scope, "instance": self.runtime_counter}
            self.runtime_counter += 1
            setattr(self, f"runtime_{scope}", runtime)
        return runtime

    async def prepare_phase(self, args, *, evaluation: bool, scope: str, reset_scope: bool) -> None:
        self.prepare_calls.append((scope, evaluation, reset_scope))

    async def recover_runtime(self, args, *, scope: str, error=None) -> bool:
        self.recover_calls.append(scope)
        setattr(self, f"runtime_{scope}", {"scope": scope, "instance": self.runtime_counter})
        self.runtime_counter += 1
        return True

    async def cleanup_scope(self, scope: str) -> None:
        self.cleanup_calls.append(scope)
        setattr(self, f"runtime_{scope}", None)

    async def run_job(self, args, runtime, job: JobSpec, *, evaluation: bool) -> RolloutResult:
        self.run_calls += 1
        if self.run_calls == 1:
            return RolloutResult(
                env_name=self.name,
                task_name="fake",
                reward=0.0,
                success=False,
                time_taken=0.1,
                failure_kind=FailureKind.RECOVERABLE_RUNTIME_FAILURE,
                recoverable=True,
                runtime_scope=runtime["scope"],
                raw_result={"attempt": 1},
            )
        return RolloutResult(
            env_name=self.name,
            task_name="fake",
            reward=0.5,
            success=True,
            time_taken=0.1,
            runtime_scope=runtime["scope"],
            raw_result={"attempt": self.run_calls},
        )

    def compute_reward(self, result: RolloutResult):
        return result.reward, {}

    def build_training_sample(self, *, sample: Sample, result: RolloutResult, tokenizer, reward: float, reward_meta: dict):
        sample.reward = reward
        sample.response = "ok"
        sample.response_length = 1
        sample.loss_mask = [1]
        sample.status = Sample.Status.COMPLETED
        return sample


def _args():
    return SimpleNamespace(
        environment_adapter_path=f"{__name__}.FakeRuntimeAdapter",
        environment_name="fake-runtime",
        hf_checkpoint="/tmp/fake",
    )


def test_adapter_rollout_state_resets_scopes(monkeypatch):
    import slime.rollout.env_adapter.rollout as rollout_module

    monkeypatch.setattr(rollout_module, "load_tokenizer", lambda *args, **kwargs: object())
    AdapterRolloutState.clear_instances()
    state = AdapterRolloutState(_args())

    train_scope, _ = state.prepare(evaluation=False, rollout_id=0)
    eval_scope, _ = state.prepare(evaluation=True, rollout_id=0)

    adapter = state.adapter
    assert train_scope == "train_rollout"
    assert eval_scope == "eval"
    assert "train_rollout" not in adapter.cleanup_calls
    assert "eval" in adapter.cleanup_calls
    train_metrics = state.snapshot_runtime_metrics(scope=train_scope)
    eval_metrics = state.snapshot_runtime_metrics(scope=eval_scope)
    assert train_metrics["worker_reuse_count"] == 0
    assert eval_metrics["worker_prepare_reset_count"] == 1


def test_recoverable_runtime_failure_retries_once(monkeypatch):
    import slime.rollout.env_adapter.rollout as rollout_module

    monkeypatch.setattr(rollout_module, "load_tokenizer", lambda *args, **kwargs: object())
    AdapterRolloutState.clear_instances()
    state = AdapterRolloutState(_args())
    scope, worker = state.prepare(evaluation=False, rollout_id=0)

    task = TaskSpec(env_name="fake-runtime", task_family="default", task_id=1, seed=1)
    job = JobSpec(
        env_name="fake-runtime",
        job_id="train:1:sample:0",
        group_id="train:1",
        index_in_group=0,
        mode="train",
        task=task,
        seed=1,
        resource_profile=ResourceProfile(),
    )
    sample = Sample(
        group_index=0,
        index=0,
        prompt="fake",
        metadata={"job_spec": job.to_dict()},
    )

    built, result = _run_single_job(state, worker, scope, sample, evaluation=False)

    assert built is not None
    assert result.success is True
    assert state.adapter.recover_calls == ["train_rollout"]
    metrics = state.snapshot_runtime_metrics(scope=scope)
    assert metrics["runtime_instance_reset_count"] >= 1
