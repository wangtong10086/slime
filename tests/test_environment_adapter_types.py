from slime.env_adapters.base import EnvironmentAdapter, FailureKind, JobSpec, ResourceProfile, RolloutResult, RuntimeCapabilities, TaskSpec


def test_task_and_job_spec_round_trip():
    task = TaskSpec(
        env_name="liveweb",
        task_family="prompt",
        task_id=123,
        seed=456,
        metadata={"phase": "warmup"},
    )
    job = JobSpec(
        env_name="liveweb",
        job_id="train:123:sample:0",
        group_id="train:123",
        index_in_group=0,
        mode="train",
        task=task,
        seed=789,
        metadata={"foo": "bar"},
        affinity_key="warmup:seed:123",
        resource_profile=ResourceProfile({"llm": 1, "env": 1}),
        prompt_hint="liveweb:123",
    )

    restored = JobSpec.from_dict(job.to_dict())

    assert restored.env_name == job.env_name
    assert restored.job_id == job.job_id
    assert restored.group_id == job.group_id
    assert restored.task.task_id == task.task_id
    assert restored.resource_profile.resources == {"llm": 1, "env": 1}


class _MetricsAdapter(EnvironmentAdapter):
    name = "metrics"

    def get_runtime_capabilities(self):
        return RuntimeCapabilities()

    def get_train_phase(self, args):
        return "train"

    def get_evaluation_plan(self, args, rollout_id):
        raise NotImplementedError

    def sample_tasks(self, *, split, count, phase, rollout_id=None):
        raise NotImplementedError

    def expand_jobs(self, *, tasks, n_samples_per_task, mode):
        raise NotImplementedError

    async def ensure_runtime(self, args, *, scope: str):
        raise NotImplementedError

    async def prepare_phase(self, args, *, evaluation: bool, scope: str, reset_scope: bool):
        raise NotImplementedError

    async def run_job(self, args, runtime, job, *, evaluation: bool):
        raise NotImplementedError

    def compute_reward(self, result):
        raise NotImplementedError

    def build_training_sample(self, *, sample, result, tokenizer, reward, reward_meta):
        raise NotImplementedError


def test_summarize_results_includes_runtime_and_pollution_rates():
    adapter = _MetricsAdapter()
    metrics = adapter.summarize_results(
        [
            RolloutResult(
                env_name="metrics",
                task_name="a",
                reward=1.0,
                success=True,
                time_taken=1.0,
                failure_kind=FailureKind.RECOVERABLE_RUNTIME_FAILURE,
                recoverable=True,
                environment_pollution=True,
            ),
            RolloutResult(
                env_name="metrics",
                task_name="b",
                reward=0.0,
                success=False,
                time_taken=2.0,
                failure_kind=FailureKind.INVALID_MODEL_OUTPUT,
                recoverable=False,
                environment_pollution=False,
            ),
        ],
        "env",
    )

    assert metrics["runtime/recoverable_failure_rate"] == 0.5
    assert metrics["env/pollution_rate"] == 0.5
    assert metrics["env/invalid_output_rate"] == 0.5
