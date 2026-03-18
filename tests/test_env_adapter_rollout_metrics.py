from slime.env_adapters.base import FailureKind, RolloutResult
from slime.rollout.env_adapter.rollout import _finalize_rollout_metrics, _merge_rollout_metrics, generate_rollout
from slime.rollout.base_types import RolloutFnTrainOutput
from slime.ray.rollout_batching import (
    choose_dynamic_global_batch_size,
    compute_train_trim_length,
    resolve_max_samples_per_rollout,
)
from types import SimpleNamespace
import pytest


class _DummyAdapter:
    def summarize_results(self, results, prefix):
        total = len(results)
        pollution = sum(1 for item in results if item.environment_pollution) / total
        return {
            f"{prefix}/pollution_rate": pollution,
            f"{prefix}/mean_score": sum(item.reward for item in results) / total,
        }


def test_rollout_metrics_do_not_sum_config_values_or_rates():
    aggregate = {}
    _merge_rollout_metrics(
        aggregate,
        {
            "env/accepted_groups": 2.0,
            "env/accepted_samples": 8.0,
            "env/runtime_reset_count": 1.0,
            "scheduler/runtime_active_job_cap": 32.0,
            "scheduler/runtime_queued_jobs": 16.0,
            "scheduler/runtime_completed_jobs": 32.0,
            "scheduler/config_max_parallel_env_jobs": 32.0,
            "scheduler/config_max_parallel_llm_jobs": 16.0,
        },
    )
    _merge_rollout_metrics(
        aggregate,
        {
            "env/accepted_groups": 1.0,
            "env/accepted_samples": 4.0,
            "env/runtime_reset_count": 1.0,
            "scheduler/runtime_active_job_cap": 24.0,
            "scheduler/runtime_queued_jobs": 24.0,
            "scheduler/runtime_completed_jobs": 16.0,
            "scheduler/config_max_parallel_env_jobs": 32.0,
            "scheduler/config_max_parallel_llm_jobs": 16.0,
        },
    )

    accepted_groups = [[object()] for _ in range(3)]
    results = [
        RolloutResult(env_name="liveweb", task_name="a", reward=1.0, success=True, time_taken=1.0),
        RolloutResult(
            env_name="liveweb",
            task_name="b",
            reward=0.0,
            success=False,
            time_taken=1.0,
            failure_kind=FailureKind.ENV_RUNTIME_FAILURE,
            environment_pollution=True,
        ),
        RolloutResult(env_name="liveweb", task_name="c", reward=0.5, success=True, time_taken=1.0),
        RolloutResult(
            env_name="liveweb",
            task_name="d",
            reward=0.0,
            success=False,
            time_taken=1.0,
            failure_kind=FailureKind.MODEL_REQUEST_FAILURE,
            environment_pollution=True,
        ),
    ]

    metrics = _finalize_rollout_metrics(
        args=None,
        adapter=_DummyAdapter(),
        accepted_groups=accepted_groups,
        all_results=results,
        aggregate_metrics=aggregate,
        total_requested_groups=5,
        total_requested_jobs=20,
        base_group_target=5,
    )

    assert metrics["scheduler/config_max_parallel_env_jobs"] == 32.0
    assert metrics["scheduler/config_max_parallel_llm_jobs"] == 16.0
    assert metrics["scheduler/runtime_active_job_cap"] == 32.0
    assert metrics["scheduler/runtime_queued_jobs"] == 24.0
    assert metrics["scheduler/runtime_completed_jobs"] == 48.0
    assert metrics["env/pollution_rate"] == 0.5
    assert metrics["env/runtime_reset_rate"] == 0.5


def test_train_trim_length_keeps_multiple_steps(monkeypatch):
    monkeypatch.setenv("TRAIN_MAX_SAMPLES_PER_ROLLOUT", "128")

    max_samples = resolve_max_samples_per_rollout(32)
    assert max_samples == 128
    assert compute_train_trim_length(56, 32, max_samples) == 32
    assert compute_train_trim_length(96, 32, max_samples) == 96
    assert compute_train_trim_length(133, 32, max_samples) == 128

    monkeypatch.delenv("TRAIN_MAX_SAMPLES_PER_ROLLOUT", raising=False)


def test_dynamic_global_batch_size_prefers_remainder_free_steps():
    assert choose_dynamic_global_batch_size(
        num_samples=60,
        dp_size=1,
        original_gbs=32,
        configured_cap=32,
        configured_min=16,
    ) == 30
    assert choose_dynamic_global_batch_size(
        num_samples=56,
        dp_size=1,
        original_gbs=32,
        configured_cap=32,
        configured_min=16,
    ) == 28


def test_env_adapter_generate_rollout_raises_on_empty_accepted_groups(monkeypatch):
    class _DummyAdapterImpl:
        name = "dummy"

        def sample_tasks(self, split, count, phase):
            return [{"task_id": i} for i in range(count)]

        def expand_jobs(self, tasks, n_samples_per_task, mode):
            jobs = []
            for idx, _task in enumerate(tasks):
                jobs.append(
                    SimpleNamespace(
                        group_id=f"group-{idx}",
                        prompt_hint=f"prompt-{idx}",
                        job_id=f"job-{idx}",
                        affinity_key=f"aff-{idx}",
                        to_dict=lambda idx=idx: {"metadata": {"combo_key": f"combo-{idx}"}, "task": {"metadata": {}}},
                    )
                )
            return jobs

    class _DummyState:
        def __init__(self, args):
            self.adapter = _DummyAdapterImpl()

    def _fake_run_groups(args, groups, evaluation, rollout_id=None):
        assert evaluation is False
        return [], [], {"env/accepted_groups": 0.0}

    monkeypatch.setattr("slime.rollout.env_adapter.rollout.AdapterRolloutState", _DummyState)
    monkeypatch.setattr("slime.rollout.env_adapter.rollout._run_groups", _fake_run_groups)

    args = SimpleNamespace(
        rollout_batch_size=2,
        n_samples_per_prompt=2,
        global_batch_size=4,
        over_sampling_batch_size=2,
    )
    data_source = SimpleNamespace(get_samples=lambda n: [[object()] for _ in range(n)])

    with pytest.raises(RuntimeError, match="produced no trainable groups"):
        generate_rollout(args, rollout_id=0, data_source=data_source, evaluation=False)
