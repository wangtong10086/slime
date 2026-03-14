from slime.env_adapters.base import FailureKind, RolloutResult
from slime.rollout.env_adapter.rollout import _finalize_rollout_metrics, _merge_rollout_metrics


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
