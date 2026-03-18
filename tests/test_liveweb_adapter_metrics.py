from slime.env_adapters.base import RolloutResult
from slime.env_adapters.liveweb import LiveWebEnvironmentAdapter
import pytest


def test_liveweb_classifies_domain_unreachable():
    adapter = LiveWebEnvironmentAdapter()
    result = {
        "error": "Required site unreachable: https://news.ycombinator.com/show",
        "extra": {"failure_reason": "site_unreachable", "cache_stats": {}},
    }
    assert adapter._classify_environment_failure_type(result) == "domain_unreachable"


def test_liveweb_summarize_results_breaks_out_failure_types():
    adapter = LiveWebEnvironmentAdapter()
    results = [
        RolloutResult(
            env_name="liveweb",
            task_name="t1",
            reward=0.0,
            success=False,
            time_taken=1.0,
            raw_result={"error": "Required site unreachable: https://news.ycombinator.com/", "extra": {"failure_reason": "site_unreachable", "cache_stats": {}}},
        ),
        RolloutResult(
            env_name="liveweb",
            task_name="t2",
            reward=0.0,
            success=False,
            time_taken=1.0,
            raw_result={"error": "prefetch timeout", "extra": {"failure_reason": "cache_error", "cache_stats": {"prefetch_timeouts": 1}}},
        ),
        RolloutResult(
            env_name="liveweb",
            task_name="t3",
            reward=0.0,
            success=False,
            time_taken=1.0,
            raw_result={
                "error": "",
                "extra": {
                    "failure_reason": "parse_failed",
                    "cache_stats": {},
                    "learning_bucket": "format_failure",
                    "progress_summary": {"progress_score": 0.0},
                    "format_recovery_attempts": 2,
                    "format_recovery_successes": 1,
                    "format_recovery_exhausted": 1,
                    "format_failure_recoverable_rate": 0.5,
                    "format_failure_terminal_rate": 0.5,
                },
            },
        ),
        RolloutResult(
            env_name="liveweb",
            task_name="t4",
            reward=0.0,
            success=False,
            time_taken=1.0,
            raw_result={
                "error": "",
                "extra": {
                    "failure_reason": "site_unreachable",
                    "cache_stats": {},
                    "learning_bucket": "environment_failure",
                    "progress_summary": {"progress_score": 0.0},
                    "reachability_audit": {
                        "classification": "env_nav_aborted",
                        "is_environment_failure": True,
                        "is_model_hallucination": False,
                    },
                },
            },
        ),
        RolloutResult(
            env_name="liveweb",
            task_name="t5",
            reward=0.4,
            success=False,
            time_taken=1.0,
            raw_result={
                "error": "",
                "extra": {
                    "failure_reason": None,
                    "cache_stats": {},
                    "learning_bucket": "near_miss",
                    "progress_summary": {"progress_score": 0.6},
                },
            },
        ),
        RolloutResult(
            env_name="liveweb",
            task_name="t6",
            reward=0.0,
            success=False,
            time_taken=1.0,
            raw_result={
                "error": "",
                "extra": {
                    "failure_reason": None,
                    "cache_stats": {},
                    "learning_bucket": "wrong_path",
                    "progress_summary": {"progress_score": 0.1},
                },
            },
        ),
    ]
    metrics = adapter.summarize_results(results, "env")
    assert metrics["env/site_unreachable_rate"] == 2 / 6
    assert metrics["env/cache_error_rate"] == 1 / 6
    assert metrics["env/parse_failed_rate"] == 1 / 6
    assert metrics["env/domain_unreachable_rate"] == 1 / 6
    assert metrics["env/prefetch_failure_rate"] == 1 / 6
    assert metrics["env/invalid_tool_format_rate"] == 1 / 6
    assert metrics["env/format_recovery_attempts"] == 2
    assert metrics["env/format_recovery_successes"] == 1
    assert metrics["env/format_recovery_exhausted"] == 1
    assert metrics["env/format_recovery_success_rate"] == 0.5
    assert metrics["env/format_failure_recoverable_rate"] == pytest.approx(1 / 12)
    assert metrics["env/format_failure_terminal_rate"] == pytest.approx(1 / 12)
    assert metrics["env/mean_progress_score"] == pytest.approx((0.6 + 0.1) / 6)
    assert metrics["env/near_miss_rate"] == pytest.approx(1 / 6)
    assert metrics["env/wrong_path_rate"] == pytest.approx(1 / 6)
    assert metrics["env/learning_bucket/environment_failure_rate"] == pytest.approx(1 / 6)
    assert metrics["env/learning_bucket/format_failure_rate"] == pytest.approx(1 / 6)
    assert metrics["env/reachability_audit_count"] == 1.0
    assert metrics["env/reachability_env_failure_rate"] == 1.0
    assert metrics["env/reachability_model_hallucination_rate"] == 0.0
    assert metrics["env/reachability_nav_aborted_rate"] == 1.0
    assert metrics["env/reachability_target_closed_rate"] == 0.0
