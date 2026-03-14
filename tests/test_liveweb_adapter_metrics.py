from slime.env_adapters.liveweb import LiveWebEnvironmentAdapter
from slime.env_adapters.base import RolloutResult


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
            raw_result={"error": "", "extra": {"failure_reason": "parse_failed", "cache_stats": {}}},
        ),
    ]
    metrics = adapter.summarize_results(results, "env")
    assert metrics["env/site_unreachable_rate"] == 1 / 3
    assert metrics["env/cache_error_rate"] == 1 / 3
    assert metrics["env/parse_failed_rate"] == 1 / 3
    assert metrics["env/domain_unreachable_rate"] == 1 / 3
    assert metrics["env/prefetch_failure_rate"] == 1 / 3
    assert metrics["env/invalid_tool_format_rate"] == 1 / 3
