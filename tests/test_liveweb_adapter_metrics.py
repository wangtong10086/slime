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
                    "abort_failed_count": 1,
                    "pending_samples": 2,
                    "pending_groups": 1,
                    "oldest_pending_age_seconds": 12.0,
                    "active_decode_requests": 1,
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
            time_taken=2.0,
            raw_result={
                "error": "",
                "extra": {
                    "failure_reason": None,
                    "cache_stats": {},
                    "learning_bucket": "near_miss",
                    "rl_failure_bucket": "premature_stop",
                    "unsupported_stop": True,
                    "hallucinated_plugin_count": 1,
                    "num_subtasks": 2,
                    "steps_used": 2,
                    "trajectory_diagnostics": {
                        "google_family_offdomain_count": 1,
                        "repeated_url_count": 2,
                        "same_page_loop_count": 1,
                    },
                    "progress_summary": {"progress_score": 0.6},
                },
            },
        ),
        RolloutResult(
            env_name="liveweb",
            task_name="t6",
            reward=0.0,
            success=False,
            time_taken=3.0,
            raw_result={
                "error": "",
                "extra": {
                    "failure_reason": "max_steps_reached",
                    "cache_stats": {"per_domain_miss_count": 2, "per_domain_miss_latency_s": 1.0},
                    "learning_bucket": "wrong_path",
                    "rl_failure_bucket": "wrong_domain_loop",
                    "num_subtasks": 3,
                    "steps_used": 3,
                    "trajectory_diagnostics": {
                        "google_family_offdomain_count": 3,
                        "repeated_url_count": 4,
                        "same_page_loop_count": 2,
                    },
                    "progress_summary": {"progress_score": 0.1},
                },
            },
        ),
        RolloutResult(
            env_name="liveweb",
            task_name="t7",
            reward=0.0,
            success=False,
            time_taken=1.0,
            raw_result={
                "error": "The input (34758 tokens) is longer than the model's context length (32768 tokens).",
                "extra": {
                    "failure_reason": "format_recovery_overflow",
                    "cache_stats": {},
                    "learning_bucket": "format_failure",
                    "progress_summary": {"progress_score": 0.0},
                },
            },
        ),
        RolloutResult(
            env_name="liveweb",
            task_name="t8",
            reward=0.0,
            success=False,
            time_taken=1.0,
            raw_result={
                "error": "401 Unauthorized while calling /abort_request",
                "extra": {
                    "failure_reason": "control_plane_auth_failure",
                    "cache_stats": {},
                    "learning_bucket": "environment_failure",
                    "progress_summary": {"progress_score": 0.0},
                },
            },
        ),
    ]
    metrics = adapter.summarize_results(results, "env")
    assert metrics["env/site_unreachable_rate"] == 2 / 8
    assert metrics["env/cache_error_rate"] == 1 / 8
    assert metrics["env/parse_failed_rate"] == 1 / 8
    assert metrics["env/domain_unreachable_rate"] == 1 / 8
    assert metrics["env/prefetch_failure_rate"] == 1 / 8
    assert metrics["env/invalid_tool_format_rate"] == 1 / 8
    assert metrics["env/format_recovery_overflow_rate"] == 1 / 8
    assert metrics["env/control_plane_auth_failure_count"] == 1.0
    assert metrics["env/abort_failed_count"] == 1.0
    assert metrics["env/format_recovery_attempts"] == 2
    assert metrics["env/format_recovery_successes"] == 1
    assert metrics["env/format_recovery_exhausted"] == 1
    assert metrics["env/format_recovery_success_rate"] == 0.5
    assert metrics["env/format_failure_recoverable_rate"] == pytest.approx(1 / 16)
    assert metrics["env/format_failure_terminal_rate"] == pytest.approx(1 / 16)
    assert metrics["env/mean_progress_score"] == pytest.approx((0.6 + 0.1) / 8)
    assert metrics["env/mean_num_tasks"] == pytest.approx(2.5)
    assert metrics["env/num_tasks_unknown_rate"] == pytest.approx(6 / 8)
    assert metrics["env/num_tasks_1_rate"] == pytest.approx(0.0)
    assert metrics["env/num_tasks_2_rate"] == pytest.approx(0.5)
    assert metrics["env/num_tasks_3_rate"] == pytest.approx(0.5)
    assert metrics["env/num_tasks_4_rate"] == pytest.approx(0.0)
    assert metrics["env/near_miss_rate"] == pytest.approx(1 / 8)
    assert metrics["env/wrong_domain_loop_rate"] == pytest.approx(1 / 8)
    assert metrics["env/premature_stop_rate"] == pytest.approx(1 / 8)
    assert metrics["env/wrong_path_rate"] == pytest.approx(1 / 8)
    assert metrics["env/max_steps_reached_rate"] == pytest.approx(1 / 8)
    assert metrics["env/unsupported_stop_rate"] == pytest.approx(1 / 8)
    assert metrics["env/hallucinated_plugin_mean"] == pytest.approx(1 / 8)
    assert metrics["env/google_family_offdomain_mean"] == pytest.approx(4 / 8)
    assert metrics["env/repeated_url_mean"] == pytest.approx(6 / 8)
    assert metrics["env/same_page_loop_mean"] == pytest.approx(3 / 8)
    assert metrics["env/browser_step_time_mean"] == pytest.approx((1 + 1 + 1 + 1 + 1 + 1 + 1 + 1) / 8)
    assert metrics["env/browser_nav_time_mean"] == pytest.approx(0.5)
    assert metrics["env/learning_bucket/environment_failure_rate"] == pytest.approx(2 / 8)
    assert metrics["env/learning_bucket/format_failure_rate"] == pytest.approx(2 / 8)
    assert metrics["env/cache_hit_rate"] == 0.0
    assert metrics["env/challenge_page_rate"] == 0.0
    assert metrics["scheduler/pending_samples"] == 2.0
    assert metrics["scheduler/pending_groups"] == 1.0
    assert metrics["scheduler/oldest_pending_age_seconds"] == 12.0
    assert metrics["scheduler/active_decode_requests"] == 1.0
    assert metrics["env/reachability_audit_count"] == 1.0
    assert metrics["env/reachability_env_failure_rate"] == 1.0
    assert metrics["env/reachability_model_hallucination_rate"] == 0.0
    assert metrics["env/reachability_nav_aborted_rate"] == 1.0
    assert metrics["env/reachability_target_closed_rate"] == 0.0


def test_liveweb_classifies_challenge_page_as_environment_failure():
    adapter = LiveWebEnvironmentAdapter()
    result = {
        "error": "CAPTCHA/challenge page detected (title: 'Just a moment...')",
        "extra": {"failure_reason": "site_unreachable", "cache_stats": {}},
    }
    assert adapter._classify_environment_failure_type(result) == "challenge_page"


def test_liveweb_summarize_results_accepts_per_domain_cache_stats_dicts():
    adapter = LiveWebEnvironmentAdapter()
    results = [
        RolloutResult(
            env_name="liveweb",
            task_name="cache-dict",
            reward=0.0,
            success=False,
            time_taken=4.0,
            raw_result={
                "error": "",
                "extra": {
                    "failure_reason": "max_steps_reached",
                    "steps_used": 4,
                    "cache_stats": {
                        "per_domain_miss_count": {"www.google.com": 2, "www.bloomberg.com": 1},
                        "per_domain_miss_latency_s": {"www.google.com": 1.2, "www.bloomberg.com": 0.9},
                    },
                },
            },
        )
    ]

    metrics = adapter.summarize_results(results, "env")

    assert metrics["env/browser_step_time_mean"] == pytest.approx(1.0)
    assert metrics["env/browser_nav_time_mean"] == pytest.approx((1.2 + 0.9) / 3)
