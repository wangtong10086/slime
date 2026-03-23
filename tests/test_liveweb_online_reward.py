import pytest

from slime.rollout.liveweb_online.common import compute_reward_from_result


def _result(score=0.0, failure_reason=None):
    return {
        "score": score,
        "success": False,
        "extra": {
            "failure_reason": failure_reason,
        },
    }


def test_parse_failed_penalty():
    reward, meta = compute_reward_from_result(_result(score=0.2, failure_reason="parse_failed"))
    assert reward == 0.1
    assert meta["environment_failure_type"] is None


def test_malformed_toolcall_parse_failed_is_dropped():
    reward, meta = compute_reward_from_result(
        {
            "score": 0.0,
            "success": False,
            "extra": {
                "failure_reason": "parse_failed",
                "raw_response_preview": '{"name":"goto","arguments":{"url":"https://example.com"}}\n</tool_call>',
                "tool_calls_preview": [],
            },
        }
    )
    assert reward is None
    assert meta["drop_reason"] == "invalid_toolcall_output"


def test_max_steps_penalty():
    reward, meta = compute_reward_from_result(_result(score=0.0, failure_reason="max_steps_reached"))
    assert reward == -0.2
    assert meta["environment_failure_type"] is None


def test_environment_failure_is_dropped():
    reward, meta = compute_reward_from_result(_result(score=0.0, failure_reason="site_unreachable"))
    assert reward is None
    assert meta["environment_failure_type"] == "site_unreachable"


def test_challenge_page_error_is_treated_as_environment_pollution():
    reward, meta = compute_reward_from_result(
        {
            "score": 0.0,
            "success": False,
            "error": "CAPTCHA/challenge page detected (title: 'Just a moment...')",
            "extra": {"failure_reason": "site_unreachable"},
        }
    )
    assert reward is None
    assert meta["environment_failure_type"] == "challenge_page"


def test_reward_is_clipped():
    reward, _ = compute_reward_from_result(_result(score=2.0, failure_reason=None))
    assert reward == 1.0


def test_progress_signals_improve_partial_reward():
    reward, meta = compute_reward_from_result(
        {
            "score": 0.2,
            "success": False,
            "extra": {
                "failure_reason": None,
                "required_domains": [],
                "visited_domains": [],
                "target_assets": ["bitcoin"],
                "collected_target_assets": ["bitcoin"],
                "confirmed_targets": ["bitcoin"],
                "answer_details": [],
                "num_subtasks": 1,
            },
            "rewards": {
                "step_rewards": [
                    {
                        "signals": [
                            {"signal": "target_asset", "value": 0.10, "reason": "+1 targets"},
                            {"signal": "detail_page_visit", "value": 0.03, "reason": "Detail: bitcoin"},
                            {"signal": "all_targets", "value": 0.15, "reason": "All targets collected!"},
                        ]
                    }
                ]
            },
        }
    )

    assert reward == pytest.approx(0.35)
    assert meta["learning_bucket"] == "near_miss"


def test_progress_summary_is_derived_from_result_fields():
    reward, meta = compute_reward_from_result(
        {
            "score": 0.0,
            "success": False,
            "extra": {
                "failure_reason": None,
                "required_domains": ["a.com", "b.com"],
                "visited_domains": ["a.com"],
                "target_assets": ["btc", "eth"],
                "collected_target_assets": ["btc"],
                "confirmed_targets": ["btc"],
                "answer_details": [{"actual": "42"}, {"actual": None}],
                "num_subtasks": 2,
            },
            "rewards": {"step_rewards": []},
        }
    )

    assert reward is not None
    assert meta["required_domains_total"] == 2
    assert meta["required_domains_visited"] == 1
    assert meta["target_assets_total"] == 2
    assert meta["target_assets_collected"] == 1
    assert meta["confirmed_targets_collected"] == 1
    assert meta["answer_slots_total"] == 2
    assert meta["valid_answers"] == 1
    assert meta["progress_summary"]["progress_score"] == 0.5
    assert meta["learning_bucket"] == "near_miss"
    assert meta["rl_failure_bucket"] == "premature_stop"


def test_all_required_domains_covered_adds_shaping_bonus():
    reward, meta = compute_reward_from_result(
        {
            "score": 0.2,
            "success": False,
            "extra": {
                "failure_reason": "incomplete_data",
                "required_domains": ["a.com", "b.com"],
                "visited_domains": ["a.com", "b.com"],
                "target_assets": [],
                "collected_target_assets": [],
                "confirmed_targets": [],
                "answer_details": [],
                "num_subtasks": 2,
            },
            "rewards": {"step_rewards": []},
        }
    )

    assert reward is not None
    assert meta["progress_summary"]["required_domain_coverage"] == 1.0
    assert meta["shaping_reward"] == pytest.approx(0.13)


def test_wrong_domain_loop_penalty_and_bucket():
    reward, meta = compute_reward_from_result(
        {
            "score": 0.0,
            "success": False,
            "extra": {
                "failure_reason": "max_steps_reached",
                "required_domains": ["coingecko.com"],
                "visited_domains": ["google.com"],
                "target_assets": [],
                "collected_target_assets": [],
                "confirmed_targets": [],
                "answer_details": [],
                "num_subtasks": 2,
                "trajectory_diagnostics": {
                    "disallowed_domain_hits": 4,
                    "google_family_offdomain_count": 3,
                    "repeated_url_count": 2,
                    "same_page_loop_count": 2,
                    "offdomain_persist_count": 3,
                    "required_domain_hits": 0,
                    "required_domains_hit": [],
                    "final_host": "www.google.com",
                    "final_google_search": True,
                },
            },
            "rewards": {"step_rewards": []},
        }
    )

    assert reward is not None
    assert reward < -0.5
    assert meta["rl_failure_bucket"] == "wrong_domain_loop"


def test_hallucinated_plugin_answers_are_penalized():
    reward, meta = compute_reward_from_result(
        {
            "score": 0.5,
            "success": False,
            "extra": {
                "failure_reason": "incomplete_data",
                "visited_domains": ["taostats.io"],
                "required_domains": ["taostats.io", "coingecko.com"],
                "target_assets": ["bitcoin"],
                "collected_target_assets": ["bitcoin"],
                "confirmed_targets": ["bitcoin"],
                "answer_details": [
                    {"actual": "42", "required_domains": ["taostats.io"]},
                    {"actual": "100", "required_domains": ["coingecko.com"]},
                ],
                "num_subtasks": 2,
            },
            "rewards": {"step_rewards": []},
        }
    )

    assert reward is not None
    assert meta["hallucinated_plugin_count"] == 1
    assert meta["unsupported_stop"] is True
    assert reward < 0.5
