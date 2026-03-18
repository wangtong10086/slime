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


def test_max_steps_penalty():
    reward, meta = compute_reward_from_result(_result(score=0.0, failure_reason="max_steps_reached"))
    assert reward == -0.05
    assert meta["environment_failure_type"] is None


def test_environment_failure_is_dropped():
    reward, meta = compute_reward_from_result(_result(score=0.0, failure_reason="site_unreachable"))
    assert reward is None
    assert meta["environment_failure_type"] == "site_unreachable"


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

    assert reward == 0.25
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
