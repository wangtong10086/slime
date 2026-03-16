import pytest

from slime.rollout.liveweb_online.common import compute_reward_from_result


def test_liveweb_reward_shaping_keeps_final_score_dominant():
    reward, meta = compute_reward_from_result(
        {
            "score": 0.4,
            "extra": {
                "failure_reason": None,
                "required_domains": ["a.com", "b.com"],
                "visited_domains": ["a.com", "b.com"],
                "answer_details": [
                    {"answer_tag": "answer1", "actual": "foo"},
                    {"answer_tag": "answer2", "actual": "bar"},
                ],
            },
            "rewards": {
                "step_rewards": [
                    {"signals": [{"signal": "new_domain", "value": 0.05, "reason": "visited"}]},
                    {"signals": [{"signal": "no_progress", "value": -0.02, "reason": "stall"}]},
                ]
            },
        }
    )

    assert reward == pytest.approx(0.48)
    assert meta["final_score"] == 0.4
    assert meta["shaping_reward"] == pytest.approx(0.08)


def test_liveweb_reward_drops_environment_failures():
    reward, meta = compute_reward_from_result(
        {
            "score": 0.0,
            "error": "Required site unreachable: https://example.com",
            "extra": {"failure_reason": "site_unreachable"},
        }
    )

    assert reward is None
    assert meta["drop_reason"] == "site_unreachable"
