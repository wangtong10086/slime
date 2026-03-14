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
