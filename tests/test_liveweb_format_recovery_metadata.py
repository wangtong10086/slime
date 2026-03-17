from slime.rollout.liveweb_online.common import format_recovery_extra


class _FakeAgentLoop:
    def get_format_recovery_stats(self):
        return {
            "format_recovery_attempts": 3,
            "format_recovery_successes": 2,
            "format_recovery_exhausted": 1,
            "format_recovery_success_rate": 2 / 3,
            "format_failure_class_counts": {
                "recoverable_truncated_tool_json": 2,
                "terminal_natural_language": 1,
            },
            "format_failure_recoverable_rate": 2 / 3,
            "format_failure_terminal_rate": 1 / 3,
        }


def test_format_recovery_extra_passthrough():
    extra = format_recovery_extra(_FakeAgentLoop())
    assert extra["format_recovery_attempts"] == 3
    assert extra["format_recovery_successes"] == 2
    assert extra["format_recovery_exhausted"] == 1
    assert extra["format_failure_class_counts"]["recoverable_truncated_tool_json"] == 2
    assert extra["format_failure_recoverable_rate"] == 2 / 3
    assert extra["format_failure_terminal_rate"] == 1 / 3


def test_format_recovery_extra_handles_missing_agent_loop():
    assert format_recovery_extra(None) == {}


def test_format_recovery_extra_handles_non_mapping_stats():
    class _BadAgentLoop:
        def get_format_recovery_stats(self):
            return None

    assert format_recovery_extra(_BadAgentLoop()) == {}
