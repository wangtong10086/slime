from slime.utils.save_guard import (
    resolve_save_plan,
    resolve_save_mode,
    resolve_hf_export_policy,
    should_skip_save_due_to_memory_guard,
)


def test_resolve_hf_export_policy_final_save(monkeypatch):
    monkeypatch.delenv("LIVEWEB_ENABLE_PERIODIC_HF_EXPORT", raising=False)
    assert resolve_hf_export_policy(rollout_id=9, num_rollout=10, force_sync=True) is True


def test_resolve_hf_export_policy_periodic_disabled(monkeypatch):
    monkeypatch.setenv("LIVEWEB_ENABLE_PERIODIC_HF_EXPORT", "0")
    assert resolve_hf_export_policy(rollout_id=4, num_rollout=20, force_sync=False) is False


def test_resolve_hf_export_policy_periodic_enabled(monkeypatch):
    monkeypatch.setenv("LIVEWEB_ENABLE_PERIODIC_HF_EXPORT", "1")
    monkeypatch.setenv("LIVEWEB_HF_EXPORT_INTERVAL", "5")
    assert resolve_hf_export_policy(rollout_id=4, num_rollout=20, force_sync=False) is True
    assert resolve_hf_export_policy(rollout_id=3, num_rollout=20, force_sync=False) is False


def test_should_skip_save_due_to_memory_guard_by_available():
    snapshot = {"save/host_mem_available_gb": 20.0, "save/host_mem_used_ratio": 0.90}
    assert should_skip_save_due_to_memory_guard(snapshot, min_available_gb=40.0, max_used_ratio=0.96) is True


def test_should_skip_save_due_to_memory_guard_by_ratio():
    snapshot = {"save/host_mem_available_gb": 80.0, "save/host_mem_used_ratio": 0.98}
    assert should_skip_save_due_to_memory_guard(snapshot, min_available_gb=40.0, max_used_ratio=0.96) is True


def test_should_not_skip_save_due_to_memory_guard():
    snapshot = {"save/host_mem_available_gb": 80.0, "save/host_mem_used_ratio": 0.90}
    assert should_skip_save_due_to_memory_guard(snapshot, min_available_gb=40.0, max_used_ratio=0.96) is False


def test_resolve_save_mode_prefers_full_on_final():
    assert resolve_save_mode(rollout_id=9, num_rollout=10, archive_interval=5, full_interval=25) == "full"


def test_resolve_save_mode_full_interval():
    assert resolve_save_mode(rollout_id=24, num_rollout=100, archive_interval=5, full_interval=25) == "full"


def test_resolve_save_mode_archive_interval():
    assert resolve_save_mode(rollout_id=4, num_rollout=100, archive_interval=5, full_interval=25) == "archive"


def test_resolve_save_mode_none_when_not_due():
    assert resolve_save_mode(rollout_id=3, num_rollout=100, archive_interval=5, full_interval=25) is None


def test_resolve_save_plan_final_none_falls_back_to_archive():
    plan = resolve_save_plan(
        rollout_id=9,
        num_rollout=10,
        archive_interval=5,
        full_interval=100,
        final_save_mode="none",
    )
    assert plan is not None
    assert plan.mode == "archive"
    assert plan.is_final is True
    assert plan.write_rollout_state is False
