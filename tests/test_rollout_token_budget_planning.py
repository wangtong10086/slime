import sys
import types
from argparse import Namespace

fake_mpu_module = types.SimpleNamespace()
fake_megatron_core = types.ModuleType("megatron.core")
fake_megatron_core.mpu = fake_mpu_module
fake_packed_seq_params = types.ModuleType("megatron.core.packed_seq_params")
fake_packed_seq_params.PackedSeqParams = object

sys.modules.setdefault("megatron", types.ModuleType("megatron"))
sys.modules["megatron.core"] = fake_megatron_core
sys.modules["megatron.core.packed_seq_params"] = fake_packed_seq_params
sys.modules.setdefault("wandb", types.SimpleNamespace())

from slime.backends.megatron_utils import data as megatron_data
from slime.ray.rollout_batching import plan_train_steps_by_token_budget


class _FakeMpu:
    @staticmethod
    def get_data_parallel_world_size(with_context_parallel=False):
        return 1

    @staticmethod
    def get_data_parallel_group():
        return None

    @staticmethod
    def get_virtual_pipeline_model_parallel_world_size():
        return None

    @staticmethod
    def get_context_parallel_world_size():
        return 1


def test_plan_train_steps_spreads_long_samples_across_steps():
    samples = [{"tokens": list(range(length))} for length in [20000, 19000, 18000, 17000, 6000, 5000, 4000, 3000]]

    plan = plan_train_steps_by_token_budget(
        samples,
        max_samples_per_step=12,
        min_samples_per_step=8,
        underfilled_min_samples=4,
        step_token_budget=48000,
        long_sample_threshold=16000,
        max_long_samples_per_step=2,
    )

    assert plan.retained_indices
    assert max(plan.step_token_counts) <= 48000
    assert len(plan.step_num_samples) >= 2
    assert all(count >= 4 for count in plan.step_num_samples)
    assert plan.underfilled_steps == len(plan.step_num_samples)
    assert max(plan.step_long_sample_counts) <= 2


def test_plan_train_steps_tracks_density_restrictions():
    samples = [{"tokens": list(range(length))} for length in [17000, 16000, 15000, 1000, 1000, 1000, 1000, 1000]]

    plan = plan_train_steps_by_token_budget(
        samples,
        max_samples_per_step=12,
        min_samples_per_step=4,
        underfilled_min_samples=2,
        step_token_budget=48000,
        long_sample_threshold=15000,
        max_long_samples_per_step=1,
    )

    assert max(plan.step_long_sample_counts) <= 1
    assert plan.density_restricted_steps >= 1


def test_plan_train_steps_keeps_underfilled_last_step():
    samples = [{"tokens": list(range(length))} for length in [12000, 11000, 10000, 9000, 8000, 7000, 6000, 5000, 4000]]

    plan = plan_train_steps_by_token_budget(
        samples,
        max_samples_per_step=12,
        min_samples_per_step=8,
        underfilled_min_samples=4,
        step_token_budget=48000,
        long_sample_threshold=16000,
        max_long_samples_per_step=2,
    )

    assert plan.step_num_samples[-1] >= 4
    assert sum(plan.step_num_samples) == len(plan.retained_indices)
    assert plan.step_boundaries[-1] == len(plan.retained_indices)


def test_plan_train_steps_drops_oversize_samples():
    samples = [{"tokens": list(range(length))} for length in [60000, 12000, 11000, 10000, 9000]]

    plan = plan_train_steps_by_token_budget(
        samples,
        max_samples_per_step=12,
        min_samples_per_step=8,
        underfilled_min_samples=4,
        step_token_budget=48000,
        long_sample_threshold=16000,
        max_long_samples_per_step=2,
    )

    assert plan.oversize_samples_dropped == 1
    assert 0 not in plan.retained_indices
    assert max(plan.step_token_counts) <= 48000
    assert plan.long_samples_trimmed >= 1


def test_get_data_iterator_uses_explicit_train_step_boundaries(monkeypatch):
    monkeypatch.setattr(megatron_data, "mpu", _FakeMpu())

    args = Namespace(
        global_batch_size=32,
        use_dynamic_batch_size=False,
        micro_batch_size=2,
        max_tokens_per_gpu=None,
    )
    rollout_data = {
        "tokens": [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12]],
        "response_lengths": [1] * 12,
        "loss_masks": [[1]] * 12,
        "total_lengths": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        "dynamic_global_batch_size": 12,
        "train_step_boundaries": [0, 4, 9, 12],
        "train_step_token_counts": [46, 80, 60],
        "train_step_num_samples": [4, 5, 3],
    }

    data_iterators, num_microbatches = megatron_data.get_data_iterator(args, [object()], rollout_data)

    assert num_microbatches == [2, 3, 2]
    iterator = data_iterators[0]
    batch0 = iterator.get_next(["tokens"])
    batch1 = iterator.get_next(["tokens"])
    batch2 = iterator.get_next(["tokens"])

    assert batch0["dynamic_global_batch_size"] == 4
    assert batch0["train_step_token_count"] == 46
    assert batch1["dynamic_global_batch_size"] == 4
    assert batch2["dynamic_global_batch_size"] == 5


def test_legacy_path_without_train_step_boundaries(monkeypatch):
    monkeypatch.setattr(megatron_data, "mpu", _FakeMpu())

    args = Namespace(
        global_batch_size=4,
        use_dynamic_batch_size=False,
        micro_batch_size=2,
        max_tokens_per_gpu=None,
    )
    rollout_data = {
        "tokens": [[1], [2], [3], [4]],
        "response_lengths": [1] * 4,
        "loss_masks": [[1]] * 4,
        "total_lengths": [10, 11, 12, 13],
    }

    data_iterators, num_microbatches = megatron_data.get_data_iterator(args, [object()], rollout_data)

    assert num_microbatches == [2]
    batch = data_iterators[0].get_next(["tokens"])
    assert "dynamic_global_batch_size" not in batch


def test_log_rollout_data_accepts_scalar_numeric_metrics(monkeypatch):
    monkeypatch.setattr(
        megatron_data,
        "mpu",
        types.SimpleNamespace(
            get_tensor_model_parallel_rank=lambda: 0,
            is_pipeline_last_stage=lambda: True,
            get_context_parallel_world_size=lambda: 1,
        ),
    )
    captured = {}

    def _fake_gather_log_data(prefix, args, rollout_id, log_dict):
        captured.update(log_dict)
        return {}

    monkeypatch.setattr(megatron_data, "gather_log_data", _fake_gather_log_data)

    rollout_data = {
        "response_lengths": [2, 3],
        "loss_masks": [[1, 1], [1, 1, 1]],
        "total_lengths": [10, 12],
        "train_step_token_budget": 48000,
        "train_underfilled_steps": 1,
        "train_oversize_samples_dropped": 2,
    }
    args = Namespace(
        qkv_format="thd",
        ci_test=False,
        log_multi_turn=False,
        log_passrate=False,
        log_correct_samples=False,
    )

    megatron_data.log_rollout_data(0, args, rollout_data)

    assert captured["train_step_token_budget"] == 48000.0
    assert captured["train_underfilled_steps"] == 1.0
    assert captured["train_oversize_samples_dropped"] == 2.0
