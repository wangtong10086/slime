import dataclasses
import sys
import types
from types import SimpleNamespace


def _install_fake_sglang_engine_deps(monkeypatch):
    fake_router = types.ModuleType("sglang_router")
    fake_router.__version__ = "0.3.0"
    fake_server_args_mod = types.ModuleType("sglang.srt.server_args")

    @dataclasses.dataclass
    class _FakeServerArgs:
        model_path: str = ""
        trust_remote_code: bool = False
        random_seed: int = 0
        enable_memory_saver: bool = False
        host: str = "127.0.0.1"
        port: int = 0
        nccl_port: int = 0
        nnodes: int = 1
        node_rank: int = 0
        dist_init_addr: str = ""
        gpu_id_step: int = 1
        base_gpu_id: int = 0
        tp_size: int = 1
        dp_size: int = 1
        pp_size: int = 1
        ep_size: int = 1
        skip_server_warmup: bool = True
        enable_draft_weights_cpu_backup: bool = True
        enable_return_routed_experts: bool = False
        dtype: str | None = None
        mem_fraction_static: float | None = None
        disaggregation_mode: str = "null"
        load_balance_method: str | None = None
        disaggregation_bootstrap_port: int | None = None
        prefill_round_robin_balance: bool = False

    fake_server_args_mod.ServerArgs = _FakeServerArgs
    fake_utils_mod = types.ModuleType("sglang.srt.utils")
    fake_utils_mod.kill_process_tree = lambda pid: None
    fake_ray_actor = types.ModuleType("slime.ray.ray_actor")
    fake_ray_actor.RayActor = object

    monkeypatch.setitem(sys.modules, "sglang_router", fake_router)
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", fake_server_args_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", fake_utils_mod)
    monkeypatch.setitem(sys.modules, "slime.ray.ray_actor", fake_ray_actor)


def _build_sglang_args(**overrides):
    base = dict(
        rollout_num_gpus_per_engine=2,
        num_gpus_per_node=8,
        seed=1234,
        offload_rollout=True,
        sglang_pp_size=1,
        sglang_dp_size=1,
        sglang_ep_size=1,
        use_rollout_routing_replay=False,
        fp16=False,
        hf_checkpoint="/models/base",
        liveweb_rollout_bootstrap_model_path=None,
        liveweb_rollout_boot_mode="eager_weights",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_resolve_sglang_startup_jit_config_defaults_disabled(monkeypatch):
    _install_fake_sglang_engine_deps(monkeypatch)
    sys.modules.pop("slime.backends.sglang_utils.sglang_engine", None)
    from slime.backends.sglang_utils.sglang_engine import resolve_sglang_startup_jit_config

    monkeypatch.delenv("LIVEWEB_SGLANG_STARTUP_JIT_ENABLED", raising=False)
    enabled, reason = resolve_sglang_startup_jit_config()

    assert enabled is False
    assert reason == "disabled_by_default_for_liveweb_resume"


def test_compute_server_args_uses_bootstrap_model_for_updatable_group(monkeypatch):
    _install_fake_sglang_engine_deps(monkeypatch)
    sys.modules.pop("slime.backends.sglang_utils.sglang_engine", None)
    from slime.backends.sglang_utils.sglang_engine import _compute_server_args

    args = _build_sglang_args(
        liveweb_rollout_boot_mode="lazy_weights",
        liveweb_rollout_bootstrap_model_path="/models/resume_export",
    )

    kwargs, _ = _compute_server_args(
        args=args,
        rank=0,
        dist_init_addr="127.0.0.1:12345",
        nccl_port=12346,
        host="127.0.0.1",
        port=12347,
        worker_type="regular",
        base_gpu_id=0,
        sglang_overrides={"update_weights": True},
        num_gpus_per_engine=2,
        is_updatable_group=True,
    )

    assert kwargs["model_path"] == "/models/resume_export"
    assert "update_weights" not in kwargs


def test_compute_server_args_keeps_hf_model_for_frozen_group(monkeypatch):
    _install_fake_sglang_engine_deps(monkeypatch)
    sys.modules.pop("slime.backends.sglang_utils.sglang_engine", None)
    from slime.backends.sglang_utils.sglang_engine import _compute_server_args

    args = _build_sglang_args(
        liveweb_rollout_boot_mode="lazy_weights",
        liveweb_rollout_bootstrap_model_path="/models/resume_export",
    )

    kwargs, _ = _compute_server_args(
        args=args,
        rank=0,
        dist_init_addr="127.0.0.1:12345",
        nccl_port=12346,
        host="127.0.0.1",
        port=12347,
        worker_type="regular",
        base_gpu_id=0,
        sglang_overrides={"update_weights": False},
        num_gpus_per_engine=2,
        is_updatable_group=False,
    )

    assert kwargs["model_path"] == "/models/base"
    assert "update_weights" not in kwargs
