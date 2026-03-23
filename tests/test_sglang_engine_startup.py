import dataclasses
import sys
import types
from types import SimpleNamespace

import requests


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
        api_key: str | None = None
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


def test_resolve_sglang_control_timeout_seconds_defaults(monkeypatch):
    _install_fake_sglang_engine_deps(monkeypatch)
    sys.modules.pop("slime.backends.sglang_utils.sglang_engine", None)
    from slime.backends.sglang_utils.sglang_engine import resolve_sglang_control_timeout_seconds

    monkeypatch.delenv("LIVEWEB_SGLANG_CONTROL_TIMEOUT_SECONDS", raising=False)
    assert resolve_sglang_control_timeout_seconds() == 60.0

    monkeypatch.setenv("LIVEWEB_SGLANG_CONTROL_TIMEOUT_SECONDS", "17.5")
    assert resolve_sglang_control_timeout_seconds() == 17.5


def test_make_request_uses_control_timeout(monkeypatch):
    _install_fake_sglang_engine_deps(monkeypatch)
    sys.modules.pop("slime.backends.sglang_utils.sglang_engine", None)
    import slime.backends.sglang_utils.sglang_engine as engine_mod

    captured: dict[str, object] = {}

    class _FakeResponse:
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True}

    class _FakeSession:
        trust_env = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json=None, headers=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            captured["timeout"] = timeout
            return _FakeResponse()

    monkeypatch.setenv("LIVEWEB_SGLANG_CONTROL_TIMEOUT_SECONDS", "23")
    monkeypatch.setattr(engine_mod, "_build_requests_session", lambda _url: _FakeSession())

    engine = object.__new__(engine_mod.SGLangEngine)
    engine.node_rank = 0
    engine.server_host = "127.0.0.1"
    engine.server_port = 18000
    engine.server_api_key = "local-liveweb"

    result = engine._make_request("resume_memory_occupation", {"tags": ["weights"]})

    assert result == {"ok": True}
    assert captured["timeout"] == 23.0
    assert captured["url"] == "http://127.0.0.1:18000/resume_memory_occupation"


def test_make_request_timeout_adds_endpoint_context(monkeypatch):
    _install_fake_sglang_engine_deps(monkeypatch)
    sys.modules.pop("slime.backends.sglang_utils.sglang_engine", None)
    import slime.backends.sglang_utils.sglang_engine as engine_mod

    class _FakeSession:
        trust_env = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json=None, headers=None, timeout=None):
            raise requests.exceptions.Timeout("timed out")

    monkeypatch.setenv("LIVEWEB_SGLANG_CONTROL_TIMEOUT_SECONDS", "5")
    monkeypatch.setattr(engine_mod, "_build_requests_session", lambda _url: _FakeSession())

    engine = object.__new__(engine_mod.SGLangEngine)
    engine.node_rank = 0
    engine.server_host = "127.0.0.1"
    engine.server_port = 18001
    engine.server_api_key = None

    try:
        engine._make_request("resume_memory_occupation", {"tags": ["weights"]})
    except requests.exceptions.Timeout as exc:
        notes = getattr(exc, "__notes__", [])
        assert any("resume_memory_occupation" in note for note in notes)
        assert any("timeout=5.0" in note for note in notes)
    else:
        raise AssertionError("expected timeout")


def test_init_normal_registers_worker_with_router_auth_headers(monkeypatch):
    _install_fake_sglang_engine_deps(monkeypatch)
    sys.modules.pop("slime.backends.sglang_utils.sglang_engine", None)
    import slime.backends.sglang_utils.sglang_engine as engine_mod

    captured: dict[str, object] = {}

    class _FakeResponse:
        status_code = 200

        def __init__(self, request_url="http://127.0.0.1:3572/workers"):
            self.request = SimpleNamespace(url=request_url)

        def raise_for_status(self):
            return None

    class _FakeSession:
        trust_env = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json=None, headers=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return _FakeResponse(url)

    monkeypatch.setattr(engine_mod, "_build_requests_session", lambda _url: _FakeSession())
    monkeypatch.setattr(engine_mod, "launch_server_process", lambda _args: SimpleNamespace(pid=12345))

    engine = object.__new__(engine_mod.SGLangEngine)
    engine.node_rank = 0
    engine.router_ip = "127.0.0.1"
    engine.router_port = 3572
    engine.server_host = "127.0.0.1"
    engine.server_port = 15000
    engine.server_api_key = "local-liveweb"
    engine.worker_type = "regular"
    engine.args = SimpleNamespace(use_slime_router=False)

    engine._init_normal(
        {
            "host": "127.0.0.1",
            "port": 15000,
            "api_key": "local-liveweb",
        }
    )

    assert captured["url"] == "http://127.0.0.1:3572/workers"
    assert captured["headers"] == {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": "Bearer local-liveweb",
    }
    assert captured["json"] == {
        "url": "http://127.0.0.1:15000",
        "worker_type": "regular",
        "api_key": "local-liveweb",
    }
