import sys
import types
from types import SimpleNamespace


def _install_fake_rollout_deps(monkeypatch):
    fake_sglang = types.ModuleType("sglang")
    fake_srt = types.ModuleType("sglang.srt")
    fake_constants = types.ModuleType("sglang.srt.constants")
    fake_constants.GPU_MEMORY_TYPE_CUDA_GRAPH = "cuda_graph"
    fake_constants.GPU_MEMORY_TYPE_KV_CACHE = "kv"
    fake_constants.GPU_MEMORY_TYPE_WEIGHTS = "weights"
    fake_server_args_mod = types.ModuleType("sglang.srt.server_args")

    class _FakeServerArgs:
        pass

    fake_server_args_mod.ServerArgs = _FakeServerArgs
    fake_utils_mod = types.ModuleType("sglang.srt.utils")
    fake_utils_mod.kill_process_tree = lambda pid: None
    fake_router = types.ModuleType("sglang_router")
    fake_router.__version__ = "0.3.0"
    fake_ray_actor = types.ModuleType("slime.ray.ray_actor")
    fake_ray_actor.RayActor = object
    fake_wandb = types.ModuleType("wandb")
    fake_wandb.sdk = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "sglang", fake_sglang)
    monkeypatch.setitem(sys.modules, "sglang.srt", fake_srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.constants", fake_constants)
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", fake_server_args_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", fake_utils_mod)
    monkeypatch.setitem(sys.modules, "sglang_router", fake_router)
    monkeypatch.setitem(sys.modules, "slime.ray.ray_actor", fake_ray_actor)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)


class _FakeActorModel:
    def __init__(self):
        self.rollout_manager = None

    def set_rollout_manager(self, rollout_manager):
        self.rollout_manager = rollout_manager
        return "actor-attached"


class _FakeRolloutManager:
    class _RemoteMethod:
        def __init__(self, tag):
            self.tag = tag
            self.calls = []

        def remote(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return (self.tag, args, kwargs)

    def __init__(self):
        self.load = self._RemoteMethod("load")


def test_attach_rollout_manager_to_training_models_without_global_dataset(monkeypatch):
    fake_rollout = types.ModuleType("slime.ray.rollout")
    fake_rollout.RolloutManager = object
    monkeypatch.setitem(sys.modules, "slime.ray.rollout", fake_rollout)
    from slime.ray import placement_group

    actor_model = _FakeActorModel()
    args = SimpleNamespace(use_critic=False, rollout_global_dataset=False, start_rollout_id=3)
    rollout_manager = _FakeRolloutManager()

    monkeypatch.setattr(placement_group.ray, "get", lambda refs: refs)

    placement_group.attach_rollout_manager_to_training_models(args, actor_model, None, rollout_manager)

    assert actor_model.rollout_manager is rollout_manager
    assert rollout_manager.load.calls == []


def test_attach_rollout_manager_to_training_models_loads_dataset_when_enabled(monkeypatch):
    fake_rollout = types.ModuleType("slime.ray.rollout")
    fake_rollout.RolloutManager = object
    monkeypatch.setitem(sys.modules, "slime.ray.rollout", fake_rollout)
    from slime.ray import placement_group

    actor_model = _FakeActorModel()
    critic_model = _FakeActorModel()
    args = SimpleNamespace(use_critic=True, rollout_global_dataset=True, start_rollout_id=5)
    rollout_manager = _FakeRolloutManager()

    monkeypatch.setattr(placement_group.ray, "get", lambda refs: refs)

    placement_group.attach_rollout_manager_to_training_models(args, actor_model, critic_model, rollout_manager)

    assert actor_model.rollout_manager is rollout_manager
    assert critic_model.rollout_manager is rollout_manager
    assert rollout_manager.load.calls == [((4,), {})]


def test_resolve_rollout_bootstrap_metadata_prefers_resume_hf_export(monkeypatch, tmp_path):
    _install_fake_rollout_deps(monkeypatch)
    from slime.ray.rollout import resolve_rollout_bootstrap_metadata

    ckpt_root = tmp_path / "source" / "checkpoints"
    ckpt_root.mkdir(parents=True)
    (ckpt_root / "latest_checkpointed_iteration.txt").write_text("7")
    hf_dir = ckpt_root.parent / "hf_exports" / "iter_0000007"
    hf_dir.mkdir(parents=True)

    monkeypatch.setenv("LIVEWEB_ROLLOUT_BOOT_MODE", "lazy_weights")
    monkeypatch.setenv("LIVEWEB_RUN_MODE", "resume")
    monkeypatch.setenv("LIVEWEB_SKIP_INITIAL_WEIGHT_SYNC_IF_RESUME_EXPORT", "1")
    args = SimpleNamespace(load=str(ckpt_root), hf_checkpoint="/base/hf")

    metadata = resolve_rollout_bootstrap_metadata(args)

    assert metadata["boot_mode"] == "lazy_weights"
    assert metadata["bootstrap_model_source"] == "resume_hf_export"
    assert metadata["bootstrap_model_path"] == str(hf_dir)
    assert metadata["skip_initial_weight_sync"] is True


def test_resolve_rollout_bootstrap_metadata_defaults_to_initial_weight_sync(monkeypatch, tmp_path):
    _install_fake_rollout_deps(monkeypatch)
    from slime.ray.rollout import resolve_rollout_bootstrap_metadata

    ckpt_root = tmp_path / "source" / "checkpoints"
    ckpt_root.mkdir(parents=True)
    (ckpt_root / "latest_checkpointed_iteration.txt").write_text("7")
    hf_dir = ckpt_root.parent / "hf_exports" / "iter_0000007"
    hf_dir.mkdir(parents=True)

    monkeypatch.setenv("LIVEWEB_ROLLOUT_BOOT_MODE", "lazy_weights")
    monkeypatch.setenv("LIVEWEB_RUN_MODE", "resume")
    monkeypatch.delenv("LIVEWEB_SKIP_INITIAL_WEIGHT_SYNC_IF_RESUME_EXPORT", raising=False)
    args = SimpleNamespace(load=str(ckpt_root), hf_checkpoint="/base/hf")

    metadata = resolve_rollout_bootstrap_metadata(args)

    assert metadata["bootstrap_model_source"] == "resume_hf_export"
    assert metadata["bootstrap_model_path"] == str(hf_dir)
    assert metadata["skip_initial_weight_sync"] is False


def test_prepare_rollout_startup_starts_routers_without_engines(monkeypatch):
    _install_fake_rollout_deps(monkeypatch)
    from slime.ray import rollout as rollout_module

    class _FakeModel:
        def __init__(self, name, has_pd):
            self.name = name
            self._has_pd = has_pd

        def resolve(self, args):
            return None

        @property
        def has_pd_disaggregation(self):
            return self._has_pd

    fake_config = SimpleNamespace(models=[_FakeModel("actor", False), _FakeModel("ref", True)])
    start_calls = []

    monkeypatch.setattr(rollout_module, "_resolve_sglang_config", lambda args: fake_config)

    def fake_start_router(args, *, has_pd_disaggregation=False, force_new=False):
        start_calls.append((has_pd_disaggregation, force_new))
        return ("127.0.0.1", 3000 + len(start_calls))

    monkeypatch.setattr(rollout_module, "_start_router", fake_start_router)

    prepared = rollout_module.prepare_rollout_startup(SimpleNamespace())

    assert prepared == {
        "actor": {"router_ip": "127.0.0.1", "router_port": 3001, "has_pd_disaggregation": False},
        "ref": {"router_ip": "127.0.0.1", "router_port": 3002, "has_pd_disaggregation": True},
    }
    assert start_calls == [(False, False), (True, True)]
