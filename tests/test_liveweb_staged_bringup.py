import sys
import types
from types import SimpleNamespace


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
