from types import SimpleNamespace

from slime.ray.actor_group import RayTrainGroup


class _RemoteMethod:
    def __init__(self):
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return (args, kwargs)


class _FakeActor:
    def __init__(self):
        self.save_model = _RemoteMethod()


def test_save_model_forwards_hf_export(monkeypatch):
    group = object.__new__(RayTrainGroup)
    group.args = SimpleNamespace(train_backend="other")
    group._actor_handlers = [_FakeActor(), _FakeActor()]
    monkeypatch.setattr("slime.ray.actor_group.ray.get", lambda refs: refs)

    results = RayTrainGroup.save_model(
        group,
        rollout_id=7,
        force_sync=True,
        hf_export=True,
        save_mode="archive",
        save_dir="/tmp/archive",
    )

    assert len(results) == 2
    for actor in group._actor_handlers:
        assert actor.save_model.calls == [
            ((7,), {"force_sync": True, "hf_export": True, "save_mode": "archive", "save_dir": "/tmp/archive"})
        ]
