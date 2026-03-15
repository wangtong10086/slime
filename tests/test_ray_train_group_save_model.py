from types import SimpleNamespace

import slime.ray.actor_group as actor_group_module


class _FakeRemoteMethod:
    def __init__(self, events, actor_name):
        self._events = events
        self._actor_name = actor_name

    def remote(self, rollout_id, force_sync=False):
        self._events.append(("remote", self._actor_name, rollout_id, force_sync))
        return f"ref:{self._actor_name}:{rollout_id}"


class _FakeActor:
    def __init__(self, events, name):
        self.save_model = _FakeRemoteMethod(events, name)


def _build_group(train_backend, events, actor_count=4):
    group = actor_group_module.RayTrainGroup.__new__(actor_group_module.RayTrainGroup)
    group.args = SimpleNamespace(train_backend=train_backend)
    group._actor_handlers = [_FakeActor(events, f"rank{idx}") for idx in range(actor_count)]
    return group


def test_megatron_save_model_forces_all_ranks_together(monkeypatch, caplog):
    events = []
    ray_get_calls = []

    def fake_ray_get(refs):
        ray_get_calls.append(list(refs))
        events.append(("ray.get", tuple(refs)))
        return list(refs)

    monkeypatch.setenv("TRAIN_SAVE_BATCH_SIZE", "1")
    monkeypatch.setattr(actor_group_module.ray, "get", fake_ray_get)

    group = _build_group("megatron", events, actor_count=4)

    with caplog.at_level("WARNING"):
        results = group.save_model(rollout_id=9, force_sync=True)

    assert results == [
        "ref:rank0:9",
        "ref:rank1:9",
        "ref:rank2:9",
        "ref:rank3:9",
    ]
    assert ray_get_calls == [[
        "ref:rank0:9",
        "ref:rank1:9",
        "ref:rank2:9",
        "ref:rank3:9",
    ]]
    assert events[:4] == [
        ("remote", "rank0", 9, True),
        ("remote", "rank1", 9, True),
        ("remote", "rank2", 9, True),
        ("remote", "rank3", 9, True),
    ]
    assert events[4] == ("ray.get", ("ref:rank0:9", "ref:rank1:9", "ref:rank2:9", "ref:rank3:9"))
    assert "requires all ranks to enter together" in caplog.text


def test_non_megatron_save_model_still_honors_chunking(monkeypatch):
    events = []
    ray_get_calls = []

    def fake_ray_get(refs):
        ray_get_calls.append(list(refs))
        events.append(("ray.get", tuple(refs)))
        return list(refs)

    monkeypatch.setenv("TRAIN_SAVE_BATCH_SIZE", "2")
    monkeypatch.setattr(actor_group_module.ray, "get", fake_ray_get)

    group = _build_group("dummy", events, actor_count=4)

    results = group.save_model(rollout_id=3, force_sync=False)

    assert results == [
        "ref:rank0:3",
        "ref:rank1:3",
        "ref:rank2:3",
        "ref:rank3:3",
    ]
    assert ray_get_calls == [
        ["ref:rank0:3", "ref:rank1:3"],
        ["ref:rank2:3", "ref:rank3:3"],
    ]
    assert events == [
        ("remote", "rank0", 3, False),
        ("remote", "rank1", 3, False),
        ("ray.get", ("ref:rank0:3", "ref:rank1:3")),
        ("remote", "rank2", 3, False),
        ("remote", "rank3", 3, False),
        ("ray.get", ("ref:rank2:3", "ref:rank3:3")),
    ]
