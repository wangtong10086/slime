from slime.env_adapters.liveweb import LiveWebEnvironmentAdapter


def test_liveweb_adapter_samples_composite_tasks(monkeypatch):
    monkeypatch.setenv("TASK_REGISTRY_VERSION", "v2")
    monkeypatch.setenv("LIVEWEB_EXCLUDE_PLUGINS", "weather,openlibrary")
    monkeypatch.setenv("LIVEWEB_MIN_UNIQUE_PLUGINS", "2")

    adapter = LiveWebEnvironmentAdapter()
    tasks = adapter.sample_tasks(split="train", count=4, phase="main")

    assert len(tasks) == 4
    for task in tasks:
        assert task.task_family == "composite_prompt"
        assert int(task.metadata["num_subtasks"]) in {2, 3, 4}
        plugin_names = task.metadata["plugin_names"]
        assert len(set(plugin_names)) >= 2
        assert "weather" not in plugin_names
        assert "openlibrary" not in plugin_names
