from slime.rollout.liveweb_online.task_sampling import LiveWebDynamicSampler


def test_liveweb_sampler_emits_composite_tasks(monkeypatch):
    monkeypatch.setenv("TASK_REGISTRY_VERSION", "v2")
    monkeypatch.setenv("LIVEWEB_EXCLUDE_PLUGINS", "weather,openlibrary")

    sampler = LiveWebDynamicSampler(excluded_plugins={"weather", "openlibrary"}, min_unique_plugins=2)
    selection = sampler.sample(seed=1234, phase="main", evaluation=False)

    assert selection["num_subtasks"] in {2, 3, 4}
    assert len(set(selection["plugin_names"])) >= 2
    assert "weather" not in selection["plugin_names"]
    assert "openlibrary" not in selection["plugin_names"]


def test_liveweb_dynamic_sampler_downweights_noisy_zero_std_combos(monkeypatch):
    monkeypatch.setenv("TASK_REGISTRY_VERSION", "v2")

    sampler = LiveWebDynamicSampler(excluded_plugins={"weather", "openlibrary"}, min_unique_plugins=2)
    low_value = sampler.candidates[0]
    high_value = next(candidate for candidate in sampler.candidates if candidate.combo_key != low_value.combo_key)

    sampler.record_group_feedback(
        [
            {
                "combo_key": low_value.combo_key,
                "mean_score": 0.0,
                "success_rate": 0.0,
                "env_error_rate": 0.8,
                "accepted": False,
                "zero_std": True,
            }
            for _ in range(8)
        ]
        + [
            {
                "combo_key": high_value.combo_key,
                "mean_score": 0.35,
                "success_rate": 0.25,
                "env_error_rate": 0.05,
                "accepted": True,
                "zero_std": False,
            }
            for _ in range(8)
        ]
    )

    assert sampler.compute_dynamic_weight(high_value, phase="main") > sampler.compute_dynamic_weight(
        low_value, phase="main"
    )
