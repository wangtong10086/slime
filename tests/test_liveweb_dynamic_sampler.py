import pytest

from slime.rollout.liveweb_online.task_sampling import LiveWebDynamicSampler, canonicalize_phase_name


def test_liveweb_sampler_emits_composite_tasks(monkeypatch):
    monkeypatch.setenv("TASK_REGISTRY_VERSION", "v2")
    monkeypatch.setenv("LIVEWEB_EXCLUDE_PLUGINS", "weather,openlibrary")

    sampler = LiveWebDynamicSampler(excluded_plugins={"weather", "openlibrary"}, min_unique_plugins=2)
    selection = sampler.sample(seed=1234, phase="main", evaluation=False)

    assert selection["num_subtasks"] in {2, 3, 4}
    assert len(set(selection["plugin_names"])) >= 2
    assert "weather" not in selection["plugin_names"]
    assert "openlibrary" not in selection["plugin_names"]


def test_liveweb_sampler_respects_num_task_bounds(monkeypatch):
    monkeypatch.setenv("TASK_REGISTRY_VERSION", "v2")
    monkeypatch.setenv("LIVEWEB_EXCLUDE_PLUGINS", "weather,openlibrary")
    monkeypatch.setenv("LIVEWEB_MIN_NUM_TASKS", "2")
    monkeypatch.setenv("LIVEWEB_MAX_NUM_TASKS", "4")

    sampler = LiveWebDynamicSampler(excluded_plugins={"weather", "openlibrary"}, min_unique_plugins=2)
    for seed in range(20):
        selection = sampler.sample(seed=seed + 2000, phase="main", evaluation=False)
        assert 2 <= selection["num_subtasks"] <= 4


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


def test_liveweb_dynamic_sampler_penalizes_noise_more_aggressively_in_warmup(monkeypatch):
    monkeypatch.setenv("TASK_REGISTRY_VERSION", "v2")

    sampler = LiveWebDynamicSampler(excluded_plugins={"weather", "openlibrary"}, min_unique_plugins=2)
    candidate = sampler.candidates[0]
    sampler.record_group_feedback(
        [
            {
                "combo_key": candidate.combo_key,
                "mean_score": 0.05,
                "success_rate": 0.05,
                "env_error_rate": 0.4,
                "accepted": False,
                "zero_std": True,
            }
            for _ in range(8)
        ]
    )

    assert sampler.compute_dynamic_weight(candidate, phase="warmup") > sampler.compute_dynamic_weight(
        candidate, phase="main"
    )


def test_liveweb_dynamic_sampler_penalizes_high_noise_sites_in_main(monkeypatch):
    monkeypatch.setenv("TASK_REGISTRY_VERSION", "v2")

    sampler = LiveWebDynamicSampler(excluded_plugins={"weather", "openlibrary"}, min_unique_plugins=2)
    taostats_candidate = next(candidate for candidate in sampler.candidates if "taostats" in candidate.plugin_names)
    stable_candidate = next(
        candidate
        for candidate in sampler.candidates
        if "taostats" not in candidate.plugin_names and "stooq" not in candidate.plugin_names
    )

    sampler.record_group_feedback(
        [
            {
                "combo_key": taostats_candidate.combo_key,
                "mean_score": 0.10,
                "success_rate": 0.05,
                "env_error_rate": 0.6,
                "accepted": False,
                "zero_std": False,
            }
            for _ in range(8)
        ]
        + [
            {
                "combo_key": stable_candidate.combo_key,
                "mean_score": 0.15,
                "success_rate": 0.10,
                "env_error_rate": 0.1,
                "accepted": True,
                "zero_std": False,
            }
            for _ in range(8)
        ]
    )

    assert sampler.compute_dynamic_weight(stable_candidate, phase="main") > sampler.compute_dynamic_weight(
        taostats_candidate, phase="main"
    )


def test_liveweb_dynamic_sampler_prefers_partial_progress_when_enabled(monkeypatch):
    monkeypatch.setenv("TASK_REGISTRY_VERSION", "v2")
    monkeypatch.setenv("LIVEWEB_ENABLE_PROGRESS_AWARE_SAMPLER", "1")

    sampler = LiveWebDynamicSampler(excluded_plugins={"weather", "openlibrary"}, min_unique_plugins=2)
    stalled_candidate = sampler.candidates[0]
    progressing_candidate = next(
        candidate for candidate in sampler.candidates if candidate.combo_key != stalled_candidate.combo_key
    )

    sampler.record_group_feedback(
        [
            {
                "combo_key": stalled_candidate.combo_key,
                "mean_score": 0.05,
                "success_rate": 0.0,
                "env_error_rate": 0.05,
                "mean_progress_score": 0.0,
                "near_miss_rate": 0.0,
                "format_failure_rate": 0.0,
                "accepted": True,
                "zero_std": False,
            }
            for _ in range(8)
        ]
        + [
            {
                "combo_key": progressing_candidate.combo_key,
                "mean_score": 0.05,
                "success_rate": 0.0,
                "env_error_rate": 0.05,
                "mean_progress_score": 0.5,
                "near_miss_rate": 0.25,
                "format_failure_rate": 0.0,
                "accepted": True,
                "zero_std": False,
            }
            for _ in range(8)
        ]
    )

    assert sampler.compute_dynamic_weight(progressing_candidate, phase="main") > sampler.compute_dynamic_weight(
        stalled_candidate, phase="main"
    )


def test_liveweb_sampler_records_failure_buckets_and_reuses_task_ids(monkeypatch):
    monkeypatch.setenv("TASK_REGISTRY_VERSION", "v2")
    monkeypatch.setenv("LIVEWEB_MIN_NUM_TASKS", "2")
    monkeypatch.setenv("LIVEWEB_MAX_NUM_TASKS", "4")
    monkeypatch.setenv("LIVEWEB_FAILURE_BUCKET_NORMAL_RATIO", "0.0")
    monkeypatch.setenv("LIVEWEB_FAILURE_BUCKET_WRONG_DOMAIN_RATIO", "1.0")
    monkeypatch.setenv("LIVEWEB_FAILURE_BUCKET_PREMATURE_STOP_RATIO", "0.0")
    monkeypatch.setenv("LIVEWEB_FAILURE_BUCKET_NEAR_MISS_RATIO", "0.0")

    sampler = LiveWebDynamicSampler(excluded_plugins={"weather", "openlibrary"}, min_unique_plugins=2)
    baseline = sampler.sample(seed=123, phase="main", evaluation=False)
    sampler.record_group_feedback(
        [
            {
                "combo_key": sampler.candidates[0].combo_key,
                "mean_score": 0.0,
                "success_rate": 0.0,
                "env_error_rate": 0.0,
                "accepted": True,
                "zero_std": False,
                "task_records": [
                    {
                        "task_id": baseline["task_id"],
                        "rl_failure_bucket": "wrong_domain_loop",
                        "score": 0.0,
                        "success": False,
                    }
                ],
            }
        ]
    )

    selection = sampler.sample(seed=123, phase="main", evaluation=False)
    assert selection["failure_bucket"] == "wrong_domain_loop"
    assert selection["task_id"] == baseline["task_id"]
    assert selection["sampling_strategy"] == "bucket:wrong_domain_loop"


def test_liveweb_sampler_bootstraps_failure_bucket_from_raw_results(tmp_path, monkeypatch):
    monkeypatch.setenv("TASK_REGISTRY_VERSION", "v2")
    monkeypatch.setenv("LIVEWEB_FAILURE_BUCKET_RESULTS_DIRS", str(tmp_path))
    payload = {
        "score": 0.0,
        "success": False,
        "extra": {
            "task_id": 174013,
            "rl_failure_bucket": "near_miss",
        },
    }
    (tmp_path / "task_174013.json").write_text(__import__("json").dumps(payload))

    sampler = LiveWebDynamicSampler(excluded_plugins={"weather", "openlibrary"}, min_unique_plugins=2)
    assert 174013 in list(sampler._bucket_pools["near_miss"])


def test_liveweb_phase_aliases_are_canonicalized():
    assert canonicalize_phase_name("warmup") == "bootstrap"
    assert canonicalize_phase_name("main") == "main_warm"


@pytest.mark.parametrize(
    ("phase", "allowed"),
    [
        ("bootstrap", {1, 2}),
        ("main_warm", {2, 3}),
        ("online_align", {2, 3, 4}),
    ],
)
def test_liveweb_sampler_uses_phase_num_task_weights(monkeypatch, phase, allowed):
    monkeypatch.setenv("TASK_REGISTRY_VERSION", "v2")
    monkeypatch.delenv("LIVEWEB_MIN_NUM_TASKS", raising=False)
    monkeypatch.delenv("LIVEWEB_MAX_NUM_TASKS", raising=False)
    sampler = LiveWebDynamicSampler(excluded_plugins={"weather", "openlibrary"}, min_unique_plugins=2)
    seen = {int(sampler.sample(seed=100 + idx, phase=phase, evaluation=False)["num_subtasks"]) for idx in range(12)}
    assert seen
    assert seen <= allowed
