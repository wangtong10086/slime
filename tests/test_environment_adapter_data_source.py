from types import SimpleNamespace

from slime.rollout.env_adapter.data_source import AdapterDataSource


def test_adapter_data_source_expands_liveweb_groups():
    args = SimpleNamespace(
        environment_adapter_path="slime.env_adapters.liveweb.LiveWebEnvironmentAdapter",
        environment_name="liveweb",
        n_samples_per_prompt=4,
        save="/tmp/slime-test-save",
        load=None,
    )

    data_source = AdapterDataSource(args)
    groups = data_source.get_samples(2)

    assert len(groups) == 2
    assert all(len(group) == 4 for group in groups)
    assert all("job_spec" in sample.metadata for group in groups for sample in group)
    for group in groups:
        assert len({sample.session_id for sample in group}) == 1
