from slime.backends.sglang_utils.memory_budget import parse_mem_fraction_overrides, resolve_engine_mem_fraction


def test_parse_mem_fraction_overrides():
    overrides = parse_mem_fraction_overrides("0:0.74, 1:0.74, 4:0.80")
    assert overrides == {0: 0.74, 1: 0.74, 4: 0.80}


def test_resolve_engine_mem_fraction_uses_most_conservative_matching_gpu():
    assert (
        resolve_engine_mem_fraction(
            default_fraction=0.80,
            base_gpu_id=0,
            num_gpus_per_engine=2,
            overrides_spec="0:0.74,1:0.76,4:0.80",
        )
        == 0.74
    )


def test_resolve_engine_mem_fraction_keeps_default_when_engine_has_no_override():
    assert (
        resolve_engine_mem_fraction(
            default_fraction=0.80,
            base_gpu_id=4,
            num_gpus_per_engine=2,
            overrides_spec="0:0.74,1:0.74",
        )
        == 0.80
    )
