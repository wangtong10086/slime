from types import SimpleNamespace

from slime.env_adapters.liveweb import LiveWebEnvironmentAdapter
from slime.env_adapters.lgc import LGCEnvironmentAdapter
from slime.env_adapters.registry import load_environment_adapter, resolve_environment_adapter_path


def test_resolve_environment_adapter_aliases():
    assert resolve_environment_adapter_path("liveweb").endswith("LiveWebEnvironmentAdapter")
    assert resolve_environment_adapter_path("lgc").endswith("LGCEnvironmentAdapter")


def test_load_environment_adapter_from_alias():
    args = SimpleNamespace()
    liveweb = load_environment_adapter("liveweb", args=args)
    lgc = load_environment_adapter("lgc", args=args)

    assert isinstance(liveweb, LiveWebEnvironmentAdapter)
    assert isinstance(lgc, LGCEnvironmentAdapter)
