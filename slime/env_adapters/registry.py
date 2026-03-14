from __future__ import annotations

import os
from typing import Any

from slime.utils.misc import load_function

from .base import EnvironmentAdapter


_ENVIRONMENT_ADAPTER_ALIASES = {
    "liveweb": "slime.env_adapters.liveweb.LiveWebEnvironmentAdapter",
    "navworld": "slime.env_adapters.navworld.NavWorldEnvironmentAdapter",
    "game": "slime.env_adapters.game.GameEnvironmentAdapter",
    "swe-pro": "slime.env_adapters.swepro.SWEProEnvironmentAdapter",
    "swepro": "slime.env_adapters.swepro.SWEProEnvironmentAdapter",
    "lgc": "slime.env_adapters.lgc.LGCEnvironmentAdapter",
}


def resolve_environment_adapter_path(path_or_alias: str | None, env_name: str | None = None) -> str:
    candidate = path_or_alias or env_name or os.getenv("SLIME_ENVIRONMENT_NAME") or os.getenv("SLIME_ENV_ADAPTER_PATH")
    if not candidate:
        raise ValueError(
            "No environment adapter configured. Set --environment-adapter-path, --environment-name, "
            "or SLIME_ENV_ADAPTER_PATH."
        )
    return _ENVIRONMENT_ADAPTER_ALIASES.get(candidate, candidate)


def load_environment_adapter(
    path_or_alias: str | None = None,
    *,
    env_name: str | None = None,
    args: Any | None = None,
) -> EnvironmentAdapter:
    resolved = resolve_environment_adapter_path(path_or_alias, env_name=env_name)
    adapter_cls = load_function(resolved)
    adapter = adapter_cls(args=args) if args is not None else adapter_cls()
    if not isinstance(adapter, EnvironmentAdapter):
        raise TypeError(f"Loaded adapter {resolved} is not an EnvironmentAdapter: {type(adapter)}")
    return adapter
