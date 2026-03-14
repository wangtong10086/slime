from __future__ import annotations

import dataclasses
import json
import os
from typing import Any

import affinetes as af

from .base import RuntimeCapabilities


@dataclasses.dataclass
class AffinetesRuntimeConfig:
    mode: str = "docker"
    image: str | None = None
    base_url: str | None = None
    connect_only: bool = False
    container_name: str | None = None
    env_vars: dict[str, str] = dataclasses.field(default_factory=dict)
    force_recreate: bool = False
    pull: bool = False
    cleanup: bool = True
    host_network: bool = False
    host_port: int | None = None
    backend_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_env(cls, prefix: str) -> "AffinetesRuntimeConfig":
        env_vars_raw = os.getenv(f"{prefix}_ENV_VARS_JSON", "").strip()
        backend_kwargs_raw = os.getenv(f"{prefix}_BACKEND_KWARGS_JSON", "").strip()
        env_vars = json.loads(env_vars_raw) if env_vars_raw else {}
        backend_kwargs = json.loads(backend_kwargs_raw) if backend_kwargs_raw else {}
        return cls(
            mode=os.getenv(f"{prefix}_MODE", "docker"),
            image=os.getenv(f"{prefix}_IMAGE") or None,
            base_url=os.getenv(f"{prefix}_BASE_URL") or None,
            connect_only=os.getenv(f"{prefix}_CONNECT_ONLY", "0") == "1",
            container_name=os.getenv(f"{prefix}_CONTAINER_NAME") or None,
            env_vars=env_vars,
            force_recreate=os.getenv(f"{prefix}_FORCE_RECREATE", "0") == "1",
            pull=os.getenv(f"{prefix}_PULL", "0") == "1",
            cleanup=os.getenv(f"{prefix}_CLEANUP", "1") == "1",
            host_network=os.getenv(f"{prefix}_HOST_NETWORK", "0") == "1",
            host_port=int(os.getenv(f"{prefix}_HOST_PORT")) if os.getenv(f"{prefix}_HOST_PORT") else None,
            backend_kwargs=backend_kwargs,
        )


class AffinetesRuntimeHandle:
    def __init__(self, config: AffinetesRuntimeConfig, *, scope: str = "default"):
        self.config = config
        self.scope = scope
        self.env = None
        self._method_names: set[str] | None = None

    async def ensure_loaded(self):
        if self.env is not None:
            return self.env
        kwargs = dict(
            mode=self.config.mode,
            connect_only=self.config.connect_only,
            container_name=self.config.container_name,
            env_vars=self.config.env_vars,
            force_recreate=self.config.force_recreate,
            pull=self.config.pull,
            cleanup=self.config.cleanup,
            host_network=self.config.host_network,
            host_port=self.config.host_port,
            **self.config.backend_kwargs,
        )
        if self.config.mode == "url":
            kwargs["base_url"] = self.config.base_url
        else:
            kwargs["image"] = self.config.image
        self.env = af.load_env(**kwargs)
        return self.env

    def is_healthy(self) -> bool:
        return self.env is not None and bool(getattr(self.env, "is_ready", lambda: False)())

    async def list_methods(self) -> list[dict[str, Any]]:
        env = await self.ensure_loaded()
        return await env.list_methods(print_info=False)

    async def get_method_names(self) -> set[str]:
        if self._method_names is None:
            methods = await self.list_methods()
            names: set[str] = set()
            for method in methods:
                if isinstance(method, dict):
                    if "name" in method:
                        names.add(str(method["name"]))
                    elif "path" in method:
                        names.add(str(method["path"]).strip("/"))
            self._method_names = names
        return set(self._method_names)

    async def detect_capabilities(self) -> RuntimeCapabilities:
        names = await self.get_method_names()
        return RuntimeCapabilities(
            supports_evaluate=("evaluate" in names),
            supports_openenv={"reset", "step"}.issubset(names),
            supports_trajectory_export=("evaluate" in names),
            supports_offline_dataset_export=False,
        )

    async def evaluate(self, **kwargs):
        env = await self.ensure_loaded()
        return await env.evaluate(**kwargs)

    async def openenv(self):
        env = await self.ensure_loaded()
        return env.openenv()

    async def reset_scope(self):
        await self.cleanup_scope()
        return await self.ensure_loaded()

    def clone_scope(self, scope: str) -> "AffinetesRuntimeHandle":
        return AffinetesRuntimeHandle(self.config, scope=scope)

    async def cleanup_scope(self):
        await self.cleanup()

    async def cleanup(self):
        if self.env is not None:
            await self.env.cleanup()
            self.env = None
            self._method_names = None
