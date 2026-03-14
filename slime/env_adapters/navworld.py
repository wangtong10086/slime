from __future__ import annotations

from .affinetes_env import AffinetesEnvironmentAdapter


class NavWorldEnvironmentAdapter(AffinetesEnvironmentAdapter):
    name = "navworld"
    env_var_prefix = "AFFINE_NAVWORLD"
    default_task_family = "navigation"
