from __future__ import annotations

from .affinetes_env import AffinetesEnvironmentAdapter


class GameEnvironmentAdapter(AffinetesEnvironmentAdapter):
    name = "game"
    env_var_prefix = "AFFINE_GAME"
    default_task_family = "game"
