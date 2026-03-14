from __future__ import annotations

from .affinetes_env import AffinetesEnvironmentAdapter


class LGCEnvironmentAdapter(AffinetesEnvironmentAdapter):
    name = "lgc"
    env_var_prefix = "AFFINE_LGC"
    default_task_family = "logical_reasoning"
