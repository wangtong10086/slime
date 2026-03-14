from __future__ import annotations

from .affinetes_env import AffinetesEnvironmentAdapter


class SWEProEnvironmentAdapter(AffinetesEnvironmentAdapter):
    name = "swe-pro"
    env_var_prefix = "AFFINE_SWEPRO"
    default_task_family = "software_engineering"
