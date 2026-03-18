import dataclasses
import itertools
import logging
import multiprocessing
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from slime.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig, SglangConfig
from slime.backends.sglang_utils.sglang_engine import SGLangEngine
from slime.rollout.base_types import call_rollout_fn
from slime.utils import logging_utils
from slime.utils.health_monitor import RolloutHealthMonitor
from slime.utils.http_utils import _wrap_ipv6, find_available_port, get_host_info, init_http_client
from slime.utils.logging_utils import configure_logger, init_tracking
from slime.utils.metric_utils import compute_pass_rate, compute_rollout_step, compute_statistics, dict_add_prefix
from slime.utils.misc import Box, group_by, load_function
from slime.utils.seqlen_balancing import get_seqlen_balanced_partitions
from slime.utils.types import Sample

from ..utils.metric_utils import has_repetition
from .utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, Lock
from .rollout_batching import (
    choose_dynamic_global_batch_size,
    compute_train_trim_length,
    plan_train_steps_by_token_budget,
    resolve_max_samples_per_rollout,
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def _rewrite_eval_aux_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    """Route eval-only auxiliary metrics onto eval-specific namespaces."""
    if not metrics:
        return {}

    prefix_map = {
        "env/": "eval_env/",
        "cache/": "eval_cache/",
        "runtime/": "eval_runtime/",
        "scheduler/": "eval_scheduler/",
    }

    rewritten: dict[str, Any] = {}
    for key, value in metrics.items():
        for old_prefix, new_prefix in prefix_map.items():
            if key.startswith(old_prefix):
                rewritten[f"{new_prefix}{key[len(old_prefix):]}"] = value
                break
        else:
            rewritten[key] = value
    return rewritten


@dataclasses.dataclass
class ServerGroup:
    """A group of homogeneous SGLang engines with the same configuration.

    All engines in a group share the same tp_size / nodes_per_engine / pg.
    A RolloutServer may contain multiple ServerGroups (e.g. prefill vs decode
    in PD disaggregation).
    """

    args: Any
    pg: Any  # (placement_group, reordered_bundle_indices, reordered_gpu_ids)
    all_engines: list
    num_gpus_per_engine: int
    num_new_engines: int
    worker_type: str = "regular"  # "regular", "prefill", or "decode"
    rank_offset: int = 0  # cumulative engine count before this group
    gpu_offset: int = 0  # cumulative GPU count before this group
    sglang_overrides: dict = dataclasses.field(default_factory=dict)
    needs_offload: bool = False  # True when this group's GPUs overlap with megatron
    model_path: str | None = None  # checkpoint path for update_weights_from_disk
    router_ip: str | None = None
    router_port: int | None = None

    @property
    def nodes_per_engine(self):
        return max(1, self.num_gpus_per_engine // self.args.num_gpus_per_node)

    @property
    def engines(self):
        """Node-0 engines only (for multi-node serving)."""
        return self.all_engines[:: self.nodes_per_engine]

    def start_engines(self, port_cursors: dict[int, int] | None = None) -> tuple[list, dict[int, int]]:
        """Create Ray actors, allocate ports, and fire ``engine.init()`` without waiting.

        Returns ``(init_handles, port_cursors)`` where *init_handles* is a list
        of Ray ObjectRefs and *port_cursors* maps node index → next free port.
        The caller should ``ray.get()`` on the handles to block until the
        engines are healthy, and pass *port_cursors* to the next server group
        so that different groups on the same node don't race for ports.

        Placeholder groups (worker_type="placeholder") skip engine creation entirely.
        """
        if port_cursors is None:
            port_cursors = {}
        if self.args.debug_train_only or self.worker_type == "placeholder":
            self.num_new_engines = 0
            return [], port_cursors

        num_gpu_per_engine = min(self.num_gpus_per_engine, self.args.num_gpus_per_node)

        pg, reordered_bundle_indices, reordered_gpu_ids = self.pg

        RolloutRayActor = ray.remote(SGLangEngine)

        rollout_engines = []
        for i in range(len(self.all_engines)):
            if self.all_engines[i] is not None:
                continue

            global_rank = self.rank_offset + i
            num_gpus = 0.2
            num_cpus = num_gpus

            # Get the base GPU ID from placement group using gpu_offset.
            gpu_index = self.gpu_offset + i * num_gpu_per_engine
            base_gpu_id = int(reordered_gpu_ids[gpu_index])

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=reordered_bundle_indices[gpu_index],
            )

            env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {
                key: os.environ.get(key, default_val)
                for key, default_val in {
                    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
                    "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                    "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                    "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
                    "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
                    "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
                    "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
                }.items()
            }

            rollout_engine = RolloutRayActor.options(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                runtime_env={
                    "env_vars": env_vars,
                },
            ).remote(
                self.args,
                rank=global_rank,
                worker_type=self.worker_type,
                base_gpu_id=base_gpu_id,
                sglang_overrides=self.sglang_overrides,
                num_gpus_per_engine=self.num_gpus_per_engine,
            )

            rollout_engines.append((global_rank, rollout_engine))
            self.all_engines[i] = rollout_engine

        self.num_new_engines = len(rollout_engines)

        if self.num_new_engines == 0:
            return [], port_cursors

        if self.args.rollout_external:
            addr_and_ports = _allocate_rollout_engine_addr_and_ports_external(
                args=self.args, rollout_engines=rollout_engines
            )
        else:
            # Compute base_port from the maximum cursor across all nodes that
            # this group's engines may land on (conservative: just use global max).
            base_port = max(port_cursors.values()) if port_cursors else 15000
            addr_and_ports, port_cursors = _allocate_rollout_engine_addr_and_ports_normal(
                args=self.args,
                rollout_engines=rollout_engines,
                worker_type=self.worker_type,
                num_gpus_per_engine=self.num_gpus_per_engine,
                rank_offset=self.rank_offset,
                base_port=base_port,
            )

        init_handles = [
            engine.init.remote(
                **(addr_and_ports[rank]),
                router_ip=self.router_ip,
                router_port=self.router_port,
            )
            for rank, engine in rollout_engines
        ]
        return init_handles, port_cursors

    def offload(self):
        """Fire release_memory_occupation on all engines (non-blocking).

        Returns a list of Ray ObjectRefs.  Skipped for groups that do not
        overlap with megatron GPUs (``needs_offload=False``).
        """
        if not self.needs_offload:
            return []
        return [engine.release_memory_occupation.remote() for engine in self.engines if engine is not None]

    def onload(self, tags: list[str] | None = None):
        """Fire resume_memory_occupation on all engines (non-blocking).

        Returns a list of Ray ObjectRefs.  Skipped for groups that do not
        overlap with megatron GPUs (``needs_offload=False``).
        """
        if not self.needs_offload:
            return []
        return [engine.resume_memory_occupation.remote(tags=tags) for engine in self.engines if engine is not None]

    def onload_weights_from_disk(self):
        """Reload weights from ``model_path`` for non-updatable groups.

        Used instead of ``resume_memory_occupation(tags=[WEIGHTS])`` so that
        CPU memory is not consumed by offloaded weight copies.
        """
        if not self.needs_offload or not self.model_path:
            return []
        return [
            engine.update_weights_from_disk.remote(self.model_path) for engine in self.engines if engine is not None
        ]


@dataclasses.dataclass
class RolloutServer:
    """A model served behind a shared router, with one or more server groups.

    Each RolloutServer represents one model deployed behind a single router.
    A server may contain multiple ServerGroups with different
    ``num_gpus_per_engine`` (e.g. prefill TP=2, decode TP=4).
    """

    server_groups: list[ServerGroup]
    router_ip: str | None = None
    router_port: int | None = None
    model_name: str = "default"
    update_weights: bool = True

    @property
    def engines(self):
        """All node-0 engines across all groups (placeholder groups contribute nothing)."""
        return [e for g in self.server_groups for e in g.engines]

    @property
    def all_engines(self):
        """All engines (including non-node-0) across all groups."""
        return [e for g in self.server_groups for e in g.all_engines]

    @property
    def num_new_engines(self):
        return sum(g.num_new_engines for g in self.server_groups)

    @num_new_engines.setter
    def num_new_engines(self, value):
        for g in self.server_groups:
            g.num_new_engines = value

    @property
    def engine_gpu_counts(self) -> list[int]:
        """Per-engine GPU count for all node-0 engines, parallel to ``engines``."""
        return [g.num_gpus_per_engine for g in self.server_groups for _ in g.engines]

    @property
    def engine_gpu_offsets(self) -> list[int]:
        """Per-engine GPU offset for all node-0 engines, parallel to ``engines``.

        Accounts for placeholder groups that occupy GPU slots without creating engines.
        """
        offsets = []
        for g in self.server_groups:
            for j in range(len(g.engines)):
                offsets.append(g.gpu_offset + j * g.num_gpus_per_engine)
        return offsets

    @property
    def nodes_per_engine(self):
        """Nodes per engine.  Only valid when all active groups share the same value."""
        values = {g.nodes_per_engine for g in self.server_groups}
        if len(values) != 1:
            raise ValueError(f"Heterogeneous nodes_per_engine across groups: {values}")
        return values.pop()

    def recover(self):
        """Recover dead engines across all active groups, overlapping init."""
        # Record dead indices per group before starting.
        dead_per_group = [[i for i, engine in enumerate(g.all_engines) if engine is None] for g in self.server_groups]

        # Start all groups concurrently.
        all_handles = []
        port_cursors: dict[int, int] = {}
        for g in self.server_groups:
            handles, port_cursors = g.start_engines(port_cursors)
            all_handles.extend(handles)
        if all_handles:
            ray.get(all_handles)

        # Post-recovery: offload then onload weights for newly created engines.
        release_handles = []
        updatable_new_engines = []
        non_updatable_groups_engines: list[tuple[str, list]] = []
        for g, dead_indices in zip(self.server_groups, dead_per_group, strict=True):
            logger.info(f"Recovered {g.num_new_engines} dead rollout engines (worker_type={g.worker_type})")
            assert g.num_new_engines == len(dead_indices), "num_new_engines does not match dead_indices length"
            if g.needs_offload and dead_indices:
                new_engines = [g.all_engines[i] for i in dead_indices]
                release_handles.extend(engine.release_memory_occupation.remote() for engine in new_engines)
                if self.update_weights:
                    updatable_new_engines.extend(new_engines)
                elif g.model_path:
                    non_updatable_groups_engines.append((g.model_path, new_engines))

        if release_handles:
            ray.get(release_handles)
            # Resume GPU memory for all engines that need offload.
            all_resume_engines = updatable_new_engines[:]
            for _model_path, engines in non_updatable_groups_engines:
                all_resume_engines.extend(engines)
            if all_resume_engines:
                ray.get(
                    [
                        engine.resume_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS])
                        for engine in all_resume_engines
                    ]
                )

    def offload(self):
        """Release memory occupation across all groups (concurrent)."""
        handles = self.offload_async()
        return ray.get(handles) if handles else []

    def offload_async(self):
        """Release memory occupation across all groups without waiting."""
        handles = []
        for g in self.server_groups:
            handles.extend(g.offload())
        return handles

    def onload(self, tags: list[str] | None = None):
        """Resume memory occupation across all groups (concurrent)."""
        handles = []
        for g in self.server_groups:
            handles.extend(g.onload(tags))
        return ray.get(handles) if handles else []

    def onload_weights(self):
        """Restore weights for offloaded groups.

        All groups resume from CPU cache via ``resume_memory_occupation``.
        For updatable servers, weights will be overwritten by
        ``update_weights`` shortly after.  For non-updatable servers the
        CPU backup already contains the correct (unchanged) weights.
        """
        handles = []
        for g in self.server_groups:
            if not g.needs_offload:
                continue
            handles.extend(g.onload(tags=[GPU_MEMORY_TYPE_WEIGHTS]))
        return ray.get(handles) if handles else []

    def onload_kv(self):
        """Resume KV cache and CUDA graphs for offloaded groups."""
        handles = []
        for g in self.server_groups:
            handles.extend(g.onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH]))
        return ray.get(handles) if handles else []


@ray.remote
class RolloutManager:
    """The class to run rollout and convert rollout data to training data."""

    def __init__(self, args, pg):
        configure_logger()

        self.pg = pg
        self.args = args

        init_tracking(args, primary=False)

        data_source_cls = load_function(self.args.data_source_path)
        self.data_source = data_source_cls(args)

        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)
        self.custom_reward_post_process_func = None
        if self.args.custom_reward_post_process_path is not None:
            self.custom_reward_post_process_func = load_function(self.args.custom_reward_post_process_path)
        self.custom_convert_samples_to_train_data_func = None
        if self.args.custom_convert_samples_to_train_data_path is not None:
            self.custom_convert_samples_to_train_data_func = load_function(
                self.args.custom_convert_samples_to_train_data_path
            )
        logger.info(f"import {self.args.rollout_function_path} as generate_rollout function.")
        logger.info(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

        if self.args.debug_train_only:
            self.servers: dict[str, RolloutServer] = {}
        else:
            init_http_client(args)
            self.servers = start_rollout_servers(args, pg)
        self.rollout_engine_lock = Lock.options(num_cpus=1, num_gpus=0).remote()
        self.rollout_id = -1
        self._save_quiesced = False
        self._full_save_engine_handles: list[Any] = []

        self._health_monitors = []
        if not self.args.debug_train_only and self.args.use_fault_tolerance:
            for srv in self.servers.values():
                for group in srv.server_groups:
                    monitor = RolloutHealthMonitor(group, args)
                    monitor.start()
                    self._health_monitors.append(monitor)
            self._ci_fault_injection_pending = self.args.ci_test  # Flag for CI fault injection
        self._full_save_quiesced = False

    def _try_ci_fault_injection(self):
        """Try to inject fault during generate (when health monitor is running)."""
        if not self._ci_fault_injection_pending:
            return

        # Only inject fault once
        self._ci_fault_injection_pending = False

        if self.server and self.server.server_groups[0].all_engines and self.server.server_groups[0].all_engines[0]:
            logger.info("CI Fault Injection: Simulating crash on engine 0 during generate")
            try:
                # This will cause the ray actor to exit
                self.server.server_groups[0].all_engines[0].simulate_crash.remote()
                # Wait for health monitor to detect the crash and mark engine as None
                # health_check_interval + health_check_timeout + buffer
                wait_time = self.args.rollout_health_check_interval + self.args.rollout_health_check_timeout + 5
                logger.info(f"CI Fault Injection: Waiting {wait_time}s for health monitor to detect crash")
                time.sleep(wait_time)
            except Exception as e:
                logger.warning(f"CI Fault Injection failed: {e}")

    def dispose(self):
        for monitor in self._health_monitors:
            monitor.stop()
        logging_utils.finish_tracking(self.args)

    @property
    def server(self) -> RolloutServer | None:
        """Default server (first model).  For backward compatibility."""
        if not self.servers:
            return None
        return next(iter(self.servers.values()))

    def _get_updatable_server(self) -> RolloutServer | None:
        """Return the server with ``update_weights=True``.

        When multiple updatable servers exist, returns the first one
        (multi-model weight update is not yet supported).
        """
        for srv in self.servers.values():
            if srv.update_weights:
                return srv
        return None

    @property
    def rollout_engines(self):
        """All node-0 engines across all servers / models."""
        return [e for srv in self.servers.values() for e in srv.engines]

    def get_updatable_engines_and_lock(self):
        """Return engines eligible for weight updates.

        Returns engines from the first model that has
        ``update_weights=True``.  Frozen models (reference, reward,
        etc.) are automatically excluded.
        """
        srv = self._get_updatable_server()
        engines = srv.engines if srv else []
        gpu_counts = srv.engine_gpu_counts if srv else []
        gpu_offsets = srv.engine_gpu_offsets if srv else []
        num_new = srv.num_new_engines if srv else 0
        return engines, self.rollout_engine_lock, num_new, gpu_counts, gpu_offsets

    def get_num_rollout_per_epoch(self):
        assert self.args.rollout_global_dataset
        return len(self.data_source) // self.args.rollout_batch_size

    def generate(self, rollout_id):
        start_time = time.time()
        self.rollout_id = rollout_id
        self.health_monitoring_resume()
        if self.args.ci_test and self.args.use_fault_tolerance and rollout_id >= 2:
            self._try_ci_fault_injection()
        data, metrics = self._get_rollout_data(rollout_id=rollout_id)
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=False)
        _log_rollout_data(rollout_id, self.args, data, metrics, time.time() - start_time)
        if self.args.debug_rollout_only:
            # if debug rollout only, we don't convert samples to train data and directly return
            return
        data = self._convert_samples_to_train_data(data)
        return self._split_train_data_by_dp(data, self.train_parallel_config["dp_size"])

    def eval(self, rollout_id):
        if self.args.debug_train_only:
            # if debug train only, we don't generate evaluation data
            return
        self.health_monitoring_resume()

        result = call_rollout_fn(self.eval_generate_rollout, self.args, rollout_id, self.data_source, evaluation=True)
        data = result.data
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=True)
        _log_eval_rollout_data(rollout_id, self.args, data, result.metrics)

    def save(self, rollout_id, save_root=None):
        self.data_source.save(rollout_id, save_root=save_root)

    def _count_live_engines(self, engine_handles: list[Any], timeout_s: float = 3.0) -> int:
        if not engine_handles:
            return 0

        refs = []
        for engine in engine_handles:
            try:
                refs.append(engine.ping.remote())
            except Exception:
                continue
        if not refs:
            return 0
        ready, remaining = ray.wait(refs, num_returns=len(refs), timeout=timeout_s)
        live = 0
        for ref in ready:
            try:
                ray.get(ref)
                live += 1
            except Exception:
                continue
        return live + len(remaining)

    def get_save_status(self):
        live_sglang_actor_count = 0
        if self._full_save_quiesced:
            live_sglang_actor_count = self._count_live_engines(self._full_save_engine_handles, timeout_s=1.0)
        else:
            live_sglang_actor_count = len([engine for engine in self.rollout_engines if engine is not None])
        return {
            "save_quiesced": self._save_quiesced or self._full_save_quiesced,
            "full_save_quiesced": self._full_save_quiesced,
            "live_sglang_actor_count": live_sglang_actor_count,
            "live_rollout_actor_count": live_sglang_actor_count,
            "num_servers": len(self.servers),
        }

    def begin_save_quiesce(self):
        self.health_monitoring_pause()
        timeout_s = float(os.environ.get("LIVEWEB_SAVE_QUIESCE_TIMEOUT_SECONDS", "60"))
        handles = []
        for srv in self.servers.values():
            handles.extend(srv.offload_async())
        if handles:
            try:
                ray.get(handles, timeout=timeout_s)
            except Exception as exc:
                logger.warning(
                    "save/quiesce offload timed out after %.1fs; proceeding with best-effort quiesce (%s)",
                    timeout_s,
                    exc,
                )
        self._save_quiesced = True
        status = self.get_save_status()
        status.update({"save_mode": "archive", "num_engines": len(self.rollout_engines)})
        return status

    def end_save_quiesce(self):
        if self._save_quiesced:
            self.health_monitoring_resume()
        self._save_quiesced = False
        status = self.get_save_status()
        status.update({"save_mode": "archive", "num_engines": len(self.rollout_engines)})
        return status

    def begin_full_save_quiesce(self):
        self.health_monitoring_pause()
        for monitor in self._health_monitors:
            monitor.stop()
        self._health_monitors = []
        self._full_save_engine_handles = [engine for engine in self.rollout_engines if engine is not None]
        timeout_s = float(os.environ.get("LIVEWEB_FULL_SAVE_SHUTDOWN_TIMEOUT_SECONDS", "180"))
        handles = [engine.shutdown.remote() for engine in self._full_save_engine_handles]
        if handles:
            try:
                ray.get(handles, timeout=timeout_s)
            except Exception as exc:
                logger.warning(
                    "save/full_quiesce shutdown timed out after %.1fs; continuing with servers dropped (%s)",
                    timeout_s,
                    exc,
                )
        live_after_shutdown = self._count_live_engines(self._full_save_engine_handles, timeout_s=2.0)
        if live_after_shutdown > 0:
            logger.warning("save/full_quiesce found %s live rollout engines after shutdown; force-killing", live_after_shutdown)
            for engine in self._full_save_engine_handles:
                try:
                    ray.kill(engine, no_restart=True)
                except Exception:
                    continue
        live_after_kill = self._count_live_engines(self._full_save_engine_handles, timeout_s=2.0)
        self.servers = {}
        self._full_save_quiesced = True
        return {
            "save_quiesced": True,
            "save_mode": "full",
            "num_servers": 0,
            "num_engines": 0,
            "live_sglang_actor_count": live_after_kill,
            "live_rollout_actor_count": live_after_kill,
        }

    def end_full_save_quiesce(self):
        if self._full_save_quiesced:
            self.servers = start_rollout_servers(self.args, self.pg)
            if self.args.use_fault_tolerance:
                for srv in self.servers.values():
                    for group in srv.server_groups:
                        monitor = RolloutHealthMonitor(group, self.args)
                        monitor.start()
                        self._health_monitors.append(monitor)
            self._full_save_quiesced = False
            self._full_save_engine_handles = []
        return {
            "save_quiesced": False,
            "save_mode": "full",
            "num_servers": len(self.servers),
            "num_engines": len(self.rollout_engines),
            "live_sglang_actor_count": len(self.rollout_engines),
            "live_rollout_actor_count": len(self.rollout_engines),
        }

    def load(self, rollout_id=None):
        self.data_source.load(rollout_id)

    def offload(self):
        self.health_monitoring_pause()
        for srv in self.servers.values():
            srv.offload()

    def onload(self, tags: list[str] | None = None):
        for srv in self.servers.values():
            srv.onload(tags)

    def onload_weights(self):
        for srv in self.servers.values():
            srv.onload_weights()

    def onload_kv(self):
        for srv in self.servers.values():
            srv.onload_kv()

    def recover_updatable_engines(self):
        """Restart any dead rollout engines and update num_new_engines for update_weights detection.

        Recovers the updatable model (the one that receives weight
        updates from training).
        """
        self.health_monitoring_pause()
        srv = self._get_updatable_server()
        if self.rollout_id == -1 or srv is None:
            engines = srv.engines if srv else []
            gpu_counts = srv.engine_gpu_counts if srv else []
            gpu_offsets = srv.engine_gpu_offsets if srv else []
            return engines, self.rollout_engine_lock, (srv.num_new_engines if srv else 0), gpu_counts, gpu_offsets

        srv.recover()
        return (
            srv.engines,
            self.rollout_engine_lock,
            srv.num_new_engines,
            srv.engine_gpu_counts,
            srv.engine_gpu_offsets,
        )

    def clear_updatable_num_new_engines(self):
        # when fault tolerance is not enabled, we need to manually clear num_new_engines after update_weights
        srv = self._get_updatable_server()
        if srv:
            srv.num_new_engines = 0

    def health_monitoring_pause(self) -> None:
        for monitor in self._health_monitors:
            monitor.pause()

    def health_monitoring_resume(self) -> None:
        for monitor in self._health_monitors:
            monitor.resume()

    def check_weights(self, action: str):
        return ray.get([engine.check_weights.remote(action=action) for engine in self.rollout_engines])

    def _get_rollout_data(self, rollout_id):
        self._train_step_boundaries = None
        self._train_step_token_counts = None
        self._train_step_num_samples = None
        self._train_step_long_sample_counts = None
        self._train_step_token_budget = None
        self._train_step_logit_budget = None
        self._train_step_logit_counts = None
        self._train_oversize_samples_dropped = 0
        self._train_oversize_logit_samples_dropped = 0
        self._train_underfilled_steps = 0
        self._train_long_samples_trimmed = 0
        self._train_density_restricted_steps = 0
        self._train_windowed_samples = 0
        self._train_windowed_token_trim = 0
        if self.args.load_debug_rollout_data:
            data = torch.load(
                self.args.load_debug_rollout_data.format(rollout_id=rollout_id),
                weights_only=False,
            )["samples"]
            data = [Sample.from_dict(sample) for sample in data]
            if (ratio := self.args.load_debug_rollout_data_subsample) is not None:
                original_num_rows = len(data)
                rough_subsample_num_rows = int(original_num_rows * ratio)
                data = data[: rough_subsample_num_rows // 2] + data[-rough_subsample_num_rows // 2 :]
                logger.info(
                    f"Subsample loaded debug rollout data using {ratio=} and change num rows {original_num_rows} -> {len(data)}"
                )
            metrics = None
        else:
            data = call_rollout_fn(self.generate_rollout, self.args, rollout_id, self.data_source, evaluation=False)
            metrics = data.metrics
            data = data.samples
            if not data:
                raise RuntimeError(
                    "Rollout returned zero samples before training trim; "
                    f"rollout_id={rollout_id}; metrics={metrics}"
                )
            # flatten the data if it is a list of lists
            while isinstance(data[0], list):
                data = list(itertools.chain.from_iterable(data))
                if not data:
                    raise RuntimeError(
                        "Rollout returned zero samples after flattening train groups; "
                        f"rollout_id={rollout_id}; metrics={metrics}"
                    )

            if not self.args.disable_rollout_trim_samples and not self.args.debug_rollout_only:
                requested_samples = len(data)
                global_batch_size = self.args.global_batch_size
                use_dynamic_global_batch_size = bool(
                    getattr(self.args, "use_dynamic_global_batch_size", False)
                    or getattr(self.args, "use_dynamic_batch_size", False)
                )
                self._train_step_boundaries = None
                self._train_step_token_counts = None
                self._train_step_num_samples = None
                self._train_step_long_sample_counts = None
                self._train_step_token_budget = None
                self._train_step_logit_budget = None
                self._train_step_logit_counts = None
                self._train_oversize_samples_dropped = 0
                self._train_oversize_logit_samples_dropped = 0
                self._train_underfilled_steps = 0
                self._train_long_samples_trimmed = 0
                self._train_density_restricted_steps = 0
                self._train_windowed_samples = 0
                self._train_windowed_token_trim = 0
                if use_dynamic_global_batch_size:
                    logger.info(f"Collected {len(data)} samples from rollout to train with dynamic global batch size")
                    self._dynamic_global_batch_size = self._compute_dynamic_global_batch_size(len(data))
                    global_batch_size = self._dynamic_global_batch_size

                token_budget = int(os.environ.get("TRAIN_STEP_TOKEN_BUDGET", "0") or "0")
                logit_budget = int(os.environ.get("TRAIN_STEP_LOGIT_BUDGET", "0") or "0")
                underfilled_min_samples = int(os.environ.get("TRAIN_UNDERFILLED_STEP_MIN_SAMPLES", "4") or "4")
                packing_strategy = os.environ.get("TRAIN_STEP_PACKING_STRATEGY", "greedy_desc")
                long_sample_threshold = int(
                    os.environ.get("TRAIN_STEP_LONG_SAMPLE_THRESHOLD", str(max(1, int(token_budget * 0.35)))) or "0"
                )
                max_long_samples_per_step = int(
                    os.environ.get("TRAIN_MAX_LONG_SAMPLES_PER_STEP", "2") or "2"
                )
                max_single_sample_tokens = int(os.environ.get("TRAIN_MAX_SINGLE_SAMPLE_TOKENS", "12000") or "12000")
                max_total_tokens_per_sample = int(
                    os.environ.get("TRAIN_MAX_TOTAL_TOKENS_PER_SAMPLE", "4096") or "4096"
                )
                max_response_tokens_per_sample = int(
                    os.environ.get("TRAIN_MAX_RESPONSE_TOKENS_PER_SAMPLE", "2048") or "2048"
                )
                dp_size = self.train_parallel_config["dp_size"]

                if token_budget > 0 and dp_size == 1:
                    configured_min = int(
                        os.environ.get(
                            "TRAIN_MIN_DYNAMIC_GLOBAL_BATCH_SIZE",
                            str(max(dp_size, global_batch_size // 2)),
                        )
                        or "0"
                    )
                    max_samples_per_rollout = resolve_max_samples_per_rollout(global_batch_size)
                    step_plan = plan_train_steps_by_token_budget(
                        data,
                        max_samples_per_step=global_batch_size,
                        min_samples_per_step=configured_min,
                        underfilled_min_samples=underfilled_min_samples,
                        step_token_budget=token_budget,
                        max_samples_per_rollout=max_samples_per_rollout,
                        packing_strategy=packing_strategy,
                        long_sample_threshold=long_sample_threshold,
                        max_long_samples_per_step=max_long_samples_per_step,
                        max_single_sample_tokens_per_step=max_single_sample_tokens,
                        step_logit_budget=logit_budget,
                        max_single_sample_logit_tokens=max_response_tokens_per_sample,
                        max_total_tokens_per_sample=max_total_tokens_per_sample,
                        max_response_tokens_per_sample=max_response_tokens_per_sample,
                    )
                    if not step_plan.retained_indices:
                        raise ValueError(
                            "Token-budget planning retained no trainable samples "
                            f"(num_samples={len(data)}, token_budget={token_budget})"
                        )
                    data = [data[index] for index in step_plan.retained_indices]
                    self._dynamic_global_batch_size = step_plan.dynamic_global_batch_size or global_batch_size
                    self._train_step_boundaries = step_plan.step_boundaries
                    self._train_step_token_counts = step_plan.step_token_counts
                    self._train_step_num_samples = step_plan.step_num_samples
                    self._train_step_long_sample_counts = step_plan.step_long_sample_counts
                    self._train_step_token_budget = token_budget
                    self._train_step_logit_budget = logit_budget
                    self._train_step_logit_counts = step_plan.step_logit_counts
                    self._train_oversize_samples_dropped = step_plan.oversize_samples_dropped
                    self._train_oversize_logit_samples_dropped = step_plan.oversize_logit_samples_dropped
                    self._train_underfilled_steps = step_plan.underfilled_steps
                    self._train_long_samples_trimmed = step_plan.long_samples_trimmed
                    self._train_density_restricted_steps = step_plan.density_restricted_steps
                    if metrics is None:
                        metrics = {}
                    metrics.update(
                        {
                            "scheduler/runtime_requested_tokens": float(step_plan.requested_tokens),
                            "scheduler/runtime_retained_tokens": float(step_plan.retained_tokens),
                            "scheduler/runtime_trimmed_tokens": float(step_plan.trimmed_tokens),
                            "scheduler/runtime_requested_logit_tokens": float(step_plan.requested_logit_tokens),
                            "scheduler/runtime_retained_logit_tokens": float(step_plan.retained_logit_tokens),
                            "scheduler/runtime_trimmed_logit_tokens": float(step_plan.trimmed_logit_tokens),
                            "train/step_token_budget": float(token_budget),
                            "train/step_logit_budget": float(logit_budget),
                            "train/actual_step_token_mean": (
                                float(np.mean(step_plan.step_token_counts)) if step_plan.step_token_counts else 0.0
                            ),
                            "train/actual_step_token_max": float(max(step_plan.step_token_counts, default=0)),
                            "train/actual_step_logit_mean": (
                                float(np.mean(step_plan.step_logit_counts)) if step_plan.step_logit_counts else 0.0
                            ),
                            "train/actual_step_logit_max": float(max(step_plan.step_logit_counts, default=0)),
                            "train/step_long_sample_mean": (
                                float(np.mean(step_plan.step_long_sample_counts))
                                if step_plan.step_long_sample_counts
                                else 0.0
                            ),
                            "train/step_long_sample_max": float(max(step_plan.step_long_sample_counts, default=0)),
                            "train/underfilled_steps": float(step_plan.underfilled_steps),
                            "train/oversize_samples_dropped": float(step_plan.oversize_samples_dropped),
                            "train/oversize_logit_samples_dropped": float(step_plan.oversize_logit_samples_dropped),
                            "train/long_samples_trimmed": float(step_plan.long_samples_trimmed),
                            "train/density_restricted_steps": float(step_plan.density_restricted_steps),
                        }
                    )
                    logger.info(
                        "Planned train steps by token budget: "
                        f"samples={requested_samples}, retained={len(step_plan.retained_indices)}, "
                        f"steps={len(step_plan.step_num_samples)}, "
                        f"max_step_tokens={max(step_plan.step_token_counts, default=0)}, "
                        f"mean_step_tokens={np.mean(step_plan.step_token_counts) if step_plan.step_token_counts else 0:.2f}, "
                        f"max_step_logit_tokens={max(step_plan.step_logit_counts, default=0)}, "
                        f"step_sizes={step_plan.step_num_samples}, "
                        f"long_step_counts={step_plan.step_long_sample_counts}"
                    )
                else:
                    if token_budget > 0 and dp_size > 1:
                        logger.warning(
                            "TRAIN_STEP_TOKEN_BUDGET is set, but dp_size=%s > 1. "
                            "Falling back to legacy sample-count trimming for this rollout.",
                            dp_size,
                        )
                    max_samples_per_rollout = resolve_max_samples_per_rollout(global_batch_size)
                    trim_len = compute_train_trim_length(len(data), global_batch_size, max_samples_per_rollout)
                    if trim_len != len(data):
                        if trim_len == 0:
                            raise ValueError(f"Not enough samples {len(data)} for global_batch_size {global_batch_size}")
                        origin_data_length = len(data)
                        data = data[:trim_len]
                        logger.info(
                            f"trim number of samples from {origin_data_length} to {trim_len} "
                            f"(global_batch_size={global_batch_size}, train_steps={trim_len // global_batch_size})"
                        )
                    logger.info(
                        f"Final collected {len(data)} samples from rollout to train "
                        f"(global_batch_size={global_batch_size}, train_steps={len(data) // global_batch_size})"
                    )

        return data, metrics

    def _compute_dynamic_global_batch_size(self, num_samples: int) -> int:
        """Choose a train step size that fits dynamic batch constraints.

        The returned value is the per-step ``global_batch_size``. A rollout may
        still contain multiple train steps when enough full samples remain after
        trimming.
        """
        dp_size = self.train_parallel_config["dp_size"]
        original_gbs = self.args.global_batch_size

        configured_cap = int(os.environ.get("TRAIN_DYNAMIC_GLOBAL_BATCH_SIZE_CAP", "0") or "0")
        configured_min = int(
            os.environ.get("TRAIN_MIN_DYNAMIC_GLOBAL_BATCH_SIZE", str(max(dp_size, original_gbs // 2))) or "0"
        )
        dynamic_gbs = choose_dynamic_global_batch_size(
            num_samples=num_samples,
            dp_size=dp_size,
            original_gbs=original_gbs,
            configured_cap=configured_cap,
            configured_min=configured_min,
        )

        if dynamic_gbs == 0:
            # Too few samples, use at least dp_size
            dynamic_gbs = dp_size
            logger.warning(f"num_samples={num_samples} < dp_size={dp_size}, using dp_size as global_batch_size")

        token_budget = int(os.environ.get("TRAIN_STEP_TOKEN_BUDGET", "0") or "0")
        if token_budget > 0 and dp_size == 1:
            if dynamic_gbs != original_gbs:
                logger.info(
                    f"Dynamic global_batch_size upper bound: {original_gbs} -> {dynamic_gbs} "
                    f"(num_samples={num_samples}, token_budget={token_budget})"
                )
            return dynamic_gbs

        max_samples_per_rollout = resolve_max_samples_per_rollout(dynamic_gbs)
        retained = compute_train_trim_length(num_samples, dynamic_gbs, max_samples_per_rollout)
        train_steps = retained // dynamic_gbs if retained > 0 else 0
        wasted = num_samples - retained

        if dynamic_gbs != original_gbs or wasted > 0 or train_steps > 1:
            logger.info(
                f"Dynamic global_batch_size: {original_gbs} -> {dynamic_gbs} "
                f"(num_samples={num_samples}, dp_size={dp_size}, train_steps={train_steps}, wasted={wasted})"
            )

        return dynamic_gbs

    def _save_debug_rollout_data(self, data, rollout_id, evaluation: bool):
        # TODO to be refactored (originally Buffer._set_data)
        if (path_template := self.args.save_debug_rollout_data) is not None:
            path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
            logger.info(f"Save debug rollout data to {path}")
            path.parent.mkdir(parents=True, exist_ok=True)

            # TODO may improve the format
            if evaluation:
                dump_data = dict(
                    samples=[sample.to_dict() for dataset_name, info in data.items() for sample in info["samples"]]
                )
            else:
                dump_data = dict(
                    samples=[sample.to_dict() for sample in data],
                )

            torch.save(dict(rollout_id=rollout_id, **dump_data), path)

    def _post_process_rewards(self, samples: list[Sample] | list[list[Sample]]):
        if self.custom_reward_post_process_func is not None:
            return self.custom_reward_post_process_func(self.args, samples)

        raw_rewards = [sample.get_reward_value(self.args) for sample in samples]
        if (
            self.args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
            and self.args.rewards_normalization
        ):
            # group norm
            rewards = torch.tensor(raw_rewards, dtype=torch.float)
            if rewards.shape[-1] == self.args.n_samples_per_prompt * self.args.rollout_batch_size:
                rewards = rewards.reshape(-1, self.args.n_samples_per_prompt)
            else:
                # when samples count are not equal in each group
                rewards = rewards.view(-1, rewards.shape[-1])
            mean = rewards.mean(dim=-1, keepdim=True)
            rewards = rewards - mean

            if self.args.advantage_estimator in ["grpo", "gspo"] and self.args.grpo_std_normalization:
                std = rewards.std(dim=-1, keepdim=True)
                rewards = rewards / (std + 1e-6)

            return raw_rewards, rewards.flatten().tolist()

        return raw_rewards, raw_rewards

    def _convert_samples_to_train_data(self, samples: list[Sample] | list[list[Sample]]):
        """
        Convert inference generated samples to training data.
        """
        if self.custom_convert_samples_to_train_data_func is not None:
            return self.custom_convert_samples_to_train_data_func(self.args, samples)

        raw_rewards, rewards = self._post_process_rewards(samples)

        assert len(raw_rewards) == len(samples)
        assert len(rewards) == len(samples)

        max_total_tokens_per_sample = int(os.environ.get("TRAIN_MAX_TOTAL_TOKENS_PER_SAMPLE", "0") or "0")
        max_response_tokens_per_sample = int(os.environ.get("TRAIN_MAX_RESPONSE_TOKENS_PER_SAMPLE", "0") or "0")
        window_policy = os.environ.get("TRAIN_SAMPLE_WINDOW_POLICY", "tail_response")

        def _window_sample(sample: Sample) -> tuple[list[int], int, list[int], int]:
            tokens = sample.tokens
            response_length = sample.response_length
            loss_mask = sample.loss_mask if sample.loss_mask is not None else [1] * response_length

            if not max_total_tokens_per_sample and not max_response_tokens_per_sample:
                return tokens, response_length, loss_mask, 0

            prompt_length = max(0, len(tokens) - response_length)
            keep_response = response_length
            if max_response_tokens_per_sample > 0:
                keep_response = min(keep_response, max_response_tokens_per_sample)

            if window_policy != "tail_response":
                raise ValueError(f"Unsupported TRAIN_SAMPLE_WINDOW_POLICY={window_policy}")

            prompt_budget = max_total_tokens_per_sample - keep_response if max_total_tokens_per_sample > 0 else prompt_length
            prompt_budget = max(prompt_budget, 0)
            keep_prompt = min(prompt_length, prompt_budget)

            kept_tokens = tokens[prompt_length - keep_prompt : prompt_length] + tokens[-keep_response:]
            kept_loss_mask = loss_mask[-keep_response:]
            trimmed = len(tokens) - len(kept_tokens)
            return kept_tokens, keep_response, kept_loss_mask, trimmed

        train_tokens = []
        train_response_lengths = []
        windowed_loss_masks = []
        windowed_samples = 0
        windowed_token_trim = 0

        for sample in samples:
            kept_tokens, kept_response_length, kept_loss_mask, trimmed = _window_sample(sample)
            train_tokens.append(kept_tokens)
            train_response_lengths.append(kept_response_length)
            windowed_loss_masks.append(kept_loss_mask)
            if trimmed > 0:
                windowed_samples += 1
                windowed_token_trim += trimmed

        self._train_windowed_samples = windowed_samples
        self._train_windowed_token_trim = windowed_token_trim

        train_data = {
            "tokens": train_tokens,
            "response_lengths": train_response_lengths,
            # some reward model, e.g. remote rm, may return multiple rewards,
            # we could use key to select the reward.
            "rewards": rewards,
            "raw_reward": raw_rewards,
            "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
            "sample_indices": [sample.index for sample in samples],
        }

        # loss mask
        # TODO: compress the loss mask
        loss_masks = []
        for sample, kept_loss_mask, kept_response_length in zip(
            samples, windowed_loss_masks, train_response_lengths, strict=True
        ):
            assert len(kept_loss_mask) == kept_response_length, (
                f"loss mask length {len(kept_loss_mask)} != response length {kept_response_length}"
            )
            if sample.remove_sample:
                kept_loss_mask = [0] * kept_response_length
            loss_masks.append(kept_loss_mask)
        train_data["loss_masks"] = loss_masks

        # overwriting the raw reward
        if samples[0].metadata and "raw_reward" in samples[0].metadata:
            train_data["raw_reward"] = [sample.metadata["raw_reward"] for sample in samples]

        # For rollout buffer
        if samples[0].metadata and "round_number" in samples[0].metadata:
            train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

        # Add rollout log probabilities for off-policy correction
        if samples[0].rollout_log_probs is not None:
            train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

        if samples[0].rollout_routed_experts is not None:
            train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]

        if samples[0].train_metadata is not None:
            train_data["metadata"] = [sample.train_metadata for sample in samples]

        if any(sample.multimodal_train_inputs is not None for sample in samples):
            train_data["multimodal_train_inputs"] = [sample.multimodal_train_inputs for sample in samples]

        if samples[0].teacher_log_probs is not None:
            train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]

        return train_data

    def set_train_parallel_config(self, config: dict):
        self.train_parallel_config = config

    def _split_train_data_by_dp(self, data, dp_size):
        """Split the train data by data parallel size."""
        rollout_data = {}

        if "prompt" in data:
            rollout_data["prompt"] = data["prompt"]

        total_lengths = [len(t) for t in data["tokens"]]
        data["total_lengths"] = total_lengths

        if self.args.balance_data:
            partitions = get_seqlen_balanced_partitions(total_lengths, dp_size, equal_size=True)
        else:
            partitions = [range(i, len(total_lengths), dp_size) for i in range(dp_size)]

        rollout_data_refs = []

        for i in range(dp_size):
            rollout_data = {}
            partition = partitions[i]
            rollout_data["partition"] = partition
            for key in [
                "tokens",
                "multimodal_train_inputs",
                "response_lengths",
                "rewards",
                "truncated",
                "loss_masks",
                "round_number",
                "sample_indices",
                "rollout_log_probs",
                "rollout_routed_experts",
                "prompt",
                "teacher_log_probs",
            ]:
                if key not in data:
                    continue
                val = [data[key][j] for j in partition]
                rollout_data[key] = val
            # keys that need to be splited at train side
            for key in [
                "raw_reward",
                "total_lengths",
            ]:
                if key not in data:
                    continue
                rollout_data[key] = data[key]
            # Pass dynamic global_batch_size to training side
            if hasattr(self, "_dynamic_global_batch_size"):
                rollout_data["dynamic_global_batch_size"] = self._dynamic_global_batch_size
            if getattr(self, "_train_step_boundaries", None) is not None and dp_size == 1:
                rollout_data["train_step_boundaries"] = list(self._train_step_boundaries)
                rollout_data["train_step_token_counts"] = list(self._train_step_token_counts or [])
                rollout_data["train_step_num_samples"] = list(self._train_step_num_samples or [])
                rollout_data["train_step_long_sample_counts"] = list(self._train_step_long_sample_counts or [])
                rollout_data["train_step_token_budget"] = self._train_step_token_budget
                rollout_data["train_step_logit_budget"] = self._train_step_logit_budget
                rollout_data["train_step_logit_counts"] = list(self._train_step_logit_counts or [])
                rollout_data["train_oversize_samples_dropped"] = self._train_oversize_samples_dropped
                rollout_data["train_oversize_logit_samples_dropped"] = self._train_oversize_logit_samples_dropped
                rollout_data["train_underfilled_steps"] = self._train_underfilled_steps
                rollout_data["train_long_samples_trimmed"] = self._train_long_samples_trimmed
                rollout_data["train_density_restricted_steps"] = self._train_density_restricted_steps
                rollout_data["train_windowed_samples"] = self._train_windowed_samples
                rollout_data["train_windowed_token_trim"] = self._train_windowed_token_trim
            rollout_data_refs.append(Box(ray.put(rollout_data)))
        return rollout_data_refs


def _allocate_rollout_engine_addr_and_ports_external(args, rollout_engines):
    addr_and_ports = {}
    for rank, _ in rollout_engines:
        addr = args.rollout_external_engine_addrs[rank]
        [host, port] = addr.split(":")
        addr_and_ports[rank] = dict(
            dist_init_addr=addr,
            nccl_port=None,
            host=host,
            port=int(port),
        )
    return addr_and_ports


def _allocate_rollout_engine_addr_and_ports_normal(
    *,
    args,
    rollout_engines,
    worker_type="regular",
    num_gpus_per_engine=None,
    rank_offset=0,
    base_port=15000,
):
    # get ports
    # there are 4 ports we need to allocate
    # 1. server port
    # 2. nccl port
    # 3. dist_init_addr port
    # 4. other ports for dp_attention, which is of size 4 + dp_size
    _gpus_per_engine = num_gpus_per_engine or args.rollout_num_gpus_per_engine
    num_engines_per_node = max(1, args.num_gpus_per_node // _gpus_per_engine)
    addr_and_ports: dict[int, dict] = {}

    # Track per-node port cursors so that different server groups (called
    # sequentially) never race for the same ports on a given node.
    node_port_cursor: dict[int, int] = {}

    visited_nodes = set()
    for rank, engine in rollout_engines:
        local_rank = rank - rank_offset
        node_index = local_rank // num_engines_per_node
        if node_index in visited_nodes:
            continue
        visited_nodes.add(node_index)
        # TODO: currently when restarting engines, we will set port for all engines on this node starting with this rank.
        # e.g. for 8 gpus, if we are restarting engine on gpu 3, we will set port for engine 3,4,5,6,7 on this node.
        num_engines_on_this_node = num_engines_per_node - (local_rank % num_engines_per_node)

        def get_addr_and_ports(engine, node_idx):
            # use small ports to prevent ephemeral port between 32768 and 65536.
            # also, ray uses port 10002-19999, thus we avoid near-10002 to avoid racing condition
            start_port = node_port_cursor.get(node_idx, base_port)

            def port(consecutive=1):
                nonlocal start_port
                _, port = ray.get(
                    engine._get_current_node_ip_and_free_port.remote(
                        start_port=start_port,
                        consecutive=consecutive,
                    )
                )
                start_port = port + consecutive
                node_port_cursor[node_idx] = start_port
                return port

            def addr():
                addr, _ = ray.get(engine._get_current_node_ip_and_free_port.remote())
                return addr

            return addr, port

        get_addr, get_port = get_addr_and_ports(engine, node_index)

        for i in range(num_engines_on_this_node):
            current_rank = rank + i
            addr_and_ports.setdefault(current_rank, {})
            addr_and_ports[current_rank]["host"] = get_addr()
            addr_and_ports[current_rank]["port"] = get_port()
            addr_and_ports[current_rank]["nccl_port"] = get_port()

            if worker_type == "prefill":
                addr_and_ports[current_rank]["disaggregation_bootstrap_port"] = get_port()

        if _gpus_per_engine > args.num_gpus_per_node:
            num_node_per_engine = _gpus_per_engine // args.num_gpus_per_node
            if local_rank % num_node_per_engine == 0:
                # this is the first node in the engine, we need to allocate the dist_init_addr port
                dist_init_addr = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"
                for i in range(num_node_per_engine):
                    addr_and_ports.setdefault(rank + i, {})
                    addr_and_ports[rank + i]["dist_init_addr"] = dist_init_addr
        else:
            for i in range(num_engines_on_this_node):
                addr_and_ports[rank + i]["dist_init_addr"] = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"

    for i, _ in rollout_engines:
        for key in ["port", "nccl_port", "dist_init_addr"]:
            assert key in addr_and_ports[i], f"Engine {i} {key} is not set."
        logger.info(f"Ports for engine {i}: {addr_and_ports[i]}")

    return addr_and_ports, node_port_cursor


def _start_router(args, *, has_pd_disaggregation: bool = False, force_new: bool = False) -> tuple[str, int]:
    """Start sgl router or slime router and return (router_ip, router_port).

    If ``args.sglang_router_ip`` is already set (e.g. by the user) and
    ``force_new`` is False, skip launching and return the existing values.
    When ``force_new`` is True (multi-model), always allocate a fresh port.
    """
    if not force_new and args.sglang_router_ip is not None:
        return args.sglang_router_ip, args.sglang_router_port

    router_ip = _wrap_ipv6(get_host_info()[1])
    if force_new:
        router_port = find_available_port(random.randint(3000, 4000))
    else:
        router_port = args.sglang_router_port
        if router_port is None:
            router_port = find_available_port(random.randint(3000, 4000))

    if args.use_slime_router:
        assert not has_pd_disaggregation, "slime router does not support PD disaggregation."
        import copy

        from slime.router.router import run_router

        router_args = copy.copy(args)
        router_args.sglang_router_ip = router_ip
        router_args.sglang_router_port = router_port

    else:
        from sglang_router.launch_router import RouterArgs

        from slime.utils.http_utils import run_router

        router_args = RouterArgs.from_cli_args(args, use_router_prefix=True)
        router_args.host = router_ip
        router_args.port = router_port
        router_args.prometheus_port = find_available_port(random.randint(4000, 5000))
        router_args.log_level = "warn"
        router_args.request_timeout_secs = args.sglang_router_request_timeout_secs

        if has_pd_disaggregation:
            router_args.pd_disaggregation = True

        logger.info(f"Launch router with args: {router_args}")

    process = multiprocessing.Process(
        target=run_router,
        args=(router_args,),
    )
    process.daemon = True  # Set the process as a daemon
    process.start()
    # Wait 3 seconds
    time.sleep(3)
    assert process.is_alive()
    logger.info(f"Router launched at {router_ip}:{router_port}")
    return router_ip, router_port


def _compute_rollout_offset(args) -> int:
    """Offset (in PG bundle slots) where rollout GPUs start."""
    if args.debug_train_only or args.debug_rollout_only or args.colocate:
        return 0
    if args.critic_train_only:
        return args.critic_num_nodes * args.critic_num_gpus_per_node
    offset = args.actor_num_nodes * args.actor_num_gpus_per_node
    if args.use_critic:
        offset += args.critic_num_nodes * args.critic_num_gpus_per_node
    return offset


def _compute_megatron_num_gpus(args) -> int:
    """Total number of megatron (actor + critic) GPU slots in the placement group."""
    if args.debug_rollout_only:
        return 0
    if args.critic_train_only:
        return args.critic_num_nodes * args.critic_num_gpus_per_node
    num = args.actor_num_nodes * args.actor_num_gpus_per_node
    if args.use_critic:
        num += args.critic_num_nodes * args.critic_num_gpus_per_node
    return num


def start_rollout_servers(args, pg) -> dict[str, RolloutServer]:
    """Start rollout servers: one per model, each with its own router.

    Each model defined in the sglang config gets its own router and set
    of server groups.  Server groups within a model may have different
    ``num_gpus_per_engine`` (e.g. for PD disaggregation where prefill
    and decode use different TP sizes).

    Returns a dict mapping model name → ``RolloutServer``.

    Note: ``init_http_client`` should be called separately before this,
    as the HTTP client is shared across all servers.
    """
    config = _resolve_sglang_config(args)

    servers: dict[str, RolloutServer] = {}
    gpu_offset = 0
    engine_offset = 0

    # Compute megatron GPU range for per-group offload decisions.
    rollout_pg_offset = _compute_rollout_offset(args)
    megatron_num_gpus = _compute_megatron_num_gpus(args)

    for model_idx, model_cfg in enumerate(config.models):
        model_cfg.resolve(args)

        has_pd = model_cfg.has_pd_disaggregation
        router_ip, router_port = _start_router(args, has_pd_disaggregation=has_pd, force_new=(model_idx > 0))

        # Write back for backward compat (first model only).
        if model_idx == 0:
            args.sglang_router_ip = router_ip
            args.sglang_router_port = router_port

        server_groups: list[ServerGroup] = []
        all_init_handles: list = []
        port_cursors: dict[int, int] = {}

        for group_cfg in model_cfg.server_groups:
            gpus_per_engine = group_cfg.num_gpus_per_engine
            num_gpu_per_engine_local = min(gpus_per_engine, args.num_gpus_per_node)
            num_engines = group_cfg.num_gpus // num_gpu_per_engine_local

            # Only offload groups whose GPUs overlap with megatron.
            group_abs_start = rollout_pg_offset + gpu_offset
            needs_offload = args.offload_rollout and group_abs_start < megatron_num_gpus
            overrides = dict(group_cfg.overrides)
            if args.offload_rollout and not needs_offload:
                overrides.setdefault("enable_memory_saver", False)
            logger.info(
                f"Engine group '{group_cfg.worker_type}' gpu_offset={gpu_offset} "
                f"(abs={group_abs_start}): needs_offload={needs_offload}"
            )

            group = ServerGroup(
                args=args,
                pg=pg,
                all_engines=[None] * num_engines if group_cfg.worker_type != "placeholder" else [],
                num_gpus_per_engine=gpus_per_engine,
                num_new_engines=0,
                worker_type=group_cfg.worker_type,
                rank_offset=engine_offset,
                gpu_offset=gpu_offset,
                sglang_overrides=overrides,
                needs_offload=needs_offload,
                model_path=overrides.get("model_path", args.hf_checkpoint),
                router_ip=router_ip,
                router_port=router_port,
            )
            handles, port_cursors = group.start_engines(port_cursors)
            all_init_handles.extend(handles)
            server_groups.append(group)

            engine_offset += num_engines
            gpu_offset += group_cfg.num_gpus

        if all_init_handles:
            ray.get(all_init_handles)

        servers[model_cfg.name] = RolloutServer(
            server_groups=server_groups,
            router_ip=router_ip,
            router_port=router_port,
            model_name=model_cfg.name,
            update_weights=model_cfg.update_weights,
        )

    # Expose per-model router info for custom rollout functions.
    args.sglang_model_routers = {name: (srv.router_ip, srv.router_port) for name, srv in servers.items()}

    return servers


def _resolve_sglang_config(args) -> SglangConfig:
    """Build a SglangConfig from args, choosing the right source."""
    if getattr(args, "sglang_config", None) is not None:
        config = SglangConfig.from_yaml(args.sglang_config)
        # Validate total GPUs match.
        expected = args.rollout_num_gpus
        actual = config.total_num_gpus
        assert actual == expected, f"sglang_config total GPUs ({actual}) != rollout_num_gpus ({expected})"
        return config

    if args.prefill_num_servers is not None:
        return SglangConfig.from_prefill_num_servers(args)

    # Default: single regular group.
    return SglangConfig(
        models=[
            ModelConfig(
                name="default",
                server_groups=[ServerGroupConfig(worker_type="regular", num_gpus=args.rollout_num_gpus)],
            )
        ]
    )


def _log_eval_rollout_data(rollout_id, args, data, extra_metrics: dict[str, Any] | None = None):
    if args.custom_eval_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_eval_rollout_log_function_path)
        if custom_log_func(rollout_id, args, data, extra_metrics):
            return

    log_dict = _rewrite_eval_aux_metrics(extra_metrics)
    for key in data.keys():
        rewards = data[key]["rewards"]
        log_dict[f"eval/{key}"] = sum(rewards) / len(rewards)
        if (samples := data[key].get("samples")) is not None:
            log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), f"eval/{key}/")
        if "truncated" in data[key]:
            truncated = data[key]["truncated"]
            log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)
        if args.log_passrate:
            log_dict |= dict_add_prefix(
                compute_pass_rate(
                    flat_rewards=rewards,
                    group_size=args.n_samples_per_eval_prompt,
                ),
                f"eval/{key}-",
            )

    logger.info(f"eval {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    logging_utils.log(args, log_dict, step_key="eval/step")

    return log_dict


def _log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    if args.custom_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_rollout_log_function_path)
        if custom_log_func(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
            return

    if args.load_debug_rollout_data:
        return

    log_dict = {**(rollout_extra_metrics or {})}
    log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), "rollout/")
    log_dict |= dict_add_prefix(compute_perf_metrics_from_samples(args, samples, rollout_time), "perf/")
    logger.info(f"perf {rollout_id}: {log_dict}")
    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    logging_utils.log(args, log_dict, step_key="rollout/step")


def compute_metrics_from_samples(args, samples):
    response_lengths = [sample.effective_response_length for sample in samples]

    log_dict = {}
    log_dict |= dict_add_prefix(compute_statistics(response_lengths), "response_len/")
    log_dict |= _compute_zero_std_metrics(args, samples)
    log_dict |= _compute_reward_cat_metrics(args, samples)
    log_dict["repetition_frac"] = np.mean([int(has_repetition(s.response)) for s in samples]).item()
    log_dict["truncated_ratio"] = np.mean([int(s.status == Sample.Status.TRUNCATED) for s in samples]).item()
    return log_dict


def compute_perf_metrics_from_samples(args, samples, rollout_time):
    non_generation_time = [sample.non_generation_time for sample in samples]

    log_dict = {}
    log_dict["rollout_time"] = rollout_time
    if max(non_generation_time) > 0:
        log_dict |= dict_add_prefix(compute_statistics(non_generation_time), "non_generation_time/")

    def token_perf(response_lengths, non_generation_time, key=""):
        max_response_length = max(response_lengths)
        if args.rollout_num_gpus:
            log_dict[f"{key}tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / args.rollout_num_gpus
        log_dict[f"longest_{key}sample_tokens_per_sec"] = max_response_length / rollout_time

        if max(non_generation_time) == 0:
            return

        non_generation_time = [
            t for t, length in zip(non_generation_time, response_lengths, strict=True) if length == max_response_length
        ]
        mean_non_generation_time = sum(non_generation_time) / len(non_generation_time)

        log_dict[f"longest_{key}sample_non_generation_time"] = mean_non_generation_time
        log_dict[f"longest_{key}sample_tokens_per_sec_without_non_generation"] = max_response_length / (
            rollout_time - mean_non_generation_time
        )

    token_perf([sample.response_length for sample in samples], non_generation_time, key="")
    token_perf([sample.effective_response_length for sample in samples], non_generation_time, key="effective_")

    return log_dict


def _compute_zero_std_metrics(args, all_samples: list[Sample]):
    # only compute in GRPO-like algorithms where one prompt has multiple responses
    if args.advantage_estimator == "ppo":
        return {}

    def _is_zero_std(samples: list[Sample]):
        rewards = [sample.get_reward_value(args) for sample in samples]
        return len(rewards) == 0 or all(rewards[0] == r for r in rewards)

    all_sample_groups = group_by(all_samples, lambda s: s.group_index)
    interesting_sample_groups = [g for g in all_sample_groups.values() if _is_zero_std(g)]

    interesting_rewards = [str(round(g[0].get_reward_value(args), 1)) for g in interesting_sample_groups]

    return {f"zero_std/count_{reward}": len(items) for reward, items in group_by(interesting_rewards).items()}


def _compute_spec_metrics(args, all_samples: list[Sample]):
    if args.sglang_speculative_algorithm is None:
        return {}
    num_samples = len(all_samples)
    metrics = {}
    metrics["spec_accept_rate"] = sum(sample.spec_info.spec_accept_rate for sample in all_samples) / num_samples
    metrics["spec_accept_length"] = sum(sample.spec_info.spec_accept_length for sample in all_samples) / num_samples
    return metrics


def _compute_prefix_cache_metrics(args, all_samples: list[Sample]):
    num_samples = len(all_samples)
    metrics = {}
    total_cached_tokens = sum(sample.prefix_cache_info.cached_tokens for sample in all_samples)
    total_prompt_tokens = sum(sample.prefix_cache_info.total_prompt_tokens for sample in all_samples)

    metrics["prefix_cache_hit_rate"] = total_cached_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0.0
    metrics["avg_cached_tokens_per_sample"] = total_cached_tokens / num_samples
    return metrics


def _compute_reward_cat_metrics(args, all_samples: list[Sample]):
    reward_cat_key = args.log_reward_category
    if reward_cat_key is None:
        return {}

    samples_of_reward_cat = group_by(all_samples, lambda s: s.reward[reward_cat_key])

    return {f"error_cat/{reward_cat}": len(s) / len(all_samples) for reward_cat, s in samples_of_reward_cat.items()}
