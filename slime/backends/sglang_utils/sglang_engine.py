import dataclasses
import ipaddress
import logging
import multiprocessing
import os
import time
from pathlib import Path
from urllib.parse import quote
from urllib.parse import urlparse

import requests
import sglang_router
from packaging.version import parse
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from urllib3.exceptions import NewConnectionError

from slime.ray.ray_actor import RayActor
from slime.backends.sglang_utils.memory_budget import resolve_engine_mem_fraction
from slime.utils.http_utils import get_host_info

logger = logging.getLogger(__name__)


def resolve_sglang_startup_jit_config() -> tuple[bool, str]:
    enabled = os.environ.get("LIVEWEB_SGLANG_STARTUP_JIT_ENABLED", "0") == "1"
    if enabled:
        return True, "enabled_by_env"
    return False, "disabled_by_default_for_liveweb_resume"


def _patch_sglang_startup_for_liveweb() -> None:
    enabled, reason = resolve_sglang_startup_jit_config()
    logging.getLogger(__name__).info(
        "sglang/startup_jit_enabled=%s reason=%s",
        str(enabled).lower(),
        reason,
    )
    if enabled:
        return

    from sglang.jit_kernel import norm as norm_module
    from sglang.srt.models import utils as model_utils

    def _disable_qknorm_jit(_head_dim: int) -> bool:
        return False

    norm_module.can_use_fused_inplace_qknorm = _disable_qknorm_jit
    model_utils.can_use_fused_inplace_qknorm = _disable_qknorm_jit


def _launch_server_entry(server_args: ServerArgs) -> None:
    _patch_sglang_startup_for_liveweb()
    from sglang.srt.entrypoints.http_server import launch_server

    launch_server(server_args)


def _should_bypass_proxy(base_url: str) -> bool:
    try:
        hostname = urlparse(base_url).hostname
        if not hostname:
            return False
        if hostname in {"127.0.0.1", "localhost"}:
            return True
        ip = ipaddress.ip_address(hostname)
        return ip.is_loopback or ip.is_private or ip.is_link_local
    except ValueError:
        return False


def _build_requests_session(base_url: str) -> requests.Session:
    session = requests.Session()
    if _should_bypass_proxy(base_url):
        session.trust_env = False
    return session


def _build_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def get_base_gpu_id(args, rank):
    num_gpus = min(args.num_gpus_per_node, args.rollout_num_gpus_per_engine)
    if args.colocate:
        start_index = (rank * num_gpus) % args.num_gpus_per_node
    else:
        num_actor_gpus = 0 if args.debug_rollout_only else args.actor_num_gpus_per_node * args.actor_num_nodes
        start_index = (num_actor_gpus + rank * num_gpus) % args.num_gpus_per_node
        if args.use_critic:
            num_critic_gpus = args.critic_num_gpus_per_node * args.critic_num_nodes
            start_index = (num_actor_gpus + num_critic_gpus + rank * num_gpus) % args.num_gpus_per_node
    return start_index


def _to_local_gpu_id(physical_gpu_id: int) -> int:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        return physical_gpu_id  # no remapping
    # CUDA_VISIBLE_DEVICES can be like "4,5,6,7"
    visible = [int(x) for x in cvd.split(",") if x.strip() != ""]
    # In a remapped process, valid torch device indices are 0..len(visible)-1
    if physical_gpu_id in visible:
        return visible.index(physical_gpu_id)
    # If we're already getting local IDs, allow them
    if 0 <= physical_gpu_id < len(visible):
        return physical_gpu_id
    raise RuntimeError(
        f"GPU id {physical_gpu_id} is not valid under CUDA_VISIBLE_DEVICES={cvd}. "
        f"Expected one of {visible} (physical) or 0..{len(visible)-1} (local)."
    )


def launch_server_process(server_args: ServerArgs) -> multiprocessing.Process:
    multiprocessing.set_start_method("spawn", force=True)
    server_args.host = server_args.host.strip("[]")
    p = multiprocessing.Process(target=_launch_server_entry, args=(server_args,))
    p.start()

    if server_args.node_rank != 0:
        return

    _wait_server_healthy(
        base_url=server_args.url(),
        api_key=server_args.api_key,
        is_process_alive=lambda: p.is_alive(),
    )

    return p


def _wait_server_healthy(base_url, api_key, is_process_alive):
    headers = _build_headers(api_key)

    with _build_requests_session(base_url) as session:
        while True:
            try:
                response = session.get(f"{base_url}/health_generate", headers=headers)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass

            if not is_process_alive():
                raise Exception("Server process terminated unexpectedly.")

            time.sleep(2)


class SGLangEngine(RayActor):
    def __init__(
        self,
        args,
        rank: int,
        worker_type: str = "regular",
        base_gpu_id: int | None = None,
        sglang_overrides: dict | None = None,
        num_gpus_per_engine: int | None = None,
        is_updatable_group: bool = True,
    ):
        self.args = args
        self.rank = rank
        self.worker_type = worker_type
        self.base_gpu_id = base_gpu_id
        self.sglang_overrides = sglang_overrides or {}
        self.num_gpus_per_engine = num_gpus_per_engine
        self.is_updatable_group = is_updatable_group

    def init(
        self,
        dist_init_addr,
        port,
        nccl_port,
        host=None,
        disaggregation_bootstrap_port=None,
        router_ip=None,
        router_port=None,
    ):
        self.router_ip = router_ip if router_ip is not None else self.args.sglang_router_ip
        self.router_port = router_port if router_port is not None else self.args.sglang_router_port

        host = host or get_host_info()[1]

        def _format_v6_uri(addr):
            if not addr or addr.startswith("["):
                return addr
            try:
                if ipaddress.ip_address(addr).version == 6:
                    return f"[{addr}]"
            except ValueError:
                pass
            return addr

        host = _format_v6_uri(host)
        ip_part, port_part = dist_init_addr.rsplit(":", 1)
        dist_init_addr = f"{_format_v6_uri(ip_part)}:{port_part}"

        server_args_dict, external_engine_need_check_fields = _compute_server_args(
            self.args,
            self.rank,
            dist_init_addr,
            nccl_port,
            host,
            port,
            self.worker_type,
            disaggregation_bootstrap_port,
            base_gpu_id=self.base_gpu_id,
            sglang_overrides=self.sglang_overrides,
            num_gpus_per_engine=self.num_gpus_per_engine,
            is_updatable_group=self.is_updatable_group,
        )

        self.node_rank = server_args_dict["node_rank"]
        self.server_host = server_args_dict["host"]  # with [] if ipv6
        self.server_port = server_args_dict["port"]
        self.server_api_key = server_args_dict.get("api_key")

        if self.args.rollout_external:
            self._init_external(server_args_dict, external_engine_need_check_fields=external_engine_need_check_fields)
        else:
            self._init_normal(server_args_dict)

    def _init_external(self, expect_server_args, external_engine_need_check_fields):
        logger.info(f"Use external SGLang engine (rank={self.rank}, expect_server_args={expect_server_args})")

        def _get_actual_server_args():
            response = requests.get(f"http://{self.server_host}:{self.server_port}/get_server_info")
            response.raise_for_status()
            return response.json()

        def _sanity_check_server_args(actual_server_args, expect_server_args):
            for name in external_engine_need_check_fields:
                expect_value = expect_server_args.get(name)
                actual_value = actual_server_args.get(name)
                assert (
                    actual_value == expect_value
                ), f"{name=} {expect_value=} {actual_value=} {expect_server_args=} {actual_server_args=}"

        _wait_server_healthy(
            base_url=f"http://{self.server_host}:{self.server_port}",
            api_key=None,
            is_process_alive=lambda: True,
        )
        actual_server_args = _get_actual_server_args()
        _sanity_check_server_args(actual_server_args, expect_server_args)

    def _init_normal(self, server_args_dict):
        logger.info(f"Launch HttpServerEngineAdapter at: {self.server_host}:{self.server_port}")
        self.process = launch_server_process(ServerArgs(**server_args_dict))

        if self.node_rank == 0 and self.router_ip and self.router_port:
            if parse(sglang_router.__version__) <= parse("0.2.1") or self.args.use_slime_router:
                assert (
                    self.worker_type == "regular"
                ), "pd disaggregation is not supported in old router or slime router."
                response = requests.post(
                    f"http://{self.router_ip}:{self.router_port}/add_worker?url=http://{self.server_host}:{self.server_port}"
                )
            else:
                payload = {
                    "url": f"http://{self.server_host}:{self.server_port}",
                    "worker_type": self.worker_type,
                }
                if self.worker_type == "prefill":
                    payload["bootstrap_port"] = server_args_dict["disaggregation_bootstrap_port"]
                response = requests.post(
                    f"http://{self.router_ip}:{self.router_port}/workers",
                    json=payload,
                )
            response.raise_for_status()

    def _make_request(self, endpoint: str, payload: dict | None = None):
        """Make a POST request to the specified endpoint with the given payload.

        Args:
            endpoint: The API endpoint to call
            payload: The JSON payload to send (default: empty dict)

        Returns:
            The JSON response from the server
        """
        if self.node_rank != 0:
            return

        url = f"http://{self.server_host}:{self.server_port}/{endpoint}"
        with _build_requests_session(url) as session:
            response = session.post(url, json=payload or {}, headers=_build_headers(self.server_api_key))
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            e.add_note(f"{response.text=}")
            raise
        return response.json()

    def health_generate(self, timeout: float = 5.0) -> bool:
        """Run /health_generate on the underlying SGLang HTTP server.

        Args:
            timeout: Timeout for the health request in seconds.

        Returns:
            True if the server responds with HTTP 200.

        Raises:
            requests.RequestException: If the request fails for any reason, including timeout.
        """
        if self.node_rank != 0:
            return True

        url = f"http://{self.server_host}:{self.server_port}/health_generate"
        with _build_requests_session(url) as session:
            response = session.get(
                url,
                timeout=timeout,
                headers=_build_headers(self.server_api_key),
            )
        response.raise_for_status()
        return True

    def update_weights_from_tensor(
        self,
        serialized_named_tensors: list[str],
        load_format: str | None = None,
        flush_cache: bool = False,
        weight_version: str | None = None,
    ):
        """
        Update model weights from tensor data. The HTTP server will only post meta data, and the real weights will be copied directly from GPUs.

        Note: The model should be on GPUs rather than CPU for this functionality to work properly.
        If you encounter issues, ensure your model is loaded on GPU devices rather than CPU.
        """
        payload = {
            "serialized_named_tensors": serialized_named_tensors,
            "load_format": load_format,
            "flush_cache": flush_cache,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        return self._make_request(
            "update_weights_from_tensor",
            payload,
        )

    def flush_cache(self):
        """Flush the cache of the server."""
        if self.node_rank != 0:
            return
        # flush cache will not return status_code 200 when there are pending requests
        url = f"http://{self.server_host}:{self.server_port}/flush_cache"
        headers = _build_headers(self.server_api_key)
        for _ in range(60):
            try:
                with _build_requests_session(url) as session:
                    response = session.get(url, headers=headers)
                if response.status_code == 200:
                    break
            except NewConnectionError as e:
                raise e
            except Exception as e:
                logger.info(f"Error flushing cache: {e}")
                time.sleep(1)
                continue
        else:
            raise TimeoutError("Timeout while flushing cache.")

    def shutdown(self):
        if self.args.rollout_external:
            return

        logger.info(f"Shutdown engine {self.server_host}:{self.server_port}...")
        if self.node_rank == 0:
            worker_url = f"http://{self.server_host}:{self.server_port}"
            response = None
            if parse(sglang_router.__version__) <= parse("0.2.1") or self.args.use_slime_router:
                response = requests.post(
                    f"http://{self.router_ip}:{self.router_port}/remove_worker?url=http://{self.server_host}:{self.server_port}"
                )
            elif parse(sglang_router.__version__) < parse("0.3.0"):
                worker_url = quote(worker_url, safe="")
                response = requests.delete(f"http://{self.router_ip}:{self.router_port}/workers/{worker_url}")
            else:
                try:
                    all_workers = requests.get(f"http://{self.router_ip}:{self.router_port}/workers").json()["workers"]
                    for worker in all_workers:
                        if worker["url"] == worker_url:
                            worker_id = worker["id"]
                            response = requests.delete(
                                f"http://{self.router_ip}:{self.router_port}/workers/{worker_id}"
                            )
                            break
                    else:
                        logger.warning(f"Worker {worker_url} not found in router during shutdown.")
                except Exception as e:
                    logger.warning(f"Failed to fetch workers list or remove worker: {e}")

            if response is not None:
                response.raise_for_status()
        kill_process_tree(self.process.pid)

    def ping(self):
        return {
            "rank": self.rank,
            "worker_type": self.worker_type,
            "server_host": getattr(self, "server_host", None),
            "server_port": getattr(self, "server_port", None),
        }

    def get_weight_version(self):
        if self.node_rank != 0:
            return
        url = f"http://{self.server_host}:{self.server_port}/get_weight_version"
        with _build_requests_session(url) as session:
            response = session.get(url, headers=_build_headers(self.server_api_key))
        response.raise_for_status()
        return response.json()["weight_version"]

    def release_memory_occupation(self):
        self.flush_cache()
        return self._make_request("release_memory_occupation")

    def resume_memory_occupation(self, tags: list[str] = None):
        """
        Available tags for multi-stage resume: weights, kv_cache
        """
        return self._make_request(
            "resume_memory_occupation",
            {"tags": tags},
        )

    def check_weights(self, action: str):
        return self._make_request("weights_checker", {"action": action})

    def update_weights_from_disk(self, model_path: str, load_format: str | None = None):
        """Reload weights from *model_path* without restarting the engine.

        Used for non-updatable (frozen) models that overlap with megatron:
        after offload, weights are restored from disk instead of CPU cache.
        """
        payload = {"model_path": model_path}
        if load_format is not None:
            payload["load_format"] = load_format
        return self._make_request("update_weights_from_disk", payload)

    def init_weights_update_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        return self._make_request(
            "init_weights_update_group",
            {
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": rank_offset,
                "world_size": world_size,
                "group_name": group_name,
                "backend": backend,
            },
        )

    def destroy_weights_update_group(self, group_name):
        try:
            return self._make_request(
                "destroy_weights_update_group",
                {
                    "group_name": group_name,
                },
            )
        except requests.exceptions.RequestException:
            # catch the case there the engine is just created and does not have the group.
            pass

    def update_weights_from_distributed(
        self, names, dtypes, shapes, group_name, flush_cache=False, weight_version: str | None = None
    ):
        payload = {
            "names": names,
            "dtypes": [str(dtype).replace("torch.", "") for dtype in dtypes],
            "shapes": shapes,
            "group_name": group_name,
            "flush_cache": flush_cache,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        return self._make_request(
            "update_weights_from_distributed",
            payload,
        )

    def pause_generation(self):
        url = f"http://{self.server_host}:{self.server_port}/pause_generation"
        with _build_requests_session(url) as session:
            response = session.post(url, json={}, headers=_build_headers(self.server_api_key))
        response.raise_for_status()
        return response

    def continue_generation(self):
        url = f"http://{self.server_host}:{self.server_port}/continue_generation"
        with _build_requests_session(url) as session:
            response = session.post(url, json={}, headers=_build_headers(self.server_api_key))
        response.raise_for_status()
        return response

    def post_process_weights(
        self,
        restore_weights_before_load: bool = False,
        post_process_quantization: bool = False,
    ):
        """
        Update model weights from tensor data. The HTTP server will only post meta data, and the real weights will be copied directly from GPUs.
        Note: The model should be on GPUs rather than CPU for this functionality to work properly.
        If you encounter issues, ensure your model is loaded on GPU devices rather than CPU.
        """

        return self._make_request(
            "post_process_weights",
            {
                "restore_weights_before_load": restore_weights_before_load,
                "post_process_quantization": post_process_quantization,
            },
        )

    def start_profile(
        self,
        # The output directory
        output_dir: str | None = None,
        # If set, it profile as many as this number of steps.
        # If it is set, profiling is automatically stopped after this step, and
        # the caller doesn't need to run stop_profile.
        start_step: int | None = None,
        num_steps: int | None = None,
        activities: list[str] | None = None,
        profile_by_stage: bool = False,
        with_stack: bool | None = None,
        record_shapes: bool | None = None,
    ):
        url = f"http://{self.server_host}:{self.server_port}/start_profile"
        with _build_requests_session(url) as session:
            response = session.post(
                url,
                json={
                    "output_dir": output_dir,
                    "start_step": start_step,
                    "num_steps": num_steps,
                    "activities": activities,
                    "profile_by_stage": profile_by_stage,
                    "with_stack": with_stack,
                    "record_shapes": record_shapes,
                },
                headers=_build_headers(self.server_api_key),
            )
        response.raise_for_status()
        return response

    def stop_profile(self):
        url = f"http://{self.server_host}:{self.server_port}/stop_profile"
        with _build_requests_session(url) as session:
            response = session.post(
                url,
                json={},
                headers=_build_headers(self.server_api_key),
            )
        response.raise_for_status()
        return response

    def simulate_crash(self):
        if self.args.rollout_external or not getattr(self, "process", None):
            logger.info(
                "simulate_crash called but no local engine process exists (rollout_external=%s); skip kill",
                self.args.rollout_external,
            )
            return

        logger.info(f"Simulating crash on engine {self.server_host}:{self.server_port}...")
        self.shutdown()


def _compute_server_args(
    args,
    rank,
    dist_init_addr,
    nccl_port,
    host,
    port,
    worker_type: str = "regular",
    disaggregation_bootstrap_port: int | None = None,
    base_gpu_id: int | None = None,
    sglang_overrides: dict | None = None,
    num_gpus_per_engine: int | None = None,
    is_updatable_group: bool = True,
):
    _gpus_per_engine = num_gpus_per_engine or args.rollout_num_gpus_per_engine
    nnodes = max(1, _gpus_per_engine // args.num_gpus_per_node)
    node_rank = rank % nnodes
    physical_base_gpu_id = base_gpu_id if base_gpu_id is not None else get_base_gpu_id(args, rank)
    base = _to_local_gpu_id(physical_base_gpu_id)
    kwargs = {
        "model_path": args.hf_checkpoint,
        "trust_remote_code": True,
        "random_seed": args.seed + rank,
        # memory
        "enable_memory_saver": args.offload_rollout,
        # distributed
        "host": host,
        "port": port,
        "nccl_port": nccl_port,
        "nnodes": nnodes,
        "node_rank": node_rank,
        "dist_init_addr": dist_init_addr,
        "gpu_id_step": 1,
        "base_gpu_id": base,
        # parallel
        "tp_size": _gpus_per_engine // args.sglang_pp_size,
        "dp_size": args.sglang_dp_size,
        "pp_size": args.sglang_pp_size,
        "ep_size": args.sglang_ep_size,
        # always skip warmup to prevent warmup timeout.
        "skip_server_warmup": True,
        # always enable draft weights cpu backup so that we run training without mtp weights.
        "enable_draft_weights_cpu_backup": True,
    }

    bootstrap_model_path = getattr(args, "liveweb_rollout_bootstrap_model_path", None)
    rollout_boot_mode = getattr(args, "liveweb_rollout_boot_mode", os.environ.get("LIVEWEB_ROLLOUT_BOOT_MODE", "eager_weights"))
    if bootstrap_model_path and rollout_boot_mode == "lazy_weights" and is_updatable_group:
        kwargs["model_path"] = str(Path(bootstrap_model_path))

    if worker_type == "prefill":
        kwargs["disaggregation_mode"] = "prefill"
        kwargs["load_balance_method"] = "round_robin"
        assert (
            disaggregation_bootstrap_port is not None
        ), "disaggregation_bootstrap_port must be set for prefill worker"
        kwargs["disaggregation_bootstrap_port"] = disaggregation_bootstrap_port
    elif worker_type == "decode":
        kwargs["disaggregation_mode"] = "decode"
        kwargs["prefill_round_robin_balance"] = True

    if args.use_rollout_routing_replay:
        kwargs["enable_return_routed_experts"] = True
    if args.fp16:
        kwargs["dtype"] = "float16"
    external_engine_need_check_fields = [k for k in kwargs.keys() if k not in _EXTERNAL_ENGINE_SKIP_CHECK_FIELDS]

    unused_keys = set(kwargs.keys())
    for attr in dataclasses.fields(ServerArgs):
        if worker_type == "decode" and attr.name == "enable_hierarchical_cache":
            continue
        if hasattr(args, f"sglang_{attr.name}") and attr.name not in kwargs:
            kwargs[attr.name] = getattr(args, f"sglang_{attr.name}")
        unused_keys.discard(attr.name)

    # Per-server-group overrides from --sglang-config YAML.
    # Applied after base args so they take highest priority.
    if sglang_overrides:
        for key, value in sglang_overrides.items():
            if key == "update_weights":
                continue
            if key in kwargs:
                logger.info(f"sglang_overrides: overriding {key}={kwargs[key]} -> {value} (rank={rank})")
            kwargs[key] = value
            unused_keys.discard(key)

    # for compatibility with old args
    if len(unused_keys) > 0:
        logger.info(f"Warning: The following arguments is not supported in the current sglang: {unused_keys}.")
        for key in unused_keys:
            kwargs.pop(key)

    resolved_mem_fraction = resolve_engine_mem_fraction(
        default_fraction=kwargs.get("mem_fraction_static"),
        base_gpu_id=physical_base_gpu_id,
        num_gpus_per_engine=_gpus_per_engine,
        gpu_id_step=kwargs.get("gpu_id_step", 1),
    )
    if resolved_mem_fraction is not None and resolved_mem_fraction != kwargs.get("mem_fraction_static"):
        logger.info(
            "Balancing rollout memory budget for rank=%s on GPUs [%s]: mem_fraction_static %s -> %s",
            rank,
            ", ".join(str(physical_base_gpu_id + idx * kwargs.get("gpu_id_step", 1)) for idx in range(_gpus_per_engine)),
            kwargs.get("mem_fraction_static"),
            resolved_mem_fraction,
        )
        kwargs["mem_fraction_static"] = resolved_mem_fraction

    return kwargs, external_engine_need_check_fields


_EXTERNAL_ENGINE_SKIP_CHECK_FIELDS = [
    "model_path",
    "trust_remote_code",
    "random_seed",
    "nccl_port",
    "dist_init_addr",
    "skip_server_warmup",
    "enable_draft_weights_cpu_backup",
    "mem_fraction_static",
]
