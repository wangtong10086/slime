from __future__ import annotations

import asyncio
import copy
import json
import os
import random
import sys
import time
import ipaddress
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample


DEFAULT_TASK_MIX_PATH = Path("/home/xmyf/slime/scripts/configs/liveweb_online_task_mix.json")
DEFAULT_LIVEWEB_ARENA_DIR = Path("/home/xmyf/liveweb-arena")
ENV_POLLUTION_FAILURES = {"site_unreachable", "cache_error", "llm_error", "rollout_exception"}


def ensure_liveweb_import_path() -> Path:
    arena_dir = Path(os.getenv("LIVEWEB_ARENA_DIR", str(DEFAULT_LIVEWEB_ARENA_DIR))).resolve()
    if str(arena_dir) not in sys.path:
        sys.path.insert(0, str(arena_dir))
    return arena_dir


def load_task_mix_config() -> dict[str, Any]:
    path = Path(os.getenv("LIVEWEB_TASK_MIX_CONFIG", str(DEFAULT_TASK_MIX_PATH)))
    return json.loads(path.read_text())


def get_phase_name(evaluation: bool = False) -> str:
    if evaluation:
        return os.getenv("LIVEWEB_EVAL_PHASE", "main")
    return os.getenv("LIVEWEB_TASK_MIX_PHASE", "warmup")


def stable_plugin_allowlist() -> list[str]:
    return load_task_mix_config()["stable_plugins"]


def full_plugin_allowlist() -> list[str]:
    return load_task_mix_config()["full_plugins"]


def phase_plugin_weights(phase: str) -> dict[str, float]:
    cfg = load_task_mix_config()["phases"][phase]
    stable = cfg.get("stable_weight", 1.0)
    full_weight = cfg.get("full_weight", 0.0)

    weights: dict[str, float] = {}
    stable_plugins = stable_plugin_allowlist()
    full_plugins = full_plugin_allowlist()
    stable_share = stable / max(1, len(stable_plugins))
    full_share = full_weight / max(1, len(full_plugins))
    for plugin in stable_plugins:
        weights[plugin] = weights.get(plugin, 0.0) + stable_share
    for plugin in full_plugins:
        weights[plugin] = weights.get(plugin, 0.0) + full_share
    return weights


def weighted_plugin_choice(task_seed: int, phase: str) -> str:
    weights = phase_plugin_weights(phase)
    items = sorted(weights.items())
    plugins = [plugin for plugin, _ in items]
    probs = [weight for _, weight in items]
    total = sum(probs)
    normalized = [p / total for p in probs]
    rng = random.Random(task_seed)
    x = rng.random()
    cumulative = 0.0
    for plugin, p in zip(plugins, normalized, strict=True):
        cumulative += p
        if x <= cumulative:
            return plugin
    return plugins[-1]


def derive_llm_seed(task_seed: int, sample_index: int) -> int:
    return abs(hash((task_seed, sample_index, "liveweb_online_rl"))) % (2**31 - 1)


def derive_route_key(prefix: str, parent_seed: int, subtask_index: int, sample_index: int | None = None) -> str:
    key = f"{prefix}:seed:{parent_seed}:subtask:{subtask_index}"
    if sample_index is not None:
        key += f":sample:{sample_index}"
    return key


@dataclass
class PromptJob:
    parent_seed: int
    task_seed: int
    llm_seed: int
    subtask_index: int
    num_subtasks: int
    templates: list[tuple[str, str | None, int | None]]
    task_name: str
    plugin_name: str
    phase: str
    route_key: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "parent_seed": self.parent_seed,
            "task_seed": self.task_seed,
            "llm_seed": self.llm_seed,
            "subtask_index": self.subtask_index,
            "num_subtasks": self.num_subtasks,
            "templates": self.templates,
            "task_name": self.task_name,
            "plugin_name": self.plugin_name,
            "phase": self.phase,
            "route_key": self.route_key,
        }


def build_prompt_job(
    *,
    parent_seed: int,
    group_index: int,
    sample_index: int,
    phase: str,
) -> PromptJob:
    plugin_name = weighted_plugin_choice(parent_seed, phase)
    subtask_index = 1
    task_seed = parent_seed
    llm_seed = derive_llm_seed(task_seed, sample_index)
    return PromptJob(
        parent_seed=parent_seed,
        task_seed=task_seed,
        llm_seed=llm_seed,
        subtask_index=subtask_index,
        num_subtasks=1,
        templates=[(plugin_name, None, None)],
        task_name=f"liveweb_arena:{plugin_name}",
        plugin_name=plugin_name,
        phase=phase,
        route_key=derive_route_key(f"{phase}:prompt", parent_seed, subtask_index, sample_index),
    )


def build_eval_jobs(
    *,
    rollout_id: int,
    dataset_name: str,
    num_prompts: int,
    phase: str,
    base_seed: int,
) -> list[PromptJob]:
    jobs = []
    for idx in range(num_prompts):
        parent_seed = base_seed + idx
        plugin_name = weighted_plugin_choice(parent_seed, phase)
        jobs.append(
            PromptJob(
                parent_seed=parent_seed,
                task_seed=parent_seed,
                llm_seed=derive_llm_seed(parent_seed + rollout_id * 10_000, idx),
                subtask_index=1,
                num_subtasks=1,
                templates=[(plugin_name, None, None)],
                task_name=f"{dataset_name}:{plugin_name}",
                plugin_name=plugin_name,
                phase=phase,
                route_key=derive_route_key(f"eval:{dataset_name}", parent_seed, 1, idx),
            )
        )
    return jobs


def read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def get_eval_profile(rollout_id: int) -> tuple[str, int, str]:
    formal_every = read_int_env("LIVEWEB_FORMAL_EVAL_EVERY", 50)
    if (rollout_id + 1) % formal_every == 0:
        return "formal_eval", read_int_env("LIVEWEB_FORMAL_EVAL_PROMPTS", 200), "main"
    return "quick_eval", read_int_env("LIVEWEB_QUICK_EVAL_PROMPTS", 32), "warmup"


def summarize_cache_stats(cache_stats_list: list[dict[str, Any]]) -> dict[str, float]:
    if not cache_stats_list:
        return {}
    totals: dict[str, float] = {}
    for stats in cache_stats_list:
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0.0) + float(value)
    return {f"cache/{k}": v / len(cache_stats_list) for k, v in totals.items()}


def discover_worker_urls(args) -> list[str]:
    configured = os.getenv("LIVEWEB_SGLANG_WORKER_PORTS", "").strip()
    host = getattr(args, "sglang_router_ip", "127.0.0.1")
    if configured:
        ports = [item.strip() for item in configured.split(",") if item.strip()]
        return [f"http://{host}:{int(port)}" for port in ports]

    router_base = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    session = requests.Session()
    session.trust_env = not _should_bypass_proxy(router_base)
    response = session.get(f"{router_base}/list_workers", timeout=15)
    response.raise_for_status()
    payload = response.json()
    urls = payload.get("urls") or payload.get("worker_urls") or []
    if not urls:
        raise RuntimeError(f"No worker urls discovered from router {router_base}")
    return urls


def is_environment_pollution(result: dict[str, Any]) -> bool:
    failure_reason = (result.get("extra") or {}).get("failure_reason")
    return failure_reason in ENV_POLLUTION_FAILURES


def classify_environment_failure(result: dict[str, Any]) -> str | None:
    failure_reason = (result.get("extra") or {}).get("failure_reason")
    return failure_reason if failure_reason in ENV_POLLUTION_FAILURES else None


def compute_reward_from_result(result: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
    extra = result.get("extra") or {}
    failure_reason = extra.get("failure_reason")
    environment_failure_type = classify_environment_failure(result)
    if environment_failure_type is not None:
        return None, {
            "drop_reason": environment_failure_type,
            "environment_failure_type": environment_failure_type,
            "raw_reward": result.get("score", 0.0),
        }

    reward = float(result.get("score", 0.0))
    if failure_reason == "parse_failed":
        reward -= 0.10
    elif failure_reason == "max_steps_reached":
        reward -= 0.05
    reward = min(1.0, max(-0.10, reward))
    return reward, {
        "drop_reason": None,
        "environment_failure_type": None,
        "raw_reward": float(result.get("score", 0.0)),
    }


def should_allow_environment_fallback() -> bool:
    return os.getenv("LIVEWEB_ALLOW_ENV_FALLBACK_GROUPS", "0") == "1"


def extract_assistant_response_text(conversation: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in conversation:
        if message.get("role") != "assistant":
            continue
        if isinstance(message.get("content"), str) and message["content"]:
            parts.append(message["content"])
            continue
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function", {})
            parts.append(
                json.dumps(
                    {
                        "name": function.get("name"),
                        "arguments": function.get("arguments"),
                    },
                    ensure_ascii=False,
                )
            )
    return "\n".join(parts)


def _assistant_message_training_text(message: dict[str, Any]) -> str:
    parts: list[str] = []
    if isinstance(message.get("content"), str) and message["content"]:
        parts.append(message["content"])
    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function", {})
        parts.append(
            json.dumps(
                {
                    "name": function.get("name"),
                    "arguments": function.get("arguments"),
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(parts)


def normalize_conversation_for_training(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in conversation:
        msg = dict(message)
        if msg.get("role") == "assistant":
            training_text = _assistant_message_training_text(msg)
            if training_text:
                msg["content"] = training_text
        normalized.append(msg)
    return normalized


def _normalize_token_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, dict):
        encoded = encoded.get("input_ids", encoded)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return list(encoded)


def build_training_tokens_and_mask(tokenizer, conversation: list[dict[str, Any]]) -> tuple[list[int], int, list[int]]:
    tools = conversation[0].get("tools") if conversation else None
    messages = []
    full_mask: list[int] = []
    previous_len = 0

    for message in conversation:
        msg = dict(message)
        msg.pop("tools", None)
        messages.append(msg)
        current_ids = _normalize_token_ids(
            tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=True,
                add_generation_prompt=False,
            )
        )
        delta = len(current_ids) - previous_len
        if delta < 0:
            raise ValueError("Tokenized conversation length decreased unexpectedly")
        mask_value = 1 if msg.get("role") == "assistant" else 0
        full_mask.extend([mask_value] * delta)
        previous_len = len(current_ids)

    full_tokens = _normalize_token_ids(
        tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    if len(full_tokens) != len(full_mask):
        raise ValueError(f"Token/mask length mismatch: {len(full_tokens)} != {len(full_mask)}")

    try:
        first_supervised_idx = full_mask.index(1)
    except ValueError:
        first_supervised_idx = len(full_tokens)

    response_length = len(full_tokens) - first_supervised_idx
    loss_mask = full_mask[first_supervised_idx:]
    return full_tokens, response_length, loss_mask


def _fallback_assistant_payload(result: dict[str, Any]) -> str:
    extra = result.get("extra") or {}
    payload = {
        "name": "stop",
        "arguments": {
            "fallback": True,
            "score": float(result.get("score", 0.0)),
            "success": bool(result.get("success", False)),
            "failure_reason": extra.get("failure_reason"),
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def make_sample_from_result(
    *,
    sample: Sample,
    result: dict[str, Any],
    tokenizer,
    reward: float,
    reward_meta: dict[str, Any],
) -> Sample:
    conversation = ((result.get("extra") or {}).get("conversation")) or []
    normalized_conversation = normalize_conversation_for_training(conversation)
    full_tokens, response_length, loss_mask = build_training_tokens_and_mask(tokenizer, normalized_conversation)
    if response_length <= 0:
        normalized_conversation = [
            *normalized_conversation,
            {"role": "assistant", "content": _fallback_assistant_payload(result)},
        ]
        full_tokens, response_length, loss_mask = build_training_tokens_and_mask(tokenizer, normalized_conversation)
    sample.tokens = full_tokens
    sample.response_length = response_length
    sample.loss_mask = loss_mask
    sample.response = extract_assistant_response_text(conversation)
    sample.reward = reward
    sample.prompt = normalized_conversation
    sample.status = Sample.Status.COMPLETED
    sample.metadata = {
        **(sample.metadata or {}),
        **reward_meta,
        "task_name": result.get("task_name"),
        "score": result.get("score", 0.0),
        "success": result.get("success", False),
        "time_taken": result.get("time_taken", 0.0),
        "steps_used": len(conversation),
        "failure_reason": (result.get("extra") or {}).get("failure_reason"),
        "cache_stats": (result.get("extra") or {}).get("cache_stats") or {},
        "usage": (result.get("extra") or {}).get("usage") or {},
        "answer_details": (result.get("extra") or {}).get("answer_details") or [],
    }
    usage = sample.metadata["usage"]
    sample.metadata["prompt_tokens"] = usage.get("prompt_tokens", 0)
    sample.metadata["completion_tokens"] = usage.get("completion_tokens", 0)
    sample.metadata["total_tokens"] = usage.get("total_tokens", 0)
    sample.metadata["conversation_length"] = len(conversation)
    sample.non_generation_time = max(
        0.0,
        float(result.get("time_taken", 0.0)) - (usage.get("total_tokens", 0) / max(1.0, read_int_env("LIVEWEB_ASSUMED_TOKENS_PER_SEC", 40))),
    )
    return sample


def import_liveweb_env_symbols():
    ensure_liveweb_import_path()
    try:
        from liveweb_arena.env import Actor, _handle_navigation_event, _handle_observation_event
    except ModuleNotFoundError:
        from env import Actor, _handle_navigation_event, _handle_observation_event
    return Actor, _handle_navigation_event, _handle_observation_event


class LiveWebRolloutState(metaclass=SingletonMeta):
    def __init__(self, args):
        ensure_liveweb_import_path()
        from liveweb_arena.utils.llm_client import LLMServerConfig, MultiServerLLMRouter
        Actor, _, _ = import_liveweb_env_symbols()

        self.args = args
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.api_key = os.getenv("LIVEWEB_API_KEY", os.getenv("API_KEY", os.getenv("SGLANG_API_KEY", "local-liveweb")))
        self.liveweb_cache_dir = Path(os.getenv("LIVEWEB_CACHE_DIR", "/data/liveweb_cache/persistent")).resolve()
        self.liveweb_cache_dir.mkdir(parents=True, exist_ok=True)

        worker_urls = discover_worker_urls(args)
        servers = []
        for idx, url in enumerate(worker_urls):
            base = url.rstrip("/")
            if not base.endswith("/v1"):
                base = f"{base}/v1"
            servers.append(
                LLMServerConfig(
                    server_id=f"server-{idx}",
                    base_url=base,
                    api_key=self.api_key,
                    model_name=args.hf_checkpoint,
                )
            )
        self.router = MultiServerLLMRouter(
            servers=servers,
            route_policy=os.getenv("LIVEWEB_ROUTE_POLICY", "sticky_steal"),
            max_inflight_requests=read_int_env("LIVEWEB_MAX_LLM_REQUESTS", 16),
            sticky_slack=read_int_env("LIVEWEB_STICKY_SLACK", 0),
            sticky_latency_slack_s=float(os.getenv("LIVEWEB_STICKY_LATENCY_SLACK_S", "10.0")),
        )
        self.actor = Actor(
            api_key=self.api_key,
            cache_dir=self.liveweb_cache_dir,
            use_cache=True,
            llm_router=self.router,
        )
        self._browser_recovery_lock = asyncio.Lock()
        self._rollout_phase = "init"
        self.browser_rebuild_count = 0
        self.browser_reuse_failures = 0
        self.browser_recovery_success_count = 0

    def snapshot_browser_metrics(self) -> dict[str, int]:
        return {
            "browser_rebuild_count": self.browser_rebuild_count,
            "browser_reuse_failures": self.browser_reuse_failures,
            "browser_recovery_success_count": self.browser_recovery_success_count,
        }

    async def ensure_browser_ready(self) -> None:
        await self.actor._ensure_browser()

    async def recover_browser(self, *, force_refresh: bool = False) -> None:
        async with self._browser_recovery_lock:
            if not force_refresh:
                try:
                    await self.actor._ensure_browser()
                    return
                except Exception:
                    pass
            self.browser_rebuild_count += 1
            await self.actor.shutdown()
            await self.actor._ensure_browser()
            self.browser_recovery_success_count += 1

    async def prepare_for_eval(self) -> None:
        self._rollout_phase = "eval"
        await self.ensure_browser_ready()

    async def prepare_for_train_rollout(self) -> None:
        if self._rollout_phase != "train":
            await self.recover_browser(force_refresh=True)
            self._rollout_phase = "train"
            return
        await self.ensure_browser_ready()


def _should_bypass_proxy(base_url: str) -> bool:
    try:
        hostname = (urlparse(base_url).hostname or "").strip()
        if not hostname:
            return False
        if hostname in {"localhost", "127.0.0.1"}:
            return True
        ip = ipaddress.ip_address(hostname)
        return ip.is_private or ip.is_loopback or ip.is_link_local
    except ValueError:
        return False


async def evaluate_prompt_job(args, state: LiveWebRolloutState, job: PromptJob) -> dict[str, Any]:
    ensure_liveweb_import_path()
    from liveweb_arena.core.browser import is_browser_transport_error
    from liveweb_arena.core.agent_protocol import FunctionCallingProtocol
    from liveweb_arena.core.gt_collector import GTCollector, set_current_gt_collector
    from liveweb_arena.core.parser import AnswerParser
    from liveweb_arena.core.reward import StepwiseRewardCalculator
    from liveweb_arena.core.validators.llm_validator import validate_answers_with_llm
    _, _handle_navigation_event, _handle_observation_event = import_liveweb_env_symbols()

    actor = state.actor
    start_time = time.time()
    last_exc: Exception | None = None
    last_stage = "init"

    for attempt in range(2):
        task = None
        trajectory = []
        session = None
        interceptor = None
        gt_collector = None
        cached_pages: dict[str, Any] = {}
        exception_stage = "init"
        try:
            exception_stage = "ensure_browser"
            await state.ensure_browser_ready()

            exception_stage = "task_generation"
            task = await actor.task_manager.generate_composite_task(
                seed=job.task_seed,
                num_subtasks=job.num_subtasks,
                templates=job.templates,
            )
            total_expected_steps = sum(subtask.expected_steps for subtask in task.subtasks)
            effective_max_steps = max(read_int_env("LIVEWEB_MAX_STEPS", 30), total_expected_steps)

            plugins_used, allowed_domains, blocked_patterns = actor._collect_plugin_info(task)

            exception_stage = "new_session"
            session = await actor.browser.new_session()

            exception_stage = "interceptor"
            interceptor = await actor._setup_interceptor(
                session=session,
                cached_pages=cached_pages,
                allowed_domains=allowed_domains,
                blocked_patterns=blocked_patterns,
                plugins_used=plugins_used,
            )

            exception_stage = "gt_collector"
            gt_collector = GTCollector(
                subtasks=task.subtasks,
                task_manager=actor.task_manager,
            )
            set_current_gt_collector(gt_collector)

            async def on_navigation(url: str):
                await _handle_navigation_event(
                    interceptor=interceptor,
                    cached_pages=cached_pages,
                    plugins_used=plugins_used,
                    url=url,
                    use_cache=actor.use_cache,
                )

            async def on_observation(obs):
                await _handle_observation_event(
                    interceptor=interceptor,
                    cached_pages=cached_pages,
                    plugins_used=plugins_used,
                    gt_collector=gt_collector,
                    obs=obs,
                    use_cache=actor.use_cache,
                )

            protocol = FunctionCallingProtocol()
            agent_llm_client = actor._build_llm_client(
                base_url=None,
                api_key=state.api_key,
                route_key=f"{job.route_key}:agent",
                max_retries=1,
                strict_serial=True,
            )

            exception_stage = "agent_loop"
            trajectory, final_answer, usage, failure_reason, error_message, _ = await actor._run_agent_loop(
                task=task,
                session=session,
                llm_client=agent_llm_client,
                protocol=protocol,
                model=os.getenv("LIVEWEB_MODEL_NAME", args.hf_checkpoint),
                max_steps=effective_max_steps,
                timeout=read_int_env("LIVEWEB_TIMEOUT_SECONDS", 1800),
                temperature=float(os.getenv("LIVEWEB_TEMPERATURE", "0.7")),
                seed=job.llm_seed,
                allowed_domains=allowed_domains,
                on_navigation=on_navigation,
                on_observation=on_observation,
            )

            exception_stage = "gt_fetch"
            await gt_collector.fetch_remaining_api_gt()
            set_current_gt_collector(None)

            ground_truths = {}
            gt_extraction_failures = {}
            for subtask in task.subtasks:
                tag = subtask.answer_tag
                gt_value = gt_collector.get_gt_for_subtask(subtask)
                if gt_value is not None:
                    ground_truths[tag] = gt_value
                else:
                    gt_extraction_failures[tag] = gt_collector.get_failure_reason(subtask)

            exception_stage = "answer_parse"
            parser = AnswerParser()
            parsed_answers = parser.parse_answers(final_answer, job.num_subtasks)
            output_format = parser.get_output_format(final_answer)
            validation_rules = {}
            for subtask in task.subtasks:
                plugin = actor.task_manager.get_plugin(subtask.plugin_name)
                if hasattr(plugin, "get_validation_rules"):
                    validation_rules[subtask.answer_tag] = plugin.get_validation_rules(subtask.validation_info)

            subtasks_to_validate = []
            answer_validations = []
            for subtask in task.subtasks:
                tag = subtask.answer_tag
                if tag in gt_extraction_failures:
                    answer_validations.append(
                        {
                            "question": subtask.intent,
                            "answer_tag": tag,
                            "expected": None,
                            "actual": parsed_answers.get(tag),
                            "score": 0.0,
                            "is_correct": False,
                            "reasoning": f"Data not collected: {gt_extraction_failures[tag]}",
                        }
                    )
                else:
                    subtasks_to_validate.append(subtask)

            if subtasks_to_validate:
                validator_llm_client = actor._build_llm_client(
                    base_url=None,
                    api_key=state.api_key,
                    route_key=f"{job.route_key}:validator",
                )
                exception_stage = "validation"
                answer_validations.extend(
                    await validate_answers_with_llm(
                        llm_client=validator_llm_client,
                        subtasks=subtasks_to_validate,
                        answers=parsed_answers,
                        ground_truths=ground_truths,
                        validation_rules=validation_rules,
                    )
                )
            answer_validations.sort(key=lambda item: item.get("answer_tag", ""))

            hard_failures = {"agent_timeout", "llm_error", "cache_error", "site_unreachable"}
            if failure_reason and failure_reason in hard_failures:
                total_score = 0.0
                success = False
            elif answer_validations:
                total_score = sum(item["score"] for item in answer_validations) / len(answer_validations)
                success = total_score >= 0.8
            else:
                total_score = 0.0
                success = False

            reward_calc = StepwiseRewardCalculator(
                target_assets=set(),
                required_domains=allowed_domains,
            )
            step_rewards = []
            for step in trajectory:
                url = step.observation.url
                reward = reward_calc.calculate_step_reward(
                    url=url,
                    action_result=step.action_result,
                    collected_asset_ids=set(),
                    is_blocked=interceptor._should_block(url) if url != "about:blank" else False,
                    parse_failed=(step.action is None),
                )
                step_rewards.append(reward.to_dict())

            terminal_reward = reward_calc.calculate_terminal_reward(
                validation_score=total_score,
                steps_used=len(trajectory),
                max_steps=effective_max_steps,
                truncated=(failure_reason == "max_steps_reached"),
            )
            interceptor_stats = interceptor.get_stats()
            final_url = trajectory[-1].observation.url if trajectory else None
            conversation = actor._build_conversation(task, trajectory, protocol)

            result = {
                "task_name": job.task_name,
                "score": total_score,
                "success": success,
                "time_taken": time.time() - start_time,
                "extra": {
                    "seed": job.task_seed,
                    "task_seed": job.task_seed,
                    "llm_seed": job.llm_seed,
                    "parent_seed": job.parent_seed,
                    "subtask_index": job.subtask_index,
                    "num_subtasks": job.num_subtasks,
                    "final_url": final_url,
                    "output_format": output_format,
                    "usage": usage,
                    "answer_details": answer_validations,
                    "conversation": conversation,
                    "failure_reason": failure_reason,
                    "cache_stats": interceptor_stats,
                    "steps_used": len(trajectory),
                    "plugin_name": job.plugin_name,
                    "browser_rebuild_count": state.browser_rebuild_count,
                    "browser_reuse_failures": state.browser_reuse_failures,
                    "browser_recovery_success_count": state.browser_recovery_success_count,
                },
                "rewards": {
                    "step_rewards": step_rewards,
                    "terminal_reward": terminal_reward.to_dict(),
                    "cumulative_step_reward": sum(item["total"] for item in step_rewards),
                    "total_reward": sum(item["total"] for item in step_rewards) + terminal_reward.total,
                },
            }

            if not error_message and gt_extraction_failures:
                system_errors = []
                for subtask in task.subtasks:
                    tag = subtask.answer_tag
                    if tag in gt_extraction_failures and gt_collector.is_system_error(subtask):
                        system_errors.append(f"[{tag}] {gt_extraction_failures[tag]}")
                if system_errors:
                    error_message = f"GT system error: {'; '.join(system_errors)}"

            if error_message:
                result["error"] = error_message
            return result
        except Exception as exc:  # pragma: no cover - broad to keep rollout robust
            last_exc = exc
            last_stage = exception_stage
            if attempt == 0 and is_browser_transport_error(exc):
                state.browser_reuse_failures += 1
                try:
                    await state.recover_browser(force_refresh=True)
                except Exception as rebuild_exc:
                    last_exc = rebuild_exc
                    last_stage = "browser_rebuild"
                    break
                continue
            break
        finally:
            set_current_gt_collector(None)
            if gt_collector is not None:
                gt_collector.cleanup()
            if interceptor is not None:
                interceptor.cleanup()
            cached_pages.clear()
            if session is not None:
                await session.close()

    exc = last_exc or RuntimeError("unknown rollout exception")
    browser_transport_closed = is_browser_transport_error(exc)
    return {
        "task_name": job.task_name,
        "score": 0.0,
        "success": False,
        "time_taken": time.time() - start_time,
        "extra": {
            "seed": job.task_seed,
            "task_seed": job.task_seed,
            "llm_seed": job.llm_seed,
            "parent_seed": job.parent_seed,
            "subtask_index": job.subtask_index,
            "num_subtasks": job.num_subtasks,
            "final_url": None,
            "usage": None,
            "answer_details": [],
            "conversation": [],
            "failure_reason": "rollout_exception",
            "cache_stats": {},
            "steps_used": 0,
            "plugin_name": job.plugin_name,
            "exception_type": type(exc).__name__,
            "exception_stage": last_stage,
            "browser_transport_closed": browser_transport_closed,
            "browser_rebuild_count": state.browser_rebuild_count,
            "browser_reuse_failures": state.browser_reuse_failures,
            "browser_recovery_success_count": state.browser_recovery_success_count,
        },
        "error": repr(exc),
    }


async def shutdown_liveweb_state():
    state = LiveWebRolloutState._instances.get(LiveWebRolloutState)  # type: ignore[attr-defined]
    if state is not None:
        try:
            await state.actor.shutdown()
        finally:
            LiveWebRolloutState.clear_instances()
