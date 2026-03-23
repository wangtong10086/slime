from __future__ import annotations

import asyncio
import copy
import json
import os
import sys
import time
import ipaddress
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from slime.backends.sglang_utils.control_plane import build_control_plane_headers
from slime.utils.toolcall_health import (
    TOOLCALL_FORMAT_DANGLING_CLOSING,
    TOOLCALL_FORMAT_TEXT_ONLY_JSON,
    classify_toolcall_output,
    extract_assistant_response_text,
    preserve_structured_conversation,
)
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

from .task_sampling import (
    DEFAULT_LIVEWEB_ARENA_DIR,
    LiveWebDynamicSampler,
    ensure_liveweb_import_path,
    full_plugin_allowlist,
    load_task_mix_config,
    phase_plugin_weights,
    stable_plugin_allowlist,
)

ENV_POLLUTION_FAILURES = {
    "site_unreachable",
    "cache_error",
    "llm_error",
    "rollout_exception",
    "prefetch_failed",
    "cache_fill_failed",
    "env_nav_timeout",
    "env_cdn_blocked",
    "challenge_page",
    "control_plane_auth_failure",
    "sample_wall_timeout",
    "group_wall_timeout",
}
FORMAT_FAILURE_REASONS = {"parse_failed", "invalid_tool_format"}
PROGRESS_SHAPING_SIGNALS = {"target_asset", "detail_page_visit", "all_targets"}
DEFAULT_PREWARM_URLS = {
    "hackernews": [
        "https://news.ycombinator.com/",
        "https://news.ycombinator.com/ask",
        "https://news.ycombinator.com/show",
        "https://news.ycombinator.com/jobs",
        "https://news.ycombinator.com/newest",
    ],
    "channelsurfer": [
        "https://channelsurfer.tv/",
    ],
}

GOOGLE_FAMILY_HOST_MARKERS = (
    "google.",
    "googleusercontent.com",
    "gstatic.com",
)


def _normalize_url_for_matching(url: str | None) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    host = (parsed.netloc or parsed.path or "").lower().strip("/")
    path = (parsed.path or "").rstrip("/")
    if not host:
        return ""
    return f"{scheme}://{host}{path}" if scheme else f"{host}{path}"


def _hostname_from_url(url: str | None) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    if parsed.netloc:
        return parsed.netloc.lower()
    if parsed.scheme:
        return ""
    return parsed.path.lower().split("/", 1)[0]


def _host_matches_domain(host: str, domain: str) -> bool:
    host = (host or "").lower()
    domain = (domain or "").lower().strip()
    if not host or not domain:
        return False
    return host == domain or host.endswith(f".{domain}")


def _host_in_domains(host: str, domains: set[str] | list[str] | tuple[str, ...]) -> bool:
    domain_set = {str(domain).lower() for domain in domains or []}
    return any(_host_matches_domain(host, domain) for domain in domain_set)


def _is_google_family_host(host: str) -> bool:
    host = (host or "").lower()
    return any(marker in host for marker in GOOGLE_FAMILY_HOST_MARKERS)


def _extract_trajectory_diagnostics(
    *,
    trajectory: list[Any],
    allowed_domains: set[str] | list[str],
    required_domains: set[str] | list[str],
) -> dict[str, Any]:
    allowed_domain_set = {str(domain).lower() for domain in allowed_domains or []}
    required_domain_set = {str(domain).lower() for domain in required_domains or []}
    observation_urls: list[str] = []
    observation_hosts: list[str] = []
    disallowed_domain_hits = 0
    google_family_offdomain_count = 0
    repeated_url_count = 0
    same_page_loop_count = 0
    offdomain_persist_count = 0
    goto_count = 0
    goto_hosts: list[str] = []
    required_domain_hits = 0
    required_domains_hit: set[str] = set()
    last_norm = ""
    last_host = ""
    same_url_streak = 0
    offdomain_host_streak = 0

    for step in trajectory:
        observation = getattr(step, "observation", None)
        url = getattr(observation, "url", None) or ""
        norm_url = _normalize_url_for_matching(url)
        host = _hostname_from_url(url)
        if norm_url:
            observation_urls.append(norm_url)
        if host:
            observation_hosts.append(host)
            if _host_in_domains(host, required_domain_set):
                required_domain_hits += 1
                required_domains_hit.update(
                    domain for domain in required_domain_set if _host_matches_domain(host, domain)
                )
            allowed_hit = _host_in_domains(host, allowed_domain_set)
            if not allowed_hit:
                disallowed_domain_hits += 1
                if _is_google_family_host(host):
                    google_family_offdomain_count += 1
            if not allowed_hit and host == last_host:
                offdomain_host_streak += 1
            else:
                offdomain_host_streak = 1 if (host and not allowed_hit) else 0
            if offdomain_host_streak >= 2:
                offdomain_persist_count += 1
        if norm_url and norm_url == last_norm:
            repeated_url_count += 1
            same_url_streak += 1
        else:
            same_url_streak = 1 if norm_url else 0
        if same_url_streak >= 3:
            same_page_loop_count += 1
        last_norm = norm_url
        last_host = host

        action = getattr(step, "action", None)
        action_type = getattr(action, "action_type", None)
        if action_type == "goto":
            goto_count += 1
            target_url = getattr(action, "url", None) or ""
            target_host = _hostname_from_url(target_url)
            if target_host:
                goto_hosts.append(target_host)

    final_url = observation_urls[-1] if observation_urls else ""
    final_host = observation_hosts[-1] if observation_hosts else ""
    return {
        "goto_count": goto_count,
        "goto_hosts": goto_hosts,
        "disallowed_domain_hits": disallowed_domain_hits,
        "google_family_offdomain_count": google_family_offdomain_count,
        "repeated_url_count": repeated_url_count,
        "same_page_loop_count": same_page_loop_count,
        "offdomain_persist_count": offdomain_persist_count,
        "required_domain_hits": required_domain_hits,
        "required_domains_hit": sorted(required_domains_hit),
        "final_host": final_host,
        "final_google_search": bool(final_host and _is_google_family_host(final_host) and "search" in final_url),
    }


def _count_hallucinated_plugins(answer_details: list[dict[str, Any]], visited_domains: set[str]) -> int:
    hallucinated = 0
    for detail in answer_details:
        if not _answer_is_present(detail.get("actual")):
            continue
        plugin_domains = {str(domain).lower() for domain in detail.get("required_domains") or []}
        if plugin_domains and not any(_host_in_domains(domain, visited_domains) for domain in plugin_domains):
            hallucinated += 1
    return hallucinated


def _classify_rl_failure_bucket(
    *,
    failure_reason: str | None,
    success: bool,
    score: float,
    progress_summary: dict[str, Any],
    unsupported_stop: bool,
    hallucinated_plugin_count: int,
    trajectory_diagnostics: dict[str, Any],
) -> str:
    if failure_reason in ENV_POLLUTION_FAILURES:
        return "environment_failure"
    if failure_reason in FORMAT_FAILURE_REASONS:
        return "format_failure"
    if success:
        return "success"
    if unsupported_stop or hallucinated_plugin_count > 0:
        return "premature_stop"
    if (
        trajectory_diagnostics.get("google_family_offdomain_count", 0) >= 2
        or trajectory_diagnostics.get("disallowed_domain_hits", 0) >= 3
        or trajectory_diagnostics.get("same_page_loop_count", 0) >= 2
        or trajectory_diagnostics.get("offdomain_persist_count", 0) >= 2
        or trajectory_diagnostics.get("final_google_search")
    ):
        return "wrong_domain_loop"
    if score >= 0.3 or float(progress_summary.get("progress_score", 0.0)) >= float(
        os.getenv("LIVEWEB_NEAR_MISS_PROGRESS_THRESHOLD", "0.35")
    ):
        return "near_miss"
    return "wrong_path"


def get_phase_name(evaluation: bool = False) -> str:
    if evaluation:
        return os.getenv("LIVEWEB_EVAL_PHASE", "online_align")
    return os.getenv("LIVEWEB_TASK_MIX_PHASE", "bootstrap")


def derive_llm_seed(task_seed: int, sample_index: int) -> int:
    return abs(hash((task_seed, sample_index, "liveweb_online_rl"))) % (2**31 - 1)


def derive_route_key(
    prefix: str,
    parent_seed: int,
    subtask_index: int,
    sample_index: int | None = None,
    *,
    include_sample_index: bool = True,
) -> str:
    key = f"{prefix}:seed:{parent_seed}:subtask:{subtask_index}"
    if include_sample_index and sample_index is not None:
        key += f":sample:{sample_index}"
    return key


@dataclass
class PromptJob:
    parent_seed: int
    task_id: int
    task_seed: int
    llm_seed: int
    subtask_index: int
    num_subtasks: int
    templates: list[tuple[str, str | None, int | None]]
    task_name: str
    plugin_name: str
    plugin_names: list[str]
    combo_index: int
    combo_key: str
    phase: str
    route_key: str
    failure_bucket: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "parent_seed": self.parent_seed,
            "task_id": self.task_id,
            "task_seed": self.task_seed,
            "llm_seed": self.llm_seed,
            "subtask_index": self.subtask_index,
            "num_subtasks": self.num_subtasks,
            "templates": self.templates,
            "task_name": self.task_name,
            "plugin_name": self.plugin_name,
            "plugin_names": list(self.plugin_names),
            "combo_index": self.combo_index,
            "combo_key": self.combo_key,
            "phase": self.phase,
            "failure_bucket": self.failure_bucket,
            "route_key": self.route_key,
        }


def build_prompt_job(
    *,
    parent_seed: int,
    group_index: int,
    sample_index: int,
    phase: str,
    sampler: LiveWebDynamicSampler | None = None,
) -> PromptJob:
    sampler = sampler or LiveWebDynamicSampler(
        excluded_plugins={item.strip() for item in os.getenv("LIVEWEB_EXCLUDE_PLUGINS", "weather,openlibrary").split(",") if item.strip()},
        min_unique_plugins=read_int_env("LIVEWEB_MIN_UNIQUE_PLUGINS", 2),
    )
    selection = sampler.sample(seed=parent_seed, phase=phase, evaluation=False)
    subtask_index = 1
    task_seed = int(selection["task_seed"])
    llm_seed = derive_llm_seed(parent_seed, sample_index)
    combo_key = str(selection["combo_key"])
    plugin_names = list(selection["plugin_names"])
    return PromptJob(
        parent_seed=parent_seed,
        task_id=int(selection["task_id"]),
        task_seed=task_seed,
        llm_seed=llm_seed,
        subtask_index=subtask_index,
        num_subtasks=int(selection["num_subtasks"]),
        templates=[tuple(item) for item in selection["templates"]],
        task_name=f"liveweb_arena:{combo_key}",
        plugin_name=combo_key,
        plugin_names=plugin_names,
        combo_index=int(selection["combo_index"]),
        combo_key=combo_key,
        phase=phase,
        failure_bucket=str(selection.get("failure_bucket") or ""),
        route_key=derive_route_key(
            f"{phase}:task",
            int(selection["task_id"]),
            subtask_index,
            sample_index,
            include_sample_index=False,
        ),
    )


def build_eval_jobs(
    *,
    rollout_id: int,
    dataset_name: str,
    num_prompts: int,
    phase: str,
    base_seed: int,
    sampler: LiveWebDynamicSampler | None = None,
) -> list[PromptJob]:
    jobs = []
    sampler = sampler or LiveWebDynamicSampler(
        excluded_plugins={item.strip() for item in os.getenv("LIVEWEB_EXCLUDE_PLUGINS", "weather,openlibrary").split(",") if item.strip()},
        min_unique_plugins=read_int_env("LIVEWEB_MIN_UNIQUE_PLUGINS", 2),
    )
    for idx in range(num_prompts):
        parent_seed = base_seed + idx
        selection = sampler.sample(seed=parent_seed, phase=phase, evaluation=True)
        jobs.append(
            PromptJob(
                parent_seed=parent_seed,
                task_id=int(selection["task_id"]),
                task_seed=int(selection["task_seed"]),
                llm_seed=derive_llm_seed(parent_seed + rollout_id * 10_000, idx),
                subtask_index=1,
                num_subtasks=int(selection["num_subtasks"]),
                templates=[tuple(item) for item in selection["templates"]],
                task_name=f"{dataset_name}:{selection['combo_key']}",
                plugin_name=str(selection["combo_key"]),
                plugin_names=list(selection["plugin_names"]),
                combo_index=int(selection["combo_index"]),
                combo_key=str(selection["combo_key"]),
                phase=phase,
                failure_bucket=str(selection.get("failure_bucket") or ""),
                route_key=derive_route_key(
                    f"eval:{dataset_name}",
                    int(selection["task_id"]),
                    1,
                    idx,
                    include_sample_index=False,
                ),
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
        return "formal_eval", read_int_env("LIVEWEB_FORMAL_EVAL_PROMPTS", 200), "online_align"
    return "quick_eval", read_int_env("LIVEWEB_QUICK_EVAL_PROMPTS", 32), "bootstrap"


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
    response = session.get(
        f"{router_base}/list_workers",
        timeout=15,
        headers=build_control_plane_headers(args),
    )
    response.raise_for_status()
    payload = response.json()
    urls = payload.get("urls") or payload.get("worker_urls") or []
    if not urls:
        raise RuntimeError(f"No worker urls discovered from router {router_base}")
    return urls


def is_environment_pollution(result: dict[str, Any]) -> bool:
    return classify_environment_failure(result) is not None


def classify_environment_failure(result: dict[str, Any]) -> str | None:
    extra = result.get("extra") or {}
    error_lower = str(result.get("error") or "").lower()
    failure_reason = str(extra.get("failure_reason") or "")
    if any(marker in error_lower for marker in ("challenge page", "captcha", "just a moment")):
        return "challenge_page"
    if "401 unauthorized" in error_lower or "403 forbidden" in error_lower:
        return "control_plane_auth_failure"
    reachability = extra.get("reachability_audit") or {}
    classification = reachability.get("classification")
    if classification in ENV_POLLUTION_FAILURES:
        return str(classification)
    if failure_reason in ENV_POLLUTION_FAILURES:
        return failure_reason
    return None


def classify_overflow_failure(result: dict[str, Any]) -> str | None:
    extra = result.get("extra") or {}
    failure_reason = str(extra.get("failure_reason") or "")
    error_lower = str(result.get("error") or "").lower()
    if failure_reason in {"format_recovery_overflow", "llm_context_overflow"}:
        return failure_reason
    if "recoverable_context_overflow" in error_lower:
        return "format_recovery_overflow"
    if "strict-serial format recovery error" in error_lower and "context length" in error_lower:
        return "format_recovery_overflow"
    if "requested token count exceeds the model's maximum context length" in error_lower:
        return "llm_context_overflow"
    if "longer than the model's context length" in error_lower:
        return "llm_context_overflow"
    return None


def _answer_is_present(value: Any) -> bool:
    return value not in (None, "", [], {})


def _coverage(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def build_progress_summary(
    *,
    required_domains: list[str] | set[str] | None = None,
    visited_domains: list[str] | set[str] | None = None,
    target_assets: list[str] | set[str] | None = None,
    collected_target_assets: list[str] | set[str] | None = None,
    confirmed_targets: list[str] | set[str] | None = None,
    answer_details: list[dict[str, Any]] | None = None,
    answer_slots_total: int | None = None,
    num_subtasks: int | None = None,
) -> dict[str, Any]:
    required_domain_set = set(required_domains or [])
    visited_domain_set = set(visited_domains or [])
    target_asset_set = set(target_assets or [])
    collected_target_set = set(collected_target_assets or [])
    confirmed_target_set = set(confirmed_targets or [])
    answer_details = answer_details or []

    if answer_slots_total is None:
        answer_slots_total = max(len(answer_details), int(num_subtasks or 0))

    required_domains_total = len(required_domain_set)
    required_domains_visited = len(required_domain_set & visited_domain_set)
    target_assets_total = len(target_asset_set)
    target_assets_collected = len(target_asset_set & collected_target_set)
    confirmed_targets_collected = len(target_asset_set & confirmed_target_set)
    valid_answers = sum(1 for item in answer_details if _answer_is_present(item.get("actual")))

    required_domain_coverage = _coverage(required_domains_visited, required_domains_total)
    target_asset_coverage = _coverage(target_assets_collected, target_assets_total)
    confirmed_target_coverage = _coverage(confirmed_targets_collected, target_assets_total)
    valid_answer_coverage = _coverage(valid_answers, answer_slots_total)

    coverage_terms: list[float] = []
    if required_domains_total > 0:
        coverage_terms.append(required_domain_coverage)
    if target_assets_total > 0:
        coverage_terms.append(target_asset_coverage)
        coverage_terms.append(confirmed_target_coverage)
    if answer_slots_total > 0:
        coverage_terms.append(valid_answer_coverage)

    progress_score = sum(coverage_terms) / len(coverage_terms) if coverage_terms else 0.0
    return {
        "required_domains_total": required_domains_total,
        "required_domains_visited": required_domains_visited,
        "required_domain_coverage": required_domain_coverage,
        "target_assets_total": target_assets_total,
        "target_assets_collected": target_assets_collected,
        "target_asset_coverage": target_asset_coverage,
        "confirmed_targets_collected": confirmed_targets_collected,
        "confirmed_target_coverage": confirmed_target_coverage,
        "answer_slots_total": answer_slots_total,
        "valid_answers": valid_answers,
        "valid_answer_coverage": valid_answer_coverage,
        "progress_score": progress_score,
    }


def classify_learning_bucket(
    *,
    failure_reason: str | None,
    success: bool,
    score: float,
    progress_score: float,
) -> str:
    if failure_reason in ENV_POLLUTION_FAILURES:
        return "environment_failure"
    if failure_reason in FORMAT_FAILURE_REASONS:
        return "format_failure"
    if success:
        return "success"
    near_miss_progress_threshold = float(os.getenv("LIVEWEB_NEAR_MISS_PROGRESS_THRESHOLD", "0.35"))
    if score >= 0.3 or progress_score >= near_miss_progress_threshold:
        return "near_miss"
    return "wrong_path"


def _sum_positive_step_signals(step_rewards: list[dict[str, Any]], signal_names: set[str]) -> float:
    total = 0.0
    for step_reward in step_rewards:
        for signal in step_reward.get("signals") or []:
            if signal.get("signal") in signal_names:
                total += max(0.0, float(signal.get("value", 0.0)))
    return total


def _extract_confirmed_targets(step_rewards: list[dict[str, Any]]) -> set[str]:
    confirmed_targets: set[str] = set()
    for step_reward in step_rewards:
        for signal in step_reward.get("signals") or []:
            if signal.get("signal") != "detail_page_visit":
                continue
            reason = str(signal.get("reason") or "")
            if reason.startswith("Detail: "):
                confirmed_targets.add(reason.split(": ", 1)[1].strip())
    return confirmed_targets


def _aligned_collected_asset_snapshots(
    snapshots: list[set[str]],
    *,
    trajectory_len: int,
) -> list[set[str]]:
    if trajectory_len <= 0:
        return []
    if not snapshots:
        return [set() for _ in range(trajectory_len)]
    if len(snapshots) == trajectory_len + 1:
        snapshots = snapshots[1:]
    elif len(snapshots) > trajectory_len:
        snapshots = snapshots[-trajectory_len:]
    elif len(snapshots) < trajectory_len:
        snapshots = [*snapshots, *([set(snapshots[-1])] * (trajectory_len - len(snapshots)))]
    return [set(snapshot) for snapshot in snapshots]


def compute_reward_from_result(result: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
    extra = result.get("extra") or {}
    failure_reason = extra.get("failure_reason")
    overflow_failure = classify_overflow_failure(result)
    if overflow_failure is not None:
        learning_bucket = "format_failure" if overflow_failure == "format_recovery_overflow" else "wrong_path"
        return None, {
            "drop_reason": overflow_failure,
            "environment_failure_type": None,
            "raw_reward": result.get("score", 0.0),
            "learning_bucket": learning_bucket,
            "rl_failure_bucket": overflow_failure,
        }
    malformed_toolcall_output = False
    if os.getenv("LIVEWEB_DROP_INVALID_TOOLCALL_OUTPUTS", "1") == "1" and failure_reason == "parse_failed":
        raw_preview = extra.get("raw_response_preview") or extra.get("last_action_raw") or ""
        output_class = classify_toolcall_output(content=raw_preview, tool_calls=extra.get("tool_calls_preview"))
        malformed_toolcall_output = output_class in {
            TOOLCALL_FORMAT_DANGLING_CLOSING,
            TOOLCALL_FORMAT_TEXT_ONLY_JSON,
        }
        if malformed_toolcall_output:
            return None, {
                "drop_reason": "invalid_toolcall_output",
                "environment_failure_type": None,
                "raw_reward": result.get("score", 0.0),
                "learning_bucket": "format_failure",
                "toolcall_output_class": output_class,
            }
    environment_failure_type = classify_environment_failure(result)
    answer_details = extra.get("answer_details") or []
    progress_summary = extra.get("progress_summary")
    if not isinstance(progress_summary, dict) or "progress_score" not in progress_summary:
        progress_summary = build_progress_summary(
            required_domains=extra.get("required_domains") or [],
            visited_domains=extra.get("visited_domains") or [],
            target_assets=extra.get("target_assets") or [],
            collected_target_assets=extra.get("collected_target_assets") or [],
            confirmed_targets=extra.get("confirmed_targets") or [],
            answer_details=answer_details,
            answer_slots_total=extra.get("answer_slots_total"),
            num_subtasks=extra.get("num_subtasks"),
        )
    visited_domains = {str(domain).lower() for domain in extra.get("visited_domains") or []}
    trajectory_diagnostics = dict(extra.get("trajectory_diagnostics") or {})
    if not trajectory_diagnostics:
        trajectory_diagnostics = {
            "disallowed_domain_hits": 0,
            "google_family_offdomain_count": 0,
            "repeated_url_count": 0,
            "same_page_loop_count": 0,
            "offdomain_persist_count": 0,
            "required_domain_hits": 0,
            "required_domains_hit": [],
            "final_host": "",
            "final_google_search": False,
        }
    unsupported_stop = bool(
        extra.get("unsupported_stop")
        or (
            failure_reason in {None, "incomplete_data"}
            and (
                (
                    int(progress_summary.get("required_domains_total", 0)) > 0
                    and float(progress_summary.get("required_domain_coverage", 0.0)) < 1.0
                )
                or (
                    int(progress_summary.get("target_assets_total", 0)) > 0
                    and float(progress_summary.get("confirmed_target_coverage", 0.0)) < 1.0
                )
            )
        )
    )
    hallucinated_plugin_count = int(
        extra.get("hallucinated_plugin_count")
        or _count_hallucinated_plugins(answer_details, visited_domains)
    )
    learning_bucket = str(
        extra.get("learning_bucket")
        or classify_learning_bucket(
            failure_reason=failure_reason,
            success=bool(result.get("success", False)),
            score=float(result.get("score", 0.0)),
            progress_score=float(progress_summary.get("progress_score", 0.0)),
        )
    )
    rl_failure_bucket = str(
        extra.get("rl_failure_bucket")
        or _classify_rl_failure_bucket(
            failure_reason=failure_reason,
            success=bool(result.get("success", False)),
            score=float(result.get("score", 0.0)),
            progress_summary=progress_summary,
            unsupported_stop=unsupported_stop,
            hallucinated_plugin_count=hallucinated_plugin_count,
            trajectory_diagnostics=trajectory_diagnostics,
        )
    )
    if environment_failure_type is not None:
        return None, {
            "drop_reason": environment_failure_type,
            "environment_failure_type": environment_failure_type,
            "raw_reward": result.get("score", 0.0),
            "learning_bucket": learning_bucket,
            "rl_failure_bucket": rl_failure_bucket,
            "progress_summary": progress_summary,
            **progress_summary,
        }

    final_score = float(result.get("score", 0.0))
    required_domains = {str(domain).lower() for domain in extra.get("required_domains") or []}
    terminal_reward = final_score
    if failure_reason == "max_steps_reached":
        terminal_reward -= float(os.getenv("LIVEWEB_RL_TRUNCATION_PENALTY", "0.20"))
    if unsupported_stop:
        terminal_reward -= float(os.getenv("LIVEWEB_RL_UNSUPPORTED_STOP_PENALTY", "0.20"))
    if hallucinated_plugin_count:
        terminal_reward -= min(
            float(os.getenv("LIVEWEB_RL_MAX_HALLUCINATION_PENALTY", "0.40")),
            float(os.getenv("LIVEWEB_RL_HALLUCINATION_PLUGIN_PENALTY", "0.10")) * hallucinated_plugin_count,
        )

    shaping = 0.0
    if int(progress_summary.get("required_domains_visited", 0)) > 0:
        shaping += float(os.getenv("LIVEWEB_RL_FIRST_REQUIRED_DOMAIN_REWARD", "0.08"))
    if int(progress_summary.get("target_assets_collected", 0)) > 0:
        shaping += float(os.getenv("LIVEWEB_RL_FIRST_TARGET_ASSET_REWARD", "0.10"))
    if int(progress_summary.get("confirmed_targets_collected", 0)) > 0:
        shaping += float(os.getenv("LIVEWEB_RL_FIRST_DETAIL_PAGE_REWARD", "0.05"))

    shaping -= min(
        float(os.getenv("LIVEWEB_RL_MAX_REPEATED_URL_PENALTY", "0.24")),
        float(os.getenv("LIVEWEB_RL_REPEATED_URL_PENALTY", "0.08"))
        * float(trajectory_diagnostics.get("repeated_url_count", 0)),
    )
    no_progress_steps = 0
    for step_reward in (result.get("rewards") or {}).get("step_rewards") or []:
        for signal in step_reward.get("signals") or []:
            if signal.get("signal") == "no_progress":
                no_progress_steps += 1
    shaping -= min(
        float(os.getenv("LIVEWEB_RL_MAX_NO_PROGRESS_PENALTY", "0.20")),
        float(os.getenv("LIVEWEB_RL_NO_PROGRESS_PENALTY", "0.05")) * no_progress_steps,
    )
    shaping -= min(
        float(os.getenv("LIVEWEB_RL_MAX_OFFDOMAIN_PENALTY", "0.30")),
        float(os.getenv("LIVEWEB_RL_OFFDOMAIN_PERSIST_PENALTY", "0.10"))
        * float(trajectory_diagnostics.get("offdomain_persist_count", 0)),
    )
    shaping -= min(
        float(os.getenv("LIVEWEB_RL_MAX_GOOGLE_PENALTY", "0.30")),
        float(os.getenv("LIVEWEB_RL_GOOGLE_FAMILY_PENALTY", "0.15"))
        * float(trajectory_diagnostics.get("google_family_offdomain_count", 0)),
    )
    shaping -= min(
        float(os.getenv("LIVEWEB_RL_MAX_SAME_PAGE_LOOP_PENALTY", "0.30")),
        float(os.getenv("LIVEWEB_RL_SAME_PAGE_LOOP_PENALTY", "0.12"))
        * float(trajectory_diagnostics.get("same_page_loop_count", 0)),
    )
    if failure_reason == "parse_failed":
        shaping -= float(os.getenv("LIVEWEB_RL_PARSE_FAILED_PENALTY", "0.10"))
    if (
        int(progress_summary.get("required_domains_total", 0)) > 0
        and float(progress_summary.get("required_domain_coverage", 0.0)) >= 1.0
        and not bool(result.get("success", False))
    ):
        shaping += float(os.getenv("LIVEWEB_RL_ALL_REQUIRED_DOMAINS_COVERED_REWARD", "0.05"))

    shaping = min(0.40, max(-0.85, shaping))
    reward = min(1.0, max(-1.0, terminal_reward + shaping))
    return reward, {
        "drop_reason": None,
        "environment_failure_type": None,
        "raw_reward": final_score,
        "final_score": final_score,
        "terminal_reward": terminal_reward,
        "shaping_reward": shaping,
        "learning_bucket": learning_bucket,
        "rl_failure_bucket": rl_failure_bucket,
        "unsupported_stop": unsupported_stop,
        "hallucinated_plugin_count": hallucinated_plugin_count,
        "trajectory_diagnostics": trajectory_diagnostics,
        "progress_summary": progress_summary,
        **progress_summary,
    }


def should_allow_environment_fallback() -> bool:
    return os.getenv("LIVEWEB_ALLOW_ENV_FALLBACK_GROUPS", "0") == "1"


def normalize_conversation_for_training(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return preserve_structured_conversation(conversation)


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
        "task_id": (result.get("extra") or {}).get("task_id"),
        "combo_key": (result.get("extra") or {}).get("combo_key"),
        "plugin_names": (result.get("extra") or {}).get("plugin_names") or [],
        "required_domains": (result.get("extra") or {}).get("required_domains") or [],
        "visited_domains": (result.get("extra") or {}).get("visited_domains") or [],
        "target_assets": (result.get("extra") or {}).get("target_assets") or [],
        "collected_target_assets": (result.get("extra") or {}).get("collected_target_assets") or [],
        "confirmed_targets": (result.get("extra") or {}).get("confirmed_targets") or [],
        "progress_summary": (result.get("extra") or {}).get("progress_summary") or reward_meta.get("progress_summary") or {},
        "learning_bucket": (result.get("extra") or {}).get("learning_bucket") or reward_meta.get("learning_bucket"),
        "rl_failure_bucket": (result.get("extra") or {}).get("rl_failure_bucket") or reward_meta.get("rl_failure_bucket"),
        "unsupported_stop": (result.get("extra") or {}).get("unsupported_stop", reward_meta.get("unsupported_stop")),
        "hallucinated_plugin_count": (result.get("extra") or {}).get(
            "hallucinated_plugin_count", reward_meta.get("hallucinated_plugin_count", 0)
        ),
        "trajectory_diagnostics": (result.get("extra") or {}).get("trajectory_diagnostics")
        or reward_meta.get("trajectory_diagnostics")
        or {},
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


class LiveWebRolloutState:
    def __init__(self, args, *, scope: str = "train_rollout"):
        ensure_liveweb_import_path()
        from liveweb_arena.utils.llm_client import LLMServerConfig, MultiServerLLMRouter
        Actor, _, _ = import_liveweb_env_symbols()

        self.args = args
        self.scope = scope
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
        self.runtime_reset_count = 0
        self.runtime_pool_hits = 0
        self.runtime_jobs_since_reset = 0
        self._force_refresh_next = False
        self._consecutive_soft_failures = 0
        self._max_reuse_jobs = read_int_env("LIVEWEB_RUNTIME_MAX_REUSE_JOBS", 24)
        self._soft_failure_reset_threshold = read_int_env("LIVEWEB_RUNTIME_SOFT_FAILURE_RESET_THRESHOLD", 3)
        self._prewarmed_urls: set[str] = set()

    def snapshot_browser_metrics(self) -> dict[str, int]:
        return {
            "browser_rebuild_count": self.browser_rebuild_count,
            "browser_reuse_failures": self.browser_reuse_failures,
            "browser_recovery_success_count": self.browser_recovery_success_count,
            "runtime_reset_count": self.runtime_reset_count,
            "runtime_pool_hits": self.runtime_pool_hits,
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
            self.runtime_reset_count += 1
            await self.actor.shutdown()
            await self.actor._ensure_browser()
            self.browser_recovery_success_count += 1
            self.runtime_jobs_since_reset = 0
            self._consecutive_soft_failures = 0
            self._force_refresh_next = False
            self._prewarmed_urls.clear()

    async def _prewarm_scope(self) -> None:
        if not getattr(self.actor, "cache_manager", None):
            return
        try:
            from liveweb_arena.core.cache import PageRequirement
            from liveweb_arena.env import _find_plugin_for_url
        except Exception:
            return

        prewarm_urls = []
        env_urls = [
            item.strip()
            for item in os.getenv("LIVEWEB_PREWARM_URLS", "").split(",")
            if item.strip()
        ]
        prewarm_urls.extend(env_urls)
        for urls in DEFAULT_PREWARM_URLS.values():
            prewarm_urls.extend(urls)
        unique_urls = [url for url in dict.fromkeys(prewarm_urls) if url not in self._prewarmed_urls]
        if not unique_urls:
            return

        plugins_used = {}
        for plugin_name in stable_plugin_allowlist():
            plugin = self.actor.task_manager.get_plugin(plugin_name)
            if plugin is not None:
                plugins_used[plugin_name] = plugin

        for url in unique_urls:
            plugin = _find_plugin_for_url(plugins_used, url)
            if plugin is None:
                continue
            try:
                need_api = plugin.needs_api_data(url)
                page_req = PageRequirement.data(url) if need_api else PageRequirement.nav(url)
                await self.actor.cache_manager.ensure_cached([page_req], plugin)
                self._prewarmed_urls.add(url)
            except Exception:
                continue

    def record_job_outcome(self, result: dict[str, Any]) -> None:
        extra = result.get("extra") or {}
        failure_reason = extra.get("failure_reason")
        environment_failure = failure_reason in {"site_unreachable", "cache_error", "llm_error", "rollout_exception"}
        if failure_reason == "rollout_exception" and extra.get("browser_transport_closed"):
            self._force_refresh_next = True
        if environment_failure:
            self._consecutive_soft_failures += 1
        else:
            self._consecutive_soft_failures = 0
        self.runtime_jobs_since_reset += 1

    async def prepare_for_eval(self) -> None:
        self._rollout_phase = "eval"
        await self.recover_browser(force_refresh=True)
        await self._prewarm_scope()

    async def prepare_for_train_rollout(self) -> None:
        needs_refresh = (
            self._rollout_phase != "train"
            or self._force_refresh_next
            or self.runtime_jobs_since_reset >= self._max_reuse_jobs
            or self._consecutive_soft_failures >= self._soft_failure_reset_threshold
        )
        if needs_refresh:
            await self.recover_browser(force_refresh=True)
            self._rollout_phase = "train"
            await self._prewarm_scope()
            return
        await self.ensure_browser_ready()
        self.runtime_pool_hits += 1


def format_recovery_extra(agent_loop: Any | None) -> dict[str, Any]:
    if agent_loop is None or not hasattr(agent_loop, "get_format_recovery_stats"):
        return {}
    stats = agent_loop.get_format_recovery_stats()
    return stats if isinstance(stats, dict) else {}


def _cleanup_interceptor_value(interceptor: Any) -> None:
    if interceptor is None:
        return
    target = interceptor
    if isinstance(interceptor, tuple) and len(interceptor) >= 2:
        target = interceptor[1]
    cleanup = getattr(target, "cleanup", None)
    if callable(cleanup):
        cleanup()


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
    from liveweb_arena.core.reward import RewardConfig, StepwiseRewardCalculator
    from liveweb_arena.core.runtime_profiles import FAST_COLLECT_PROFILE, STRICT_EVAL_PROFILE
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
        collected_asset_snapshots: list[set[str]] = []
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
            phase_config = load_task_mix_config().get("phases", {}).get(str(job.phase), {})
            configured_max_steps = int(phase_config.get("max_steps", read_int_env("LIVEWEB_MAX_STEPS", 30)))
            if (job.failure_bucket or "") == "near_miss":
                configured_max_steps += int(os.getenv("LIVEWEB_NEAR_MISS_STEP_BONUS", "4"))
            effective_max_steps = max(configured_max_steps, total_expected_steps)
            target_assets: set[str] = set()
            required_domains: set[str] = set()
            reward_overrides: dict[str, float] = {}
            for subtask in task.subtasks:
                template = getattr(subtask, "template", None)
                if template is None:
                    continue
                target_assets.update(template.get_target_assets(subtask.validation_info))
                required_domains.update(template.get_required_domains(subtask.validation_info))
                overrides = template.get_reward_overrides()
                if overrides:
                    reward_overrides.update(overrides)
            reward_config = RewardConfig(**reward_overrides) if reward_overrides else RewardConfig()

            plugins_used, allowed_domains, blocked_patterns = actor._collect_plugin_info(task)

            exception_stage = "new_session"
            session = await actor.browser.new_session()

            exception_stage = "interceptor"
            session, interceptor = await actor._setup_interceptor(
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
                collected_asset_snapshots.append(set(gt_collector.get_collected_api_data().keys()))

            protocol = FunctionCallingProtocol()
            agent_llm_client = actor._build_llm_client(
                base_url=None,
                api_key=state.api_key,
                route_key=f"{job.route_key}:agent",
                max_retries=1,
                strict_serial=True,
            )
            runtime_profile = FAST_COLLECT_PROFILE if state._rollout_phase == "train" else STRICT_EVAL_PROFILE

            exception_stage = "agent_loop"
            rollout_temperature = float(
                os.getenv(
                    "LIVEWEB_EVAL_TEMPERATURE" if state._rollout_phase == "eval" else "LIVEWEB_TRAIN_TEMPERATURE",
                    os.getenv("LIVEWEB_TEMPERATURE", "0.7"),
                )
            )
            trajectory, final_answer, usage, failure_reason, error_message, agent_loop = await actor._run_agent_loop(
                task=task,
                session=session,
                llm_client=agent_llm_client,
                protocol=protocol,
                model=os.getenv("LIVEWEB_MODEL_NAME", args.hf_checkpoint),
                max_steps=effective_max_steps,
                timeout=read_int_env("LIVEWEB_TIMEOUT_SECONDS", 1800),
                temperature=rollout_temperature,
                seed=job.llm_seed,
                allowed_domains=allowed_domains,
                on_navigation=on_navigation,
                on_observation=on_observation,
                runtime_profile=runtime_profile,
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
                            "plugin_name": subtask.plugin_name,
                            "required_domains": sorted(
                                set(getattr(subtask.template, "get_required_domains")(subtask.validation_info))
                            )
                            if getattr(subtask, "template", None) is not None
                            else [],
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
            subtask_by_tag = {subtask.answer_tag: subtask for subtask in task.subtasks}
            for item in answer_validations:
                subtask = subtask_by_tag.get(str(item.get("answer_tag") or ""))
                if subtask is None:
                    continue
                item.setdefault("plugin_name", subtask.plugin_name)
                template = getattr(subtask, "template", None)
                if template is not None:
                    item.setdefault(
                        "required_domains",
                        sorted(set(template.get_required_domains(subtask.validation_info))),
                    )
                else:
                    item.setdefault("required_domains", [])

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
                config=reward_config,
                target_assets=target_assets,
                required_domains=required_domains,
            )
            step_rewards = []
            aligned_snapshots = _aligned_collected_asset_snapshots(
                collected_asset_snapshots,
                trajectory_len=len(trajectory),
            )
            for step, collected_asset_ids in zip(trajectory, aligned_snapshots, strict=False):
                url = step.observation.url
                reward = reward_calc.calculate_step_reward(
                    url=url,
                    action_result=step.action_result,
                    collected_asset_ids=collected_asset_ids,
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
            collected_assets = set(gt_collector.get_collected_api_data().keys())
            collected_target_assets = target_assets & collected_assets
            confirmed_targets = target_assets & _extract_confirmed_targets(step_rewards)
            visited_domains = set(reward_calc.get_state().get("visited_domains", []))
            progress_summary = build_progress_summary(
                required_domains=required_domains,
                visited_domains=visited_domains,
                target_assets=target_assets,
                collected_target_assets=collected_target_assets,
                confirmed_targets=confirmed_targets,
                answer_details=answer_validations,
                answer_slots_total=len(task.subtasks),
                num_subtasks=job.num_subtasks,
            )
            learning_bucket = classify_learning_bucket(
                failure_reason=failure_reason,
                success=success,
                score=total_score,
                progress_score=float(progress_summary["progress_score"]),
            )
            trajectory_diagnostics = _extract_trajectory_diagnostics(
                trajectory=trajectory,
                allowed_domains=allowed_domains,
                required_domains=required_domains,
            )
            unsupported_stop = (
                failure_reason in {None, "incomplete_data"}
                and (
                    (
                        int(progress_summary.get("required_domains_total", 0)) > 0
                        and float(progress_summary.get("required_domain_coverage", 0.0)) < 1.0
                    )
                    or (
                        int(progress_summary.get("target_assets_total", 0)) > 0
                        and float(progress_summary.get("confirmed_target_coverage", 0.0)) < 1.0
                    )
                )
            )
            hallucinated_plugin_count = _count_hallucinated_plugins(answer_validations, visited_domains)
            rl_failure_bucket = _classify_rl_failure_bucket(
                failure_reason=failure_reason,
                success=success,
                score=total_score,
                progress_summary=progress_summary,
                unsupported_stop=unsupported_stop,
                hallucinated_plugin_count=hallucinated_plugin_count,
                trajectory_diagnostics=trajectory_diagnostics,
            )

            result = {
                "task_name": job.task_name,
                "score": total_score,
                "success": success,
                "time_taken": time.time() - start_time,
                "extra": {
                    "seed": job.task_seed,
                    "task_id": job.task_id,
                    "task_seed": job.task_seed,
                    "llm_seed": job.llm_seed,
                    "parent_seed": job.parent_seed,
                    "subtask_index": job.subtask_index,
                    "num_subtasks": job.num_subtasks,
                    "combo_index": job.combo_index,
                    "combo_key": job.combo_key,
                    "final_url": final_url,
                    "output_format": output_format,
                    "usage": usage,
                    "answer_details": answer_validations,
                    "conversation": conversation,
                    "failure_reason": failure_reason,
                    "cache_stats": interceptor_stats,
                    "steps_used": len(trajectory),
                    "plugin_name": job.plugin_name,
                    "plugin_names": list(job.plugin_names),
                    "required_domains": sorted(required_domains),
                    "visited_domains": sorted(visited_domains),
                    "target_assets": sorted(target_assets),
                    "collected_target_assets": sorted(collected_target_assets),
                    "confirmed_targets": sorted(confirmed_targets),
                    "progress_summary": progress_summary,
                    "learning_bucket": learning_bucket,
                    "runtime_profile": runtime_profile,
                    "trajectory_diagnostics": trajectory_diagnostics,
                    "unsupported_stop": unsupported_stop,
                    "hallucinated_plugin_count": hallucinated_plugin_count,
                    "rl_failure_bucket": rl_failure_bucket,
                    "browser_rebuild_count": state.browser_rebuild_count,
                    "browser_reuse_failures": state.browser_reuse_failures,
                    "browser_recovery_success_count": state.browser_recovery_success_count,
                    **format_recovery_extra(agent_loop),
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
            _cleanup_interceptor_value(interceptor)
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
            "task_id": job.task_id,
            "task_seed": job.task_seed,
            "llm_seed": job.llm_seed,
            "parent_seed": job.parent_seed,
            "subtask_index": job.subtask_index,
            "num_subtasks": job.num_subtasks,
            "combo_index": job.combo_index,
            "combo_key": job.combo_key,
            "final_url": None,
            "usage": None,
            "answer_details": [],
            "conversation": [],
            "failure_reason": "rollout_exception",
            "cache_stats": {},
            "steps_used": 0,
            "plugin_name": job.plugin_name,
            "plugin_names": list(job.plugin_names),
            "required_domains": [],
            "visited_domains": [],
            "target_assets": [],
            "collected_target_assets": [],
            "confirmed_targets": [],
            "progress_summary": build_progress_summary(),
            "learning_bucket": "environment_failure",
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
    return None
