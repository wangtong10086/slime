#!/usr/bin/env python3
import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List

import httpx


def parse_server_specs(specs: str) -> List[Dict[str, str]]:
    parsed = []
    for idx, item in enumerate(specs.split()):
        gpu_ids, port = item.split(":")
        parsed.append(
            {
                "server_id": f"server-{idx}",
                "gpu_ids": gpu_ids,
                "port": int(port),
                "base_url": f"http://127.0.0.1:{int(port)}/v1",
            }
        )
    return parsed


def load_server_pool(args) -> List[Dict[str, str]]:
    if args.server_pool_file:
        payload = json.loads(Path(args.server_pool_file).read_text())
        return payload.get("servers", [])
    if args.server_specs:
        return parse_server_specs(args.server_specs)
    if args.base_url:
        return [
            {
                "server_id": "server-0",
                "base_url": args.base_url.rstrip("/"),
                "api_key": args.api_key,
                "model_name": args.model_name,
            }
        ]
    raise ValueError("One of --server-pool-file, --server-specs, or --base-url is required")


async def prewarm_server(server: Dict[str, str], model_name: str, api_key: str):
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Return {}"}],
        "temperature": 0.0,
        "max_tokens": 8,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{server['base_url'].rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
            )
            return {"server_id": server.get("server_id"), "ok": response.status_code == 200, "status_code": response.status_code}
    except Exception as exc:
        return {"server_id": server.get("server_id"), "ok": False, "error": repr(exc)}


async def prewarm_actor(actor):
    await actor._ensure_browser()
    await actor.cache_manager._ensure_browser()


def summarize_results(result_dir: Path, mode: str, router_snapshot: Dict) -> Dict:
    rows = []
    failure_reasons = {}
    environment_failures = {
        "site_unreachable": 0,
        "cache_error": 0,
        "prefetch_timeout": 0,
        "data_not_collected": 0,
        "parse_failed": 0,
    }
    aggregate_cache = {
        "hits": 0,
        "misses": 0,
        "blocked": 0,
        "passed": 0,
        "errors": 0,
        "stale_hits": 0,
        "prefetch_timeouts": 0,
        "soft_failures": 0,
        "per_domain_prefetch_timeouts": {},
        "per_domain_miss_count": {},
        "per_domain_miss_latency_s": {},
    }
    for path in sorted(result_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if path.name == "summary.json":
            continue
        cache_stats = data.get("extra", {}).get("cache_stats", {})
        failure_reason = data.get("extra", {}).get("failure_reason")
        if failure_reason:
            failure_reasons[failure_reason] = failure_reasons.get(failure_reason, 0) + 1
            if failure_reason in environment_failures:
                environment_failures[failure_reason] += 1
        for answer_detail in data.get("extra", {}).get("answer_details", []):
            reasoning = str(answer_detail.get("reasoning", "")).lower()
            if "data not collected" in reasoning:
                environment_failures["data_not_collected"] += 1
            if "pre-fetch timeout" in reasoning or "prefetch timeout" in reasoning:
                environment_failures["prefetch_timeout"] += 1
        for key in ("hits", "misses", "blocked", "passed", "errors", "stale_hits", "prefetch_timeouts", "soft_failures"):
            aggregate_cache[key] += int(cache_stats.get(key, 0))
        for dict_key in ("per_domain_prefetch_timeouts", "per_domain_miss_count", "per_domain_miss_latency_s"):
            for domain, value in cache_stats.get(dict_key, {}).items():
                aggregate_cache[dict_key][domain] = aggregate_cache[dict_key].get(domain, 0) + value
        rows.append(
            {
                "job_id": path.stem,
                "seed": data.get("extra", {}).get("seed"),
                "parent_seed": data.get("extra", {}).get("parent_seed"),
                "subtask_index": data.get("extra", {}).get("subtask_index"),
                "schedule_unit": data.get("extra", {}).get("schedule_unit", "run"),
                "score": float(data.get("score", 0.0)),
                "success": bool(data.get("success", False)),
                "time_taken": float(data.get("time_taken", 0.0)),
                "mode": data.get("extra", {}).get("mode", mode),
                "failure_reason": failure_reason,
            }
        )

    return {
        "mode": mode,
        "num_runs": len(rows),
        "mean_run_score": (sum(r["score"] for r in rows) / len(rows)) if rows else 0.0,
        "success_count": sum(r["success"] for r in rows),
        "failure_reasons": failure_reasons,
        "environment_failures": environment_failures,
        "runs": rows,
        "router": router_snapshot,
        "cache": aggregate_cache,
    }


def build_jobs(actor, args):
    template_overrides = None
    if args.templates_json:
        template_overrides = json.loads(args.templates_json)

    jobs = []
    for seed in range(args.start_seed, args.start_seed + args.num_runs):
        if args.schedule_unit == "run":
            jobs.append(
                {
                    "job_id": f"seed_{seed}",
                    "seed": seed,
                    "parent_seed": seed,
                    "subtask_index": None,
                    "num_subtasks": args.num_tasks,
                    "templates": template_overrides,
                    "route_key": f"{args.mode}:run:{seed}",
                    "task_name": f"liveweb_arena:{args.num_tasks}tasks",
                }
            )
            continue

        plan = actor.task_manager.plan_subtasks(
            seed=seed,
            num_subtasks=args.num_tasks,
            templates=template_overrides,
        )
        for item in plan:
            plugin_name = str(item["plugin_name"])
            template_name = item["template_name"]
            variant = item["variant"]
            subtask_index = int(item["subtask_index"])
            subtask_seed = int(item["subtask_seed"])
            job_id = f"seed_{seed}_subtask_{subtask_index + 1:02d}"
            jobs.append(
                {
                    "job_id": job_id,
                    "seed": subtask_seed,
                    "parent_seed": seed,
                    "subtask_index": subtask_index + 1,
                    "num_subtasks": 1,
                    "templates": [(plugin_name, template_name, variant)],
                    "route_key": f"{args.mode}:prompt:{seed}:{subtask_index + 1}",
                    "task_name": "liveweb_arena:1task",
                }
            )
    return jobs


async def worker_loop(
    worker_name: str,
    actor,
    queue: asyncio.Queue,
    result_dir: Path,
    model_name: str,
    num_tasks: int,
    max_steps: int,
    timeout: int,
    temperature: float,
    progress_log: Path,
    mode: str,
    router,
):
    while True:
        try:
            job = queue.get_nowait()
        except asyncio.QueueEmpty:
            return

        try:
            result = await actor.evaluate(
                model=model_name,
                seed=job["seed"],
                num_subtasks=job["num_subtasks"],
                templates=job["templates"],
                max_steps=max_steps,
                timeout=timeout,
                temperature=temperature,
                mode=mode,
                route_key=job["route_key"],
            )
            extra = result.setdefault("extra", {})
            extra["schedule_unit"] = job.get("schedule_unit", "run")
            extra["parent_seed"] = job.get("parent_seed")
            extra["subtask_index"] = job.get("subtask_index")
            out_path = result_dir / f"{job['job_id']}.json"
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
            router_stats = router.snapshot() if router else {}
            with progress_log.open("a") as f:
                f.write(
                    f"done worker={worker_name} job={job['job_id']} seed={job['seed']} mode={mode} "
                    f"score={result.get('score', 0.0)} time_taken={result.get('time_taken', 0.0)} "
                    f"router={json.dumps(router_stats, ensure_ascii=False)}\n"
                )
        except Exception as exc:
            err_path = result_dir / f"{job['job_id']}.json"
            err_payload = {
                "task_name": job["task_name"],
                "score": 0.0,
                "success": False,
                "time_taken": 0.0,
                "error": repr(exc),
                "extra": {
                    "mode": mode,
                    "seed": job["seed"],
                    "parent_seed": job.get("parent_seed"),
                    "subtask_index": job.get("subtask_index"),
                    "schedule_unit": job.get("schedule_unit", "run"),
                    "conversation": [],
                    "answer_details": [],
                },
            }
            err_path.write_text(json.dumps(err_payload, indent=2, ensure_ascii=False))
            with progress_log.open("a") as f:
                f.write(f"error worker={worker_name} job={job['job_id']} seed={job['seed']} error={repr(exc)}\n")
        finally:
            queue.task_done()


async def main():
    parser = argparse.ArgumentParser(description="Dynamic LiveWeb Arena batch runner")
    parser.add_argument("--liveweb-arena-dir", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--progress-log", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--server-pool-file")
    parser.add_argument("--server-specs")
    parser.add_argument("--base-url")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--num-runs", type=int, required=True)
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--num-tasks", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--mode", choices=("eval", "collect"), default="eval")
    parser.add_argument("--cache-profile", choices=("hot", "cold"), default="hot")
    parser.add_argument("--schedule-unit", choices=("run", "prompt"), default="run")
    parser.add_argument("--templates-json")
    parser.add_argument("--route-policy", default="sticky_steal")
    parser.add_argument("--sticky-slack", type=int, default=0)
    parser.add_argument("--sticky-latency-slack-s", type=float, default=10.0)
    parser.add_argument("--max-browser-sessions", type=int, required=True)
    parser.add_argument("--max-llm-requests", type=int, required=True)
    parser.add_argument("--prewarm-browsers", action="store_true")
    parser.add_argument("--prewarm-servers", action="store_true")
    args = parser.parse_args()

    liveweb_arena_dir = Path(args.liveweb_arena_dir)
    result_dir = Path(args.result_dir)
    progress_log = Path(args.progress_log)
    summary_path = Path(args.summary_path)
    cache_dir = Path(args.cache_dir)

    result_dir.mkdir(parents=True, exist_ok=True)
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(liveweb_arena_dir))
    from dotenv import load_dotenv

    load_dotenv(liveweb_arena_dir / ".env")
    from env import Actor
    from liveweb_arena.utils.llm_client import LLMServerConfig, MultiServerLLMRouter

    servers = load_server_pool(args)
    router = MultiServerLLMRouter.from_server_pool_file(
        args.server_pool_file,
        route_policy=args.route_policy,
        max_inflight_requests=args.max_llm_requests,
        sticky_slack=args.sticky_slack,
        sticky_latency_slack_s=args.sticky_latency_slack_s,
        default_api_key=args.api_key,
    ) if args.server_pool_file else MultiServerLLMRouter(
        servers=[
            LLMServerConfig(
                server_id=str(server.get("server_id") or f"server-{idx}"),
                base_url=str(server["base_url"]).rstrip("/"),
                api_key=str(server.get("api_key") or args.api_key),
                model_name=server.get("model_name") or args.model_name,
                metadata=server,
            )
            for idx, server in enumerate(servers)
        ],
        route_policy=args.route_policy,
        max_inflight_requests=args.max_llm_requests,
        sticky_slack=args.sticky_slack,
        sticky_latency_slack_s=args.sticky_latency_slack_s,
    )

    actor = Actor(
        api_key=args.api_key,
        cache_dir=cache_dir,
        use_cache=True,
        llm_router=router,
    )
    actor._semaphore = asyncio.Semaphore(args.max_browser_sessions)
    jobs = build_jobs(actor, args)
    for job in jobs:
        job["schedule_unit"] = args.schedule_unit

    queue: asyncio.Queue = asyncio.Queue()
    for job in jobs:
        queue.put_nowait(job)

    with progress_log.open("a") as f:
        f.write(
            f"batch_start num_runs={args.num_runs} start_seed={args.start_seed} mode={args.mode} "
            f"schedule_unit={args.schedule_unit} num_jobs={len(jobs)} "
            f"cache_profile={args.cache_profile} max_browser_sessions={args.max_browser_sessions} "
            f"max_llm_requests={args.max_llm_requests} route_policy={args.route_policy} "
            f"sticky_slack={args.sticky_slack} sticky_latency_slack_s={args.sticky_latency_slack_s}\n"
        )

    browser_prewarm_task = None
    if args.prewarm_browsers and args.cache_profile == "cold":
        with progress_log.open("a") as f:
            f.write("prewarm browsers start (background)\n")
        browser_prewarm_task = asyncio.create_task(prewarm_actor(actor))

    if args.prewarm_servers:
        with progress_log.open("a") as f:
            f.write("prewarm servers start\n")
        prewarm_results = await asyncio.gather(
            *(prewarm_server(server, args.model_name, args.api_key) for server in servers),
            return_exceptions=False,
        )
        with progress_log.open("a") as f:
            ok_count = sum(1 for item in prewarm_results if item.get("ok"))
            failed = [item for item in prewarm_results if not item.get("ok")]
            f.write(
                "prewarm servers done "
                f"ok={ok_count} failed={len(failed)} details={json.dumps(failed, ensure_ascii=False)}\n"
            )

    worker_count = min(len(jobs), args.max_browser_sessions)
    tasks = [
        asyncio.create_task(
            worker_loop(
                worker_name=f"worker-{worker_idx}",
                actor=actor,
                queue=queue,
                result_dir=result_dir,
                model_name=args.model_name,
                num_tasks=args.num_tasks,
                max_steps=args.max_steps,
                timeout=args.timeout,
                temperature=args.temperature,
                progress_log=progress_log,
                mode=args.mode,
                router=router,
            )
        )
        for worker_idx in range(worker_count)
    ]

    await asyncio.gather(*tasks)

    if browser_prewarm_task is not None:
        await asyncio.gather(browser_prewarm_task, return_exceptions=True)
        with progress_log.open("a") as f:
            f.write("prewarm browsers done\n")

    await actor.shutdown()

    summary = summarize_results(result_dir, args.mode, router.snapshot())
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
