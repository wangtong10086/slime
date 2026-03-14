#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import httpx

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


DEFAULT_TEMPLATES = [
    "stooq/stooq_price",
    "stooq/stooq_comparison",
    "stooq/stooq_ranking",
    "coingecko/coingecko_price",
    "coingecko/coingecko_rank",
    "coingecko/coingecko_performance",
    "hackernews/hackernews_multi_condition_filter",
    "hackernews/hackernews_news_summary",
]


def flatten_numeric_metrics(data: dict, prefix: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in data.items():
        metric_key = f"{prefix}/{key}"
        if isinstance(value, bool):
            metrics[metric_key] = float(value)
        elif isinstance(value, (int, float)):
            metrics[metric_key] = float(value)
        elif isinstance(value, dict):
            metrics.update(flatten_numeric_metrics(value, metric_key))
    return metrics


def log(message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch slime checkpoints and evaluate them on LiveWeb Arena.")
    parser.add_argument("--slime-dir", type=Path, required=True)
    parser.add_argument("--watch-dir", type=Path, required=True)
    parser.add_argument("--origin-hf-dir", type=Path, required=True)
    parser.add_argument("--converted-root", type=Path, required=True)
    parser.add_argument("--arena-dir", type=Path, required=True)
    parser.add_argument("--arena-cache-dir", type=Path, required=True)
    parser.add_argument("--arena-output-dir", type=Path, required=True)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--eval-gpus", type=str, default="4,5,6,7")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--server-host", type=str, default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=31000)
    parser.add_argument("--server-api-key", type=str, default="local-liveweb")
    parser.add_argument("--served-model-name", type=str, default="qwen3-32b-liveweb-sft")
    parser.add_argument("--mem-fraction-static", type=float, default=0.82)
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--disable-cuda-graph", action="store_true", default=False)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--server-ready-timeout", type=int, default=1800)
    parser.add_argument("--steps-per-epoch", type=int, required=True)
    parser.add_argument("--num-epoch", type=int, required=True)
    parser.add_argument("--stop-after-iteration", type=int, required=True)
    parser.add_argument("--keep-last-converted", type=int, default=1)
    parser.add_argument("--skip-initial-eval", action="store_true", default=False)
    parser.add_argument("--only-initial-eval", action="store_true", default=False)
    parser.add_argument("--only-iteration", type=int, default=None)
    parser.add_argument("--num-tasks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--templates", nargs="+", default=DEFAULT_TEMPLATES)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-run-id", type=str, default=None)
    parser.add_argument("--wandb-team", type=str, default=None)
    parser.add_argument("--wandb-dir", type=Path, default=None)
    parser.add_argument("--wandb-key", type=str, default=None)
    parser.add_argument(
        "--wandb-settings-file",
        type=Path,
        default=Path.home() / ".config" / "wandb" / "settings",
    )
    return parser.parse_args()


class ServerManager:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.process: subprocess.Popen[str] | None = None
        self.model_dir: Path | None = None

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            log("Stopping SGLang server.")
            self.process.terminate()
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                log("SGLang server did not stop in time, killing it.")
                self.process.kill()
                self.process.wait(timeout=30)
        self.process = None
        self.model_dir = None

    def start(self, model_dir: Path) -> None:
        self.stop()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.args.eval_gpus
        cmd = [
            self.args.python_bin,
            "-m",
            "sglang.launch_server",
            "--model-path",
            str(model_dir),
            "--trust-remote-code",
            "--host",
            self.args.server_host,
            "--port",
            str(self.args.server_port),
            "--api-key",
            self.args.server_api_key,
            "--served-model-name",
            self.args.served_model_name,
            "--tp-size",
            str(self.args.tp_size),
            "--mem-fraction-static",
            str(self.args.mem_fraction_static),
            "--context-length",
            str(self.args.context_length),
            "--dtype",
            self.args.dtype,
        ]
        if self.args.disable_cuda_graph:
            cmd.append("--disable-cuda-graph")

        log(f"Starting SGLang server for {model_dir}.")
        self.process = subprocess.Popen(cmd, env=env, cwd=self.args.slime_dir)
        self.model_dir = model_dir
        self._wait_ready()

    def _wait_ready(self) -> None:
        deadline = time.time() + self.args.server_ready_timeout
        headers = {"Authorization": f"Bearer {self.args.server_api_key}"}
        url = f"http://{self.args.server_host}:{self.args.server_port}/v1/models"
        last_error: Exception | None = None

        while time.time() < deadline:
            if self.process is None:
                raise RuntimeError("SGLang server process was not created.")
            if self.process.poll() is not None:
                raise RuntimeError(f"SGLang server exited early with code {self.process.returncode}.")

            try:
                response = httpx.get(url, headers=headers, timeout=15.0)
                if response.status_code == 200:
                    log("SGLang server is ready.")
                    return
            except Exception as exc:  # pragma: no cover
                last_error = exc
            time.sleep(5)

        raise RuntimeError(f"SGLang server did not become ready in time. last_error={last_error!r}")


def init_wandb(args: argparse.Namespace):
    if wandb is None or not args.wandb_project or not args.wandb_run_id:
        return None

    wandb_key = args.wandb_key
    if not wandb_key and args.wandb_settings_file.exists():
        for line in args.wandb_settings_file.read_text().splitlines():
            if line.startswith("api_key = "):
                wandb_key = line.split(" = ", 1)[1].strip()
                break
    if wandb_key:
        wandb.login(key=wandb_key)

    init_kwargs = {
        "id": args.wandb_run_id,
        "project": args.wandb_project,
        "resume": "allow",
        "reinit": True,
        "settings": wandb.Settings(
            mode="shared",
            x_primary=False,
            x_update_finish_state=False,
        ),
    }
    if args.wandb_team:
        init_kwargs["entity"] = args.wandb_team
    if args.wandb_dir:
        args.wandb_dir.mkdir(parents=True, exist_ok=True)
        init_kwargs["dir"] = str(args.wandb_dir)

    run = wandb.init(**init_kwargs)
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    wandb.define_metric("liveweb_arena/checkpoint_iteration")
    wandb.define_metric("liveweb_arena/*", step_metric="liveweb_arena/checkpoint_iteration")
    return run


def read_latest_iteration(watch_dir: Path) -> int | None:
    tracker = watch_dir / "latest_checkpointed_iteration.txt"
    if not tracker.exists():
        return None

    content = tracker.read_text().strip()
    if not content.isdigit():
        return None
    return int(content)


def get_iteration_dir(watch_dir: Path, iteration: int) -> Path:
    return watch_dir / f"iter_{iteration:07d}"


def convert_checkpoint(args: argparse.Namespace, iteration_dir: Path, converted_dir: Path) -> None:
    if converted_dir.exists():
        shutil.rmtree(converted_dir)
    converted_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python_bin,
        "tools/convert_torch_dist_to_hf.py",
        "--input-dir",
        str(iteration_dir),
        "--output-dir",
        str(converted_dir),
        "--origin-hf-dir",
        str(args.origin_hf_dir),
        "--force",
    ]
    log(f"Converting checkpoint {iteration_dir.name} to HF format.")
    subprocess.run(cmd, cwd=args.slime_dir, check=True)


def run_liveweb_eval(args: argparse.Namespace, iteration: int) -> dict:
    output_path = args.arena_output_dir / f"iter_{iteration:07d}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python_bin,
        "eval.py",
        "--model",
        args.served_model_name,
        "--base-url",
        f"http://{args.server_host}:{args.server_port}/v1",
        "--api-key",
        args.server_api_key,
        "--seed",
        str(args.seed),
        "--num-tasks",
        str(args.num_tasks),
        "--max-steps",
        str(args.max_steps),
        "--timeout",
        str(args.timeout),
        "--temperature",
        str(args.temperature),
        "--output",
        str(output_path),
        "--quiet",
        "--templates",
        *args.templates,
    ]

    env = os.environ.copy()
    env["API_KEY"] = args.server_api_key
    env["LIVEWEB_CACHE_DIR"] = str(args.arena_cache_dir)
    env["PYTHONPATH"] = str(args.arena_dir) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    log(f"Running LiveWeb Arena evaluation for iteration {iteration}.")
    try:
        subprocess.run(cmd, cwd=args.arena_dir, env=env, check=True)
    except subprocess.CalledProcessError as exc:
        if output_path.exists():
            log(
                f"LiveWeb Arena evaluation exited with code {exc.returncode}, "
                "but result file exists; logging the partial result."
            )
        else:
            log(f"LiveWeb Arena evaluation failed before producing output. returncode={exc.returncode}")
            return {
                "task_name": f"liveweb_arena:{args.num_tasks}tasks",
                "score": 0.0,
                "success": False,
                "time_taken": 0.0,
                "extra": {
                    "num_subtasks": args.num_tasks,
                    "returncode": exc.returncode,
                },
                "error": f"eval.py exited with code {exc.returncode}",
            }

    result = json.loads(output_path.read_text())
    if isinstance(result, dict) and "error" in result and result["error"]:
        extra = result.setdefault("extra", {})
        if isinstance(extra, dict):
            extra["had_error"] = True
    return result


def log_metrics(args: argparse.Namespace, result: dict, iteration: int, source: str) -> None:
    epoch = iteration / args.steps_per_epoch
    had_error = float(bool(result.get("error")))
    metrics = {
        "eval/step": iteration,
        "eval/liveweb_arena/checkpoint_iteration": iteration,
        "eval/liveweb_arena/epoch": epoch,
        "eval/liveweb_arena/source": source,
        "eval/liveweb_arena/score": result["score"],
        "eval/liveweb_arena/success": float(bool(result["success"])),
        "eval/liveweb_arena/time_taken": result["time_taken"],
        "eval/liveweb_arena/had_error": had_error,
        "liveweb_arena/checkpoint_iteration": iteration,
        "liveweb_arena/epoch": epoch,
        "liveweb_arena/source": source,
        "liveweb_arena/score": result["score"],
        "liveweb_arena/success": float(bool(result["success"])),
        "liveweb_arena/time_taken": result["time_taken"],
        "liveweb_arena/had_error": had_error,
    }
    if "extra" in result and isinstance(result["extra"], dict):
        metrics.update(flatten_numeric_metrics(result["extra"], "liveweb_arena/extra"))
        metrics.update(flatten_numeric_metrics(result["extra"], "eval/liveweb_arena/extra"))
    if wandb is not None and wandb.run is not None:
        wandb.log(metrics)

    summary_path = args.arena_output_dir / "summary.jsonl"
    with summary_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"iteration": iteration, "epoch": epoch, "source": source, **result}, ensure_ascii=True) + "\n")


def maybe_run_initial_eval(args: argparse.Namespace, server: ServerManager) -> None:
    if args.skip_initial_eval:
        return

    marker_path = args.arena_output_dir / "initial_eval_done.json"
    if marker_path.exists():
        return

    log("Running initial LiveWeb Arena baseline evaluation on origin HF model.")
    server.start(args.origin_hf_dir)
    result = run_liveweb_eval(args, 0)
    log_metrics(args, result, iteration=0, source="origin_hf")
    server.stop()
    marker_path.write_text(json.dumps(result, ensure_ascii=True, indent=2))


def prune_converted_dirs(history: deque[Path], keep_last: int) -> None:
    while len(history) > keep_last:
        stale = history.popleft()
        if stale.exists():
            log(f"Removing old converted checkpoint {stale}.")
            shutil.rmtree(stale, ignore_errors=True)


def main() -> int:
    args = parse_args()
    args.watch_dir.mkdir(parents=True, exist_ok=True)
    args.converted_root.mkdir(parents=True, exist_ok=True)
    args.arena_cache_dir.mkdir(parents=True, exist_ok=True)
    args.arena_output_dir.mkdir(parents=True, exist_ok=True)

    server = ServerManager(args)
    converted_history: deque[Path] = deque()
    last_evaluated = 0
    init_wandb(args)

    def handle_signal(signum, _frame):
        log(f"Received signal {signum}, shutting down.")
        server.stop()
        if wandb is not None and wandb.run is not None:
            wandb.finish()
        raise SystemExit(130)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        maybe_run_initial_eval(args, server)

        if args.only_initial_eval:
            log("Initial-only LiveWeb Arena evaluation finished.")
            return 0

        if args.only_iteration is not None:
            iteration_dir = get_iteration_dir(args.watch_dir, args.only_iteration)
            if not iteration_dir.exists():
                raise FileNotFoundError(f"Checkpoint iteration directory does not exist: {iteration_dir}")

            converted_dir = args.converted_root / iteration_dir.name
            convert_checkpoint(args, iteration_dir, converted_dir)
            server.start(converted_dir)
            result = run_liveweb_eval(args, args.only_iteration)
            log_metrics(args, result, args.only_iteration, source=iteration_dir.name)
            server.stop()
            log(f"Single-iteration LiveWeb Arena evaluation finished for iteration {args.only_iteration}.")
            return 0

        while True:
            latest_iteration = read_latest_iteration(args.watch_dir)
            if latest_iteration is None or latest_iteration <= last_evaluated:
                if last_evaluated >= args.stop_after_iteration:
                    break
                time.sleep(args.poll_seconds)
                continue

            iteration_dir = get_iteration_dir(args.watch_dir, latest_iteration)
            if not iteration_dir.exists():
                time.sleep(10)
                continue

            converted_dir = args.converted_root / iteration_dir.name
            convert_checkpoint(args, iteration_dir, converted_dir)
            converted_history.append(converted_dir)
            prune_converted_dirs(converted_history, args.keep_last_converted)

            server.start(converted_dir)
            result = run_liveweb_eval(args, latest_iteration)
            log_metrics(args, result, latest_iteration, source=iteration_dir.name)
            server.stop()

            last_evaluated = latest_iteration
            if last_evaluated >= args.stop_after_iteration:
                break

            time.sleep(args.poll_seconds)
    finally:
        server.stop()
        if wandb is not None and wandb.run is not None:
            wandb.finish()

    log("LiveWeb Arena watcher finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
