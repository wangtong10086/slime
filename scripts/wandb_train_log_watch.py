#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import time
from pathlib import Path

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse slime train.log and mirror train metrics to W&B.")
    parser.add_argument("--log-glob", type=str, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--wandb-project", type=str, required=True)
    parser.add_argument("--wandb-run-id", type=str, required=True)
    parser.add_argument("--wandb-team", type=str, default=None)
    parser.add_argument("--wandb-dir", type=Path, default=None)
    parser.add_argument("--wandb-key", type=str, default=None)
    parser.add_argument("--wandb-settings-file", type=Path, default=Path.home() / ".config" / "wandb" / "settings")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser.parse_args()


def init_wandb(args: argparse.Namespace):
    if wandb is None:
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
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    return run


def load_state(state_file: Path) -> dict[str, int]:
    if not state_file.exists():
        return {"last_train_step": -1, "offsets": {}}
    return json.loads(state_file.read_text())


def save_state(state_file: Path, state: dict[str, int]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state))


def extract_train_metrics(line: str) -> tuple[int, dict[str, float]] | None:
    marker = "step "
    idx = line.find(marker)
    if idx < 0:
        return None

    colon = line.find(":", idx)
    if colon < 0:
        return None

    step_str = line[idx + len(marker) : colon].strip()
    if not step_str.isdigit():
        return None

    payload = line[colon + 1 :].strip()
    if not (payload.startswith("{") and payload.endswith("}")):
        return None

    try:
        metrics = ast.literal_eval(payload)
    except Exception:
        return None

    if "train/step" not in metrics:
        metrics["train/step"] = int(step_str)
    return int(step_str), metrics


def process_new_lines(args: argparse.Namespace, state: dict[str, int]) -> dict[str, int]:
    offsets = state.setdefault("offsets", {})
    for log_path_str in sorted(glob.glob(args.log_glob)):
        log_path = Path(log_path_str)
        offset = offsets.get(log_path_str, 0)
        with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
            fh.seek(offset)
            while True:
                line = fh.readline()
                if not line:
                    break

                parsed = extract_train_metrics(line.rstrip("\n"))
                if parsed is None:
                    continue

                step, metrics = parsed
                if step <= state["last_train_step"]:
                    continue

                if wandb is not None and wandb.run is not None:
                    wandb.log(metrics)
                state["last_train_step"] = step

            offsets[log_path_str] = fh.tell()

    save_state(args.state_file, state)
    return state


def main() -> int:
    args = parse_args()
    init_wandb(args)
    state = load_state(args.state_file)

    try:
        while True:
            state = process_new_lines(args, state)
            time.sleep(args.poll_seconds)
    finally:
        if wandb is not None and wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    raise SystemExit(main())
