#!/usr/bin/env python3
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import httpx


def parse_server_specs(specs: str) -> List[Dict[str, object]]:
    servers = []
    for idx, item in enumerate(specs.split()):
        gpu_ids, port = item.split(":")
        servers.append(
            {
                "server_id": f"server-{idx}",
                "gpu_ids": gpu_ids,
                "port": int(port),
            }
        )
    return servers


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def wait_for_server(base_url: str, api_key: str, timeout_s: int) -> bool:
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"{base_url.rstrip('/')}/models"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            response = httpx.get(url, headers=headers, timeout=15.0)
            if response.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def get_listener_pid(port: int) -> int:
    try:
        result = subprocess.run(
            ["lsof", f"-iTCP:{port}", "-sTCP:LISTEN", "-t", "-n", "-P"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return 0

    for line in result.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            return int(line)
    return 0


def terminate_pid(pid: int):
    if not pid:
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            return


def wait_for_pid_exit(pid: int, timeout_s: int = 20) -> bool:
    if not pid:
        return True
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not is_pid_alive(pid):
            return True
        time.sleep(1)
    return not is_pid_alive(pid)


def log_path_for_server(log_dir: Path, server: Dict[str, object]) -> Path:
    return log_dir / f"sglang_{server['port']}.log"


def server_log_recently_active(log_path: Path, grace_s: int) -> bool:
    if grace_s <= 0 or not log_path.exists():
        return False
    now = time.time()
    try:
        if now - log_path.stat().st_mtime > grace_s:
            return False
    except FileNotFoundError:
        return False

    request_markers = ("POST /v1/chat/completions", "Prefill batch", "Decode batch", "#running-req:")
    try:
        text = log_path.read_text(errors="ignore")
    except Exception:
        return True

    recent_lines = text.splitlines()[-200:]
    return any(any(marker in line for marker in request_markers) for line in recent_lines)


def should_reuse_server(
    server: Dict[str, object],
    prev: Dict[str, object],
    args,
    log_dir: Path,
) -> bool:
    base_url = f"http://127.0.0.1:{server['port']}/v1"
    pid = int(prev.get("pid", 0)) if prev.get("pid") else 0
    if not pid or not is_pid_alive(pid):
        return False
    if not wait_for_server(base_url, args.api_key, timeout_s=5):
        return False
    if args.generation_id and prev.get("generation_id") not in (None, args.generation_id):
        return False
    if args.require_clean and server_log_recently_active(log_path_for_server(log_dir, server), args.activity_grace_s):
        return False
    return True


def load_state(state_path: Path) -> Dict[str, object]:
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text())


def save_state(state_path: Path, payload: Dict[str, object]):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def launch_server(server: Dict[str, object], args, log_dir: Path) -> int:
    tp_size = len(str(server["gpu_ids"]).split(","))
    log_path = log_dir / f"sglang_{server['port']}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(server["gpu_ids"])
    nccl_port = args.nccl_port_base + int(server["port"])

    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_dir,
        "--trust-remote-code",
        "--host",
        "127.0.0.1",
        "--port",
        str(server["port"]),
        "--api-key",
        args.api_key,
        "--served-model-name",
        args.served_model_name,
        "--nccl-port",
        str(nccl_port),
        "--tp-size",
        str(tp_size),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--context-length",
        str(args.context_length),
        "--dtype",
        args.dtype,
        "--disable-cuda-graph",
    ]

    if args.tool_call_parser:
        cmd.extend(["--tool-call-parser", args.tool_call_parser])
    if args.reasoning_parser:
        cmd.extend(["--reasoning-parser", args.reasoning_parser])
    if args.chat_template:
        cmd.extend(["--chat-template", args.chat_template])

    log_file = log_path.open("a")
    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
        close_fds=True,
    )
    return process.pid


def command_start(args):
    service_root = Path(args.service_root)
    log_dir = service_root / "logs"
    state_path = service_root / "state.json"
    pool_path = service_root / "server_pool.json"
    log_dir.mkdir(parents=True, exist_ok=True)

    servers = parse_server_specs(args.server_specs)
    previous_state = load_state(state_path)
    previous_servers = {entry["server_id"]: entry for entry in previous_state.get("servers", [])}
    previous_model_dir = previous_state.get("model_dir")
    previous_served_model_name = previous_state.get("served_model_name")
    if previous_servers and (
        previous_model_dir not in (None, args.model_dir)
        or previous_served_model_name not in (None, args.served_model_name)
    ):
        raise RuntimeError(
            f"Existing service_root {service_root} is bound to model_dir={previous_model_dir} "
            f"served_model_name={previous_served_model_name}; use another service root or stop it first"
        )

    running_servers = []
    for server in servers:
        server_id = server["server_id"]
        base_url = f"http://127.0.0.1:{server['port']}/v1"
        log_path = log_path_for_server(log_dir, server)
        prev = previous_servers.get(server_id, {})
        pid = int(prev.get("pid", 0)) if prev.get("pid") else 0
        if should_reuse_server(server, prev, args, log_dir):
            running_servers.append(
                {
                    **server,
                    "pid": pid,
                    "base_url": base_url,
                    "api_key": args.api_key,
                    "model_name": args.served_model_name,
                    "log_file": str(log_path),
                    "generation_id": args.generation_id,
                    "status": "reused",
                }
            )
            continue

        if pid and is_pid_alive(pid):
            terminate_pid(pid)
            wait_for_pid_exit(pid)

        listener_pid = get_listener_pid(int(server["port"]))
        if listener_pid and listener_pid != pid:
            active_prev = (
                server_log_recently_active(log_path, args.activity_grace_s)
                if args.require_clean else False
            )
            generation_matches = prev.get("generation_id") in (None, args.generation_id)
            if generation_matches and not active_prev and wait_for_server(base_url, args.api_key, timeout_s=5):
                running_servers.append(
                    {
                        **server,
                        "pid": listener_pid,
                        "base_url": base_url,
                        "api_key": args.api_key,
                        "model_name": args.served_model_name,
                        "log_file": str(log_path),
                        "generation_id": args.generation_id,
                        "status": "reused_port",
                    }
                )
                continue
            terminate_pid(listener_pid)
            wait_for_pid_exit(listener_pid)

        pid = launch_server(server, args, log_dir)
        running_servers.append(
            {
                **server,
                "pid": pid,
                "base_url": base_url,
                "api_key": args.api_key,
                "model_name": args.served_model_name,
                "log_file": str(log_path),
                "generation_id": args.generation_id,
                "status": "started",
            }
        )

    for server in running_servers:
        ready = wait_for_server(server["base_url"], args.api_key, timeout_s=args.ready_timeout)
        if not ready:
            raise RuntimeError(f"Server not ready: {server['server_id']} {server['base_url']}")
        if not is_pid_alive(int(server["pid"])):
            listener_pid = get_listener_pid(int(server["port"]))
            if listener_pid and wait_for_server(server["base_url"], args.api_key, timeout_s=5):
                server["pid"] = listener_pid
                server["status"] = "adopted_port_listener"
            else:
                raise RuntimeError(
                    f"Server pid exited and no healthy listener remained: {server['server_id']} {server['base_url']}"
                )
        server["status"] = "ready"

    state_payload = {
        "service_root": str(service_root),
        "model_dir": args.model_dir,
        "served_model_name": args.served_model_name,
        "server_specs": args.server_specs,
        "generation_id": args.generation_id,
        "servers": running_servers,
    }
    save_state(state_path, state_payload)
    save_state(pool_path, {"generation_id": args.generation_id, "servers": running_servers})
    print(json.dumps({"status": "ok", "server_pool_file": str(pool_path), "servers": running_servers}, indent=2))


def command_status(args):
    state_path = Path(args.service_root) / "state.json"
    state = load_state(state_path)
    if not state:
        print(json.dumps({"status": "empty", "service_root": args.service_root}, indent=2))
        return

    servers = []
    for server in state.get("servers", []):
        alive = is_pid_alive(int(server["pid"])) if server.get("pid") else False
        ready = alive and wait_for_server(server["base_url"], server.get("api_key", args.api_key or ""), timeout_s=3)
        servers.append({**server, "alive": alive, "ready": ready})
    print(json.dumps({"status": "ok", "servers": servers}, indent=2))


def command_stop(args):
    service_root = Path(args.service_root)
    state_path = service_root / "state.json"
    pool_path = service_root / "server_pool.json"
    state = load_state(state_path)
    stopped = []
    for server in state.get("servers", []):
        pid = int(server.get("pid", 0)) if server.get("pid") else 0
        if not pid or not is_pid_alive(pid):
            continue
        try:
            os.killpg(pid, signal.SIGTERM)
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                continue
        stopped.append({"server_id": server.get("server_id"), "pid": pid})

    if args.remove_state:
        if state_path.exists():
            state_path.unlink()
        if pool_path.exists():
            pool_path.unlink()

    print(json.dumps({"status": "ok", "stopped": stopped}, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Persistent SGLang service manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--service-root", required=True)
    common.add_argument("--api-key", default="local-liveweb")

    start_parser = subparsers.add_parser("start", parents=[common])
    start_parser.add_argument("--server-specs", required=True)
    start_parser.add_argument("--model-dir", required=True)
    start_parser.add_argument("--served-model-name", required=True)
    start_parser.add_argument("--context-length", type=int, default=32768)
    start_parser.add_argument("--mem-fraction-static", type=float, default=0.82)
    start_parser.add_argument("--dtype", default="bfloat16")
    start_parser.add_argument("--tool-call-parser", default=None)
    start_parser.add_argument("--reasoning-parser", default=None)
    start_parser.add_argument("--chat-template", default=None)
    start_parser.add_argument("--ready-timeout", type=int, default=1800)
    start_parser.add_argument("--nccl-port-base", type=int, default=10000)
    start_parser.add_argument("--generation-id", default=None)
    start_parser.add_argument("--require-clean", action="store_true")
    start_parser.add_argument("--activity-grace-s", type=int, default=20)

    status_parser = subparsers.add_parser("status", parents=[common])

    stop_parser = subparsers.add_parser("stop", parents=[common])
    stop_parser.add_argument("--remove-state", action="store_true")

    args = parser.parse_args()

    if args.command == "start":
        command_start(args)
    elif args.command == "status":
        command_status(args)
    elif args.command == "stop":
        command_stop(args)


if __name__ == "__main__":
    main()
