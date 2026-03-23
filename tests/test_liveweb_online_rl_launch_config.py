from pathlib import Path


RUN_SCRIPT = Path("/home/xmyf/slime/scripts/run-qwen3-32B-liveweb-online-rl.sh")
LAUNCH_SCRIPT = Path("/home/xmyf/slime/scripts/launch-qwen3-32B-liveweb-online-rl.sh")


def test_online_rl_run_script_uses_colocated_offload_train_defaults():
    script = RUN_SCRIPT.read_text(encoding="utf-8")

    assert 'TRAIN_PHASE="${TRAIN_PHASE:-bootstrap}"' in script
    assert 'warmup) TRAIN_PHASE="bootstrap"' in script
    assert 'main) TRAIN_PHASE="main_warm"' in script
    assert 'export LIVEWEB_ENABLE_AUTO_CURRICULUM="${LIVEWEB_ENABLE_AUTO_CURRICULUM:-1}"' in script
    assert 'export LIVEWEB_BROWSER_PROXY_MODE="${LIVEWEB_BROWSER_PROXY_MODE:-system}"' in script
    assert 'export LIVEWEB_BROWSER_STOOQ_DIRECT="${LIVEWEB_BROWSER_STOOQ_DIRECT:-0}"' in script
    assert 'slime_liveweb_sft_*/checkpoints' in script
    assert 'slime_liveweb_sft_*/checkpoints/iter_*' not in script
    assert 'TRAIN_MEMORY_MARGIN_BYTES="${TRAIN_MEMORY_MARGIN_BYTES:-0}"' in script
    assert 'TRAIN_PYTORCH_CUDA_ALLOC_CONF="${TRAIN_PYTORCH_CUDA_ALLOC_CONF:-}"' in script
    assert 'ROLLOUT_PYTORCH_CUDA_ALLOC_CONF="${ROLLOUT_PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"' in script
    assert 'ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-32768}"' in script
    assert 'LIVEWEB_COMPLETION_HEADROOM="${LIVEWEB_COMPLETION_HEADROOM:-4096}"' in script
    assert 'SAFE_LIVEWEB_MAX_COMPLETION_TOKENS=$((SGLANG_CONTEXT_LENGTH - LIVEWEB_COMPLETION_HEADROOM))' in script
    assert 'LIVEWEB_MAX_COMPLETION_TOKENS="${LIVEWEB_MAX_COMPLETION_TOKENS:-512}"' in script
    assert 'LIVEWEB_RL_SAMPLE_WALL_TIMEOUT_SECONDS="${LIVEWEB_RL_SAMPLE_WALL_TIMEOUT_SECONDS:-300}"' in script
    assert 'LIVEWEB_RL_ROLLOUT_ROUND_TIMEOUT_SECONDS="${LIVEWEB_RL_ROLLOUT_ROUND_TIMEOUT_SECONDS:-900}"' in script
    assert '--train-memory-margin-bytes "${TRAIN_MEMORY_MARGIN_BYTES}"' in script
    assert '"train_memory_margin_bytes": ${TRAIN_MEMORY_MARGIN_BYTES}' in script
    assert '--train-env-vars "${TRAIN_ENV_VARS_JSON}"' in script
    assert '"train_pytorch_cuda_alloc_conf": "${TRAIN_PYTORCH_CUDA_ALLOC_CONF}"' in script
    assert '"rollout_pytorch_cuda_alloc_conf": "${ROLLOUT_PYTORCH_CUDA_ALLOC_CONF}"' in script
    assert '"browser_proxy_mode": "${LIVEWEB_BROWSER_PROXY_MODE}"' in script
    assert '"browser_stooq_direct": ${LIVEWEB_BROWSER_STOOQ_DIRECT}' in script
    assert '"liveweb_enable_auto_curriculum": ${LIVEWEB_ENABLE_AUTO_CURRICULUM}' in script
    assert '"liveweb_curriculum_metrics_window": ${LIVEWEB_CURRICULUM_METRICS_WINDOW}' in script
    assert '--offload-train' in script
    assert '--no-offload-train' not in script
    assert '--offload-rollout' in script
    assert '"offload_train": true' in script
    assert '"offload_rollout": true' in script
    assert 'set_default_if_unset ROLLOUT_BATCH_SIZE 4' in script
    assert 'set_default_if_unset LIVEWEB_MAX_COMPLETION_TOKENS "${SAFE_LIVEWEB_MAX_COMPLETION_TOKENS}"' in script
    assert 'RL_DEFAULT_MAX_COMPLETION_TOKENS="${RL_DEFAULT_MAX_COMPLETION_TOKENS:-512}"' in script
    assert '--rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"' in script
    assert '"rollout_max_response_len": ${ROLLOUT_MAX_RESPONSE_LEN}' in script
    assert 'WARM_START_FROM_EXTERNAL_CHECKPOINT=0' in script
    assert 'COMMON_ARGS+=(--no-load-optim --no-load-rng --finetune)' in script
    assert '"warm_start_from_external_checkpoint": ${WARM_START_FROM_EXTERNAL_CHECKPOINT}' in script
    assert 'bootstrap)' in script
    assert 'main_warm)' in script
    assert 'online_align)' in script


def test_online_rl_launch_script_exports_zero_margin_default():
    script = LAUNCH_SCRIPT.read_text(encoding="utf-8")

    assert 'TRAIN_MEMORY_MARGIN_BYTES="${TRAIN_MEMORY_MARGIN_BYTES:-0}"' in script
    assert "TRAIN_MEMORY_MARGIN_BYTES='${TRAIN_MEMORY_MARGIN_BYTES}'" in script
