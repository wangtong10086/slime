from pathlib import Path


RUN_SCRIPT = Path("/home/xmyf/slime/scripts/run-qwen3-32B-liveweb-sft.sh")
LAUNCH_SCRIPT = Path("/home/xmyf/slime/scripts/launch-qwen3-32B-liveweb-sft-tmux.sh")


def test_sft_run_script_wires_oom_guardrails():
    script = RUN_SCRIPT.read_text(encoding="utf-8")

    assert 'MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-512}"' in script
    assert 'LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-2048}"' in script
    assert 'RECOMPUTE_LOSS_FUNCTION="${RECOMPUTE_LOSS_FUNCTION:-1}"' in script
    assert 'TRAIN_STEP_TOKEN_BUDGET="${TRAIN_STEP_TOKEN_BUDGET:-48000}"' in script
    assert 'TRAIN_STEP_LOGIT_BUDGET="${TRAIN_STEP_LOGIT_BUDGET:-24000}"' in script
    assert 'TRAIN_MAX_TOTAL_TOKENS_PER_SAMPLE="${TRAIN_MAX_TOTAL_TOKENS_PER_SAMPLE:-4096}"' in script
    assert 'TRAIN_MAX_RESPONSE_TOKENS_PER_SAMPLE="${TRAIN_MAX_RESPONSE_TOKENS_PER_SAMPLE:-2048}"' in script
    assert '--log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}"' in script
    assert '--recompute-loss-function' in script
    assert '"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"' in script
    assert '"TRAIN_STEP_TOKEN_BUDGET": os.environ["TRAIN_STEP_TOKEN_BUDGET"]' in script
    assert '"TRAIN_STEP_LOGIT_BUDGET": os.environ["TRAIN_STEP_LOGIT_BUDGET"]' in script
    assert '"train_step_token_budget": int(os.environ["TRAIN_STEP_TOKEN_BUDGET"])' in script
    assert '"train_step_logit_budget": int(os.environ["TRAIN_STEP_LOGIT_BUDGET"])' in script
    assert '"use_dynamic_batch_size": True' in script


def test_sft_launch_script_exports_recommended_defaults():
    script = LAUNCH_SCRIPT.read_text(encoding="utf-8")

    assert 'TRAIN_TP_SIZE="${TRAIN_TP_SIZE:-2}"' in script
    assert 'TRAIN_PP_SIZE="${TRAIN_PP_SIZE:-2}"' in script
    assert 'TRAIN_CP_SIZE="${TRAIN_CP_SIZE:-2}"' in script
    assert 'MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-512}"' in script
    assert 'LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-2048}"' in script
    assert 'RECOMPUTE_LOSS_FUNCTION="${RECOMPUTE_LOSS_FUNCTION:-1}"' in script
    assert 'TRAIN_STEP_TOKEN_BUDGET="${TRAIN_STEP_TOKEN_BUDGET:-48000}"' in script
    assert 'TRAIN_STEP_LOGIT_BUDGET="${TRAIN_STEP_LOGIT_BUDGET:-24000}"' in script
    assert 'TRAIN_MAX_TOTAL_TOKENS_PER_SAMPLE="${TRAIN_MAX_TOTAL_TOKENS_PER_SAMPLE:-4096}"' in script
    assert 'TRAIN_MAX_RESPONSE_TOKENS_PER_SAMPLE="${TRAIN_MAX_RESPONSE_TOKENS_PER_SAMPLE:-2048}"' in script
    assert "export LOG_PROBS_CHUNK_SIZE='${LOG_PROBS_CHUNK_SIZE}'" in script
    assert "export RECOMPUTE_LOSS_FUNCTION='${RECOMPUTE_LOSS_FUNCTION}'" in script
    assert "export TRAIN_STEP_TOKEN_BUDGET='${TRAIN_STEP_TOKEN_BUDGET}'" in script
    assert "export TRAIN_STEP_LOGIT_BUDGET='${TRAIN_STEP_LOGIT_BUDGET}'" in script
