# AGENT.md

This file records the current operational rules for `slime` when running
LiveWeb SFT / RL training.

## Scope

`slime` is responsible for:

- SFT and RL training launch
- rollout orchestration
- worker control-plane calls
- rollout/sample timeout enforcement
- RL reward shaping and failure-bucket sampling
- training-time observability

It is **not** the source of truth for strict benchmark scoring semantics.
Strict benchmark semantics stay aligned with downstream strict judge and
upstream-compatible `strict_eval`.

## Preferred Launch Skill

When launching or updating LiveWeb SFT / RL runs, prefer using:

- `/home/xmyf/.codex/skills/slime-liveweb-training-launcher`

This skill exists to preserve the currently validated launch paths and to avoid
repeating known-bad experiments such as unsafe completion caps, proxy-polluted
control-plane traffic, or incompatible offload/allocator combinations.

## LiveWeb RL Current Priorities

The current main failure mode is **not** tool-call formatting.
The primary risks are:

- wrong-domain attraction
- bad-path persistence
- unsupported stop / answering without evidence
- long-tail samples that can stall an entire rollout round

When debugging RL, prioritize:

1. control-flow safety
2. sample cleanup / pending release
3. rollout throughput
4. reward shaping quality

Do not jump straight to reward tuning if rollout cleanup is still unreliable.

## Control-Plane Rules

### All Worker Control Calls Must Be Authenticated

Any control-plane endpoint must use the shared authenticated helper.

This includes:

- `abort_request`
- `server_info`
- `model_info`
- worker-list / worker-health calls

Authentication sources may come from:

- `args.sglang_api_key`
- `LIVEWEB_API_KEY`
- `SGLANG_API_KEY`
- `API_KEY`

But all control-plane calls must go through the same header-builder path.

### Control-Plane Traffic Must Bypass Proxy

Local SGLang/router/control-plane requests must not inherit the browser/system
proxy by accident.

Current invariant:

- external browser traffic may use proxy
- local control-plane traffic must use `trust_env=False`
- local service traffic must respect `NO_PROXY`

If a control-plane request hits a proxy unexpectedly, treat it as a bug.

### Auth Failures Are Infrastructure Failures

`401/403` on control-plane endpoints must be treated as fatal infrastructure
signals.

Do not silently keep retrying for long periods after:

- `abort_request 401`
- `server_info 401`
- `model_info 401`

These should be surfaced through explicit metrics and fail-fast handling.

## Rollout Safety Rules

### A Single Sample Must Not Stall a Whole Rollout Round

The old catastrophic failure mode was:

- bad trajectory
- recovery/context overflow
- abort/cleanup failure
- pending request never clears
- entire rollout round hangs for hours

This is not acceptable.

Current requirement:

- sample-level timeout
- group/round timeout
- bounded abort grace period
- force release from pending wait chain when cleanup fails

Prefer dropping one bad sample over hanging an entire training round.

### Context Overflow Must Fail Fast

If a sample hits:

- `400 Bad Request`
- model context overflow
- recovery-context overflow

it should become a sample-level failure such as:

- `llm_context_overflow`
- `format_recovery_overflow`

and terminate quickly.

Do not keep retrying the same pathological sample indefinitely.

### RL Completion Budget Must Stay Conservative

In current LiveWeb RL, the model mostly needs to emit short tool calls.
Therefore:

- rollout completion cap should stay conservative
- recovery completion cap should stay very small

Do not treat "response length can approach 32k" as a normal RL assumption.

Current working rule of thumb:

- RL main completion cap: conservative (`256-512` range by default)
- recovery cap: `<= 128`

## Offload / Memory Rules

### `offload_train` and `expandable_segments` Must Stay Split by Process

Known limitation:

- `offload_train` relies on `torch_memory_saver`
- `torch_memory_saver` is incompatible with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` inside the same process

So the allowed configuration is:

- train actor:
  - may use `offload_train`
  - must **not** inject `expandable_segments`
- rollout worker / SGLang:
  - may use `expandable_segments:True`

Do not attempt same-process coexistence again unless the underlying allocator
path changes.

### Prefer Role-Split Env Vars Over Global Env Assumptions

When rollout and train need different allocator behavior, split environment
variables by role instead of pushing the same env into every process.

## Reward / Sampling Notes

### RL Reward Still Centers on Official Score

Reward shaping exists to correct policy behavior, not replace the main target.

Keep the terminal objective anchored on official `score`, while using shaping to
penalize:

- wrong-domain persistence
- google-family offdomain behavior
- repeated URLs / same-page loops
- unsupported stop
- hallucinated plugin evidence

### Environment Pollution Must Be Neutralized

The following should not drive policy updates as ordinary model failures:

- challenge pages
- `site_unreachable`
- prefetch / cache fill failures
- environment navigation timeouts

These should be counted and logged, but neutralized or excluded from policy
update.

### Failure Buckets Are a First-Class Training Control

Current RL relies on bucketed sampling such as:

- `normal`
- `wrong_domain_loop`
- `premature_stop`
- `near_miss`

When modifying sampling, preserve explicit bucket accounting and logging.
Do not hide failure-bucket behavior behind opaque heuristics.

## Observability Requirements

The following signals must remain observable during RL debugging:

- `pending_samples`
- `pending_groups`
- `oldest_pending_age_seconds`
- `active_decode_requests`
- `abort_failed_count`
- `sample_wall_timeout_count`
- `llm_context_overflow_rate`
- `format_recovery_overflow_rate`
- `control_plane_auth_failure_count`

If a rollout appears stuck, these are the first signals to inspect.

## Bootstrap / Smoke Expectations

For small smoke runs, success means:

- the job enters rollout/update main loop
- bad samples time out and clear
- no repeated `abort_request 401`
- no infinite pending growth
- no multi-hour single-round stall

Bad policy behavior during smoke is acceptable.
A control-flow stall is not.

## Save / Load Smoke Rules

The RL launcher must support a minimal checkpoint lifecycle smoke:

1. launch a tiny fresh run
2. save a checkpoint
3. launch a second tiny run in `resume/checkpoint` mode
4. confirm the second run really loads the saved checkpoint and re-enters
   rollout/update

### Required Save-Side Checks

For a save smoke to count as successful, verify all of the following:

- `checkpoints_full/latest_checkpointed_iteration.txt` exists
- the log contains:
  - `successfully saved checkpoint`
  - `save/commit_end`
- the checkpoint directory contains actual shard files, not only a temporary
  staging directory

### Required Load-Side Checks

For a load smoke to count as successful, verify all of the following:

- `run_config.json` shows:
  - `liveweb_run_mode = resume`
  - `liveweb_resume_mode = checkpoint`
  - `resume_checkpoint_dir` points at the source run's checkpoint directory
- the actual launch command uses `--load <saved_checkpoint_dir>`
- the log contains:
  - `loading distributed checkpoint from ...`
  - `successfully loaded checkpoint from ...`
- after restore, the run reaches at least:
  - `startup/restore_phase_end`
  - `startup/rollout_phase_begin`

### Launcher Requirement

The tmux launcher must forward resume-related env vars into the child process.

Do not forget to pass:

- `LIVEWEB_RUN_MODE`
- `LIVEWEB_RESUME_MODE`
- `RESUME_CHECKPOINT_DIR`

If these are not forwarded, a so-called load smoke may silently become a fresh
run.

## Run Documentation Requirement

Every newly launched training run must include a human-readable run overview
document in the run directory.

Required file:

- `RUN_OVERVIEW.md`

Preferred skill:

- use `run-overview-writer` at
  `/home/xmyf/.codex/skills/run-overview-writer`
  when creating or updating this file

Default expectations:

- write it in Chinese unless there is a strong reason not to
- create/update it immediately after launch, not hours later
- keep it concise but complete enough that someone can understand the run
  without reconstructing launch arguments from logs

Each `RUN_OVERVIEW.md` should record at least:

- what the run is training
- the goal of the run
- model / checkpoint source
- algorithm used
- reward / shaping summary
- rollout / batch / completion-cap settings
- important timeout / offload / proxy settings
- what "success" means for this run

This applies to:

- SFT runs
- RL runs
- bootstrap / smoke runs
- eval-like training validation runs that create a dedicated run directory
