# Agent Notes

## Current Objective

- Manage the 2026-07-23 experiment schedule for `/home/jiale/MBRL/ContinualMBRL-wsc`.
- Active policy: keep only existing `no_wsc` and `wsc_WSC_grad_scale_constant_all` walker/dog runs from the old schedule; kill old lower-priority `init`, `factor`, `nograd`, and `wsc_no_scale` runs and old queueing shells.
- New experiments are managed by `auto_scripts/codex_experiment_scheduler.py` with state under `logdir/scheduler/`.
- Latest user instruction: keep the scheduler in the foreground of the active Codex session; do not rely only on a background daemon, do not voluntarily stop polling, and maximize concurrency under FPS/GPU/RAM safety constraints.
- Keep `task_checklist.md` compact. Since `codex-cli-executor.log` exists, experiment polling/progress belongs there, while durable operational context belongs here.

## W&B And Logging

- W&B project/group/name are inferred by `dreamerv3/main.py` from the last three `logdir` path components: `logdir/<project>/<group>/<run>`.
- W&B credentials belong only in `.env.wandb.local`, which is ignored by git via `.env*.local`.
- Do not print, copy, or commit the W&B API key.
- Periodically compact `codex-cli-executor.log`: retain one concise summary plus recent meaningful scheduler records, remove repetitive heartbeat detail, and avoid duplicating progress in `task_checklist.md`.
- W&B health is part of log legality: the active run must have fresh files under `logdir/.../wandb/wandb/run-*`, a fresh `logs/debug-internal.log`, and no unrecovered filestream fatal after the last `200 OK` upload.
- Remote W&B run state can drift to `crashed` even while active local filestream uploads keep returning `200 OK`. The scheduler runs a remote state check every foreground poll interval currently configured as `CODEX_SCHED_REMOTE_WANDB_CHECK_SECONDS=120`; when local health is clean and a remote active run is `crashed` or `failed`, it repairs the remote state to `pending`. W&B does not allow direct API transition to `running`; continued curve upload is verified via fresh local W&B logs and remote summary/filestream behavior.

## Existing Runs To Preserve

- Existing target experiments should be the 12 combinations:
  - Tasks: `walker_run|hopper_hop|fish_swim` and `dog_stand|dog_walk|dog_trot`.
  - Mechanisms/groups: `no_wsc` and `wsc_WSC_grad_scale_constant_all`.
  - Seeds: `1000`, `2000`, `3000`.
- If any of these are stopped, the scheduler may resume them, but it must still enforce the `5.8` FPS floor and OOM protection.

## New Queue

- Priority 1: DMC-prior continual task string `swimmer_swimmer6|cheetah_run|reacher_hard`.
  - This is the repo-compatible interpretation of the user wording `swimmer|halfcheetah|reacher_hard`.
  - Use `--run.task_interval 500000` and `--run.steps 7500000` for 5 cycles over 3 tasks.
  - Use `no_wsc` and `WSC_grad_scale_constant_all`, seeds `1000/2000/3000`.
  - Use a separate W&B project from the existing walker/hopper/fish project.
- Priority 2: Crafter single-task runs.
  - Use default `crafter` config except WSC/no-WSC setting, seed, logdir, and EGL/CUDA placement.
  - Use `no_wsc` and `WSC_grad_scale_constant_all`, seeds `1000/2000/3000`.
  - Latest user update: Crafter runs must use `--run.task_interval 100000000` and `--run.steps 100000000`; runs previously stopped at about `1100000` steps were stopped by an incorrect scheduler completion threshold and must resume from the existing logdir/checkpoint and W&B run id.
- Priority 3: Quadruped hard-task comparison.
  - Task: `quadruped_walk|quadruped_escape|quadruped_fetch`.
  - Use `WSC_grad_scale_constant_all`, seeds `1000/2000/3000`.
  - Hyperparameters follow `/home/jiale/MBRL/ContinualMBRL-soft-reset/auto_scripts/hard_task_quadruped.sh`: `size1m`, `train_ratio=1024`, `task_interval=1000000`, `reset_frequency=50000`, `reset_alpha=0.8`, `revive_epoch=0`, `revive_strategy=threshold`, `imag_length=15`, ReDo logging enabled.
  - Latest user correction: W&B project remains `continual_dreamer_soft_reset_quadruped_walk|quadruped_escape|quadruped_fetch_size1m`, but group must be `wsc_WSC_grad_scale_constant_all`, not a `sandp_*` group; run names remain `seed_<seed>`.
  - This WSC repo rejects `--run.reset_alpha`; keep the comparable W&B group name but do not pass that unsupported CLI flag.
- Priority 4: Humanoid continual task.
  - Task: `humanoid_stand|humanoid_run`.
  - Use `WSC_grad_scale_constant_all`, seeds `1000/2000/3000`.
  - Hyperparameters follow the humanoid entry in `auto_scripts/wsc_continual_scheduler.sh`: `size1m`, `train_ratio=1024`, `task_interval=3000000`, `reset_frequency=0`, `revive_epoch=0`, replay chunksize/cache from that script, WSC target `all`.
  - W&B project `continual_dreamer_soft_reset_humanoid_stand|humanoid_run_size1m`, group `wsc_WSC_grad_scale_constant_all`, run `seed_<seed>`.

## Safety Policy

- FPS floor: policy FPS should be at least `5.8` for every non-Crafter running experiment after a fresh continuous metrics sample exists. Crafter uses a relaxed scheduler floor, currently `0.5`, plus `21600` seconds freshness grace because Crafter logs every 10k steps and can be much slower.
- Launch only when all current fresh running samples are above the floor, system RAM is safe, and the selected GPU has enough free memory.
- Current high-concurrency foreground run uses up to `32` total runs, `4` runs per GPU, and `3500` MB minimum free GPU memory; GPU selection prioritizes the fewest active training processes, then lower memory-use ratio and lower utilization to keep CUDA load balanced. Health protection can roll back managed jobs or pause unmanaged low-FPS runs if throughput falls below threshold.
- If a process exits successfully, mark `scheduler_done` and launch the next queued job when safe.
- If a process fails or repeatedly falls below FPS after the grace window, stop it, log the reason, and continue managing remaining work without exposing secrets.

## Latest Snapshot

- 2026-07-25 HKT: wrong W&B remote runs are absent (`wrong_remote_present_count=0`), and local wrong archives/sandp quadruped dirs are absent (`local_wrong_dirs=0/0`).
- Scheduler treats any logdir containing `crafter` as a Crafter run, so the current foreground scheduler uses the relaxed `0.5` FPS threshold and `21600` second freshness grace for Crafter.
- Scheduler blocks new launches on the first fresh low-FPS sample, but stops/requeues only after two consecutive low-FPS samples; manual foreground management may pause a low fresh-FPS process immediately to satisfy the user's strict running-FPS rule.
- Quadruped seeds `1000`, `2000`, and `3000` were relaunched with corrected logdir-derived W&B naming, but each produced fresh FPS below `5.8` and was paused. Three later-discovered wrong remote runs in group `sandp_all_a0p8_50k_no_revive` were deleted on 2026-07-25. A later single-run seed `3000` probe with only six DMC-prior runs active produced no fresh metrics after about 2 minutes and was paused again.
- Crafter no_wsc and WSC seeds `1000/2000/3000` are resumed toward `100000000` total steps with `--run.task_interval 100000000`, original logdirs, and original W&B run ids.
- 2026-07-26 11:01 HKT snapshot: 12 active runs are healthy: six DMC-prior and six Crafter. GPU distribution is balanced as `0:1, 1:2, 2:1, 3:2, 4:2, 5:1, 6:2, 7:1`; JSONL metrics are parseable and monotonic, train logs are fresh, W&B local files/internal logs are fresh, and remote project/group/run names match. Remote W&B state continues to drift on some active runs, so the foreground scheduler now auto-repairs `crashed/failed -> pending` every 120 seconds after local health passes.
- DMC-prior seed `2000` no_wsc and WSC had unrecovered W&B filestream fatal logs on 2026-07-25 while local training continued. They were externally synced, then restarted from checkpoint with `WANDB_RUN_ID=o4f60wjm` and `WANDB_RUN_ID=c9z2r69x`, `WANDB_RESUME=must`; on 2026-07-26 they were repaired again after remote state drift, relaunched with original run ids, and remote W&B state is running. Local nonmonotonic metrics rows from checkpoint catch-up were removed from no_wsc seed `2000` with backups.
- 2026-07-26 19:50 HKT: 18 long-paused `T` state Dreamer processes from old walker/dog, humanoid, and quadruped probes were terminated because they were not in the active scheduler jobs, were not writing logs, and were holding about 150GiB RSS plus GPU allocations. This released GPU memory and reduced swap from about 6.3GiB to about 581MiB while preserving the current 12 active DMC/Crafter runs.
- 2026-07-26 20:38 HKT: scheduler retried quadruped seeds `1000/2000` and humanoid seeds `1000/2000/3000`. Quadruped seeds `1000/2000` produced fresh FPS below `5.8` and were stopped/marked failed; quadruped seed `3000` was already failed after max attempts. Humanoid seeds `1000/2000/3000` are active and healthy with fresh FPS above `5.8`.
