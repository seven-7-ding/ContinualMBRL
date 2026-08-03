# Agent Notes

## 2026-07-28 WSC Skip Last Layer Run

- Foreground scheduler session `88402` was stopped by explicit user request at 2026-07-30 15:52 HKT; do not assume active Codex monitoring is still running.
- Scheduler command uses `auto_scripts/codex_experiment_scheduler.py` with state `logdir/scheduler/codex_wsc_skip_last_layer_state.json` and PID file `logdir/scheduler/codex_wsc_skip_last_layer.pid`.
- Latest user requirement: monitoring has ended; Dreamer experiments should continue in the background and should not be stopped unless explicitly requested.
- Current foreground scheduler was restarted at 2026-07-28 22:58 HKT with `CODEX_SCHED_MIN_FPS=0`, `CODEX_SCHED_CRAFTER_MIN_FPS=0`, and very large freshness grace values so FPS or short logging gaps do not trigger health stops.
- Managed new queue is exactly 9 runs under group `wsc_skip_last_layer_constant_all`: dog seeds `1000/2000/3000`, crafter seeds `1000/2000/3000`, and walker/hopper/fish seeds `1000/2000/3000`.
- Six pre-existing `no_wsc` Dreamer processes are passive monitoring targets only.
- Dog seed1000 hit transient `ptxas` 139 twice; scheduler was restarted with `CODEX_SCHED_MAX_ATTEMPTS=100`, `CODEX_SCHED_LAUNCH_BATCH=1`, and the job was requeued without stopping other runs.

## 2026-08-01 WSC Last L2 Init Run

- Added `wsc_{constant/init/factor}_last_l2_init_{weight_decay}_{target}` support. It behaves like skip-last-layer WSC for RMSNorm-followed target layers and applies l2-init gradient penalty to target layers skipped because they are not followed by RMSNorm.
- `wsc_constant_last_l2_init_2e-5_all` experiments were launched directly, not under the old foreground scheduler.
- Old `wsc_skip_last_layer_constant_all` Crafter seeds were stopped by explicit user request; dog/simple skip-last-layer runs were left untouched.
- New live PIDs at launch confirmation: Crafter `606833/606834/606835` on GPUs `3/4/1`; continual MuJoCo `606836/613245/606838` on GPUs `0/5/2`.
- MuJoCo seed `2000` hit transient JAX/XLA `ptxas` 139 twice on compile and was relaunched successfully as PID `613245`.
- Final launch confirmation used process liveness, clean current train.log tails, W&B run directories, and `debug-internal.log` `200 OK` upload entries. Local `metrics.jsonl` had not reached its first write interval yet.

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

- Current scheduler state: stopped by user request after pausing all live WSC jobs. Only six `no_wsc` Dreamer processes remain running. WSC jobs were marked queued with `paused_reason=user_requested_pause_all_wsc_keep_no_wsc` and far-future `retry_after`.
- Launch only when all current fresh running samples are above the floor, system RAM is safe, and the selected GPU has enough free memory.
- Current high-concurrency foreground run uses up to `40` total runs, `5` runs per GPU, and `2000` MB minimum free GPU memory. New launches exclude `cuda:0,1,2,4` to leave `cuda:0` for the user and protect all humanoid GPUs. Health protection may only roll back scheduler-owned jobs. Do not pause, kill, repair, or otherwise manage external/unowned processes, especially user experiments on `cuda:0`.
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
- 2026-07-26 21:32 HKT: to satisfy the user's higher concurrency request, the scheduler now includes the old target walker/dog `no_wsc` and `wsc_WSC_grad_scale_constant_all` jobs as lower-priority resume jobs. With FPS enforcement, the system reached 24 active jobs (`3` per GPU). Six jobs failed/are excluded for FPS or errors: quadruped seeds `1000/2000/3000`, walker no_wsc seed `1000`, dog no_wsc seed `1000`, and dog WSC seed `3000`. Do not force these just to reach 4 per GPU unless the FPS requirement is changed.
## 2026-07-26 22:23 HKT CUDA-0 Boundary
- User may deploy additional experiments on `cuda:0`; scheduler must not stop/pause/repair/manage external processes.
- `auto_scripts/codex_experiment_scheduler.py` now filters health and remote W&B repair to scheduler-owned running PIDs only, skips unmanaged cuda:0 Dreamer processes during bootstrap cleanup, and supports `CODEX_SCHED_EXCLUDE_GPUS=0` for new launches.
- Current foreground scheduler should be run with `CODEX_SCHED_MIN_FPS=5.6` for the user-approved “around 5.8” floor; strict logging/freshness checks remain enabled.
- Current foreground scheduler session is `98334` after the 2026-07-27 12:17 HKT Crafter pause/requeue. Restart command should include `CODEX_SCHED_HUMANOID_LOW_FPS_CONFIRM_COUNT=4`, `CODEX_SCHED_LOW_FPS_REPEAT_SECONDS=600`, and `CODEX_SCHED_HUMANOID_LOW_FPS_REPEAT_SECONDS=1200`.
- Health logic now increments low-FPS confirmation counts only when a new metrics sample appears (`step` or metrics mtime changes). This prevents one stale low-FPS sample from being counted repeatedly on every poll.
- If the same low-FPS metrics sample remains current for the repeat window, it can count again: 600 seconds for normal non-Crafter jobs and 1200 seconds for humanoid. This avoids both rapid false repeated counts and leaving very slow low-FPS probes alive until full stale timeout.
- For simultaneous bad jobs, health protection ranks by FPS/threshold severity before static job priority, then requeues only scheduler-owned PIDs. This prevents very low-FPS probe jobs from unnecessarily crowding healthier long-running jobs.

## 2026-07-27 12:15 HKT Crafter Pause
- User requested temporarily stopping the Crafter runs to free capacity for downstream tasks.
- The six scheduler-owned Crafter processes were terminated cleanly and marked queued with a far-future `retry_after`; do not treat this as completion or deletion.
- Crafter logdirs, checkpoints, and W&B run directories are preserved for later resume.
- Quadruped WSC seeds were requeued immediately to use the freed resources.

## 2026-07-27 13:09 HKT Resource-Priority Concurrency
- User clarified resources are abundant and requested as much concurrent experiment scheduling as possible.
- Current foreground scheduler session is `50579`.
- Current run uses `CODEX_SCHED_MAX_TOTAL_RUNNING=40`, `CODEX_SCHED_MAX_RUNS_PER_GPU=5`, `CODEX_SCHED_LAUNCH_BATCH=12`, `CODEX_SCHED_MIN_FPS=2.5`, `CODEX_SCHED_GPU_MIN_FREE_MB=2000`, and `CODEX_SCHED_MAX_ATTEMPTS=100` to keep more work alive while preserving stale-log, OOM, W&B, and scheduler-owned PID protections.
- After the user repeated that CUDA load was far from saturated, Crafter was resumed from the temporary pause and previous failed/queued lower-priority jobs were requeued to fill capacity.

## 2026-07-27 14:25 HKT No-Crafter Focus
- User corrected scope: do not restart Crafter for now.
- Latest foreground scheduler session is `21357`.
- Focus only on humanoid, dog, quadruped, and the two three-task MuJoCo/DMC chains: `swimmer_swimmer6|cheetah_run|reacher_hard` and `walker_run|hopper_hop|fish_swim`.
- Six Crafter Dreamer processes and their W&B child processes were stopped again; Crafter jobs are queued with a far-future retry to preserve resumability without launching.
- Dog no_wsc runs are also delayed for this round per user request; currently running dog no_wsc seed1000/3000 plus W&B child processes were stopped, and all `p6_old_dog_no_wsc_*` jobs are queued with far-future retry.
- Dog-series jobs now have a 6m-step stop rule in `auto_scripts/codex_experiment_scheduler.py`: new launches include `--run.steps 6000000`, and running jobs are actively stopped/marked done once metrics reach 6,000,000 steps.
- Pay extra attention to humanoid/quadruped actual training: metrics freshness, train.log freshness, W&B local/internal log freshness, and whether JSONL metrics are parseable and meaningful.
- Current priority mode uses `CODEX_SCHED_MIN_FPS=2.0`: low FPS below the old 2.5 threshold should not block humanoid/quadruped recovery, but stale metrics/logs still must trigger restart.

## 2026-07-27 16:12 HKT Priority Health Snapshot
- User requested priority order: keep humanoid and quadruped smooth first, then use remaining CUDA for dog WSC, DMC-prior, and walker-chain tasks until load is close to saturated.
- Current foreground scheduler session is `2114`; new launches exclude `cuda:0`, Crafter remains paused, and dog `no_wsc` remains delayed.
- Audit found 21 scheduler-owned running Dreamer jobs, no unmanaged Dreamer jobs, no Crafter/dog-no_wsc active jobs, and no queued allowed jobs left to launch.
- Humanoid/quadruped metrics JSONL tails are parseable and train/W&B logs are writing. Some metrics files update slowly, so scheduler should keep prioritizing freshness-based repair for those six jobs before adding any future lower-priority work.
- Foreground manager now uses `CODEX_SCHED_FRESH_GRACE_SECONDS=7200` because humanoid/quadruped are running around 2-3 FPS and write metrics about every 10k steps; train.log and W&B freshness remain the primary short-interval liveness signals between metrics samples.

## 2026-07-27 18:48 HKT GPU1 Humanoid Protection
- Humanoid seed1000 produced a fresh legal metrics sample at step 950000 with fps about 1.92. To avoid killing or destabilizing humanoid, the foreground threshold was relaxed from `2.0` to `1.8`.
- DMC-prior no_wsc seed1000 was moved off GPU1: the first relaunch landed back on GPU1, so the foreground scheduler was restarted with `CODEX_SCHED_EXCLUDE_GPUS=0,1` for new launches. It relaunched on GPU3 as PID `3958625`.
- Current foreground scheduler session is `58904`; keep using `CODEX_SCHED_MIN_FPS=1.8`, `CODEX_SCHED_FRESH_GRACE_SECONDS=7200`, and `CODEX_SCHED_EXCLUDE_GPUS=0,1` unless the user changes the policy.
- 2026-07-27 20:17 HKT update: humanoid seed1000 remained GPU1-isolated but logged about 1.75 FPS at step 960000, so the foreground threshold was relaxed again to `CODEX_SCHED_MIN_FPS=1.6` to keep priority humanoid runs alive while legal train/W&B logging continues. Current foreground scheduler session is `47220`.
- 2026-07-27 22:25 HKT update: humanoid seed2000 logged about 1.57 FPS at step 1000000 while sharing GPU4 with DMC WSC and walker WSC. Walker WSC seed2000 was paused for about one hour to free GPU4, and the foreground threshold was relaxed to `CODEX_SCHED_MIN_FPS=1.5`. Current foreground scheduler session is `44706`.
- 2026-07-27 23:28 HKT update: scheduler relaunched walker WSC seed2000 back onto GPU4 from in-memory state, so it was stopped again with the manager paused and requeued for about 30 minutes. Current foreground scheduler session is `23043`; keep GPU4 free until dog WSC seed3000 reaches 6m and frees GPU5.
- 2026-07-28 00:24 HKT update: dog WSC seed3000 reached 6m and was marked done. DMC WSC seed2000 and walker WSC seed2000 were then relaunched onto GPU5 with `CODEX_SCHED_EXCLUDE_GPUS=0,1,4`, keeping GPU4 exclusive for humanoid seed2000. Current foreground scheduler session is `40561`; active count is 19 with dog WSC seed1000/3000 done and seed2000 still running.
- 2026-07-28 01:00 HKT update: walker WSC seed3000 recovered from stale W&B logging but first relaunched onto GPU2, which hosts humanoid seed3000. It was moved again and relaunched on GPU3. Current foreground scheduler session is `26551`; use `CODEX_SCHED_EXCLUDE_GPUS=0,1,2,4` to protect user/GPU0 and all humanoid GPUs.
- 2026-07-28 01:32 HKT update: H/Q metrics cadence at 1.4-1.7 FPS is close to 7200 seconds per 10k-step metrics interval, while train.log and W&B are fresh. Foreground scheduler was restarted as session `73311` with `CODEX_SCHED_FRESH_GRACE_SECONDS=10000` to avoid false stale-metrics kills for priority H/Q runs.
- 2026-07-28 03:44 HKT update: humanoid seed1000, already isolated on GPU1, logged about 1.36 FPS at step 1000000. Foreground scheduler was restarted as session `8102` with `CODEX_SCHED_MIN_FPS=1.3` to keep priority humanoid runs alive.
- 2026-07-28 04:02 HKT update: W&B health logic was patched so stale `debug-internal.log` alone does not trigger a restart when other files in the active W&B run directory are fresh. This avoids false restarts for priority H/Q runs whose W&B files keep updating. Current foreground scheduler session is `69479`.
- 2026-07-28 17:05 HKT update: user requested pausing all WSC and exiting the task. Foreground scheduler session `34262` was stopped. Thirteen scheduler-owned live WSC Dreamer main processes and their descendants were terminated and marked queued/paused. Verification found `live_wsc=0`, `live_scheduler=0`, and six live `no_wsc` Dreamer processes: DMC-prior seeds `1000/2000/3000` and walker-chain seeds `1000/2000/3000`.
- 2026-08-03 22:13 HKT update: activation diagnostics task completed. Backup/resume data for the 27 paused Dreamer runs is in `logdir/resume_backups/dreamer_resume_20260803_211744.{json,sh}`. All 27 backup records were resumed and matched to live Dreamer processes with no duplicates. W&B internal logs were fresh for all 27; local metrics were fresh for 24 and included the new `act_redo/Zombie_Percentage`, `act_redo/Saturation_Percentage`, and `act_redo/Variation_Rank_0.9/0.95/0.99` keys. Baseline Crafter seeds 1000/2000/3000 reached `Start training loop` after loading checkpoints with legacy `wsc/init_params` extras ignored; their local metrics will update at the next natural log interval.
