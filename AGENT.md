# Agent Notes

## 2026-07-28 Mechanism Experiments

- Use only the existing conda env `dreamer`; do not create a new Python environment.
- New mechanisms are implemented through `embodied.jax.wsc.MechanismController` while keeping the old `WSC` alias for import/checkpoint compatibility.
- Requested mechanisms all target `all`: `l2_decay`, `l2_init`, and `continual_backprop`.
- `continual_backprop` is scoped to feature selection plus weight reset only; optimizer momentum/step-size logic and model architecture stay unchanged.
- W&B credentials are stored only in `.env.wandb.local`, which is ignored by git.
- New scheduler: `auto_scripts/codex_mechanism_scheduler.py`. It manages only PIDs it launches and must not kill, pause, or repair unrelated processes.
- New experiment logdirs are under `logdir/continual_dreamer_soft_reset_size1m/{l2_decay_2e-5,l2_init_2e-5,continual_backprop}/seed_{1000,2000,3000}`.

## 2026-07-31 Crafter/Dog L2-Init Runs

- Six corrected l2-init runs were launched with `auto_scripts/launch_crafter_dog_l2_init.py`; the launcher exits after starting processes and does not monitor or signal unrelated processes.
- Correct local logdirs are under `logdir/continual_dreamer_soft_reset_crafter_size1m/l2_init_2e-5/seed_{1000,2000,3000}` and `logdir/continual_dreamer_soft_reset_dog_stand|dog_walk|dog_trot_size1m/l2_init_2e-5/seed_{1000,2000,3000}` so W&B uses the requested existing projects.
- Correct launched PIDs: crafter seeds 1000/2000/3000 -> `460873`/`460874`/`460875` on GPUs 0/3/5; dog seeds 1000/2000/3000 -> `460876`/`460877`/`460878` on GPUs 7/6/4.
- Dog task sequence is `dog_stand|dog_walk|dog_trot` with `obs_dim=223`, `task_action_space=[38]`, `task_interval=2000000`, and `steps=6000000`; crafter follows the WSC repo reference with `steps=100000000` and `task_interval=100000000`.
- The earlier mistaken `logdir/crafter_dog_l2_init` runs were stopped, matching W&B runs were deleted, and the local wrong directory was archived to `_archived_wrong_crafter_dog_l2_init_20260731_1914`.

## 2026-08-01 L2 Decay Preupdate Runs

- Mechanism `l2_decay_preupdate` is implemented as pre-update parameter scaling: `new = old * (1 - weight_decay) + (optimizer_new - old)`, wired through `MechanismController.preupdate_step()`.
- The mistaken replacement runs under local group `l2_decay_2e-5` were stopped (`595956`/`595957`/`595958`), their W&B run ids ending in `preupdate-v1` were deleted, and the wrong local folder was removed.
- Correct replacement runs were launched with `auto_scripts/launch_l2_decay_preupdate_replacement.py` under `logdir/continual_dreamer_soft_reset_size1m/l2_decay_preupdate_2e-5/seed_{1000,2000,3000}`.
- Correct launched PIDs: seeds 1000/2000/3000 -> `600498`/`600499`/`600500` on GPUs 5/4/0. W&B project is `continual_dreamer_soft_reset_size1m`, group is `l2_decay_preupdate_2e-5`, run ids end in `groupfix-v1`.

## 2026-08-07 Crafter Data Augmentation Runs

- Data augmentation baselines are configured by `agent.data_augmentation.mode` with values `disabled`, `batch_align`, and `batch_aug`; aliases `data_augmentation_batch_align` and `data_augmentation_batch_aug` are accepted.
- The augmentation is a DrQ-v2-style random shift with replicate padding `pad=4`, applied only when `Agent.loss(..., training=True)` runs. Policy inference and report/eval paths keep raw observations.
- Formal Crafter runs should live under `logdir/continual_dreamer_soft_reset_crafter_size1m/data_augmentation_batch_{align,aug}/seed_{1000,2000,3000}` so W&B uses the existing Crafter project and the mechanism group names.

## 2026-08-13 DMC Vision L2-Init 2e-4 Runs

- Deployed the baseline vision sweep under `logdir/continual_dreamer_soft_reset_walker_run|hopper_hop|fish_swim_vision_size12m_1m_x5/l2_init_2e-4/seed_{1000,2000,3000}` using `dmc_vision continual_dmc_vision size12m`, task `walker_run|hopper_hop|fish_swim`, `agent.data_augmentation.mode=batch_align`, `agent.data_augmentation.pad=4`, and `l2_init` mechanism/weight `2e-4`.
- Active main PIDs at validation: seed 1000 -> `3883359` on `--egl_device 5`, seed 2000 -> `3868633` on `--egl_device 1`, seed 3000 -> `3868637` on `--egl_device 7`.
- Validation completed at 2026-08-13 23:44 HKT with valid `metrics.jsonl` logging counts seed 1000 `5/5` at step 50000, seed 2000 `6/5` at step 60000, and seed 3000 `5/5` at step 50000; all three main processes were still alive.
- Seed 1000 hit transient JAX compile `ptxas exited with non-zero error code 139` on earlier attempts; the stable run was launched via `setsid` and compiled successfully before entering the training loop.

## 2026-08-16 DMC Vision Pause

- User requested stopping all live tasks for W&B project `continual_dreamer_soft_reset_walker_run|hopper_hop|fish_swim_vision_size12m_1m_x5` and freeing CPU/CUDA.
- Cross-repo backup/resume bundle is `/home/jiale/MBRL/vision_project_pause_backup_20260816T165045_HKT`; it records 12 main Dreamer training PIDs, 9 `wandb_metrics_proxy.py` PIDs, 45 descendants, log tails, process snapshots, and restart scripts.
- All 66 target PIDs exited after SIGTERM. Verification found no exact-project process alive and no target PID in `nvidia-smi`; unrelated size1m tasks were left running.

## 2026-08-16 Walker/Hopper/Cheetah Vision Size1m

- Foreground supervisor is running from the WSC repo at `/home/jiale/MBRL/ContinualMBRL-wsc/tools/launch_walker_hopper_cheetah_size1m_supervisor.py`.
- New project: `continual_dreamer_soft_reset_walker_run|hopper_hop|cheetah_run_vision_size1m_1m_x5`; baseline-owned groups are `no_wsc`, `l2_init_2e-4`, and `data_augmentation_batch_align`, each with seeds `1000/2000/3000`.
- Settings mirror the prior DMC vision run except `size1m` and task `walker_run|hopper_hop|cheetah_run`: `task_interval=1000000`, `task_repeat=5`, `steps=15000000`, train ratio `256`, data augmentation `batch_align` where applicable, l2-init weight `2e-4`, and activation/gradient logging `log+erank+srank`.
- The first baseline launch attempt failed because this repo does not define `--run.log_policy_video`; the supervisor was patched to omit that flag and the baseline jobs were relaunched to initialization/compile. Supervisor state is in `/home/jiale/MBRL/ContinualMBRL-wsc/logdir/continual_dreamer_soft_reset_walker_run|hopper_hop|cheetah_run_vision_size1m_1m_x5/_supervisor/state.json`.
- 2026-08-16 17:36 HKT: all 9 baseline-owned jobs wrote valid `metrics.jsonl` rows at step `10000` while the foreground supervisor remained active. Data augmentation rows had `592` keys, no_wsc rows `588`, and l2_init rows `764`; FPS was roughly `15.49-26.04` across all 12 jobs.

## 2026-10-15 Two-player Dreamer sweep

- User explicitly requires conda `dreamer` (overrides repo-experiment-builder's default `bit`). Interpreter: `/home/jdingav/.conda/envs/dreamer/bin/python`.
- User requires preserving the original 40 training processes. Only manage PIDs created by this sweep, verify PID creation time and ownership, and never signal unrelated processes.
- Entry point: `auto_script/two_player.sh`; adjustable training constants are at the top of `auto_scripts/launch_two_player.py`.
- Sweep: default plus wm_as_leader/agent_as_leader/best_response at k=32/128, seeds 1000/2000/3000; walker_run|finger_spin|fish_swim, priori size1m, 500k per task, 10 full cycles, 15M total, base train_ratio=256. Dog_stand is deferred.
- Each run: `logdir/two_player_dreamer_walker_run|finger_spin|fish_swim_size1m_500k_x10/<setting>/seed_<seed>/`. Logs, replay, checkpoints and private caches/temp belong here. Controller state/evidence: `logdir/two_player_sweep_20261015/manifest.json`.
- W&B IDs are stable in the manifest. Resume the same logdir and ID using `--retry-failed`; use `--only setting:seed` for a single run. Do not create replacement W&B runs on resume.
- Original default optimizer behavior is preserved; nondefault settings use independent WM/agent optimizers, resample and re-imagine every agent update, and publish the completed block's latest policy. All parameter reset mechanisms are disabled.
- `embodied/run/diagnostics.py` adapts elements 3.22 malloc initialization, GC aggregation, and dead weak proxies. Preserve these adapters.
- The completion condition for this deployment task is all 21 live runs with at least five finite local logs and verified W&B uploads; full 15M-step training continues afterward.

- Latest user steering: all metric logs strictly at environment steps 10000, 20000, ...; W&B common axis `env_step`. V1 wall-time-logging runs were explicitly deleted (19 W&B/local runs), original 40 PIDs preserved. Restart from step 0 with v2-step10k IDs only after speed optimization validation.
- Work in progress: `train_many` fuses at most 8 fresh batches with sequential current-parameter rollouts and optimizer updates; metrics stay on GPU until each 10k environment-step boundary. Prefetch is bounded at 8. CPU group=1 and group=4 equivalence passed after fixing catastrophic cancellation in `TwoHot.pred` using a stable expm1 formulation of the same expectation. Final 16-test suite passed, including default joint fusion and metric checkpoint restoration. Final 256-update GPU benchmark (including ReDo checks) measured 21.07 -> 47.31 actor updates/s (2.246x), peak JAX 333 MB and RSS 1.85 GiB; model parameter max difference versus sequential GPU training was 6.41e-7. Results in controller benchmark_gpu/result.json. V2 default/1000 produced a valid step-10000 row with 712 metrics, checkpointed step 12080. Found elements WandBOutput leaves explicit-step rows pending; added StepAlignedWandBOutput with commit=True and corrected RGB image shape validation (additional test passes). Controlled restart of 6 sweep-owned processes only; same W&B IDs and checkpoints retained. Current full launcher is running with --retry-failed; default/1000 new PID 1198611 on GPU 7. Manifest is authoritative for other PIDs. Startup readiness is reset per attempt. GPU allocation uses measured WM and agent costs plus pending-only swaps; all cards have 2–3 sweep jobs. Skip unused single-step precompilation in formal runs; reports start at 20k rather than startup.
- Runtime media fix: MoviePy/imageio/ffmpeg installed, Pillow pinned 11.3; `WANDB_SILENT=true` avoids a hanging W&B video spinner in headless mode. Silent GIF encoding verified.

- 2026-10-15 22:14 HKT read-only audit found historical original 800xxx PIDs no longer present, while 40 other GPU training processes with 109xxxx/112xxxx/113xxxx/115xxxx PIDs are active. No signal was sent to them. Snapshot of the current protected processes is `logdir/two_player_sweep_20261015/protected_processes_current.json`; preserve both historical audit and these current processes. Do not claim historical PIDs remain alive.

- 2026-10-15 22:52 HKT: all 21 V2 specs launched, 19 training-ready. Foreground launcher session 37341 monitors five local valid rows each; final remote verification still pending. Default/1000 has local+remote 10k and 20k logs, update counter 1989 -> 4489 (2500 per 10k as required), checkpoint step12080. Optional isolated CUDA Graph=2 benchmark passed model numeric checks but gave no evidence for production adoption under shared GPU load; production flags unchanged. A no-barrier/larger-group benchmark is still running (session50208) in controller benchmark_gpu_no_barrier; do not change production without results.

- 2026-10-15 23:05 HKT: all 21 training-ready and alive; default seeds1000/2000/3000 have 3/1/1 local logs (30k/10k/10k), remote verified accordingly. Full checkpoint audit found 19 trained snapshots with all eight saved losses finite; remaining two new defaults still had initial snapshots. Current protected40 alive. Matched full-load fused8 medians: baseline8.277 updates/s, CUDA Graph2 8.268 (no gain). No-barrier group sizes1/4/8/16/32: 8.22/8.64/8.31/8.23/8.33 updates/s (no gain); production stays barrier + group8. Experimental sparse_intermediate_metrics (private opt-in attribute; defaults false) peels final update, retains scheduled sparse analyses each intermediate update, permits DCE of unused dense metrics. New CPU group4 consistency/resume case passes; GPU benchmark session64481 running in benchmark_gpu_sparse_metrics. Full18-test suite session under validation/pytest_final_v3.log is running. Foreground launcher remains session37341; never stop unrelated jobs.

## 2026-10-15 用户要求立即交接，程序不要停

先读取 `logdir/two_player_sweep_20261015/HANDOFF.txt`。21个训练保持运行；仅监控协调器PID1198602已结束、锁已释放。下一位agent用 `auto_script/two_player.sh --monitor-only` 接续前台监控，先运行verify。用户要求当前agent尽快结束，未完成每组五条日志验收，不得把交接当成验收完成。实际快照 `handoff_process_snapshot.json`。最终18项测试全部通过；唯一GPU候选基准PID1221510继续运行，正式21组未启用该候选。

## 当前任务覆盖旧交接（2026-10-15）

- 用户要求停止并删除全部k=128实验的进程、W&B曲线及本地logdir；仅保留default与k=32，共12组。
- k=32的run.log_every改为2000环境步（覆盖旧10000限制）；default仍10000，report_every仍20000。需要从现有checkpoint受控恢复k=32，保持run ID及优化器状态。
- 当前agent必须持续前台轮询直到12组各至少5条有效logging，并核验远端/checkpoint，不能以交接代替完成。

- 2026-10-15 23:43 HKT：12组均ready过，前台协调器使用 --retry-failed --max-attempts 4；best_response_32/2000因GPU2有两组重任务，从step3200同ID迁至GPU1，证据k32_gpu_rebalance.json。不要恢复旧GPU分配。验收尚未完成。

## 2026-10-16 前台验收完成

- 2026-10-16T02:08:14.889912+08:00：用户本轮任务完成。9组k128已停止且W&B曲线/本地logdir已删除，旧进程及其子进程无活跃残留；项目远端恰好12个保留ID。
- 12/12保留训练存活，本地与W&B均至少5条有效logging；all_verified=true。k32每2000环境步logging，default每10000步，report_every仍20000；所有连续优化器增量通过。
- 全部checkpoint完整、step>0、8项loss有限；40/40受保护进程仍存活，未向其发信号。前台协调器正常退出，训练继续15M目标，完整训练尚未结束。
- 最终证据：logdir/two_player_sweep_20261015/two_player_verification.json 与 final_process_cleanup_audit.json；管理删除证据k128_retirement.json，单组GPU迁移证据k32_gpu_rebalance.json。不要恢复已删除k128，不要重建新W&B ID。
