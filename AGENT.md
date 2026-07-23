# Agent Notes

## Current Objective

- `wsc_no_scale_{init,constant,factor}` is implemented and validated.
- Continue monitoring/managing active continual Dreamer experiments.
- Keep `task_checklist.md` compact. Since `codex-cli-executor.log` exists, experiment polling/progress belongs there, while only durable operational context belongs here.

## WSC No-Scale Semantics

- `wsc_no_scale_*` mechanisms must not create or train per-layer `wsc_scale`.
- Effective output scale is implicit `1.0`.
- After optimizer updates, no-scale WSC only rescales selected layer `kernel`/`bias` parameters according to the selected Frobenius mode.
- For WSC modes with a target Frobenius scale (`init` and `constant`), selected layers are normalized at parameter creation before the first optimizer update.

## Validation Evidence

- Focused validation passed on 2026-07-18:
  - `python -m compileall embodied/jax/wsc.py embodied/jax/nets.py embodied/jax/opt.py dreamerv3/agent.py`
  - CPU smoke: `wsc_no_scale_{init,constant,factor}` parse correctly.
  - CPU smoke: `wsc_no_scale_constant` init norm reached target `2.0`.
  - CPU smoke: `wsc_no_scale_factor` keeps `wsc_scale=1.0` even if a stale scale key exists.
  - CPU smoke: `WSC_USE_OUTPUT_SCALE=False` creates no `wsc_scale` parameter.

## Active Experiment Policy

- Latest scheduling request from the user: pause current `wsc_no_scale` experiments and bring other settings back onto the schedule.
- Current priority policy:
  - P1: protect FPS for `no_wsc` and `WSC_grad_scale_constant_all` across both tasks.
  - P2: `WSC_grad_scale_factor_all`, `WSC_grad_scale_init_all`; resume cautiously when P1 has clear FPS headroom, prioritizing lagging seeds/settings to keep progress reasonably balanced.
  - Lowest priority: `wsc_no_scale_*` and `WSC_nograd_*`; keep stopped unless explicitly re-enabled or clear FPS headroom exists.
- Preserve active healthy experiments. Do not interrupt runs unless recovery is needed.
- Use larger replay cache for resumed/new jobs where possible (`replay.cache_chunks=4096` for the latest no-scale size1m launches).
- Use `run.save_every=1800` for new/resumed runs to reduce checkpoint I/O pressure.
- Pause/resume FPS caveat: the first metrics line after `SIGSTOP`/`SIGCONT` can include wall-clock pause time and underreport true throughput. Require a later continuous-running metrics line before deciding FPS is truly below threshold.

## Current Experiment State

- As of 2026-07-23 03:14 HKT, managed Dreamer state is `27` total processes: `17` running and `10` stopped.
- Current intended running schedule is `9` P1 plus `8` P2. No no-scale or low-priority `WSC_nograd_*` jobs are running.
- Latest manual action result: P2 dog `WSC_grad_scale_factor_all` seed1000 PID `2566929` on GPU3 produced a second continuous sample at step `830000`/FPS `7.52`, so keep it running. GPU3 P1 walker `WSC_grad_scale_constant_all` seed3000 PID `2543964` and walker `no_wsc` seed3000 PID `2542215` remained safe at FPS `8.33` and `9.13`.
- Current holding point: `P1=9/P2=8/low=0/no_scale=0`. Dog `WSC_grad_scale_factor_all` now has all three seeds running. Hold this schedule; P1 limiter range is about `8.3-8.6` FPS, so do not resume lagging `WSC_grad_scale_init_all` seeds unless P1 headroom improves.
- Latest manual action result: P2 dog `WSC_grad_scale_factor_all` seed2000 PID `2572637` on GPU5 produced a second continuous sample at step `1020000`/FPS `9.09`, so keep it running. GPU5 P1 dog `WSC_grad_scale_constant_all` seed3000 PID `2582355` remained safe at step `4370000`/FPS `8.69`.
- Previous holding point: `P1=9/P2=7/low=0/no_scale=0` stayed stable for another observation window; active P1 limiter range was about `8.7-8.8` FPS before PID `2566929` was resumed.
- Previous resume result: P2 walker `WSC_grad_scale_factor_all` seed1000 PID `2588459` on GPU1 produced a second continuous sample at step `1220000`/FPS `9.74`, so keep it running. GPU1 P1 dog `WSC_grad_scale_constant_all` seed1000 PID `2581161` remained safe at step `4410000`/FPS `8.61`.
- Latest manual poll before the GPU5 resume: monitor PID `2913891` was alive, P1/P2 below-counts were all `0`, and current non-contaminated P1/P2 FPS remained above the `5.8` protection floor.
- Latest action: dog `no_wsc` seed3000 PID `2462714` was given GPU7 by pausing GPU7 P2 jobs, but it produced no new metric for about 36 minutes. Since a 10k step at the 5.8 FPS threshold should complete in about 29 minutes, PID `2462714` was paused again and GPU7 P2 jobs dog `WSC_grad_scale_init_all` seed2000 PID `2573081` plus walker `WSC_grad_scale_init_all` seed1000 PID `2561799` were restored. Their first post-restore metrics were low, then continuous samples recovered to FPS `9.21` and `10.36`.
- Running P1:
  - dog `WSC_grad_scale_constant_all`: seed1000 PID `2581161` FPS `10.33`, seed2000 PID `2581645` FPS `9.18`, seed3000 PID `2582355` FPS `10.72`.
  - walker `WSC_grad_scale_constant_all`: seed1000 PID `2542765` FPS `10.29`, seed2000 PID `2543365` FPS `10.51`, seed3000 PID `2543964` FPS `10.58`.
  - walker `no_wsc`: seed1000 PID `2540858` FPS `13.17`, seed2000 PID `2541402` FPS `11.97`, seed3000 PID `2542215` FPS `12.03`.
- dog `no_wsc` is fully stopped. seed1000 PID `2461551`, seed2000 PID `2461997`, and seed3000 PID `2462714` all failed throughput probes; the latest GPU7 seed3000 retry produced no new metrics for about 36 minutes, below the 5.8 FPS progress target.
- Running P2:
  - dog `WSC_grad_scale_init_all` seed2000 PID `2573081` on GPU7, step `2290000`/FPS `9.13`.
  - dog `WSC_grad_scale_factor_all` seed2000 PID `2572637` on GPU5, resumed at 2026-07-23 01:22 HKT from step `1000000`; wait for fresh metrics before judging FPS.
  - walker `WSC_grad_scale_init_all` seed1000 PID `2561799` on GPU7, step `2900000`/FPS `10.00`.
  - walker `WSC_grad_scale_factor_all` seed3000 PID `2561803` on GPU6, step `2510000`/FPS `10.17`.
  - dog `WSC_grad_scale_factor_all` seed3000 PID `2589952` on GPU6, step `1950000`/FPS `8.50`.
  - walker `WSC_grad_scale_factor_all` seed1000 PID `2588459` on GPU1, resumed at 2026-07-23 00:54 HKT from step `1200000`; wait for fresh metrics before judging FPS.
  - walker `WSC_grad_scale_factor_all` seed2000 PID `2551163` on GPU4, step `3390000`/FPS `9.83`. It shares GPU4 with P1 dog `WSC_grad_scale_constant_all` seed2000 PID `2581645`; GPU4 P1 remains safe but limiting, so do not add more P2 there while this remains the minimum.
- Stopped P2 with recent caveats:
  - GPU6 P2 jobs remain stopped while dog `no_wsc` has demonstrated poor throughput there; avoid GPU6 unless there is clear headroom and no P1 probe is active.
- Stopped no-scale deployment under `logdir/continual_dreamer_soft_reset_size1m/wsc_wsc_no_scale_constant_all/`: seed1000 PID `2672938`, seed2000 PID `2677387`, seed3000 PID `2681197`.
- Continue with P1 protection first. P2 shares GPU4, GPU1, GPU3, and GPU5 with P1; current P1 limiter range is about `8.3-8.7` FPS while staying safe. Hold at `8` P2 for another observation window before any further resume. Do not retry dog `no_wsc` without a materially different resource plan. Do not resume no-scale or `WSC_nograd_*` unless explicitly re-enabled or there is clear FPS headroom.

## Background Monitor

- Monitor script: `/tmp/codex_wsc_monitor.py`.
- PID file: `/tmp/codex_wsc_monitor.pid`.
- As of 2026-07-23 03:14 HKT, monitor PID `2913891` is alive. It replaced old PID `2854162` so the running process uses the fixed log-compaction timestamp sorting.
- It checks every 10 minutes, writes compact heartbeats every 30 minutes to `codex-cli-executor.log`, pauses any low-priority running job (`wsc_no_scale`, `WSC_nograd_*`), logs warnings only after two consecutive fresh P1 steps below `5.8`, pauses running P2 jobs to protect P1 after such warnings, pauses P2 jobs after two consecutive fresh below-threshold P2 metrics, and periodically compacts repetitive heartbeat records while preserving interventions and recent heartbeats. As of 2026-07-20 06:45 HKT, compaction sorts retained log records by timestamp prefix so the log stays chronological.
- Fresh-metrics rule: monitor tracks per-PID `running_since` after `SIGCONT`/status transition and ignores metrics written before the current running window when counting below-threshold samples.
- If the monitor dies, inspect `/tmp/codex_wsc_monitor.out`, restart with `setsid python /tmp/codex_wsc_monitor.py >/tmp/codex_wsc_monitor.out 2>&1 < /dev/null & echo $! > /tmp/codex_wsc_monitor.pid`, then update this file and `codex-cli-executor.log`.

## Logging Policy

- Keep `task_checklist.md` as the concise task contract.
- Keep `codex-cli-executor.log` concise: retain a summary plus recent meaningful experiment records; remove repetitive low-value polling detail when it grows.
- Move only durable operational context into this file.
- Do not expose or commit W&B secrets.
