Context for follow-up agents

- Task scope: implement continual soft reset support with separate reset mechanism and reset target controls.
- Requested mechanisms: `sandp` and `merge`; keep legacy hard reset behavior for compatibility.
- Requested targets: `agent_head`, `wm_head`, `all_head`, `only_rssm`, `all`, plus `no_reset` at the script level.
- Revive should stay available but default `revive_epoch=0`.
- Validation constraint: if GPU validation is needed, prefer `cuda:3-7`, avoid disturbing existing jobs, and keep memory usage low.
- Validation status: shell syntax check passed; Python compile check passed; fake-agent repeated reset validation passed; full Dreamer CPU smoke test was not completed because initialization remained too slow.
- Current debug constraint: runtime validation should prefer `cuda:6` and `cuda:7`, at most one new experiment per card, with `size0.5m` and small memory fraction.
- Repository note: `codex-cli-executor.log` is being maintained for this task.

Dog resume monitoring state as of 2026-07-05:

- Polluted dog resume runs from the previous incorrect resume were stopped: PIDs `4034334`, `4034675`, `4035100`, `4035574`, `4036001`.
- Clean dog resumes currently being monitored:
- `sandp_ab_agent_head_a0p8_50k_no_revive/seed_1000`, PID `4180159`, GPU `0`, W&B run `3z102x9a`, resumed from step `2030000`; expected and observed task after checkpoint load is `dog_walk`.
- `sandp_ab_agent_head_a0p8_50k_no_revive/seed_2000`, PID `4180509`, GPU `1`, W&B run `1q6vnugo`, resumed from step `1570000`; expected and observed task after checkpoint load is `dog_stand`.
- Pre-resume checkpoint backups were saved inside each run directory as `ckpt_pre_resume_20260705T173229`.
- Resume logs are `resume_20260705T173243.log` inside each run directory.
- These two clean resumes use `REPLAY_CACHE_CHUNKS=512` and `REPLAY_CHUNKSIZE=1024` to keep memory lower than the prior high-cache dog resumes.
- W&B API `state` for resumed crashed runs can remain stale or show `crashed`; prefer checking local PIDs, W&B internal `history_lines`, and remote `_step`/summary values.

Dog resume acceleration update as of 2026-07-05 23:22 HKT:

- Slow resumed PIDs `4180159` and `4180509` were stopped.
- Current accelerated dog resumes:
- `sandp_ab_agent_head_a0p8_50k_no_revive/seed_1000`, PID `35430`, GPU `0`, W&B `3z102x9a`, correct task `dog_walk`, latest local metrics step `2040000`, `fps/policy=9.129`.
- `sandp_ab_agent_head_a0p8_50k_no_revive/seed_2000`, PID `39268`, GPU `1`, W&B `1q6vnugo`, correct task `dog_stand`, latest local metrics step `1570000`, `fps/policy=7.433`.
- They were relaunched with `REPLAY_CACHE_CHUNKS=20000` after restoring `ckpt_pre_resume_20260705T173229`.
- Polluted 20260705 replay chunks were moved under each run's `replay/bad_resume_20260705T230216/`; polluted checkpoints and metrics were backed up under each run directory.
- Code change: checkpoint resume now re-windows replay only when `switch_count > 0`; task switches now archive top-level replay `.npz` files via `replay.clear(disk=True, ...)`.

Dog resume expansion as of 2026-07-06 00:23 HKT:

- Existing accelerated dog resumes still running: `35430` (ab_agent_head seed_1000, GPU0), `39268` (ab_agent_head seed_2000, GPU1).
- First additional batch launched with pre-resume backups `ckpt_pre_resume_20260706T001041`: `50750` (agent_head seed_2000, GPU4), `51089` (wm_head seed_1000, GPU5), `51531` (agent_head seed_3000, GPU6), `52038` (wm_head seed_2000, GPU7).
- `51531` failed before new metrics with replay sampler empty; do not count it as active. A code fix was applied afterward to lazily create train replay streams after replay restore.
- Second additional batch launched with pre-resume backups `ckpt_pre_resume_20260706T002258`: `59359` (ab_agent_head seed_3000, GPU2), `59704` (ab_wm_head seed_3000, GPU3), `60161` (agent_head seed_1000, GPU6), `60581` (wm_head seed_3000, GPU4).
- Active monitor should protect humanoid jobs by killing latest dog batch first if `MemAvailable < 120GiB`.
