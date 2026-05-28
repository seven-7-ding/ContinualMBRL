# Agent Context

- Periodic reset logic uses `run.reset_frequency`, `run.reset_mode`, and revive settings.
- Revive strategies:
  - `fixed`: always run exactly `run.revive_epoch` updates.
  - `threshold`: compare loss component-wise against `last_loss_component * revive_threshold` and only stop early when all compared components satisfy threshold.
- For `threshold`, early-stop checks start only after a minimum warmup revive count:
  - `min_check_steps = max(10, ceil(revive_epoch / 100))`.
- Rolling pre-reset losses are tracked per-component with window `run.last_loss_num`.
- Reset must reset optimizer state for the corresponding module(s).
- Runtime constraint from user (current): only use CUDA 6/7 for debug/testing; do not kill or modify existing running processes.
- Recent verification runs:
  - `logdir/debug_threshold_componentwise/reset_all/seed_0`
  - `logdir/debug_fixed_regression_after_threshold/reset_only_agent/seed_0`
