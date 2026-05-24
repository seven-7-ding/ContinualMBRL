# Agent Context

- Periodic reset logic is implemented via `run.reset_frequency`, `run.revive_epoch`, and `run.reset_mode` (`no_reset`, `reset_only_agent`, `reset_only_wm`, `reset_all`).
- Task-switch-triggered reset has been removed from continual training flow.
- Keep ReDo / gradient ReDo / data_diversity analysis logic unchanged.
- Reset requirement: when resetting wm/agent/all, optimizer state resets together.
- Runtime/debug constraints (latest): CUDA 0-7 are allowed; do not kill/modify existing running processes.
- Recent debug: fixed JAX donated-buffer deletion crashes around reset+revive in `embodied/jax/agent.py` and validated with a short run under `logdir/debug_reset_only_agent_fix7/reset_only_agent/seed_0`.
- Active deployment: `auto_scripts/continual_mujoco_dreamer_priori_reset-agent.sh` currently running in session 32524.
