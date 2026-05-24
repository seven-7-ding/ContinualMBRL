# Agent Context

- User requirement: periodic reset logic by `run.reset_frequency`, with revive phases controlled by `run.revive_epoch` and `run.reset_mode` (`no_reset`, `reset_only_agent`, `reset_only_wm`, `reset_all`).
- Remove reset-at-task-switch behavior from continual training flow.
- Keep ReDo / gradient ReDo / data_diversity analysis logic unchanged.
- Additional requirement: when resetting wm/agent/all, optimizer state must be reset accordingly.
- Runtime/debug constraint from user: only use CUDA devices 6/7 for debugging; do not modify existing running processes.
