Context for follow-up agents

- Task scope: implement continual soft reset support with separate reset mechanism and reset target controls.
- Requested mechanisms: `sandp` and `merge`; keep legacy hard reset behavior for compatibility.
- Requested targets: `agent_head`, `wm_head`, `all_head`, `only_rssm`, `all`, plus `no_reset` at the script level.
- Revive should stay available but default `revive_epoch=0`.
- Validation constraint: if GPU validation is needed, prefer `cuda:3-7`, avoid disturbing existing jobs, and keep memory usage low.
- Validation status: shell syntax check passed; Python compile check passed; fake-agent repeated reset validation passed; full Dreamer CPU smoke test was not completed because initialization remained too slow.
- Repository note: `codex-cli-executor.log` is being maintained for this task.
