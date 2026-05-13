# Agent Context

- User requested Dreamer continual reset modes: `reset_only_wm` and `reset_only_agent`.
- Only use CUDA devices 6 and 7 for debug/deployment. Do not use other CUDA devices.
- Deployment order should alternate 6/7/6/7/6/7 to avoid errors.
- Existing `auto_scripts/continual_mujoco_dreamer_priori.sh` had user edits before this task: CUDA devices set to `(6 7 6 7 6 7)`, max runs per GPU set to `1`, settings set to reset_wm/reset_agent x seeds 1000/2000/3000, imag_length override set to 2.
