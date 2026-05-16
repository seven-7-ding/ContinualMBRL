Important context for future agents:

- Primacy-bias debugging and deployment requested by the user must only use CUDA devices 6 and 7.
- Do not stop, kill, reset, or otherwise disturb already-running training processes.
- The primacy-bias sweep script intentionally writes logdirs under `logdir/continual_dreamer_full_loop_size1m/...` so wandb uses the same project as the full-loop baseline and only changes the group.
