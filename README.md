# Mastering Diverse Domains through World Models

A reimplementation of [DreamerV3][paper], a scalable and general reinforcement
learning algorithm that masters a wide range of applications with fixed
hyperparameters.

![DreamerV3 Tasks](https://user-images.githubusercontent.com/2111293/217647148-cbc522e2-61ad-4553-8e14-1ecdc8d9438b.gif)

If you find this code useful, please reference in your paper:

```
@article{hafner2025dreamerv3,
  title={Mastering diverse control tasks through world models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={Nature},
  pages={1--7},
  year={2025},
  publisher={Nature Publishing Group}
}
```

To learn more:

- [Research paper][paper]
- [Project website][website]
- [Twitter summary][tweet]

## DreamerV3

DreamerV3 learns a world model from experiences and uses it to train an actor
critic policy from imagined trajectories. The world model encodes sensory
inputs into categorical representations and predicts future representations and
rewards given actions.

![DreamerV3 Method Diagram](https://user-images.githubusercontent.com/2111293/217355673-4abc0ce5-1a4b-4366-a08d-64754289d659.png)

DreamerV3 masters a wide range of domains with a fixed set of hyperparameters,
outperforming specialized methods. Removing the need for tuning reduces the
amount of expert knowledge and computational resources needed to apply
reinforcement learning.

![DreamerV3 Benchmark Scores](https://github.com/danijar/dreamerv3/assets/2111293/0fe8f1cf-6970-41ea-9efc-e2e2477e7861)

Due to its robustness, DreamerV3 shows favorable scaling properties. Notably,
using larger models consistently increases not only its final performance but
also its data-efficiency. Increasing the number of gradient steps further
increases data efficiency.

![DreamerV3 Scaling Behavior](https://user-images.githubusercontent.com/2111293/217356063-0cf06b17-89f0-4d5f-85a9-b583438c98dd.png)

# Instructions

The code has been tested on Linux and Mac and requires Python 3.11+.

## Docker

You can either use the provided `Dockerfile` that contains instructions or
follow the manual instructions below.

## Manual

Install [JAX][jax] and then the other dependencies:

```sh
pip install -U -r requirements.txt
```

Training script:

```sh
python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/{timestamp} \
  --configs crafter \
  --run.train_ratio 32
```

To reproduce results, train on the desired task using the corresponding config,
such as `--configs atari --task atari_pong`.

View results:

```sh
pip install -U scope
python -m scope.viewer --basedir ~/logdir --port 8000
```

Scalar metrics are also writting as JSONL files.

# Two-player continual DMC experiments

`--run.two_player` selects a schedule relative to the existing `train_ratio`
clock (a baseline update uses `batch_size * batch_length` replay steps):

| Setting | World-model updates | Agent updates |
| --- | ---: | ---: |
| `default` | 1 | 1 (original joint optimizer call) |
| `wm_as_leader_k` | 1 | k |
| `agent_as_leader_k` | k | 1 |
| `best_response_k` | k | k |

Replace `k` with a positive integer, e.g. `wm_as_leader_128`. Nondefault
schedules finish the WM block before the agent block. Each update samples a
fresh replay batch, and every agent update computes a fresh imagination
rollout. WM trains the encoder, RSSM, decoder, reward and continuation heads;
agent trains policy/value (including replay value), target value and return
normalizers. The two optimizers have independent state and learning-rate
clocks. Formal runs prefetch at most eight batches and fuse at most eight
sequential updates into a compiled scan. Every scan iteration uses its own
data and latest parameters, and recomputes its imagination rollout. Metrics
stay on the device until the exact 10k-step logging boundary. The default keeps
the original joint training behavior. A 256-update RTX 3090 benchmark including
ReDo checks measured 21.1 to 47.3 agent updates/s (2.25x), with 318 MiB peak
JAX allocation and 1.85 GiB process RAM. Shared GPU load affects these rates.

The requested 21-run sweep (7 schedules, seeds 1000/2000/3000) can be deployed
using the `dreamer` conda environment:

```sh
./auto_script/two_player.sh --plan
./auto_script/two_player.sh
```

The launcher uses `walker_run|finger_spin|fish_swim`, priori observations,
size1m, train ratio 256, 500k steps per task, 10 full cycles (15M total).
`dog_stand` is deferred. Credentials are read privately from `.env`, `.env.local`
(`wandb_api` or `WANDB_API_KEY`) or `.env.wandb.local`. W&B project:
`two_player_dreamer_walker_run|finger_spin|fish_swim_size1m_500k_x10`.
Groups identify schedules and run names identify seeds; full configuration
is saved both locally and in W&B.

All parameter reset/regularization mechanisms are disabled. Activation and
gradient ReDo run in logging mode, with effective/stable ranks, every 100
updates of the corresponding optimizer. Layer gradients, update magnitudes,
data diversity, report gradient norms, task performance, replay, timers and
available resource diagnostics are enabled. Logging occurs strictly every 10k environment steps. W&B explicitly uses
`env_step` as the common x-axis for all logged metrics. Losses and other
training statistics use each module's latest update; sparse diagnostics keep
their latest finite measurement. Reports occur every 20k steps. Checkpoints are saved every 900 seconds,
at task switches, and at completion.

Each run contains its own logs, replay, checkpoints, JAX/CUDA/W&B caches and
temporary files. Completed JAX cache entries can be copied into the next run's
private cache. Controller logs, reference metrics and validation evidence live
in `logdir/two_player_sweep_20261015/`. The shell entry point works from any
working directory and preserves arguments and failure exit codes.

The launcher saves its PIDs, GPU allocation and stable W&B IDs in
`logdir/two_player_sweep_20261015/manifest.json`. It checks memory before launch and never
signals unrelated processes. It monitors until all selected processes have
five finite training logs. To resume an exited run, rerun the launcher with
`--retry-failed` (optionally `--only wm_as_leader_128:1000`); the same logdir
loads model/optimizer/replay/step and restores the task phase, while
`WANDB_RESUME=allow` reconnects the same W&B run. `--monitor-only` observes
existing runs without launching anything. Audit local/remote logs, checkpoints
and resource headroom with `auto_script/verify_two_player.sh`.
Before starting pending runs, `--rebalance-pending --plan` can reassign only
never-started runs using measured WM/agent update costs and current GPU load.

# Tips

- All config options are listed in `dreamerv3/configs.yaml` and you can
  override them as flags from the command line.
- The `debug` config block reduces the network size, batch size, duration
  between logs, and so on for fast debugging (but does not learn a good model).
- By default, the code tries to run on GPU. You can switch to CPU or TPU using
  the `--jax.platform cpu` flag.
- You can use multiple config blocks that will override defaults in the
  order they are specified, for example `--configs crafter size50m`.
- By default, metrics are printed to the terminal, appended to a JSON lines
  file, and written as Scope summaries. Other outputs like WandB and
  TensorBoard can be enabled in the training script.
- If you get a `Too many leaves for PyTreeDef` error, it means you're
  reloading a checkpoint that is not compatible with the current config. This
  often happens when reusing an old logdir by accident.
- If you are getting CUDA errors, scroll up because the cause is often just an
  error that happened earlier, such as out of memory or incompatible JAX and
  CUDA versions. Try `--batch_size 1` to rule out an out of memory error.
- Many environments are included, some of which require installing additional
  packages. See the `Dockerfile` for reference.
- To continue stopped training runs, simply run the same command line again and
  make sure that the `--logdir` points to the same directory.

# Disclaimer

This repository contains a reimplementation of DreamerV3 based on the open
source DreamerV2 code base. It is unrelated to Google or DeepMind. The
implementation has been tested to reproduce the official results on a range of
environments.

[jax]: https://github.com/google/jax#pip-installation-gpu-cuda
[paper]: https://arxiv.org/pdf/2301.04104
[website]: https://danijar.com/dreamerv3
[tweet]: https://twitter.com/danijarh/status/1613161946223677441
