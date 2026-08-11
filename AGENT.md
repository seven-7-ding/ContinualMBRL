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
