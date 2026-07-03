# Dog Run Resume Script

Use `auto_scripts/resume_dog.sh` to continue interrupted dog experiments that
were started from `auto_scripts/hard_task_dog.sh`.

The script scans existing run directories under:

```text
logdir/continual_dreamer_soft_reset_dog_stand|dog_walk|dog_trot_size1m
```

For each run, it reads the saved `config.yaml`, reuses the same `logdir`, loads
the local checkpoint from `ckpt`, and resumes the original wandb run by passing
the previous run id through `WANDB_RUN_ID` and `WANDB_RESUME`.

## Basic Usage

Dry run without launching training:

```bash
DRY_RUN=1 bash auto_scripts/resume_dog.sh
```

Resume all incomplete dog runs:

```bash
bash auto_scripts/resume_dog.sh
```

Resume selected targets or seeds:

```bash
TARGETS="agent_head wm_head" SEEDS="1000 2000" bash auto_scripts/resume_dog.sh
```

Choose GPUs explicitly:

```bash
CUDA_DEVICES_STR="6 7" MAX_CONCURRENT=2 bash auto_scripts/resume_dog.sh
```

By default, the script uses:

```text
CUDA_DEVICES_STR="2 3 4 5 6 7 0 1"
MAX_CONCURRENT=8
WANDB_RESUME_MODE=allow
REQUIRE_WANDB_ID=1
REPLAY_CACHE_CHUNKS=512
```

Set `MAX_CONCURRENT` conservatively when other experiments are running on the
same machine. The previous interruptions were caused by system memory OOM, so
restarting too many runs at once can reproduce the same failure mode.

`REPLAY_CACHE_CHUNKS` limits how many completed replay chunks stay resident in
RAM. Older chunks remain on disk and are loaded on demand when sampled. This
keeps the replay sampling distribution and stored training data unchanged, but
trades additional disk I/O for lower RAM usage. The default `512` is conservative
for the dog runs; set it to `0` to restore the old behavior where all loaded
replay chunks stay in memory.

## Outputs

Dry-run mode only prints the planned resumes and does not write a global
manifest. Real resume launches append a per-run manifest inside each run
directory:

```text
<logdir>/resume_manifest.tsv
```

Each resumed run appends process output to:

```text
<logdir>/resume_<timestamp>.log
```

The original `train.log`, `metrics.jsonl`, `scores.jsonl`, `ckpt`, and wandb
local run directory are left in place. Replay chunks are stored as `.npz` files
under `<logdir>/replay/` for training replay and, for eval-style runs, under the
corresponding eval replay subdirectory inside the same `<logdir>`.

## Wandb Continuity

`dreamerv3/main.py` now forwards these environment variables to `wandb.init()`:

```text
WANDB_RUN_ID
WANDB_RESUME
```

`resume_dog.sh` extracts `WANDB_RUN_ID` from the existing local wandb directory,
for example `wandb/wandb/run-20260629_002452-fu0m3js2` gives run id
`fu0m3js2`. This keeps resumed metrics attached to the original wandb run
instead of creating a new run with the same display name.
