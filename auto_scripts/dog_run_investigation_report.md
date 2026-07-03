# Dog Run Interruption Investigation

## Summary

The dog experiments did not stop because of Python exceptions in `train.log`.
The available evidence points to system memory exhaustion. The Linux kernel OOM
killer repeatedly killed long-running `python` processes that correspond to the
dog experiment PIDs recorded by wandb debug logs.

I also intentionally stopped the dog runs that were still active on
2026-07-03 so they can be resumed in a controlled way:

```text
400772 no_reset seed_1000
402656 no_reset seed_3000
2658593 sandp_agent_head seed_3000
2663876 sandp_wm_head seed_2000
```

No dog `dreamerv3/main.py` process was left running after that stop.

## Evidence

`train.log` files generally end after normal metric/checkpoint writes and do
not contain `Traceback`, `XlaRuntimeError`, `ValueError`, or explicit Python
fatal errors for the interruption.

Kernel logs from `journalctl -k` contain repeated OOM records such as:

```text
Jun 30 20:40:07 kernel: Out of memory: Killed process 401325 (python) ...
Jul 01 00:24:31 kernel: Out of memory: Killed process 406566 (python) ...
Jul 01 06:22:17 kernel: Out of memory: Killed process 415509 (python) ...
Jul 01 10:00:17 kernel: Out of memory: Killed process 2643498 (python) ...
Jul 01 12:34:34 kernel: Out of memory: Killed process 2652333 (python) ...
Jul 01 15:27:08 kernel: Out of memory: Killed process 2641017 (python) ...
Jul 01 22:36:24 kernel: Out of memory: Killed process 2646329 (python) ...
Jul 02 01:48:12 kernel: Out of memory: Killed process 2655260 (python) ...
Jul 02 05:49:18 kernel: Out of memory: Killed process 2661341 (python) ...
Jul 02 10:13:58 kernel: Out of memory: Killed process 2644709 (python) ...
Jul 02 15:21:42 kernel: Out of memory: Killed process 2649388 (python) ...
Jul 02 21:17:38 kernel: Out of memory: Killed process 2642296 (python) ...
Jul 03 04:51:35 kernel: Out of memory: Killed process 403593 (python) ...
```

The PIDs map back to dog runs through each run's local wandb debug log:

```text
400772  no_reset seed_1000
401325  no_reset seed_2000
402656  no_reset seed_3000
2644709 sandp_ab_agent_head seed_1000
2646329 sandp_ab_agent_head seed_2000
2649388 sandp_ab_agent_head seed_3000
2641017 sandp_ab_wm_head seed_1000
2642296 sandp_ab_wm_head seed_2000
2643498 sandp_ab_wm_head seed_3000
2652333 sandp_agent_head seed_1000
2655260 sandp_agent_head seed_2000
2658593 sandp_agent_head seed_3000
403593  sandp_all seed_1000
415509  sandp_all seed_2000
406566  sandp_all seed_3000
2661341 sandp_wm_head seed_1000
2663876 sandp_wm_head seed_2000
2666494 sandp_wm_head seed_3000
```

## Likely Cause

The launch scripts start many `size1m` Dreamer processes concurrently. Each
long-running process grows to multi-GB resident memory, and the combined memory
pressure triggers global system OOM. The kernel then kills one Python process at
a time. Because the process is killed externally, the training log ends
abruptly after the last successful metric/checkpoint write.

The dominant per-process growth source is the replay buffer. The dog runs use
`replay.size=5e6`, and the replay implementation used to keep all loaded chunks
resident in RAM while also saving them to disk. Because `replay_context=1`, the
stored transitions also include model context tensors such as `dyn/deter` and
`dyn/stoch`, so a long run accumulates several GB to tens of GB of replay data
inside each Python process.

Some wandb logs also show transient network errors such as HTTP 502 and request
timeouts, but those are not the primary stopping cause: the decisive local
evidence is the kernel OOM kill record.

## Mitigation Added

Replay now supports `replay.cache_chunks`. When this value is positive, old
completed replay chunks remain indexed but their array data can be unloaded from
RAM after they have been saved. If a sampled sequence touches an unloaded chunk,
it is loaded back from disk on demand. This preserves the replay contents,
stepids, FIFO capacity behavior, and sampling distribution, while reducing
resident memory at the cost of extra disk I/O.

For dog experiments, both `auto_scripts/hard_task_dog.sh` and
`auto_scripts/resume_dog.sh` now set:

```text
--replay.cache_chunks 512
```

Set `REPLAY_CACHE_CHUNKS=0` in `resume_dog.sh` if the old in-memory behavior is
needed for debugging. Lower values reduce RAM further but increase replay disk
reads.

## Recovery Status

All inspected dog run directories have:

```text
ckpt/latest
metrics.jsonl
wandb/wandb/run-...-<run_id>
```

That means they can be resumed from local checkpoints and can continue logging
to the same wandb run id using `auto_scripts/resume_dog.sh`.

Replay chunk files are written under each run directory, for example
`<logdir>/replay/*.npz`. Resume output files are also kept inside the same run
directory as `<logdir>/resume_<timestamp>.log` and
`<logdir>/resume_manifest.tsv`; no global resume manifest is written outside the
run directories.

## Recommendation

Resume fewer runs at once than the original launch pattern. Start with:

```bash
CUDA_DEVICES_STR="6 7" MAX_CONCURRENT=2 bash auto_scripts/resume_dog.sh
```

If system memory remains stable, increase concurrency gradually. Avoid launching
a full 18-run dog batch together with other large active experiment batches.
