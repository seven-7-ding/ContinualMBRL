#!/usr/bin/env python3
"""Repair dog resume W&B runs that were logged with a wrong task phase.

The original W&B history is immutable at row granularity. This script moves the
corrupted remote run out of the active group and uploads a corrected replacement
run with history truncated to the last known-clean step. Future resumes should
use the generated ``wandb_corrected_id.txt`` in each logdir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import wandb


ENTITY = "stevendiffdiff-"
PROJECT = "continual_dreamer_soft_reset_dog_stand|dog_walk|dog_trot_size1m"

RUNS = [
    {
        "label": "sandp_all_seed2000",
        "logdir": "logdir/continual_dreamer_soft_reset_dog_stand|dog_walk|dog_trot_size1m/sandp_all_a0p8_50k_no_revive/seed_2000",
        "archive_ids": ["mhb7zy02", "fixmhb72"],
        "new_id": "r2mhb72",
        "name": "seed_2000",
        "group": "sandp_all_a0p8_50k_no_revive",
        "cutoff_step": 3_800_000,
    },
    {
        "label": "sandp_all_seed3000",
        "logdir": "logdir/continual_dreamer_soft_reset_dog_stand|dog_walk|dog_trot_size1m/sandp_all_a0p8_50k_no_revive/seed_3000",
        "archive_ids": ["ktujo5si", "fixktuj3"],
        "new_id": "r2ktuj3",
        "name": "seed_3000",
        "group": "sandp_all_a0p8_50k_no_revive",
        "cutoff_step": 3_810_000,
    },
    {
        "label": "no_reset_seed2000",
        "logdir": "logdir/continual_dreamer_soft_reset_dog_stand|dog_walk|dog_trot_size1m/no_reset_50k_no_revive/seed_2000",
        "archive_ids": ["1v1viqpy", "fix1v1v2"],
        "new_id": "r21v1v2",
        "name": "seed_2000",
        "group": "no_reset_50k_no_revive",
        "cutoff_step": 4_110_000,
    },
    {
        "label": "ab_wm_head_seed2000",
        "logdir": "logdir/continual_dreamer_soft_reset_dog_stand|dog_walk|dog_trot_size1m/sandp_ab_wm_head_a0p8_50k_no_revive/seed_2000",
        "archive_ids": ["wgs73ky5", "fixwgs72"],
        "new_id": "r2wgs72",
        "name": "seed_2000",
        "group": "sandp_ab_wm_head_a0p8_50k_no_revive",
        "cutoff_step": 2_050_000,
    },
]


def load_rows(path: Path, cutoff: int):
  rows = []
  with path.open() as f:
    for line in f:
      if not line.strip():
        continue
      row = json.loads(line)
      if int(row.get("step", -1)) <= cutoff:
        rows.append(row)
  return rows


def move_old_run(api, spec, run_id, archive_suffix, dry_run):
  old_path = f"{ENTITY}/{PROJECT}/{run_id}"
  run = api.run(old_path)
  archive_group = f"{spec['group']}_corrupted_{archive_suffix}"
  print(
      f"archive {run_id}: group {run.group!r} -> "
      f"{archive_group!r}, name -> {spec['name']}_corrupted_{run_id}")
  if dry_run:
    return
  run.group = archive_group
  run.name = f"{spec['name']}_corrupted_{run_id}"
  tags = list(run.tags or [])
  for tag in ("corrupted_task_phase", "archived_by_codex"):
    if tag not in tags:
      tags.append(tag)
  run.tags = tags
  note = (
      "Archived because continual_train resumed with switch_count=0, causing "
      "dog task phase and performance keys to be wrong after resume. "
      f"Replacement run id: {spec['new_id']}."
  )
  run.notes = (run.notes + "\n\n" + note) if run.notes else note
  run.update()


def upload_new_run(spec, rows, dry_run):
  print(
      f"upload {spec['new_id']}: {len(rows)} rows through "
      f"step {spec['cutoff_step']} into group {spec['group']!r}")
  if dry_run:
    return
  logdir = Path(spec["logdir"])
  run = wandb.init(
      entity=ENTITY,
      project=PROJECT,
      id=spec["new_id"],
      name=spec["name"],
      group=spec["group"],
      resume="never",
      dir=str(logdir / "wandb"),
      settings=wandb.Settings(init_timeout=300),
      config={
          "corrected_from_run_ids": spec["archive_ids"],
          "corrected_cutoff_step": spec["cutoff_step"],
          "corrected_reason": "resume_task_phase_bug",
      },
  )
  for row in rows:
    global_step = int(row["step"])
    metrics = {key: value for key, value in row.items() if key != "step"}
    wandb.log(metrics, step=global_step)
  run.finish()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--archive-suffix", default="20260704")
  args = parser.parse_args()

  api = wandb.Api(timeout=60)
  manifest = []
  for spec in RUNS:
    logdir = Path(spec["logdir"])
    metrics = logdir / "metrics.jsonl"
    if not metrics.exists():
      raise FileNotFoundError(metrics)
    rows = load_rows(metrics, spec["cutoff_step"])
    if not rows:
      raise RuntimeError(f"No rows selected for {spec['label']}")
    selected_path = logdir / "wandb_corrected_history.jsonl"
    if not args.dry_run:
      with selected_path.open("w") as f:
        for row in rows:
          f.write(json.dumps(row, sort_keys=True) + "\n")
    for run_id in spec["archive_ids"]:
      move_old_run(api, spec, run_id, args.archive_suffix, args.dry_run)
    upload_new_run(spec, rows, args.dry_run)
    if not args.dry_run:
      (logdir / "wandb_corrected_id.txt").write_text(
          spec["new_id"] + "\n")
    manifest.append({
        **spec,
        "rows_uploaded": len(rows),
        "first_step": rows[0].get("step"),
        "last_step": rows[-1].get("step"),
    })
    time.sleep(2)

  manifest_path = Path("logdir/dog_wandb_repair_manifest.json")
  if not args.dry_run:
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
  print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
  main()
