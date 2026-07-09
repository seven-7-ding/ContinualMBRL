#!/usr/bin/env python3
"""Audit local Dreamer runs against W&B remote state.

The script is intentionally read-only. It combines W&B state, local process
presence, and local metrics progress so failed or stale runs can be triaged.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import subprocess
import time
from dataclasses import asdict, dataclass

import wandb


ENTITY = "stevendiffdiff-"
PROJECTS = (
    "continual_dreamer_soft_reset_dog_stand|dog_walk|dog_trot_size1m",
    "continual_dreamer_soft_reset_walker_stand|walker_walk|walker_run_size1m",
    "continual_dreamer_soft_reset_quadruped_run|dog_stand|humanoid_stand_size1m",
)

PROJECT_TARGET_STEPS = {
    "continual_dreamer_soft_reset_dog_stand|dog_walk|dog_trot_size1m":
        6_000_000,
    "continual_dreamer_soft_reset_walker_stand|walker_walk|walker_run_size1m":
        1_500_000,
    "continual_dreamer_soft_reset_quadruped_run|dog_stand|humanoid_stand_size1m":
        7_000_000,
}


@dataclass
class Row:
  project: str
  group: str
  name: str
  run_id: str
  remote_state: str
  remote_step: int
  updated_age_sec: int | None
  local_pid: str
  local_gpu: str
  local_step: int
  local_fps: float | None
  local_wandb_id: str
  logdir: str
  status: str
  action: str


def _local_runs(repo: pathlib.Path) -> tuple[
    dict[tuple[str, str, str], dict], dict[str, dict]]:
  by_key = {}
  by_id = {}
  for proc in pathlib.Path("/proc").iterdir():
    if not proc.name.isdigit():
      continue
    try:
      args = [
          item.decode("utf-8", "replace")
          for item in (proc / "cmdline").read_bytes().split(b"\0")
          if item
      ]
    except Exception:
      continue
    if "dreamerv3/main.py" not in args:
      continue
    logdir = None
    for idx, arg in enumerate(args):
      if arg == "--logdir" and idx + 1 < len(args):
        path = pathlib.Path(args[idx + 1])
        logdir = (repo / path).resolve() if not path.is_absolute() else path
        break
    if not logdir:
      continue
    parts = logdir.parts
    if len(parts) < 3:
      continue
    project, group, name = parts[-3:]
    gpu = "?"
    try:
      for item in (proc / "environ").read_bytes().decode(
          "utf-8", "replace").split("\0"):
        if item.startswith("CUDA_VISIBLE_DEVICES="):
          gpu = item.split("=", 1)[1]
    except Exception:
      pass
    step, fps = _read_metrics(logdir / "metrics.jsonl")
    wandb_id = _wandb_id_for_logdir(logdir)
    item = {
        "pid": proc.name,
        "gpu": gpu,
        "step": step,
        "fps": fps,
        "wandb_id": wandb_id,
        "logdir": str(logdir),
    }
    by_key[(project, group, name)] = item
    if wandb_id:
      by_id[wandb_id] = item
  return by_key, by_id


def _local_disk_run(repo: pathlib.Path, project: str, group: str,
                    name: str) -> dict | None:
  logdir = repo / "logdir" / project / group / name
  if not logdir.exists():
    return None
  step, fps = _read_metrics(logdir / "metrics.jsonl")
  return {
      "pid": "",
      "gpu": "",
      "step": step,
      "fps": fps,
      "wandb_id": _wandb_id_for_logdir(logdir),
      "logdir": str(logdir),
  }


def _wandb_id_for_logdir(logdir: pathlib.Path) -> str:
  if (logdir / "wandb_corrected_id.txt").exists():
    return (logdir / "wandb_corrected_id.txt").read_text().strip().splitlines()[0]
  root = logdir / "wandb" / "wandb"
  try:
    runs = sorted(root.glob("run-*"), key=lambda path: path.stat().st_mtime)
  except Exception:
    return ""
  return runs[-1].name.rsplit("-", 1)[-1] if runs else ""


def _read_metrics(path: pathlib.Path) -> tuple[int, float | None]:
  step = -1
  fps = None
  if not path.exists():
    return step, fps
  try:
    lines = path.read_text(errors="replace").splitlines()[-500:]
  except Exception:
    return step, fps
  for line in lines:
    try:
      row = json.loads(line)
    except Exception:
      continue
    if "step" in row:
      step = max(step, int(row["step"]))
    if "fps/policy" in row:
      fps = row["fps/policy"]
  return step, fps


def _remote_step(run) -> int:
  candidates = []
  for key in ("_step", "step", "global_step"):
    try:
      value = run.summary.get(key)
    except Exception:
      value = None
    if isinstance(value, (int, float)):
      candidates.append(int(value))
  return max(candidates) if candidates else -1


def _updated_age(run) -> int | None:
  updated = getattr(run, "updated_at", None)
  if not updated:
    return None
  try:
    from datetime import datetime, timezone
    if isinstance(updated, str):
      updated = updated.replace("Z", "+00:00")
      dt = datetime.fromisoformat(updated)
    else:
      dt = updated
    if dt.tzinfo is None:
      dt = dt.replace(tzinfo=timezone.utc)
    return int(time.time() - dt.timestamp())
  except Exception:
    return None


def _classify(project: str, state: str, local: dict | None, age: int | None,
              remote_step: int) -> tuple[str, str]:
  bad_states = {"crashed", "failed", "killed"}
  target_steps = PROJECT_TARGET_STEPS.get(project)
  observed_step = max(remote_step, local["step"] if local else -1)
  if target_steps and observed_step >= target_steps:
    if state in bad_states:
      return "locally_complete", "no training; optionally verify W&B final sync"
    return "finished", "no action"
  if local and local.get("pid"):
    if state in bad_states:
      return "state_mismatch", "inspect; local process exists but remote is bad"
    return "active", "monitor"
  if state in bad_states:
    return "failed", "resume if checkpoint exists and resources available"
  if state == "finished":
    return "finished", "no action unless expected steps incomplete"
  if state == "running" and age is not None and age > 900:
    return "stale_remote_running", "resume or restart after checking checkpoint"
  if state == "running":
    return "remote_running_no_local", "wait briefly; verify if W&B stale"
  return "unknown", "inspect"


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--entity", default=ENTITY)
  parser.add_argument("--project", action="append", dest="projects")
  parser.add_argument("--out-prefix", default="logdir/wandb_run_audit")
  args = parser.parse_args()

  repo = pathlib.Path.cwd().resolve()
  projects = args.projects or list(PROJECTS)
  local_by_key, local_by_id = _local_runs(repo)
  api = wandb.Api(timeout=60)
  rows = []
  for project in projects:
    try:
      runs = api.runs(f"{args.entity}/{project}")
    except Exception as exc:
      print(f"ERROR project={project}: {exc}")
      continue
    for run in runs:
      key = (project, run.group or "", run.name or "")
      loc = local_by_id.get(run.id)
      if not loc:
        by_name = local_by_key.get(key)
        if by_name and not by_name.get("wandb_id"):
          loc = by_name
      if not loc:
        loc = _local_disk_run(repo, *key)
      age = _updated_age(run)
      remote_step = _remote_step(run)
      status, action = _classify(project, run.state, loc, age, remote_step)
      rows.append(Row(
          project=project,
          group=run.group or "",
          name=run.name or "",
          run_id=run.id,
          remote_state=run.state,
          remote_step=remote_step,
          updated_age_sec=age,
          local_pid=loc["pid"] if loc else "",
          local_gpu=loc["gpu"] if loc else "",
          local_step=loc["step"] if loc else -1,
          local_fps=loc["fps"] if loc else None,
          local_wandb_id=loc["wandb_id"] if loc else "",
          logdir=loc["logdir"] if loc else "",
          status=status,
          action=action,
      ))

  rows.sort(key=lambda r: (r.status, r.project, r.group, r.name, r.run_id))
  prefix = pathlib.Path(args.out_prefix)
  prefix.parent.mkdir(parents=True, exist_ok=True)
  json_path = prefix.with_suffix(".json")
  tsv_path = prefix.with_suffix(".tsv")
  json_path.write_text(json.dumps(
      [asdict(row) for row in rows], indent=2, sort_keys=True))
  with tsv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys())
                            if rows else list(Row.__annotations__.keys()),
                            delimiter="\t")
    writer.writeheader()
    for row in rows:
      writer.writerow(asdict(row))

  counts = {}
  for row in rows:
    counts[row.status] = counts.get(row.status, 0) + 1
  print("counts", counts)
  print("json", json_path)
  print("tsv", tsv_path)
  for row in rows:
    if row.status not in ("active", "finished"):
      print(
          f"{row.status}\t{row.project}\t{row.group}/{row.name}\t"
          f"id={row.run_id}\tstate={row.remote_state}\t"
          f"remote_step={row.remote_step}\tage={row.updated_age_sec}\t"
          f"local={row.local_pid or '-'}\taction={row.action}")


if __name__ == "__main__":
  main()
