#!/usr/bin/env python3
"""Attach to retained L2-init sweep PIDs and wait for first valid logging."""

from __future__ import annotations

import json
import time

from launch_l2_init_weight_sweep import (
    MAX_RESTARTS,
    POLL_SECONDS,
    STATE_FILE,
    append_event,
    launch,
    load_env,
    pid_alive,
    save_state,
    valid_logging,
)


ACTIVE_WEIGHTS = ("2e-4", "2e-6")


def load_state() -> dict:
  if not STATE_FILE.exists():
    raise SystemExit(f"Missing scheduler state: {STATE_FILE}")
  state = json.loads(STATE_FILE.read_text())
  retained = []
  for job in state.get("jobs", []):
    if job.get("weight") in ACTIVE_WEIGHTS:
      retained.append(job)
    elif job.get("weight") == "2e-3":
      job["status"] = "abandoned"
      job["valid_logging"] = False
      job["last_reason"] = "abandoned by user instruction"
  state["jobs"] = retained
  return state


def main() -> None:
  state = load_state()
  base_env = load_env()
  append_event(
      state,
      f"monitor_retained_start jobs={len(state['jobs'])} weights={','.join(ACTIVE_WEIGHTS)}",
  )
  while True:
    all_valid = True
    for job in state["jobs"]:
      if not pid_alive(job.get("pid")):
        if job.get("restarts", 0) >= MAX_RESTARTS:
          job["status"] = "blocked"
          job["last_reason"] = "pid missing; max restarts reached"
          append_event(state, f"blocked {job['id']} pid_missing")
          raise SystemExit(f"{job['id']} blocked: pid missing")
        job["restarts"] = job.get("restarts", 0) + 1
        append_event(state, f"restarting_retained {job['id']} missing_pid restarts={job['restarts']}")
        proc = launch(job, base_env)
        job["pid"] = proc.pid
        all_valid = False
        continue
      ok, reason = valid_logging(job)
      job["valid_logging"] = ok
      job["last_reason"] = reason
      if ok:
        if job.get("status") != "valid_logging":
          job["status"] = "valid_logging"
          append_event(state, f"valid_logging {job['id']} pid={job['pid']} {reason}")
      else:
        if job.get("status") != "running":
          job["status"] = "running"
        all_valid = False
    save_state(state)
    if all_valid:
      append_event(state, "all_retained_runs_have_valid_logging")
      return
    pending = [job["id"] for job in state["jobs"] if not job.get("valid_logging")]
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S HKT')}] "
        f"waiting retained_pending={len(pending)} sample={pending[:6]}",
        flush=True,
    )
    time.sleep(POLL_SECONDS)


if __name__ == "__main__":
  main()
