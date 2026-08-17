#!/usr/bin/env python3
import argparse
import json
import math
import pathlib
import time
import traceback

import wandb


ENTITY = "stevendiffdiff-"


def clean(value):
  if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
    return None
  return value


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--interval", type=float, default=30.0)
  parser.add_argument("--project", required=True)
  parser.add_argument("--group", required=True)
  parser.add_argument("--seed", required=True)
  parser.add_argument("--run-id", required=True)
  parser.add_argument("--logdir", required=True)
  parser.add_argument("--flush-step", action="store_true")
  parser.add_argument("--min-step", type=int, default=None)
  parser.add_argument("--heartbeat-interval", type=float, default=60.0)
  args = parser.parse_args()

  logdir = pathlib.Path(args.logdir)
  proxy_dir = logdir / "wandb_proxy"
  proxy_dir.mkdir(exist_ok=True)
  run = wandb.init(
      entity=ENTITY,
      project=args.project,
      group=args.group,
      name=args.seed,
      id=args.run_id,
      resume="allow",
      dir=str(proxy_dir),
      settings=wandb.Settings(_disable_stats=True),
      reinit="create_new",
  )
  path = logdir / "metrics.jsonl"
  pos = 0
  seen = set()
  last_heartbeat = 0.0
  print("initialized", args.project, args.group, args.seed, args.run_id, flush=True)

  while True:
    try:
      if path.exists():
        with path.open() as file:
          file.seek(pos)
          lines = file.readlines()
          pos = file.tell()
        for line in lines:
          if not line.strip():
            continue
          row = json.loads(line)
          step = int(row.get("step", 0))
          if args.min_step is not None and step <= args.min_step:
            continue
          if step in seen:
            continue
          metrics = {key: clean(value) for key, value in row.items() if key != "step"}
          metrics["step"] = step
          run.log(metrics, step=step)
          if args.flush_step:
            run.log({"proxy/flush": time.time()}, step=step + 1)
          seen.add(step)
          print(
              "logged",
              args.project,
              args.group,
              args.seed,
              "step",
              step,
              "keys",
              len(metrics),
              flush=True,
          )
      now = time.time()
      if args.heartbeat_interval > 0 and now - last_heartbeat >= args.heartbeat_interval:
        run.summary.update({"proxy/heartbeat": now})
        last_heartbeat = now
        print("heartbeat", args.project, args.group, args.seed, now, flush=True)
    except Exception:
      print("ERROR", args.project, args.group, args.seed, flush=True)
      traceback.print_exc()
    time.sleep(args.interval)


if __name__ == "__main__":
  main()
