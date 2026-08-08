#!/usr/bin/env python3
import argparse
import json
import math
import pathlib
import time
import traceback

import wandb


ENTITY = "stevendiffdiff-"
GROUP = "wsc_WSC_skip_last_layer_constant_all"
RUNS = [
    ("continual_dreamer_soft_reset_dmcprior_swimmer_cheetah_reacher_size1m", "seed_1000", "yij43b8i"),
    ("continual_dreamer_soft_reset_dmcprior_swimmer_cheetah_reacher_size1m", "seed_2000", "operwvdt"),
    ("continual_dreamer_soft_reset_dmcprior_swimmer_cheetah_reacher_size1m", "seed_3000", "s1enk2tb"),
    ("continual_dreamer_soft_reset_quadruped_walk|quadruped_escape|quadruped_fetch_size1m", "seed_1000", "474kedmt"),
    ("continual_dreamer_soft_reset_quadruped_walk|quadruped_escape|quadruped_fetch_size1m", "seed_2000", "ui0bjkbk"),
    ("continual_dreamer_soft_reset_quadruped_walk|quadruped_escape|quadruped_fetch_size1m", "seed_3000", "wj1um5zm"),
]


def clean(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="logdir")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--project")
    parser.add_argument("--seed")
    parser.add_argument("--run-id")
    parser.add_argument("--flush-step", action="store_true")
    args = parser.parse_args()

    base = pathlib.Path(args.base)
    run_specs = RUNS
    if args.project or args.seed or args.run_id:
        if not (args.project and args.seed and args.run_id):
            raise SystemExit("--project, --seed, and --run-id must be provided together")
        run_specs = [(args.project, args.seed, args.run_id)]
    states = []
    for project, seed, run_id in run_specs:
        logdir = base / project / GROUP / seed
        proxy_dir = logdir / "wandb_proxy"
        proxy_dir.mkdir(exist_ok=True)
        run = wandb.init(
            entity=ENTITY,
            project=project,
            group=GROUP,
            name=seed,
            id=run_id,
            resume="allow",
            dir=str(proxy_dir),
            settings=wandb.Settings(_disable_stats=True),
            reinit="create_new",
        )
        states.append({
            "project": project,
            "seed": seed,
            "path": logdir / "metrics.jsonl",
            "pos": 0,
            "seen": set(),
            "run": run,
        })
        print("initialized", project, seed, run_id, flush=True)

    while True:
        for state in states:
            try:
                path = state["path"]
                if not path.exists():
                    continue
                with path.open() as file:
                    file.seek(state["pos"])
                    lines = file.readlines()
                    state["pos"] = file.tell()
                for line in lines:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    step = int(row.get("step", 0))
                    if step in state["seen"]:
                        continue
                    metrics = {key: clean(value) for key, value in row.items() if key != "step"}
                    metrics["step"] = step
                    state["run"].log(metrics, step=step)
                    if args.flush_step:
                        state["run"].log({"proxy/flush": time.time()}, step=step + 1)
                    state["seen"].add(step)
                    print(
                        "logged",
                        state["project"],
                        state["seed"],
                        "step",
                        step,
                        "fps",
                        row.get("fps/policy", row.get("fps")),
                        "keys",
                        len(metrics),
                        flush=True,
                    )
            except Exception:
                print("ERROR", state.get("project"), state.get("seed"), flush=True)
                traceback.print_exc()
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
