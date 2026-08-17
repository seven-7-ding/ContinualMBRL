#!/usr/bin/env python
"""Continual DrQ training on pixel observations.

This script intentionally does not use VideoRecorder or any train/eval video
writers. Rendering is used only by PixelObservationWrapper to create the image
observations required by dmc-vision training.
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.25")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import json
import pickle
import shutil
import sys
from contextlib import contextmanager

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import gym
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from jaxrl2.agents import DrQLearner
from jaxrl2.data import MemoryEfficientReplayBuffer
from jaxrl2.envs import ContinualDMCEnv
from jaxrl2.evaluation import evaluate
from jaxrl2.wrappers import wrap_pixels


FLAGS = flags.FLAGS

flags.DEFINE_string("tasks", "walker_run,hopper_hop,cheetah_run",
                    "Comma-separated dmc task names.")
flags.DEFINE_integer("obs_dim", 24, "Fixed proprio obs dim inside base env.")
flags.DEFINE_integer("act_dim", 6, "Fixed action dim inside base env.")
flags.DEFINE_integer("task_steps", 1_000_000,
                     "Single-env environment transitions per task.")
flags.DEFINE_integer("task_repeats", 5, "Cycles through the task list.")
flags.DEFINE_integer("initial_global_step", 0,
                     "Global step to resume logging/task schedule from.")
flags.DEFINE_integer("start_training", 4_000,
                     "Per-task warmup transitions before updates.")
flags.DEFINE_integer("batch_size", 256, "Mini-batch size.")
flags.DEFINE_integer("utd", 1, "Updates per environment step.")
flags.DEFINE_integer("replay_buffer_size", 1_000_000, "Per-task replay capacity.")
flags.DEFINE_integer("action_repeat", 2, "DMC action repeat.")
flags.DEFINE_integer("image_size", 84, "Rendered pixel observation size.")
flags.DEFINE_integer("num_stack", 3, "Frame stack count.")
flags.DEFINE_integer("eval_episodes", 10, "Episodes per evaluation.")
flags.DEFINE_integer("eval_interval", 10_000, "Transitions between evals.")
flags.DEFINE_integer("log_interval", 10_000, "Transitions between logs.")
flags.DEFINE_integer("diagnostics_interval", 10_000,
                     "Transitions between WB_FNorm/RMSNorm diagnostics.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("save_dir", "./logdir/continual_drq_vision/default/seed_none",
                    "Run log directory.")
flags.DEFINE_string("project", "continual_mfrl_size1m_dmc_vision",
                    "wandb project.")
flags.DEFINE_string("group", "no_wsc_lr_default", "wandb group.")
flags.DEFINE_string("run_name", "seed_42", "wandb run name.")
flags.DEFINE_string("wandb_run_id", "",
                    "Existing W&B run id to resume; empty creates a run.")
flags.DEFINE_string("wandb_resume", "allow",
                    "W&B resume policy used when a run id is provided.")
flags.DEFINE_integer("wandb_resume_from_step", -1,
                     "Use W&B rewind/resume_from at this step when supported.")
flags.DEFINE_string("restore_checkpoint", "",
                    "Checkpoint directory to restore, or 'latest'.")
flags.DEFINE_integer("checkpoint_interval", 0,
                     "Save agent checkpoint every N global steps; 0 disables.")
flags.DEFINE_boolean("checkpoint_replay", False,
                     "Also save replay buffer in checkpoints. This can be large.")
flags.DEFINE_integer("checkpoint_keep", 2,
                     "Number of recent checkpoints to keep per run.")
flags.DEFINE_string("checkpoint_lock_path",
                    "logdir/scheduler/drq_vision_checkpoint.lock",
                    "File lock used to serialize checkpoint writes.")
flags.DEFINE_string("compile_lock_path",
                    "logdir/scheduler/drq_vision_compile.lock",
                    "File lock used to serialize first JAX/XLA compile.")
flags.DEFINE_string("diagnostics_lock_path",
                    "logdir/scheduler/drq_vision_diagnostics.lock",
                    "File lock used to serialize diagnostics JAX forwards.")
flags.DEFINE_string("egl_device_id", "",
                    "Physical GPU id for EGL/MuJoCo rendering.")
flags.DEFINE_boolean("wandb", True, "Log to wandb.")
flags.DEFINE_boolean("tqdm", False, "Show tqdm progress bar.")
config_flags.DEFINE_config_file(
    "config",
    "configs/continual_drq_vision.py",
    "Training hyperparameter config.",
    lock_config=False,
)


def _load_dotenv(path=".env"):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value


def _as_plain(value):
    if hasattr(value, "to_dict"):
        return _as_plain(value.to_dict())
    if isinstance(value, dict):
        return {str(k): _as_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _write_run_config(save_dir, config, flags_obj):
    os.makedirs(save_dir, exist_ok=True)
    payload = {
        "config": _as_plain(config),
        "flags": _as_plain(flags_obj.flag_values_dict()),
    }
    path = os.path.join(save_dir, "config.yaml")
    try:
        import yaml
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=True)
    except Exception:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("# YAML fallback: JSON is valid YAML 1.2\n")
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")


def _append_metrics(save_dir, step, log_dict):
    if not log_dict:
        return
    path = os.path.join(save_dir, "metrics.jsonl")
    payload = _as_plain({"step": int(step), "metrics": dict(log_dict)})
    with open(path, "a", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")


def _checkpoint_root(save_dir):
    return os.path.join(save_dir, "checkpoints")


def _checkpoint_step(path):
    name = os.path.basename(path.rstrip(os.sep))
    if not name.startswith("step_"):
        return -1
    try:
        return int(name.split("_", 1)[1])
    except Exception:
        return -1


def _latest_checkpoint(save_dir):
    root = _checkpoint_root(save_dir)
    if not os.path.isdir(root):
        return ""
    candidates = [
        os.path.join(root, name) for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name)) and _checkpoint_step(name) >= 0
    ]
    if not candidates:
        return ""
    return max(candidates, key=_checkpoint_step)


def _write_json_atomic(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(_as_plain(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _save_checkpoint(save_dir, agent, replay_buffer, step, task_idx, task_local,
                     first_update_done, include_replay, keep):
    ckpt_dir = os.path.join(_checkpoint_root(save_dir), f"step_{int(step):012d}")
    tmp_dir = f"{ckpt_dir}.tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    agent.save_checkpoint(os.path.join(tmp_dir, "agent.msgpack"))
    _write_json_atomic(os.path.join(tmp_dir, "metadata.json"), {
        "global_step": int(step),
        "task_idx": int(task_idx),
        "task_local": int(task_local),
        "first_update_done": bool(first_update_done),
        "has_replay": bool(include_replay),
        "replay_size": int(len(replay_buffer)),
    })
    if include_replay:
        with open(os.path.join(tmp_dir, "replay_buffer.pkl"), "wb") as handle:
            pickle.dump(replay_buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    os.replace(tmp_dir, ckpt_dir)
    checkpoints = [
        os.path.join(_checkpoint_root(save_dir), name)
        for name in os.listdir(_checkpoint_root(save_dir))
        if os.path.isdir(os.path.join(_checkpoint_root(save_dir), name))
        and _checkpoint_step(name) >= 0
    ]
    for old in sorted(checkpoints, key=_checkpoint_step)[:-max(keep, 1)]:
        shutil.rmtree(old)
    return ckpt_dir


def _restore_checkpoint_path(save_dir, requested):
    if not requested:
        return ""
    if requested == "latest":
        return _latest_checkpoint(save_dir)
    return requested


def _load_checkpoint(ckpt_dir, agent):
    metadata_path = os.path.join(ckpt_dir, "metadata.json")
    agent_path = os.path.join(ckpt_dir, "agent.msgpack")
    if not os.path.exists(metadata_path) or not os.path.exists(agent_path):
        raise FileNotFoundError(f"Incomplete checkpoint: {ckpt_dir}")
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    agent.restore_checkpoint(agent_path)
    replay_buffer = None
    replay_path = os.path.join(ckpt_dir, "replay_buffer.pkl")
    if os.path.exists(replay_path):
        with open(replay_path, "rb") as handle:
            replay_buffer = pickle.load(handle)
    return metadata, replay_buffer


def _finite_scalar(value):
    try:
        value = float(np.asarray(value))
    except Exception:
        return None
    return value if np.isfinite(value) else None


def _make_env(task_list, obs_dim, act_dim, seed, action_repeat, image_size, num_stack):
    env = ContinualDMCEnv(task_list, obs_dim=obs_dim, act_dim=act_dim, seed=seed)
    env = wrap_pixels(
        env,
        action_repeat=action_repeat,
        image_size=image_size,
        num_stack=num_stack,
        camera_id=0,
    )
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    return env


def _switch_task(env, task_name):
    env.unwrapped._load_task(task_name)
    env.unwrapped.task_step = 0
    return env.reset()


def _new_replay_buffer(env, seed, capacity):
    replay_buffer = MemoryEfficientReplayBuffer(
        env.observation_space, env.action_space, capacity)
    replay_buffer.seed(seed)
    return replay_buffer


def _replay_buffer_metrics(replay_buffer):
    usage = replay_buffer.ram_usage()
    return {
        "replay_buffer/ram_allocated_bytes": float(usage["allocated_bytes"]),
        "replay_buffer/ram_used_bytes": float(usage["used_bytes"]),
        "replay_buffer/ram_allocated_mb": float(usage["allocated_mb"]),
        "replay_buffer/ram_used_mb": float(usage["used_mb"]),
        "replay_buffer/ram_allocated_gb": float(usage["allocated_gb"]),
        "replay_buffer/ram_used_gb": float(usage["used_gb"]),
        "replay_buffer/size": float(usage["size"]),
        "replay_buffer/capacity": float(usage["capacity"]),
    }


def _map_metrics(update_info):
    log_dict = {}
    loss_remap = {
        "actor_loss": "loss/policy",
        "critic_loss": "loss/value",
        "temperature_loss": "loss/temp",
    }
    train_remap = {
        "entropy": "train/ent/action",
        "temperature": "train/temperature",
        "q": "train/q_mean",
    }
    for key, value in update_info.items():
        scalar = _finite_scalar(value)
        if scalar is None:
            continue
        if key in loss_remap:
            log_dict[loss_remap[key]] = scalar
        elif key in train_remap:
            log_dict[train_remap[key]] = scalar
        elif key.startswith("mechanism/l2_init/"):
            log_dict[f"loss/{key}"] = scalar
        elif key.startswith("mechanism/"):
            log_dict[f"opt/{key}"] = scalar
        elif key.startswith("wsc/"):
            log_dict[f"opt/{key}"] = scalar
        elif key.startswith(("opt/", "act_redo/", "grad_redo/")):
            log_dict[key] = scalar
        elif key.startswith("data_augmentation/"):
            log_dict[f"train/{key}"] = scalar
        else:
            log_dict[f"train/{key}"] = scalar
    return log_dict


@contextmanager
def _compile_lock(path):
    if not path:
        yield
        return
    import fcntl
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def main(_):
    _load_dotenv()
    if FLAGS.egl_device_id:
        os.environ["EGL_DEVICE_ID"] = str(FLAGS.egl_device_id)
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(FLAGS.egl_device_id)
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    kwargs = dict(FLAGS.config)
    kwargs.pop("jax_mem_fraction", None)

    task_list = [task.strip() for task in FLAGS.tasks.split(",") if task.strip()]
    task_schedule = task_list * FLAGS.task_repeats
    total_steps = FLAGS.task_steps * len(task_schedule)
    initial_global_step = max(0, min(int(FLAGS.initial_global_step), total_steps))
    initial_task_idx = min(
        initial_global_step // FLAGS.task_steps,
        max(len(task_schedule) - 1, 0),
    )
    initial_task_local = initial_global_step - initial_task_idx * FLAGS.task_steps

    _write_run_config(FLAGS.save_dir, FLAGS.config, FLAGS)
    if FLAGS.wandb:
        wandb_kwargs = dict(
            project=FLAGS.project,
            group=FLAGS.group,
            name=FLAGS.run_name,
            dir=FLAGS.save_dir,
        )
        wandb_run_id = FLAGS.wandb_run_id or os.environ.get("WANDB_RUN_ID", "")
        if wandb_run_id and FLAGS.wandb_resume_from_step >= 0:
            wandb_kwargs["resume_from"] = (
                f"{wandb_run_id}?_step={int(FLAGS.wandb_resume_from_step)}"
            )
        elif wandb_run_id:
            wandb_kwargs["id"] = wandb_run_id
            wandb_kwargs["resume"] = (
                FLAGS.wandb_resume or os.environ.get("WANDB_RESUME", "allow")
            )
        wandb.init(**wandb_kwargs)
        allow_config_change = bool(wandb_run_id)
        wandb.config.update(
            _as_plain(FLAGS.config),
            allow_val_change=allow_config_change,
        )
        wandb.config.update(
            _as_plain(FLAGS.flag_values_dict()),
            allow_val_change=allow_config_change,
        )

    env = _make_env(
        task_list, FLAGS.obs_dim, FLAGS.act_dim, FLAGS.seed,
        FLAGS.action_repeat, FLAGS.image_size, FLAGS.num_stack)
    eval_env = _make_env(
        [task_schedule[initial_task_idx]], FLAGS.obs_dim, FLAGS.act_dim, FLAGS.seed + 42,
        FLAGS.action_repeat, FLAGS.image_size, FLAGS.num_stack)

    obs = _switch_task(env, task_schedule[initial_task_idx])
    agent = DrQLearner(
        FLAGS.seed,
        env.observation_space.sample(),
        env.action_space.sample(),
        **kwargs,
    )
    replay_buffer = _new_replay_buffer(
        env, FLAGS.seed + initial_task_idx,
        min(FLAGS.replay_buffer_size, FLAGS.task_steps))

    global_step = initial_global_step
    task_local = initial_task_local
    task_idx = initial_task_idx
    completed_scores = []
    completed_lengths = []
    last_batch = None
    update_info = {}
    first_update_done = False

    restore_path = _restore_checkpoint_path(FLAGS.save_dir, FLAGS.restore_checkpoint)
    if restore_path:
        metadata, restored_replay = _load_checkpoint(restore_path, agent)
        global_step = int(metadata["global_step"])
        task_idx = int(metadata["task_idx"])
        task_local = int(metadata["task_local"])
        first_update_done = bool(metadata.get("first_update_done", True))
        if task_idx >= len(task_schedule):
            task_idx = len(task_schedule) - 1
            task_local = FLAGS.task_steps
        cur_task = task_schedule[task_idx]
        obs = _switch_task(env, cur_task)
        eval_env.close()
        eval_env = _make_env(
            [cur_task], FLAGS.obs_dim, FLAGS.act_dim, FLAGS.seed + 42,
            FLAGS.action_repeat, FLAGS.image_size, FLAGS.num_stack)
        if restored_replay is not None:
            replay_buffer = restored_replay
        else:
            replay_buffer = _new_replay_buffer(
                env, FLAGS.seed + task_idx,
                min(FLAGS.replay_buffer_size, FLAGS.task_steps))
        print(f"Restored checkpoint {restore_path} at step {global_step}.")

    pbar = tqdm.tqdm(
        total=total_steps, initial=global_step, smoothing=0.1,
        disable=not FLAGS.tqdm)

    while global_step < total_steps:
        if task_local >= FLAGS.task_steps:
            task_idx += 1
            if task_idx >= len(task_schedule):
                break
            cur_task = task_schedule[task_idx]
            cycle = task_idx // len(task_list) + 1
            print(f"\n[step {global_step}] cycle {cycle}/{FLAGS.task_repeats}: {cur_task}")
            obs = _switch_task(env, cur_task)
            eval_env.close()
            eval_env = _make_env(
                [cur_task], FLAGS.obs_dim, FLAGS.act_dim, FLAGS.seed + 42,
                FLAGS.action_repeat, FLAGS.image_size, FLAGS.num_stack)
            replay_buffer = _new_replay_buffer(
                env, FLAGS.seed + task_idx,
                min(FLAGS.replay_buffer_size, FLAGS.task_steps))
            task_local = 0
            update_info = {}

        if task_local < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(obs)

        next_obs, reward, done, info = env.step(action)
        truncated = info.get("TimeLimit.truncated", False)
        mask = 0.0 if (done and not truncated) else 1.0
        replay_buffer.insert(dict(
            observations=obs,
            actions=action,
            rewards=float(reward),
            masks=mask,
            dones=bool(done),
            next_observations=next_obs,
        ))
        obs = next_obs
        if done:
            if env.return_queue:
                completed_scores.append(float(env.return_queue[-1]))
                completed_lengths.append(float(env.length_queue[-1]))
            obs = env.reset()

        global_step += 1
        task_local += 1
        pbar.update(1)

        if task_local >= FLAGS.start_training and len(replay_buffer) >= FLAGS.batch_size:
            for _ in range(FLAGS.utd):
                last_batch = replay_buffer.sample(FLAGS.batch_size)
                if first_update_done:
                    update_info.update(agent.update(last_batch))
                else:
                    print(f"Waiting for compile lock: {FLAGS.compile_lock_path}")
                    with _compile_lock(FLAGS.compile_lock_path):
                        print("Running first JAX update compile.")
                        update_info.update(agent.update(last_batch))
                    first_update_done = True

        log_due = global_step % FLAGS.log_interval == 0
        diag_due = (
            last_batch is not None and
            global_step % FLAGS.diagnostics_interval == 0)
        if log_due or diag_due:
            log_dict = _map_metrics(update_info)
            if log_due:
                log_dict.update(_replay_buffer_metrics(replay_buffer))
            if completed_scores:
                log_dict["episode/score"] = float(np.mean(completed_scores))
                log_dict["episode/length"] = float(np.mean(completed_lengths))
                completed_scores.clear()
                completed_lengths.clear()
            if diag_due:
                with _compile_lock(FLAGS.diagnostics_lock_path):
                    diagnostics = agent.collect_diagnostics(last_batch)
                log_dict.update(_map_metrics(diagnostics))
            if log_dict:
                _append_metrics(FLAGS.save_dir, global_step, log_dict)
                if FLAGS.wandb:
                    wandb.log(log_dict, step=global_step)

        if (
            task_local >= FLAGS.start_training and
            global_step % FLAGS.eval_interval == 0
        ):
            cur_task = task_schedule[task_idx]
            eval_info = evaluate(agent, eval_env, FLAGS.eval_episodes)
            print(f"[{cur_task} | step {global_step}] return={eval_info['return']:.1f}")
            eval_log = {
                f"performance/{cur_task}/"
                f"{'score' if key == 'return' else key}": value
                for key, value in eval_info.items()
            }
            _append_metrics(FLAGS.save_dir, global_step, eval_log)
            if FLAGS.wandb:
                wandb.log(eval_log, step=global_step)

        if FLAGS.checkpoint_interval > 0 and global_step % FLAGS.checkpoint_interval == 0:
            with _compile_lock(FLAGS.checkpoint_lock_path):
                ckpt_dir = _save_checkpoint(
                    FLAGS.save_dir,
                    agent,
                    replay_buffer,
                    global_step,
                    task_idx,
                    task_local,
                    first_update_done,
                    FLAGS.checkpoint_replay,
                    FLAGS.checkpoint_keep,
                )
            print(f"Saved checkpoint {ckpt_dir}.")

    pbar.close()
    env.close()
    eval_env.close()
    if FLAGS.wandb:
        wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    app.run(main)
