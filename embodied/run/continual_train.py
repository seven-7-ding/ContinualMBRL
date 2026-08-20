import collections
from functools import partial as bind

import elements
import embodied
from embodied.jax import reset_targets
from embodied.jax import wsc as wsc_lib
from embodied.jax.internal import stats
import numpy as np


def continual_train(make_agent, make_replay, make_env, make_stream, make_logger, args):

  agent = make_agent()
  replay = make_replay()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  step = logger.step
  usage = elements.Usage(**args.usage)
  train_agg = elements.Agg()
  epstats = elements.Agg()
  performance_agg = collections.defaultdict(elements.Agg)
  episodes = collections.defaultdict(elements.Agg)
  policy_fps = elements.FPS()
  train_fps = elements.FPS()

  # Parse task list for per-task performance logging.
  _raw_task = getattr(args, 'task', '')
  task_list = [t.strip() for t in _raw_task.split('|')] if '|' in _raw_task else [_raw_task or 'task']

  def parse_task_intervals(value):
    if value in (None, False, '', 'none', 'None'):
      return None
    if isinstance(value, str):
      raw_items = value.replace(',', '|').split('|')
    else:
      raw_items = list(value)
    intervals = [int(float(item)) for item in raw_items if str(item).strip()]
    if len(intervals) != len(task_list):
      raise ValueError(
          f'run.task_intervals must provide one interval per task: '
          f'got {intervals} for tasks {task_list}')
    if any(interval <= 0 for interval in intervals):
      raise ValueError(f'run.task_intervals must be positive, got {intervals}')
    return intervals

  task_intervals = parse_task_intervals(getattr(args, 'task_intervals', ''))
  task_boundaries = None
  if task_intervals:
    task_boundaries = np.cumsum(task_intervals).astype(np.int64).tolist()

  batch_steps = args.batch_size * args.batch_length
  should_train = elements.when.Ratio(args.train_ratio / batch_steps)
  print(f'Train ratio: {should_train._ratio}, Batch steps: {batch_steps}, Train calls per step: {args.train_ratio / batch_steps}')
  # should_log = embodied.LocalClock(args.log_every)
  should_log = elements.when.Every(args.log_every, initial=False)
  should_report = elements.when.Every(args.report_every)
  should_save = embodied.LocalClock(args.save_every)
  # TODO: enable env switching.
  should_switch = None
  if not task_intervals:
    should_switch = elements.when.Every(args.task_interval, initial=True)
  reset_frequency = int(getattr(args, 'reset_frequency', 0) or 0)
  revive_epoch = int(getattr(args, 'revive_epoch', 0) or 0)
  revive_strategy = str(getattr(args, 'revive_strategy', 'fixed')).lower()
  last_loss_num = int(getattr(args, 'last_loss_num', 10) or 0)
  revive_threshold = float(getattr(args, 'revive_threshold', 1.0))

  def canonical_reset_mechanism(mechanism):
    if mechanism in (None, False, '', 'false', 'none', 'disabled', 'off'):
      return 'disabled'
    enabled, _, _ = wsc_lib.parse_mechanism(mechanism)
    if not enabled:
      return 'disabled'
    return str(mechanism)

  def canonical_reset_target(target):
    return reset_targets.canonical_target(target)

  def legacy_reset_mode_target(mode):
    if mode in (None, False, '', 'false', 'none'):
      return None
    return canonical_reset_target(mode)

  reset_mechanism = canonical_reset_mechanism(
      getattr(args, 'reset_mechanism', 'disabled'))
  reset_target = canonical_reset_target(
      getattr(args, 'reset_target', 'all'))
  legacy_target = legacy_reset_mode_target(getattr(args, 'reset_mode', None))
  if legacy_target:
    reset_target = legacy_target
  if reset_frequency < 0:
    raise ValueError(f'reset_frequency must be >= 0, got {reset_frequency}')
  if revive_epoch < 0:
    raise ValueError(f'revive_epoch must be >= 0, got {revive_epoch}')
  if reset_frequency > 0 and reset_mechanism == 'disabled':
      raise ValueError(
        'run.reset_frequency > 0 requires an enabled parameter mechanism. '
        'Set run.reset_frequency=0 to disable reset scheduling.')
  reset_alpha = float(getattr(args, 'reset_alpha', 0.5))
  if not 0.0 <= reset_alpha <= 1.0:
    raise ValueError(f'reset_alpha must be in [0, 1], got {reset_alpha}')
  if revive_strategy not in ('fixed', 'threshold'):
    raise ValueError(
        f"revive_strategy must be 'fixed' or 'threshold', got "
        f'{revive_strategy}')
  if last_loss_num <= 0:
    raise ValueError(f'last_loss_num must be > 0, got {last_loss_num}')
  if revive_threshold <= 0:
    raise ValueError(f'revive_threshold must be > 0, got {revive_threshold}')
  next_reset_step = [reset_frequency if reset_frequency > 0 else None]
  min_replay_for_train = args.batch_size * args.batch_length
  agent_terms = {'policy', 'value', 'repval'}
  loss_windows = {
      'wm': collections.defaultdict(
          lambda: collections.deque(maxlen=last_loss_num)),
      'agent': collections.defaultdict(
          lambda: collections.deque(maxlen=last_loss_num)),
  }

  def extract_loss_items(mets):
    vals = {}
    for key, value in mets.items():
      if not key.startswith('loss/'):
        continue
      name = key.split('/', 1)[1]
      val = float(value)
      if np.isfinite(val):
        vals[name] = val
    return vals

  def split_mode_losses(mets):
    vals = extract_loss_items(mets)
    wm_vals = {k: v for k, v in vals.items() if k not in agent_terms}
    agent_vals = {k: v for k, v in vals.items() if k in agent_terms}
    return wm_vals, agent_vals

  def update_loss_windows(mets):
    wm_losses, agent_losses = split_mode_losses(mets)
    for name, val in wm_losses.items():
      loss_windows['wm'][name].append(val)
    for name, val in agent_losses.items():
      loss_windows['agent'][name].append(val)

  def loss_bucket(mode):
    if mode in ('dyn', 'wm_head'):
      return 'wm'
    if mode in ('agent_head',):
      return 'agent'
    if mode in ('all', 'all_head'):
      return 'all'
    raise ValueError(f'Unknown loss bucket mode: {mode}')

  def current_mode_losses(mode, mets):
    wm_losses, agent_losses = split_mode_losses(mets)
    bucket = loss_bucket(mode)
    if bucket == 'wm':
      return wm_losses
    if bucket == 'agent':
      return agent_losses
    return {**wm_losses, **agent_losses}

  def get_last_loss(mode):
    bucket = loss_bucket(mode)
    if bucket == 'all':
      wm = get_last_loss('wm')
      agent = get_last_loss('agent')
      if wm and agent:
        return {**wm, **agent}
      return wm or agent
    result = {}
    for name, history in loss_windows[bucket].items():
      values = list(history)
      if values:
        result[name] = float(np.mean(values))
    return result or None

  def mean_loss_dict(loss_dict):
    if not loss_dict:
      return np.nan
    return float(np.mean(list(loss_dict.values())))

  @elements.timer.section('logfn')
  def logfn(tran, worker):
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')
    for key, value in tran.items():
      if value.dtype == np.uint8 and value.ndim == 3:
        if worker == 0:
          episode.add(f'policy_{key}', value, agg='stack')
      elif key.startswith('log/'):
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key + '/avg', value, agg='avg')
        episode.add(key + '/max', value, agg='max')
        episode.add(key + '/sum', value, agg='sum')
    if tran['is_last']:
      result = episode.result()
      # Per-task performance logging (mirrors SAC's performance/{task}/score).
      current_task = task_list[switch_count % len(task_list)]
      performance = performance_agg[f'{switch_count % len(task_list)}_{current_task}']
      performance.add('score', result.pop('score'), agg='avg')
      performance.add('length', result.pop('length'), agg='avg')
      rew = result.pop('rewards')
      if len(rew) > 1:
        result['delta_reward>0.01_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      result.update(stats(rew, "real_reward"))
      epstats.add(result)

  stream_train = [None]
  # Create report stream lazily. Some small-model configs use a train replay
  # length shorter than report_length, and eager prefetch would fail before the
  # first report is actually needed.
  stream_report = [None]

  carry_train = [agent.init_train(args.batch_size)]
  carry_report = agent.init_report(args.batch_size)

  def add_train_metrics(mets):
    for key, value in mets.items():
      if key.startswith('mechanism/cbp/reset_count_since_log/'):
        train_agg.add(key, value, agg='sum', prefix='train')
      elif key.startswith('opt/mechanism/l2_init/module_delta_sq/'):
        train_agg.add(key, value, agg='last', prefix='train')
      else:
        train_agg.add(key, value, prefix='train')

  def trainfn(tran, worker):
    if len(replay) < args.batch_size * args.batch_length:
      return
    if stream_train[0] is None:
      stream_train[0] = iter(agent.stream(make_stream(replay, 'train')))
    for _ in range(should_train(step)):
      with elements.timer.section('stream_next'):
        batch = next(stream_train[0])
      carry_train[0], outs, mets = agent.train(carry_train[0], batch)
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay.update(outs['replay'])
      update_loss_windows(mets)
      add_train_metrics(mets)

  def revive(mode, max_steps, label, last_losses):
    if max_steps <= 0:
      print(f'Skip {label}: revive_epoch={max_steps}.')
      return
    if len(replay) < min_replay_for_train:
      print(
          f'Skip {label}: replay too small ({len(replay)} < '
          f'{min_replay_for_train}).')
      return
    target_losses = None
    if last_losses is not None:
      target_losses = {
          name: value * revive_threshold
          for name, value in last_losses.items()}
    min_steps_before_threshold_check = max(10, int(np.ceil(max_steps / 100.0)))
    done = 0
    final_losses = {}
    for _ in range(max_steps):
      if len(replay) < min_replay_for_train:
        break
      if stream_train[0] is None:
        stream_train[0] = iter(agent.stream(make_stream(replay, 'train')))
      with elements.timer.section('stream_next'):
        batch = next(stream_train[0])
      carry_train[0], outs, mets = agent.train(
          carry_train[0], batch, train_mode=mode)
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay.update(outs['replay'])
      update_loss_windows(mets)
      add_train_metrics(mets)
      done += 1
      cur_losses = current_mode_losses(mode, mets)
      if cur_losses:
        final_losses = cur_losses
      if revive_strategy == 'threshold' and target_losses:
        if done < min_steps_before_threshold_check:
          continue
        all_matched = True
        for name, target in target_losses.items():
          cur = final_losses.get(name, np.inf)
          if not np.isfinite(cur) or cur > target:
            all_matched = False
            break
        if all_matched:
          break
    revive_mets = {
        'revive_epoch': float(done),
        'last_loss': mean_loss_dict(last_losses),
        'target_loss': mean_loss_dict(target_losses),
        'final_loss': mean_loss_dict(final_losses),
        'min_steps_before_threshold_check': float(
            min_steps_before_threshold_check),
    }
    for name, value in (last_losses or {}).items():
      revive_mets[f'last_loss/{name}'] = value
    for name, value in (target_losses or {}).items():
      revive_mets[f'target_loss/{name}'] = value
    for name, value in (final_losses or {}).items():
      revive_mets[f'final_loss/{name}'] = value
    logger.add(revive_mets, prefix=f'revive/{mode}')
    logger.write()
    print(
        f'Finished {label}: {done}/{max_steps} updates '
        f'(strategy={revive_strategy}, min_check_steps='
        f'{min_steps_before_threshold_check}, last_losses={last_losses}, '
        f'target_losses={target_losses}, final_losses={final_losses}).')

  def periodic_reset():
    print(
        f'Mechanism schedule reached at step {step.value}: '
        f'mechanism={reset_mechanism}, target={reset_target}, alpha={reset_alpha}. '
        'Invoking scheduled parameter reset if supported by agent.reset_params.')

    if reset_mechanism in ('sandp', 'sandp_wo_opt', 'hard'):
      agent.reset_params(reset_target, mechanism=reset_mechanism, alpha=reset_alpha)
      if revive_epoch <= 0:
        return

    if revive_epoch <= 0:
      return
    if reset_target == 'all':
      revive('wm', revive_epoch, 'world model mechanism revive', get_last_loss('wm'))
      revive(
          'agent', revive_epoch, 'agent mechanism revive', get_last_loss('agent'))
    else:
      revive(
          reset_target, revive_epoch, f'{reset_target} mechanism revive',
          get_last_loss(reset_target))

  cp = elements.Checkpoint(logdir / 'ckpt')
  cp.step = step
  cp.agent = agent
  cp.replay = replay
  if args.from_checkpoint:
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=bind(agent.load, regex=args.from_checkpoint_regex)))
  def switch_count_from_step(value):
    if task_intervals:
      return sum(int(value) >= boundary for boundary in task_boundaries)
    if not task_list or args.task_interval <= 0:
      return 0
    return int(value) // int(args.task_interval)

  def phase_start_from_switch_count(count):
    if task_intervals:
      return 0 if count <= 0 else int(task_boundaries[count - 1])
    return count * int(args.task_interval)

  def should_switch_now(value, count):
    if task_intervals:
      return count < len(task_boundaries) and int(value) >= task_boundaries[count]
    return should_switch(value)

  checkpoint_exists = cp.exists()
  if checkpoint_exists:
    cp.load()
  else:
    cp.save()
  if reset_frequency > 0:
    current_step = int(step.value)
    next_reset_step[0] = (
        current_step // reset_frequency + 1) * reset_frequency

  print('Start training loop')
  policy = lambda *args: agent.policy(*args, mode='train')

  # Restore the environment phase from the checkpointed global step. Without
  # this, resumed continual runs restart from task 0 and log scores under the
  # wrong task keys until future switches catch up.
  switch_count = switch_count_from_step(step.value)
  if checkpoint_exists and switch_count > 0:
    phase_steps = int(step.value) - phase_start_from_switch_count(switch_count)
    replay.clear()
    if phase_steps > 0:
      replay.load(amount=phase_steps)
  if should_switch is not None:
    should_switch(step)
  fns = [bind(make_env, i, switch_count=switch_count) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=not args.debug)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(replay.add)
  driver.on_step(logfn)
  driver.on_step(trainfn)
  driver.reset(agent.init_policy)
  
  while step < args.steps:
    if next_reset_step[0] is not None:
      while step >= next_reset_step[0]:
        periodic_reset()
        next_reset_step[0] += reset_frequency

    if should_switch_now(step, switch_count):
      switch_count += 1
      fns = [bind(make_env, i, switch_count=switch_count) for i in range(args.envs)]
      driver.switch_envs(
        fns, parallel=not args.debug
      )
      driver.reset(agent.init_policy)
      replay.clear(disk=True, archive_prefix=f'task_switch_{int(step.value)}')
      stream_train[0] = None
      stream_report[0] = None
      print(f"Switched to new environment at step {step.value}.")

    driver(policy, steps=10)

    report_length = getattr(replay, 'length', 0)
    can_report = report_length >= args.report_length + args.replay_context
    if should_report(step) and len(replay) and can_report:
      if stream_report[0] is None:
        stream_report[0] = iter(agent.stream(make_stream(replay, 'report')))
      agg = elements.Agg()
      for _ in range(args.consec_report * args.report_batches):
        carry_report, mets = agent.report(carry_report, next(stream_report[0]))
        agg.add(mets)
      logger.add(agg.result(), prefix='report')

    if should_log(step):
      train_metrics = train_agg.result()
      train_metrics_new = {}
      loss_metrics = {}
      opt_metrics = {}
      act_redo_metrics = {}
      grad_redo_metrics = {}
      data_diversity_metrics = {}
      for k, v in train_metrics.items():
        if "train/loss/" in k and "opt" not in k:
          loss_metrics[k.replace('train/loss/', '')] = v
        elif "train/opt/mechanism/l2_init/module_delta_sq/" in k:
          loss_metrics[k.replace('train/opt/', '')] = v
        elif "train/opt/" in k and "grad_redo" not in k:
          opt_metrics[k.replace('train/opt/', '')] = v
        elif "train/act_redo/" in k:
          if not np.isnan(float(v)):
            act_redo_metrics[k.replace('train/act_redo/', '')] = v
        elif "train/opt/grad_redo/" in k:
          if not np.isnan(float(v)):
            grad_redo_metrics[k.replace('train/opt/grad_redo/', '')] = v
        elif "train/data_diversity/" in k:
          if not np.isnan(float(v)):
            data_diversity_metrics[k.replace('train/data_diversity/', '')] = v
        else:
          train_metrics_new[k.replace('train/', '')] = v
      logger.add(train_metrics_new, prefix='train')
      logger.add(loss_metrics, prefix='loss')
      logger.add(opt_metrics, prefix='opt')
      logger.add(act_redo_metrics, prefix='act_redo')
      logger.add(grad_redo_metrics, prefix='grad_redo')
      logger.add(data_diversity_metrics, prefix='data_diversity')
      for key, agg in performance_agg.items():
        logger.add(agg.result(), prefix=f'performance/{key}')
      logger.add(epstats.result(), prefix='epstats')
      logger.add(replay.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'reset/frequency': float(reset_frequency)})
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()

    if should_save(step):
      cp.save()

  logger.close()
