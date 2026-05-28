import collections
from functools import partial as bind

import elements
from jax import grad
import embodied
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

  batch_steps = args.batch_size * args.batch_length
  should_train = elements.when.Ratio(args.train_ratio / batch_steps)
  print(f'Train ratio: {should_train._ratio}, Batch steps: {batch_steps}, Train calls per step: {args.train_ratio / batch_steps}')
  # should_log = embodied.LocalClock(args.log_every)
  should_log = elements.when.Every(args.log_every, initial=False)
  should_report = elements.when.Every(args.report_every)
  should_save = embodied.LocalClock(args.save_every)
  # TODO: enable env switching.
  should_switch = elements.when.Every(args.task_interval, initial=True)
  reset_frequency = int(getattr(args, 'reset_frequency', 0) or 0)
  revive_epoch = int(getattr(args, 'revive_epoch', 100) or 0)
  revive_strategy = str(getattr(args, 'revive_strategy', 'fixed')).lower()
  last_loss_num = int(getattr(args, 'last_loss_num', 10) or 0)
  revive_threshold = float(getattr(args, 'revive_threshold', 1.0))

  def canonical_reset_mode(mode):
    aliases = {
        None: 'no_reset',
        False: 'no_reset',
        'false': 'no_reset',
        'none': 'no_reset',
        'no_reset': 'no_reset',
        'reset_only_agent': 'reset_only_agent',
        'agent': 'reset_only_agent',
        'reset_only_wm': 'reset_only_wm',
        'wm': 'reset_only_wm',
        'world_model': 'reset_only_wm',
        'worldmodel': 'reset_only_wm',
        True: 'reset_all',
        'true': 'reset_all',
        'all': 'reset_all',
        'reset_all': 'reset_all',
    }
    mode = aliases.get(mode, mode)
    if mode not in ('no_reset', 'reset_only_agent', 'reset_only_wm', 'reset_all'):
      raise ValueError(f'Unknown reset_mode: {mode}')
    return mode

  reset_mode = canonical_reset_mode(getattr(args, 'reset_mode', 'no_reset'))
  if reset_frequency < 0:
    raise ValueError(f'reset_frequency must be >= 0, got {reset_frequency}')
  if revive_epoch < 0:
    raise ValueError(f'revive_epoch must be >= 0, got {revive_epoch}')
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

  def current_mode_losses(mode, mets):
    wm_losses, agent_losses = split_mode_losses(mets)
    return wm_losses if mode == 'wm' else agent_losses

  def get_last_loss(mode):
    result = {}
    for name, history in loss_windows[mode].items():
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

  stream_train = iter(agent.stream(make_stream(replay, 'train')))
  # Create report stream lazily. Some small-model configs use a train replay
  # length shorter than report_length, and eager prefetch would fail before the
  # first report is actually needed.
  stream_report = [None]

  carry_train = [agent.init_train(args.batch_size)]
  carry_report = agent.init_report(args.batch_size)

  def trainfn(tran, worker):
    if len(replay) < args.batch_size * args.batch_length:
      return
    for _ in range(should_train(step)):
      with elements.timer.section('stream_next'):
        batch = next(stream_train)
      carry_train[0], outs, mets = agent.train(carry_train[0], batch)
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay.update(outs['replay'])
      update_loss_windows(mets)
      train_agg.add(mets, prefix='train')

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
      with elements.timer.section('stream_next'):
        batch = next(stream_train)
      carry_train[0], outs, mets = agent.train(
          carry_train[0], batch, train_mode=mode)
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay.update(outs['replay'])
      update_loss_windows(mets)
      train_agg.add(mets, prefix='train')
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
    if reset_mode == 'no_reset':
      print(
          f'Reset trigger reached at step {step.value}, '
          'reset_mode=no_reset so no reset/revive executed.')
      return
    if reset_mode == 'reset_only_agent':
      last_loss = get_last_loss('agent')
      agent.reset_params('agent')
      print(f'Reset agent at step {step.value}. last_loss={last_loss}')
      revive('agent', revive_epoch, 'agent revive', last_loss)
      return
    if reset_mode == 'reset_only_wm':
      last_loss = get_last_loss('wm')
      agent.reset_params('wm')
      print(f'Reset world model at step {step.value}. last_loss={last_loss}')
      revive('wm', revive_epoch, 'world model revive', last_loss)
      return
    if reset_mode == 'reset_all':
      last_wm_loss = get_last_loss('wm')
      last_agent_loss = get_last_loss('agent')
      agent.reset_params('all')
      print(
          f'Reset world model and agent at step {step.value}. '
          f'last_wm_loss={last_wm_loss}, last_agent_loss={last_agent_loss}')
      revive('wm', revive_epoch, 'world model revive', last_wm_loss)
      revive('agent', revive_epoch, 'agent revive', last_agent_loss)
      return
    raise ValueError(f'Unsupported reset_mode: {reset_mode}')

  cp = elements.Checkpoint(logdir / 'ckpt')
  cp.step = step
  cp.agent = agent
  cp.replay = replay
  if args.from_checkpoint:
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=bind(agent.load, regex=args.from_checkpoint_regex)))
  cp.load_or_save()

  print('Start training loop')
  policy = lambda *args: agent.policy(*args, mode='train')

  # TODO: first env.
  should_switch(step)
  switch_count = 0
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

    if should_switch(step):
      switch_count += 1
      fns = [bind(make_env, i, switch_count=switch_count) for i in range(args.envs)]
      driver.switch_envs(
        fns, parallel=not args.debug
      )
      driver.reset(agent.init_policy)
      replay.clear()
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
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()

    if should_save(step):
      cp.save()

  logger.close()
