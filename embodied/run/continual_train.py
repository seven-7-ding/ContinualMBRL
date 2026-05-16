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
      train_agg.add(mets, prefix='train')

  def add_train_metrics(mets, outs=None, steps=batch_steps):
    train_fps.step(steps)
    if outs and 'replay' in outs:
      replay.update(outs['replay'])
    train_agg.add(mets, prefix='train')

  def write_logs():
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

  def random_policy(carry, obs, mode='train'):
    del mode
    rng = random_policy.rng
    act = {}
    batch = len(obs['is_first'])
    for key, space in random_policy.act_space.items():
      if key == 'reset':
        continue
      low, high = space.low, space.high
      if np.issubdtype(space.dtype, np.floating):
        low = np.maximum(np.ones(space.shape) * np.finfo(space.dtype).min, low)
        high = np.minimum(np.ones(space.shape) * np.finfo(space.dtype).max, high)
      if space.discrete:
        if space.dtype == bool:
          values = rng.integers(0, 2, (batch, *space.shape))
        else:
          values = rng.integers(low, high, (batch, *space.shape))
        act[key] = values.astype(space.dtype)
      else:
        act[key] = rng.uniform(low, high, (batch, *space.shape)).astype(space.dtype)
    return carry, act, {}
  random_policy.rng = np.random.default_rng(getattr(args, 'seed', 0))
  random_policy.act_space = None

  def stream_lengths():
    return [len(x) for x in getattr(replay, 'streams', {}).values()]

  def getseq_len(chunkid, index, length):
    chunk = replay.chunks[chunkid]
    available = chunk.length - index
    if available >= length:
      seq = chunk.slice(index, length)
      return {k: [v] for k, v in seq.items()}
    parts = [chunk.slice(index, available)]
    remaining = length - available
    while remaining > 0:
      chunk = replay.chunks[chunk.succ]
      used = min(remaining, chunk.length)
      parts.append(chunk.slice(0, used))
      remaining -= used
    return {k: [p[k] for p in parts] for k in parts[0].keys()}

  def make_prim_stream(batch_size, raw_length):
    rng = np.random.default_rng(getattr(args, 'seed', 0) + 17)

    def sample():
      with replay.rwlock.reading:
        candidates = []
        for stream in replay.streams.values():
          stream = list(stream)
          limit = len(stream) - raw_length + 1
          candidates += stream[:max(0, limit)]
        if not candidates:
          raise RuntimeError(
              'Prim replay has no sequence candidate with raw length '
              f'{raw_length}. Current stream lengths: {stream_lengths()}')
        seqs = []
        for _ in range(batch_size):
          chunkid, index = candidates[rng.integers(0, len(candidates))]
          seqs.append(getseq_len(chunkid, index, raw_length))
        data = replay._assemble_batch(seqs, 0, raw_length)
        data = replay._annotate_batch(data, [False] * batch_size, True)
        data['consec'] = np.zeros(data['is_first'].shape, np.int32)
        return data

    return iter(agent.stream(embodied.streams.Stateless(sample)))

  def make_prim_train_state():
    lengths = stream_lengths()
    maxlen = max(lengths) if lengths else 0
    target_raw = args.batch_length + args.replay_context
    if len(replay) > 0:
      return stream_train, carry_train, batch_steps, args.batch_length
    if maxlen <= args.replay_context:
      raise RuntimeError(
          'Prim collection did not produce enough data for even one '
          f'contexted sequence. Max stream length={maxlen}, '
          f'replay_context={args.replay_context}.')
    max_batch_length = min(maxlen, target_raw) - args.replay_context
    divisors = [x for x in range(1, max_batch_length + 1) if batch_steps % x == 0]
    prim_batch_length = max(divisors)
    prim_batch_size = batch_steps // prim_batch_length
    prim_raw = prim_batch_length + args.replay_context
    prim_steps = prim_batch_size * prim_batch_length
    print(
        'Prim uses short replay batches because no full-length replay item is '
        f'available: stream_lengths={lengths}, max_raw_length={maxlen}, raw_length={prim_raw}, '
        f'batch_length={prim_batch_length}, batch_size={prim_batch_size}, '
        f'sampled_transitions={prim_steps} (target={batch_steps}).')
    prim_stream = make_prim_stream(prim_batch_size, prim_raw)
    prim_carry = [agent.init_train(prim_batch_size)]
    return prim_stream, prim_carry, prim_steps, prim_batch_length

  def run_prim_phase():
    prim_mode = getattr(args, 'prim_mode', 'none')
    prim_mode = str(prim_mode or 'none')
    if prim_mode in ('none', 'false', 'False', '0'):
      return
    modes = (
        'only_wm', 'only_agent_multi_rollout',
        'only_agent_one_rollou', 'only_agent_one_rollout', 'both')
    if prim_mode not in modes:
      raise NotImplementedError(f'Unknown prim_mode: {prim_mode}')

    collect_steps = int(getattr(args, 'prim_collect_steps', 128))
    train_steps = int(getattr(args, 'prim_train_steps', 100000))
    print(
        f'Start prim phase: mode={prim_mode}, '
        f'collect_steps={collect_steps}, train_steps={train_steps}')
    if collect_steps > 0:
      use_random = getattr(args, 'prim_random_collect', False)
      if use_random and args.replay_context:
        print('Prim random collection disabled because replay_context is on.')
        use_random = False
      collect_policy = random_policy if use_random else policy
      prim_fns = [bind(make_env, 0, switch_count=switch_count)]
      prim_driver = embodied.Driver(prim_fns, parallel=not args.debug)
      prim_driver.on_step(lambda tran, _: step.increment())
      prim_driver.on_step(lambda tran, _: policy_fps.step())
      prim_driver.on_step(replay.add)
      prim_driver.on_step(logfn)
      prim_driver.reset(agent.init_policy)
      random_policy.act_space = prim_driver.act_space
      try:
        prim_driver(collect_policy, steps=collect_steps)
      finally:
        prim_driver.close()
    prim_stream, prim_carry, prim_steps, prim_batch_length = make_prim_train_state()

    cached_rollout = None
    if prim_mode in ('only_agent_multi_rollout', 'only_agent_one_rollou',
                     'only_agent_one_rollout'):
      batch = next(prim_stream)
      prim_carry[0], outs, mets = agent.train_wm(prim_carry[0], batch)
      add_train_metrics(mets, outs, prim_steps)
      step.increment()
      if should_log(step):
        write_logs()
      if prim_mode in ('only_agent_one_rollou', 'only_agent_one_rollout'):
        cached_rollout = agent.prim_rollout(prim_carry[0], next(prim_stream))

    for prim_step in range(train_steps):
      if prim_mode == 'only_wm':
        prim_carry[0], outs, mets = agent.train_wm(
            prim_carry[0], next(prim_stream))
      elif prim_mode == 'only_agent_multi_rollout':
        prim_carry[0], outs, mets = agent.train_agent(
            prim_carry[0], next(prim_stream))
      elif prim_mode in ('only_agent_one_rollou', 'only_agent_one_rollout'):
        prim_carry[0], outs, mets = agent.train_agent_rollout(
            prim_carry[0], cached_rollout)
      elif prim_mode == 'both':
        prim_carry[0], outs, mets = agent.train_prim_both(
            prim_carry[0], next(prim_stream))
      add_train_metrics(mets, outs, prim_steps)
      step.increment()
      if should_log(step):
        write_logs()
      if (prim_step + 1) % 10000 == 0:
        print(f'Prim phase progress: {prim_step + 1}/{train_steps}')
    if prim_carry is not carry_train:
      carry_train[0] = agent.init_train(args.batch_size)
    print(f'Finished prim phase at step {step.value}.')

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
  run_prim_phase()

  fns = [bind(make_env, i, switch_count=switch_count) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=not args.debug)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(replay.add)
  driver.on_step(logfn)
  driver.reset(agent.init_policy)
  driver.on_step(trainfn)
  
  while step < args.steps:
    if should_switch(step):
      switch_count += 1
      fns = [bind(make_env, i, switch_count=switch_count) for i in range(args.envs)]
      driver.switch_envs(
        fns, parallel=not args.debug
      )
      driver.reset(agent.init_policy)
      replay.clear()
      reset_mode = getattr(args, 'reset_mode', 'none')
      if getattr(args, 'reset_on_switch', False):
        reset_mode = 'all'
      if reset_mode not in ('none', 'false', False, None):
        agent.reset_params(reset_mode)
        print(f"Agent reset at step {step.value} (reset_mode={reset_mode}).")
      else:
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
      write_logs()

    if should_save(step):
      cp.save()

  logger.close()
