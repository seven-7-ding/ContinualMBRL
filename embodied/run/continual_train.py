import collections
from functools import partial as bind

import elements
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
  episodes = collections.defaultdict(elements.Agg)
  policy_fps = elements.FPS()
  train_fps = elements.FPS()

  batch_steps = args.batch_size * args.batch_length
  should_train = elements.when.Ratio(args.train_ratio / batch_steps)
  print(f'Train ratio: {should_train._ratio}, Batch steps: {batch_steps}, Train calls per step: {args.train_ratio / batch_steps}')
  # should_log = embodied.LocalClock(args.log_every)
  should_log = elements.when.Every(args.log_every, initial=False)
  should_report = embodied.LocalClock(args.report_every)
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
      logger.add({
          'score': result.pop('score'),
          'length': result.pop('length'),
      }, prefix='episode')
      rew = result.pop('rewards')
      if len(rew) > 1:
        result['delta_reward>0.01_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      result.update(stats(rew, "real_reward"))
      epstats.add(result)

  stream_train = iter(agent.stream(make_stream(replay, 'train')))
  stream_report = iter(agent.stream(make_stream(replay, 'report')))

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

    if should_report(step) and len(replay):
      agg = elements.Agg()
      for _ in range(args.consec_report * args.report_batches):
        carry_report, mets = agent.report(carry_report, next(stream_report))
        agg.add(mets)
      logger.add(agg.result(), prefix='report')

    if should_log(step):
      train_metrics = train_agg.result()
      train_metrics_new = {}
      loss_metrics = {}
      opt_metrics = {}
      for k, v in train_metrics.items():
        if "train/loss/" in k and "opt" not in k:
          loss_metrics[k.replace('train/loss/', '')] = v
        elif "train/opt/" in k:
          opt_metrics[k.replace('train/opt/', '')] = v
        else:
          train_metrics_new[k.replace('train/', '')] = v
      logger.add(train_metrics_new, prefix='train')
      logger.add(loss_metrics, prefix='loss')
      logger.add(opt_metrics, prefix='opt')
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
