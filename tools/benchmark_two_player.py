"""GPU throughput, memory and numerical checks using real walker observations."""
import os
from pathlib import Path
import sys
import time
import json
from copy import deepcopy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dreamerv3.main import _set_default_jax_env
_set_default_jax_env()
import elements
import jax
import numpy as np
from ruamel.yaml import YAML
from dreamerv3.agent import Agent

if '--cuda-graphs' in sys.argv:
  # Benchmark-only override before the first backend is initialized.
  from embodied.jax import internal
  original_setup = internal.setup
  def graph_setup(**kwargs):
    original_setup(**kwargs)
    flags = [flag for flag in os.environ.get('XLA_FLAGS', '').split()
             if not flag.startswith('--xla_gpu_graph_level=')]
    os.environ['XLA_FLAGS'] = ' '.join(flags + ['--xla_gpu_graph_level=2'])
  internal.setup = graph_setup

if '--no-barrier' in sys.argv:
  jax.lax.optimization_barrier = lambda value: value


def main():
  folder='benchmark_gpu_graphs' if '--cuda-graphs' in sys.argv else 'benchmark_gpu'
  if '--no-barrier' in sys.argv:
    folder += '_no_barrier'
  if '--sparse-metrics' in sys.argv:
    folder += '_sparse_metrics'
  for arg in sys.argv:
    if arg.startswith('--output-folder='):
      folder=arg.split('=',1)[1]
      assert Path(folder).name == folder
  directory=ROOT/'logdir/two_player_sweep_20261015'/folder
  configs=YAML(typ='safe').load((ROOT/'dreamerv3/configs.yaml').read_text())
  c=elements.Config(configs['defaults']).update(configs['continual_dmc_priori']).update(configs['size1m'])
  c=c.update({'logdir':str(directory),'agent.two_player':True,'jax.precompile':False,
      'jax.profiler':False,'agent.log_gradients':True,'agent.redo.frequency':100})
  cfg=elements.Config(**c.agent,logdir=c.logdir,seed=1000,jax=c.jax,run=c.run,
      batch_size=c.batch_size,batch_length=c.batch_length,replay_context=c.replay_context,
      report_length=c.report_length)
  obs={k:elements.Space(t,shape) for k,t,shape in (
      ('state',np.float32,(32,)),('reward',np.float32,()),
      ('is_first',bool,()),('is_last',bool,()),('is_terminal',bool,()))}
  act={'action':elements.Space(np.float32,(6,),-1,1)}
  agent=Agent(obs,act,cfg)
  agent.sparse_intermediate_metrics = '--sparse-metrics' in sys.argv
  from dm_control import suite
  env=suite.load('walker','run',task_kwargs={'random':1000})
  rng=np.random.default_rng(1000)
  frames=[]; rewards=[]
  ts=env.reset()
  for _ in range(1024):
    state=np.concatenate([np.asarray(v).ravel() for v in ts.observation.values()])
    frames.append(np.pad(state,(0,32-len(state))))
    rewards.append(ts.reward or 0.)
    ts=env.step(rng.uniform(-1,1,6)) if not ts.last() else env.reset()
  frames=np.asarray(frames,np.float32);rewards=np.asarray(rewards,np.float32)
  data=[]
  groups=(1,4,8,16,32) if '--large-groups' in sys.argv else (1,4,8)
  total=max(groups)
  for index in range(total):
    d=agent._zeros(agent.spaces,(c.batch_size,c.batch_length+c.replay_context))
    starts=rng.integers(0,len(frames)-65,c.batch_size)
    ids=starts[:,None]+np.arange(65)[None]
    d['state']=frames[ids]; d['reward']=rewards[ids]
    with jax._src.config.explicit_device_get_scope():
      d=jax.device_put(d,agent.train_sharded)
    d['seed']=agent._seeds(index,agent.train_mirrored)
    data.append(d)
  carry=agent.init_train(c.batch_size)
  initial=agent.save()
  if '--timing-only' in sys.argv:
    print('Warm fused-8 load comparison',flush=True)
    carry,_=agent.train_many(carry,data[:8],'agent')
    rates=[]
    for _ in range(3):
      begin=time.perf_counter()
      for _ in range(32): carry,_=agent.train_many(carry,data[:8],'agent')
      jax.block_until_ready(agent.params)
      rates.append(256/(time.perf_counter()-begin))
      print('fused_8',rates[-1],'updates/s',flush=True)
    import psutil
    result=dict(rates=rates,median=float(np.median(rates)),
        rss_gib=psutil.Process().memory_info().rss/2**30,
        jax_memory=agent.train_devices[0].memory_stats())
    (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return
  if '--wm-only' in sys.argv:
    print('Warm WM fused compilation',flush=True)
    carry,_=agent.train_many(carry,data,'wm')
    begin=time.perf_counter()
    for _ in range(256//total): carry,_=agent.train_many(carry,data,'wm')
    jax.block_until_ready(agent.params)
    throughput=256/(time.perf_counter()-begin)
    path=directory/'result.json'
    result=json.loads(path.read_text()) if path.exists() else {}
    result['wm_updates_per_second']=throughput
    path.write_text(json.dumps(result,indent=2)+'\n')
    print('WM',throughput,'updates/s',flush=True)
    return
  print('Warm sequential compilation',flush=True)
  for batch in data[:8]:
    carry,_,_=agent.train(carry,dict(batch),train_mode='agent')
  sequential_state=deepcopy(agent.save())
  agent.load(deepcopy(initial));carry=agent.init_train(c.batch_size)
  print('Warm fused compilation',flush=True)
  carry,_=agent.train_many(carry,data[:8],'agent')
  fused_state=agent.save()
  differences={key:float(np.max(np.abs(sequential_state['params'][key]-fused_state['params'][key])))
      for key in sequential_state['params']}
  model_difference=max(value for key,value in differences.items()
      if key.startswith(('pol/','val/','slowval/')))
  assert model_difference < 2e-5, model_difference
  timings={}
  for size in groups:
    agent.load(initial);carry=agent.init_train(c.batch_size)
    # Each shape gets one untimed compile/warm-up.
    for i in range(0,total,size): carry,_=agent.train_many(carry,data[i:i+size],'agent')
    begin=time.perf_counter()
    for repeat in range(256//total):
      for i in range(0,total,size): carry,_=agent.train_many(carry,data[i:i+size],'agent')
    jax.block_until_ready(agent.params)
    timings[f'fused_{size}']=256/(time.perf_counter()-begin)
    print(size,timings[f'fused_{size}'],'updates/s',flush=True)
  agent.load(initial);carry=agent.init_train(c.batch_size)
  begin=time.perf_counter()
  for repeat in range(256//total):
    for batch in data: carry,_,_=agent.train(carry,dict(batch),train_mode='agent')
  jax.block_until_ready(agent.params)
  timings['sequential']=256/(time.perf_counter()-begin)
  import psutil
  result={'updates_per_second':timings,
      'speedup_8':timings['fused_8']/timings['sequential'],
      'fused_parameter_max_absolute_difference':differences,
      'fused_model_max_absolute_difference':model_difference,
      'rss_gib':psutil.Process().memory_info().rss/2**30,
      'jax_memory':agent.train_devices[0].memory_stats()}
  (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
