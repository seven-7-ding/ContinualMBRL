"""Recovery-policy checks without CUDA, networking, or signaling real jobs."""
import json
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest import mock

from auto_scripts import experiment_watchdog as w


class WatchdogTests(unittest.TestCase):

  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name)
    self.control = self.root/'watchdog'
    self.control.mkdir()
    self.state = self.root/'manifest.json'
    self.job = dict(setting='default', seed=1000, pid=999999,
        create_time=1, gpu=0, logdir=str(self.root/'run'),
        wandb_id='existing-id', attempts=2)
    self.state.write_text(json.dumps(dict(jobs=[self.job])))
    self.patches = []
    self.patch(w, 'CONTROL', self.control)
    self.patch(w.launcher, 'STATE', self.state)

  def patch(self, obj, name, value=mock.DEFAULT, **kwargs):
    patcher = mock.patch.object(obj, name, value, **kwargs)
    self.addCleanup(patcher.stop)
    return patcher.start()

  def make_checkpoint(self, done=True):
    root = Path(self.job['logdir'])/'ckpt'
    cp = root/'snapshot'
    cp.mkdir(parents=True)
    (root/'latest').write_text('snapshot')
    (cp/'step.pkl').write_bytes(pickle.dumps(1234))
    for name in ('agent.pkl','replay.pkl'):
      (cp/name).write_bytes(pickle.dumps({}))
    if done:
      (cp/'done').touch()

  def policy_mocks(self, *, step=1234, finite=True, ram=200):
    self.patch(w, 'owned', return_value=None)
    self.patch(w, 'matching_processes', return_value=[])
    self.patch(w, 'checkpoint', return_value=dict(
        complete=True, step=step, name='snapshot', age_s=60, losses_finite=finite))
    self.patch(w, 'latest_metrics', return_value=dict(step=1200, losses_finite=True))
    self.patch(w, 'gpu_stats', return_value=[dict(index=0,used_mib=8000,
        total_mib=24576,free_mib=16576,temp_c=70,util=60)])
    self.patch(w.launcher, 'available_ram_gib', return_value=ram)
    self.patch(w.launcher, 'save')
    self.patch(w, 'event')
    self.patch(w, 'cleanup_orphans', return_value=True)
    self.patch(w.shutil if hasattr(w,'shutil') else __import__('shutil'), 'copytree')
    return self.patch(w.launcher, 'launch', return_value=True)

  def test_checkpoint_requires_done_and_all_payloads(self):
    self.make_checkpoint(done=False)
    self.assertFalse(w.checkpoint(self.job['logdir'])['complete'])
    cp = Path(self.job['logdir'])/'ckpt/snapshot'
    (cp/'done').touch()
    self.assertTrue(w.checkpoint(self.job['logdir'])['complete'])
    (cp/'agent.pkl').unlink()
    self.assertFalse(w.checkpoint(self.job['logdir'])['complete'])

  def test_read_only_poll_cannot_restart(self):
    launch = self.policy_mocks()
    w.poll(recover=False)
    launch.assert_not_called()

  def test_completed_run_is_not_restarted(self):
    launch = self.policy_mocks(step=w.TARGET)
    result = w.poll()
    launch.assert_not_called()
    self.assertTrue(result['runs'][0]['completed'])

  def test_explicit_user_pause_is_never_recovered(self):
    self.job.update(user_paused=True, pause_reason='free_resources')
    self.state.write_text(json.dumps(dict(jobs=[self.job])))
    launch = self.policy_mocks()
    result = w.poll()
    launch.assert_not_called()
    self.assertTrue(result['runs'][0]['user_paused'])
    self.assertEqual(result['runs'][0]['pause_reason'], 'free_resources')

  def test_performance_profile_keeps_training_semantics_and_caps_cache(self):
    baseline = w.launcher.command(self.job)
    for profile in ('bounded_replay_v1','bounded_replay_v2'):
      with self.subTest(profile=profile):
        self.job['performance_profile'] = profile
        command = w.launcher.command(self.job)
        for flag in ('--run.two_player', '--run.train_ratio', '--run.update_batch',
                     '--run.prefetch_batches', '--replay.size', '--run.save_every'):
          self.assertEqual(command[command.index(flag)+1], baseline[baseline.index(flag)+1])
        for flag, expected in (('--replay.cache_chunks','0'),
                               ('--replay.cache_bytes',str(2 * 1024**3)),
                               ('--run.usage.malloc','False')):
          self.assertEqual(command[command.index(flag)+1], expected)

  def test_launcher_refuses_explicitly_paused_run(self):
    self.job['user_paused'] = True
    self.assertFalse(w.launcher.launch(self.job, resume='must'))

  def test_restart_requires_ram_reserve(self):
    launch = self.policy_mocks(ram=115)
    w.poll()
    launch.assert_not_called()

  def test_scheduled_quiesce_prevents_automatic_recovery(self):
    launch=self.policy_mocks()
    w.write_json(self.control/'scheduled_stop_plan.json',dict(enabled=True,
        quiesce_at=1,deadline=10**20,plan_id='deadline',run_ids=[self.job['wandb_id']]))
    w.write_json(self.control/'scheduled_stop_status.json',dict(phase='complete'))
    result=w.poll()
    launch.assert_not_called()
    self.assertTrue(result['runs'][0]['user_paused'])

  def test_restart_requires_finite_checkpoint(self):
    launch = self.policy_mocks(finite=False)
    w.poll()
    launch.assert_not_called()

  def test_recovery_keeps_id_and_requires_existing_wandb(self):
    launch = self.policy_mocks()
    w.poll()
    launch.assert_called_once()
    args, kwargs = launch.call_args
    self.assertEqual(args[0]['wandb_id'], 'existing-id')
    self.assertEqual(args[0]['logdir'], self.job['logdir'])
    self.assertEqual(kwargs['resume'], 'must')

  def test_retired_setting_is_rejected(self):
    self.state.write_text(json.dumps(dict(jobs=[{**self.job, 'setting':'best_response_128'}])))
    with self.assertRaises(RuntimeError):
      w.poll()

  def test_unsafe_gpu_cannot_be_selected(self):
    gpus = [dict(index=0,free_mib=6500,temp_c=60),
            dict(index=1,free_mib=20000,temp_c=90)]
    self.assertIsNone(w.safe_gpu(gpus, 0))

  def test_actual_free_gpu_memory_excludes_driver_reservations(self):
    self.patch(w.subprocess, 'check_output', return_value='0, 19000, 24576, 5100, 80, 70\n')
    self.assertEqual(w.gpu_stats()[0]['free_mib'], 5100)

  def test_bridge_pid_reuse_wakes_foreground(self):
    w.write_json(self.control/'wandb_bridge.json',dict(pid=999999,create_time=1))
    proc=mock.Mock()
    proc.create_time.return_value=99
    self.patch(w.psutil,'Process',return_value=proc)
    event=self.patch(w,'event')
    w.check_bridge_health()
    self.assertEqual(event.call_args.args[0],'wandb_bridge_down')

  def test_bridge_missing_metadata_is_optional(self):
    event=self.patch(w,'event')
    w.check_bridge_health()
    event.assert_not_called()

  def test_concurrent_alerts_remain_durable_and_keep_cooldowns(self):
    keys = [f'run_{i}' for i in range(32)]
    def alert(key):
      w.event('health', dict(id=key), key=key, cooldown=600)
    with ThreadPoolExecutor(max_workers=8) as pool:
      list(pool.map(alert, keys))
      list(pool.map(alert, keys))
    rows = [json.loads(line) for line in
            (self.control/'events.jsonl').read_text().splitlines()]
    self.assertEqual(len(rows), len(keys))
    self.assertEqual({r['detail']['id'] for r in rows}, set(keys))
    self.assertEqual(set(w.read_json(self.control/'notices.json')), set(keys))
    self.assertFalse(list(self.control.glob('*.tmp')))

  def test_pid_reuse_never_signaled(self):
    self.patch(w.launcher, 'alive', return_value=False)
    signal_proc = self.patch(w.psutil, 'Process')
    self.assertIsNone(w.owned(self.job))
    signal_proc.assert_not_called()


if __name__ == '__main__':
  unittest.main()
