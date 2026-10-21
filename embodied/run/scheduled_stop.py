"""Optional cooperative checkpoint-and-exit control for explicitly scheduled runs."""
import json
import os
from pathlib import Path
import tempfile
import time

PROTOCOL = 'checkpoint_stop_v1'


def write_json(path, value):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = None
  try:
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,
        prefix=path.name+'.',delete=False) as stream:
      temporary = Path(stream.name)
      json.dump(value,stream,allow_nan=False)
      stream.write('\n')
      stream.flush()
      os.fsync(stream.fileno())
    temporary.replace(path)
  finally:
    if temporary is not None:
      temporary.unlink(missing_ok=True)


class ScheduledStop:

  def __init__(self, logdir):
    self.logdir = Path(str(logdir))
    self.path = os.environ.get('CONTINUALMBRL_STOP_PLAN')
    self.runid = os.environ.get('WANDB_RUN_ID')
    self.plan = {}
    self.requested = False
    self.next_read = 0
    if self.path:
      self._read()
      write_json(self.logdir/'scheduled_stop_ready.json',dict(
          protocol=PROTOCOL,pid=os.getpid(),run_id=self.runid,
          plan_path=self.path,registered_at=time.time()))

  def _read(self):
    try:
      plan = json.loads(Path(self.path).read_text())
      if (isinstance(plan.get('enabled'),bool) and
          isinstance(plan.get('run_ids'),list) and
          isinstance(plan.get('quiesce_at'),(int,float)) and
          isinstance(plan.get('deadline'),(int,float)) and
          plan['quiesce_at'] <= plan['deadline']):
        self.plan = plan
    except (OSError,ValueError,TypeError):
      pass  # Keep the last valid intent; the independent deadline fence remains.

  def due(self):
    if not self.path:
      return False
    now = time.time()
    if now >= self.next_read:
      self._read()
      self.next_read = now+2
    self.requested = bool(self.plan.get('enabled') and
        self.runid in self.plan.get('run_ids',[]) and
        now >= self.plan.get('quiesce_at',float('inf')))
    return self.requested

  def saved(self, checkpoint, step):
    if not self.requested:
      return
    latest = Path(str(checkpoint.latest()))
    assert (latest/'done').exists()
    write_json(self.logdir/'scheduled_stop_saved.json',dict(
        protocol=PROTOCOL,pid=os.getpid(),run_id=self.runid,
        plan_id=self.plan['plan_id'],step=int(step),
        checkpoint=latest.name,saved_at=time.time(),deadline=self.plan['deadline']))
    print(f'Scheduled stop saved full checkpoint at step {int(step)}.',flush=True)
