"""Update schedules relative to the original train-ratio clock."""

import re


def update_counts(setting):
  if setting == 'default':
    return 1, 1
  match = re.fullmatch(r'(wm_as_leader|agent_as_leader|best_response)_([1-9][0-9]*)', setting)
  if not match:
    raise ValueError(f'Invalid two-player setting: {setting!r}')
  mode, k = match[1], int(match[2])
  return {
      'wm_as_leader': (1, k),
      'agent_as_leader': (k, 1),
      'best_response': (k, k),
  }[mode]


def update_modes(setting):
  wm, agent = update_counts(setting)
  # Preserve the original joint update, including its optimizer, for default.
  return ('all',) if setting == 'default' else ('wm',) * wm + ('agent',) * agent
