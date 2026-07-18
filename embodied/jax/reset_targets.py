"""Single entry point for reset/WSC component targets.

To add a new target:
  1. Inspect SYSTEM_COMPONENTS for available component names.
  2. Add a RESET_TARGETS entry mapping the new target name to components.
  3. Use the target name via --run.reset_target in scripts.
"""

SYSTEM_COMPONENTS = {
    'encoder': ('enc',),
    'dynamics': ('dyn',),
    'decoder': ('dec',),
    'reward_head': ('rew',),
    'continuation_head': ('con',),
    'policy_head': ('pol',),
    'value_head': ('val',),
    'slow_value_head': ('slowval',),
    'return_normalizer': ('retnorm',),
    'value_normalizer': ('valnorm',),
    'advantage_normalizer': ('advnorm',),
}

RESET_TARGETS = {
    'all': tuple(SYSTEM_COMPONENTS),
    'agent_head': (
        'policy_head', 'value_head', 'slow_value_head',
        'return_normalizer', 'value_normalizer', 'advantage_normalizer'),
    'wm_head': ('decoder', 'reward_head', 'continuation_head'),
    'dyn': ('dynamics',),
    'all_head': (
        'decoder', 'reward_head', 'continuation_head',
        'policy_head', 'value_head', 'slow_value_head',
        'return_normalizer', 'value_normalizer', 'advantage_normalizer'),
}

ALIASES = {
    True: 'all',
    'true': 'all',
    'all': 'all',
    'agent_head': 'agent_head',
    'agent_heads': 'agent_head',
    'wm_head': 'wm_head',
    'wm_heads': 'wm_head',
    'dyn': 'dyn',
    'dynamics': 'dyn',
    'rssm': 'dyn',
    'all_head': 'all_head',
    'all_heads': 'all_head',
}


def canonical_target(target):
  target = ALIASES.get(target, target)
  if target not in RESET_TARGETS:
    known = ', '.join(sorted(RESET_TARGETS))
    raise ValueError(f'Unknown reset target {target!r}. Known targets: {known}')
  return target


def target_components(target):
  target = canonical_target(target)
  components = []
  for name in RESET_TARGETS[target]:
    if name not in SYSTEM_COMPONENTS:
      known = ', '.join(sorted(SYSTEM_COMPONENTS))
      raise ValueError(
          f'Unknown component {name!r} in target {target!r}. '
          f'Known components: {known}')
    components.extend(SYSTEM_COMPONENTS[name])
  return tuple(components)


def matches_target(key, target):
  modules = target_components(target)
  return any(
      key == module or key.startswith(f'{module}/') or f'/{module}/' in key
      for module in modules)
