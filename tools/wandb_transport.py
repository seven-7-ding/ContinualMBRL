"""Route this sweep's verified W&B API host directly, without global changes."""
import urllib.parse
import urllib.request

API_HOST = 'api.wandb.ai'
PROFILE = 'direct_api_v1'


def sdk_environment(environment):
  env = dict(environment)
  host = urllib.parse.urlparse(env.get('WANDB_BASE_URL',
                                      'https://api.wandb.ai')).hostname
  if host != API_HOST:
    return env
  excluded = []
  for key in ('NO_PROXY', 'no_proxy'):
    for value in env.get(key, '').split(','):
      value = value.strip()
      if value and value not in excluded:
        excluded.append(value)
  if API_HOST not in excluded:
    excluded.append(API_HOST)
  env['NO_PROXY'] = env['no_proxy'] = ','.join(excluded)
  env['CONTINUALMBRL_WANDB_TRANSPORT'] = PROFILE
  return env


def urlopen(request, timeout):
  url = request.full_url if hasattr(request, 'full_url') else request
  if urllib.parse.urlparse(url).hostname == API_HOST:
    # A fresh direct opener also avoids reusing a blocked proxy tunnel.
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({})).open(request, timeout=timeout)
  return urllib.request.urlopen(request, timeout=timeout)
