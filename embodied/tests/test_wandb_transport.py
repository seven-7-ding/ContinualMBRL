from unittest import mock
import urllib.request

from tools import wandb_transport as t


def test_sdk_bypasses_only_api_and_preserves_other_proxy_settings():
  original = dict(HTTP_PROXY='http://proxy:80', HTTPS_PROXY='http://proxy:80',
                  NO_PROXY='localhost,127.0.0.1', no_proxy='internal,localhost')
  result = t.sdk_environment(original)
  assert result['NO_PROXY'] == result['no_proxy'] == (
      'localhost,127.0.0.1,internal,api.wandb.ai')
  assert result['HTTPS_PROXY'] == original['HTTPS_PROXY']
  assert result['HTTP_PROXY'] == original['HTTP_PROXY']
  assert result['CONTINUALMBRL_WANDB_TRANSPORT'] == t.PROFILE
  assert 'CONTINUALMBRL_WANDB_TRANSPORT' not in original
  assert t.sdk_environment(result) == result


def test_custom_server_environment_keeps_existing_network_policy():
  env = dict(WANDB_BASE_URL='https://wandb.internal', NO_PROXY='localhost')
  assert t.sdk_environment(env) == env


def test_api_requests_use_direct_connection_and_preserve_request_and_timeout():
  request = urllib.request.Request('https://api.wandb.ai/graphql', data=b'{}')
  opener = mock.Mock()
  with mock.patch.object(t.urllib.request, 'build_opener', return_value=opener) as build, \
       mock.patch.object(t.urllib.request, 'urlopen') as ordinary:
    result = t.urlopen(request, 7)
  assert result is opener.open.return_value
  assert build.call_args.args[0].proxies == {}
  opener.open.assert_called_once_with(request, timeout=7)
  ordinary.assert_not_called()


def test_other_hosts_use_their_existing_proxy_route():
  request = urllib.request.Request('https://wandb.internal/graphql')
  with mock.patch.object(t.urllib.request, 'build_opener') as build, \
       mock.patch.object(t.urllib.request, 'urlopen') as ordinary:
    t.urlopen(request, 7)
  ordinary.assert_called_once_with(request, timeout=7)
  build.assert_not_called()
