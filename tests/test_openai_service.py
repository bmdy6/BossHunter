"""Official OpenAI service must not inherit compatible-provider extensions."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import yaml

from bosshunter.ai import credentials
from bosshunter.config import AI_SERVICE_PRESETS, load_config


class OpenAIServiceTests(unittest.TestCase):
    def test_load_and_resolve_openai_preset(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            path = Path(tmp) / 'config.yaml'
            path.write_text(yaml.safe_dump({'ai': {'service': 'openai', 'model': 'gpt-6.1-sol', 'service_tier': 'priority'}}))
            config = load_config(path)
            self.assertEqual(config['ai']['provider'], 'openai_compatible')
            self.assertEqual(config['ai']['service_tier'], 'priority')
            self.assertEqual(credentials.get_ai_base_url(config), 'https://api.openai.com/v1')
            self.assertEqual(credentials.get_ai_base_url_source(config), '服务商预设')
            self.assertEqual(AI_SERVICE_PRESETS['openai']['key_env'], 'OPENAI_API_KEY')
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'openai-test', 'ANTHROPIC_API_KEY': 'unrelated'}):
                self.assertEqual(credentials.get_ai_api_key(config), 'openai-test')
                self.assertEqual(credentials.get_ai_key_source(config), 'OPENAI_API_KEY')
            self.assertIsNone(credentials.get_ai_api_key(config))
            self.assertIsNone(credentials.get_ai_key_source(config))

    def test_config_and_env_resolution_precedence(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'env-key', 'OPENAI_BASE_URL': 'https://env.test/v1'}, clear=True):
            config = {'ai': {'service': 'openai'}}
            self.assertEqual(credentials.get_ai_base_url(config), 'https://env.test/v1')
            self.assertEqual(credentials.get_ai_base_url_source(config), 'OPENAI_BASE_URL')
            config['ai'].update(api_key='local-key', base_url='https://local.test/v1')
            self.assertEqual(credentials.get_ai_api_key(config), 'local-key')
            self.assertEqual(credentials.get_ai_key_source(config), '本地配置')
            self.assertEqual(credentials.get_ai_base_url(config), 'https://local.test/v1')

    def test_payload_for_every_thinking_mode_and_optional_tier(self):
        for mode in ('auto', 'enabled', 'disabled', 'off'):
            for tier in (None, '', 'priority'):
                with self.subTest(mode=mode, tier=tier):
                    config = {'ai': {'service': 'openai', 'model': 'gpt-6.1-sol', 'api_key': 'test-key', 'thinking': mode, 'service_tier': tier}}
                    response = httpx.Response(200, json={'choices': [{'message': {'content': 'ok'}, 'finish_reason': 'stop'}]}, request=httpx.Request('POST', 'https://api.openai.com/v1/chat/completions'))
                    with patch.dict(os.environ, {}, clear=True), patch('httpx.post', return_value=response) as post:
                        self.assertEqual(credentials.call_openai_compatible_text('prompt', config, 8192), 'ok')
                    post.assert_called_once()
                    self.assertEqual(post.call_args.args[0], 'https://api.openai.com/v1/chat/completions')
                    payload = post.call_args.kwargs['json']
                    self.assertEqual(payload['max_completion_tokens'], 8192)
                    self.assertNotIn('max_tokens', payload)
                    self.assertNotIn('thinking', payload)
                    self.assertNotIn('temperature', payload)
                    if tier:
                        self.assertEqual(payload['service_tier'], tier)
                    else:
                        self.assertNotIn('service_tier', payload)

    def test_other_services_keep_legacy_payload(self):
        for service in ('deepseek', 'doubao', 'custom'):
            config = {'ai': {'service': service, 'thinking': 'auto', 'model': 'test', 'api_key': 'test-key', 'base_url': 'https://test.invalid/v1', 'service_tier': 'priority'}}
            response = httpx.Response(200, json={'choices': [{'message': {'content': 'ok'}}]}, request=httpx.Request('POST', 'https://test.invalid'))
            with self.subTest(service=service), patch('httpx.post', return_value=response) as post:
                self.assertEqual(credentials.call_openai_compatible_text('prompt', config, 8192), 'ok')
                payload = post.call_args.kwargs['json']
                self.assertEqual(payload['max_tokens'], 8192)
                self.assertEqual(payload['temperature'], 0.2)
                self.assertEqual(payload['thinking'], {'type': 'disabled'})
                self.assertNotIn('max_completion_tokens', payload)
                self.assertNotIn('service_tier', payload)

    def test_quota_or_rate_limit_stops_without_compatibility_retry(self):
        config = {'ai': {'service': 'openai', 'model': 'gpt-6.1-sol', 'api_key': 'test-key'}}
        for status, code, kind in ((429, 'rate_limit_exceeded', 'rate_limit'), (429, 'insufficient_quota', 'token_quota'), (402, 'billing', 'token_quota')):
            response = httpx.Response(status, json={'error': {'code': code}}, request=httpx.Request('POST', 'https://api.openai.com/v1/chat/completions'))
            with self.subTest(status=status, code=code), patch('httpx.post', return_value=response) as post:
                with self.assertRaises(credentials.AIRequestError) as caught:
                    credentials.call_openai_compatible_text('prompt', config, 8192)
                self.assertEqual(caught.exception.kind, kind)
                self.assertEqual(post.call_count, 1)
