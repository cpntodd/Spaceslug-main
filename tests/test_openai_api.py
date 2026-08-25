import json
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from spaceslug.openai_api import (
    ModelResponder,
    OpenAICompatibleServer,
    ResponderResult,
)


def request(server, method, path, payload=None):
    url = server.base_url + path
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=5) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body)
    except HTTPError as exc:
        body = exc.read().decode("utf-8")
        return exc.code, json.loads(body)


class _FixedResponder(ModelResponder):
    backend_id = "test-backend"
    model_id = "test-model"

    def respond(self, messages):
        return ResponderResult(
            content="fixed reply",
            finish_reason="stop",
            prompt_tokens=3,
            completion_tokens=2,
        )


class OpenAICompatibleServerTest(unittest.TestCase):
    def setUp(self):
        self.server = OpenAICompatibleServer(port=0)
        self.server.start()
        self.addCleanup(self.server.stop)

    def test_health_reports_ok_and_truthful_identity(self):
        status, body = request(self.server, "GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["backend"], "cpu-tiny")
        self.assertEqual(body["model"], "spaceslug-tiny-attention-v1")

    def test_models_lists_tiny_model(self):
        status, body = request(self.server, "GET", "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(body["object"], "list")
        self.assertEqual(len(body["data"]), 1)
        self.assertEqual(body["data"][0]["id"], "spaceslug-tiny-attention-v1")
        self.assertEqual(body["data"][0]["object"], "model")
        self.assertEqual(body["data"][0]["owned_by"], "cpu-tiny")

    def test_chat_completion_uses_openai_shape(self):
        payload = {
            "model": "spaceslug-tiny-attention-v1",
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        }
        status, body = request(self.server, "POST", "/v1/chat/completions", payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["model"], "spaceslug-tiny-attention-v1")
        self.assertEqual(body["system_fingerprint"], "cpu-tiny")
        choice = body["choices"][0]
        self.assertEqual(choice["message"]["role"], "assistant")
        self.assertIn("hello", choice["message"]["content"])
        self.assertIn("cpu-tiny", choice["message"]["content"])
        self.assertIn("spaceslug-tiny-attention-v1", choice["message"]["content"])
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertIsNone(choice["logprobs"])
        self.assertEqual(
            body["usage"]["total_tokens"],
            body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"],
        )

    def test_stream_true_is_rejected_explicitly(self):
        payload = {"stream": True, "messages": [{"role": "user", "content": "hi"}]}
        status, body = request(self.server, "POST", "/v1/chat/completions", payload)
        self.assertEqual(status, 400)
        self.assertIn("streaming", body["error"]["message"].lower())
        self.assertEqual(body["error"]["param"], "stream")

    def test_missing_messages_is_rejected(self):
        status, body = request(
            self.server, "POST", "/v1/chat/completions", {"messages": []}
        )
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["param"], "messages")

    def test_injectable_responder_is_used(self):
        with OpenAICompatibleServer(port=0, responder=_FixedResponder()) as server:
            status, body = request(server, "GET", "/health")
            self.assertEqual(status, 200)
            self.assertEqual(body["backend"], "test-backend")
            self.assertEqual(body["model"], "test-model")
            status, body = request(
                server,
                "POST",
                "/v1/chat/completions",
                {"messages": [{"role": "user", "content": "x"}]},
            )
            self.assertEqual(status, 200)
            self.assertEqual(body["model"], "test-model")
            self.assertEqual(body["choices"][0]["message"]["content"], "fixed reply")

    def test_binds_loopback_only(self):
        host, _port = self.server.address
        self.assertEqual(host, "127.0.0.1")
        with self.assertRaises(ValueError):
            OpenAICompatibleServer(host="0.0.0.0")

    def test_unknown_path_returns_not_found(self):
        status, body = request(self.server, "GET", "/nope")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"]["code"], "not_found")

    def test_start_stop_lifecycle(self):
        self.assertTrue(self.server.is_running)
        self.server.stop()
        self.assertFalse(self.server.is_running)
        self.server.stop()  # idempotent


if __name__ == "__main__":
    unittest.main()
