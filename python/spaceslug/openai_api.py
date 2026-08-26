"""Dependency-free, loopback-only OpenAI-compatible HTTP API service.

This module exposes a small ``http.server``-based service that speaks the subset
of the OpenAI HTTP API needed by local tools: ``GET /health``, ``GET /v1/models``
and ``POST /v1/chat/completions``. It uses only the Python standard library plus
the in-process Spaceslug-Tiny CPU reference model, binds to loopback, supports
non-streaming completions only, and never depends on the Vulkan runtime or any
third-party package.

The completion backend is injected through :class:`ModelResponder`. The default
:class:`TinyCpuEchoResponder` runs the current CPU Tiny attention reference and
returns an echo-safe (non-generative) message that truthfully identifies the
backend and model.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Self
from urllib.parse import urlsplit

from .backend import BackendError, BackendSession
from .projected_attention_reference import ProjectedTinyAttentionModel
from .lora import TinyLoRAAdapter
from .tokenizer import ByteTokenizer, default_tokenizer

DEFAULT_HOST = "127.0.0.1"
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
MODEL_CREATED = 1_700_000_000

__all__ = [
    "DEFAULT_HOST",
    "LOOPBACK_HOSTS",
    "ModelResponder",
    "OpenAICompatibleServer",
    "ResponderResult",
    "TinyCpuEchoResponder",
    "TinyGpuResponder",
]


@dataclass(frozen=True)
class ResponderResult:
    """Structured assistant reply produced by a :class:`ModelResponder`."""

    content: str
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ModelResponder(ABC):
    """Injectable completion backend for the OpenAI-compatible service.

    Implementations must truthfully report the backend and model they represent
    through :attr:`backend_id` and :attr:`model_id`, and turn a list of OpenAI
    chat messages into a :class:`ResponderResult`.
    """

    @property
    @abstractmethod
    def backend_id(self) -> str:
        """Truthful identifier of the compute backend (e.g. ``cpu-tiny``)."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Truthful identifier of the served model."""

    @abstractmethod
    def respond(self, messages: list[dict]) -> ResponderResult:
        """Produce an assistant reply for a list of OpenAI chat messages."""


def _message_text(message: dict) -> str:
    """Extract the plain-text payload of one chat message safely."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return "" if content is None else str(content)


def last_user_text(messages: list[dict]) -> str:
    """Return the text of the last ``user`` message, else the last message."""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            text = _message_text(message)
            if text:
                return text
    for message in reversed(messages):
        if isinstance(message, dict):
            text = _message_text(message)
            if text:
                return text
    return ""


class TinyGpuResponder(ModelResponder):
    """Deterministic Tiny responder using the configured Vulkan forward path."""

    backend_id = "vulkan-radv-tiny"
    model_id = "spaceslug-tiny-attention-v1"

    def __init__(self, backend: BackendSession, model=None, tokenizer=None, *, adapter_state=None, max_echo_chars: int = 1000, max_new_tokens: int = 32):
        if max_new_tokens <= 0 or max_new_tokens > 128:
            raise ValueError("max_new_tokens must be between 1 and 128")
        self.backend = backend
        self.model = model if model is not None else ProjectedTinyAttentionModel(259, 64)
        self.adapter = TinyLoRAAdapter.from_state_dict(adapter_state) if adapter_state else None
        self.tokenizer = tokenizer if tokenizer is not None else default_tokenizer()
        self.max_echo_chars = max_echo_chars
        self.max_new_tokens = max_new_tokens

    def respond(self, messages: list[dict]) -> ResponderResult:
        prompt = last_user_text(messages)
        tokens = self.tokenizer.encode(prompt)
        result = self.backend.execute_projected_attention_gpu(tokens, self.model)
        if result.status != "ok" or not result.metrics.get("gpu_execution", False):
            raise BackendError(result.metrics.get("reason", "native GPU inference unavailable"))
        logits = result.output["logits"]
        generated = []
        context = list(tokens)
        for _ in range(self.max_new_tokens):
            if self.adapter is not None:
                current = self.backend.execute_tiny_lora_forward(context[-128:], self.model, self.adapter)
            else:
                current = self.backend.execute_projected_attention_gpu(context[-128:], self.model)
            if current.status != "ok" or not current.metrics.get("gpu_execution", False):
                raise BackendError(current.metrics.get("reason", "native GPU generation unavailable"))
            token = max(range(len(current.output["logits"])), key=current.output["logits"].__getitem__)
            if token == self.tokenizer.eos_token:
                break
            generated.append(token)
            context.append(token)
        completion = self.tokenizer.decode(generated)
        content = (f"Spaceslug-Tiny GPU responder (backend={self.backend_id}, model={self.model_id}).\\n"
                   f"Native Vulkan/RADV deterministic generation ({len(generated)} tokens).\\n\\n{completion}")
        return ResponderResult(content=content, prompt_tokens=len(tokens), completion_tokens=len(generated))


class TinyCpuEchoResponder(ModelResponder):
    """Default responder: current CPU Tiny attention inference, echo-safe.

    Runs :class:`~spaceslug.projected_attention_reference.ProjectedTinyAttentionModel`
    (the current CPU Tiny reference) to compute a single argmax next-token
    prediction and returns a deterministic, non-generative message that echoes
    the prompt and truthfully names the backend and model. It never fabricates
    natural-language completions.
    """

    backend_id = "cpu-tiny"
    model_id = "spaceslug-tiny-attention-v1"

    def __init__(
        self,
        model: ProjectedTinyAttentionModel | None = None,
        tokenizer: ByteTokenizer | None = None,
        *,
        max_echo_chars: int = 1000,
    ) -> None:
        self.model = model if model is not None else ProjectedTinyAttentionModel(259)
        self.tokenizer = tokenizer if tokenizer is not None else default_tokenizer()
        self.max_echo_chars = max_echo_chars

    def respond(self, messages: list[dict]) -> ResponderResult:
        prompt = last_user_text(messages)
        tokens = self.tokenizer.encode(prompt)
        logits = self.model.logits_for_tokens(tokens)
        next_token = max(range(len(logits)), key=logits.__getitem__)
        echo = (
            prompt
            if len(prompt) <= self.max_echo_chars
            else prompt[: self.max_echo_chars] + "…"
        )
        content = (
            f"Spaceslug-Tiny CPU responder (backend={self.backend_id}, model={self.model_id}).\n"
            "This is a deterministic, non-generative CPU reference: it does not "
            "produce natural-language completions and reports only a single "
            "argmax next-token prediction from the Tiny attention model.\n\n"
            f"next_token={next_token}\n"
            f"echo={echo}\n"
        )
        completion_tokens = len(
            self.tokenizer.encode(content, add_bos=False, add_eos=False)
        )
        return ResponderResult(
            content=content,
            finish_reason="stop",
            prompt_tokens=len(tokens),
            completion_tokens=completion_tokens,
        )


def _format_host(host: str) -> str:
    """Wrap an IPv6 literal in brackets for URL construction."""
    return f"[{host}]" if ":" in host else host


class _OpenAIRequestHandler(BaseHTTPRequestHandler):
    """Stdlib handler implementing the loopback OpenAI-compatible routes."""

    server_version = "SpaceslugOpenAI/0.1"

    def _responder(self) -> ModelResponder:
        return self.server.responder  # type: ignore[attr-defined]

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _error(
        message: str,
        type_: str = "invalid_request_error",
        *,
        param: str | None = None,
        code: str | None = None,
    ) -> dict:
        return {
            "error": {"message": message, "type": type_, "param": param, "code": code}
        }

    def _read_json(self) -> tuple[dict | None, dict | None]:
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length) if length > 0 else b""
        if not raw:
            return None, self._error("missing request body", code="missing_body")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None, self._error(
                "request body is not valid JSON", code="invalid_json"
            )
        if not isinstance(payload, dict):
            return None, self._error(
                "request body must be a JSON object", code="invalid_body"
            )
        return payload, None

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        responder = self._responder()
        if path == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "backend": responder.backend_id,
                    "model": responder.model_id,
                },
            )
            return
        if path == "/v1/models":
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": responder.model_id,
                            "object": "model",
                            "created": MODEL_CREATED,
                            "owned_by": responder.backend_id,
                        }
                    ],
                },
            )
            return
        self._send_json(404, self._error("not found", code="not_found"))

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path != "/v1/chat/completions":
            self._send_json(404, self._error("not found", code="not_found"))
            return
        payload, error = self._read_json()
        if error is not None:
            self._send_json(400, error)
            return
        if payload.get("stream"):
            self._send_json(
                400,
                self._error(
                    "streaming is not supported; this service only supports "
                    "non-streaming completions",
                    param="stream",
                    code="stream_unsupported",
                ),
            )
            return
        messages = payload.get("messages")
        if (
            not isinstance(messages, list)
            or not messages
            or not all(isinstance(message, dict) for message in messages)
        ):
            self._send_json(
                400,
                self._error(
                    "messages must be a non-empty list of message objects",
                    param="messages",
                    code="invalid_messages",
                ),
            )
            return
        responder = self._responder()
        result = responder.respond(messages)
        self._send_json(
            200,
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": responder.model_id,
                "system_fingerprint": responder.backend_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result.content},
                        "finish_reason": result.finish_reason,
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.prompt_tokens + result.completion_tokens,
                },
            },
        )

    def log_message(self, format, *args) -> None:
        """Silence per-request logging so test output stays deterministic."""


class OpenAICompatibleServer:
    """Start/stop wrapper around a loopback :class:`ThreadingHTTPServer`.

    Example::

        server = OpenAICompatibleServer(port=0)
        server.start()
        try:
            print(server.base_url)
        finally:
            server.stop()
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = 8000,
        responder: ModelResponder | None = None,
    ) -> None:
        if host not in LOOPBACK_HOSTS:
            raise ValueError(f"host must be loopback, got {host!r}")
        self.host = host
        self.port = port
        self.responder = responder if responder is not None else TinyCpuEchoResponder()
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._httpd is not None

    @property
    def address(self) -> tuple[str, int]:
        if self._httpd is None:
            raise RuntimeError("server is not running")
        host, port = self._httpd.server_address
        return host, int(port)

    @property
    def base_url(self) -> str:
        if self._httpd is None:
            return f"http://{_format_host(self.host)}:{self.port}"
        host, port = self._httpd.server_address
        return f"http://{_format_host(host)}:{port}"

    def start(self) -> None:
        if self._httpd is not None:
            raise RuntimeError("server is already running")
        httpd = ThreadingHTTPServer((self.host, self.port), _OpenAIRequestHandler)
        httpd.responder = self.responder  # type: ignore[attr-defined]
        self.port = int(httpd.server_address[1])
        self._httpd = httpd
        self._thread = threading.Thread(
            target=httpd.serve_forever,
            name="spaceslug-openai-api",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        httpd = self._httpd
        if httpd is None:
            return
        self._httpd = None
        httpd.shutdown()
        httpd.server_close()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)
        self._thread = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()
