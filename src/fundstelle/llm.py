"""Chat model backends that return JSON matching a schema.

- ``LlamaCppBackend``: a local GGUF model, with the schema compiled into a grammar so a
  small model cannot produce invalid JSON.
- ``OpenAICompatibleBackend``: any server that speaks the OpenAI chat API (vLLM,
  llama.cpp server, Ollama, hosted APIs). The key comes from an environment variable.
- ``ScriptedBackend``: fixed responses, for tests.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float


class ChatBackend(Protocol):
    name: str

    def complete(
        self, messages: list[dict[str, str]], schema: dict[str, Any], max_tokens: int = 600
    ) -> Completion: ...


MODELS = {
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q8_0.gguf"),
    "qwen3-4b": ("Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"),
}


class LlamaCppBackend:
    def __init__(self, model: str = "qwen3-4b", n_ctx: int = 8192, threads: int | None = None):
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        repo, filename = MODELS.get(model, (None, model))
        path = hf_hub_download(repo, filename) if repo else filename
        self.name = model
        self.llm = Llama(model_path=path, n_ctx=n_ctx, n_threads=threads, verbose=False)

    def complete(
        self, messages: list[dict[str, str]], schema: dict[str, Any], max_tokens: int = 600
    ) -> Completion:
        t0 = time.perf_counter()
        out = self.llm.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object", "schema": schema},
            temperature=0.0,
            max_tokens=max_tokens,
        )
        usage = out["usage"]
        return Completion(
            out["choices"][0]["message"]["content"] or "",
            usage["prompt_tokens"],
            usage["completion_tokens"],
            time.perf_counter() - t0,
        )


class OpenAICompatibleBackend:
    def __init__(self, model: str, base_url: str, api_key_env: str = "OPENAI_API_KEY") -> None:
        self.name, self.base_url = model, base_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "")

    def complete(
        self, messages: list[dict[str, str]], schema: dict[str, Any], max_tokens: int = 600
    ) -> Completion:
        body = {
            "model": self.name,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema, "strict": True},
            },
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.load(r)
        usage = out.get("usage", {})
        return Completion(
            out["choices"][0]["message"]["content"] or "",
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            time.perf_counter() - t0,
        )


class ScriptedBackend:
    def __init__(self, responses: list[dict[str, Any] | str], name: str = "scripted") -> None:
        self.responses, self.name = list(responses), name
        self.calls: list[list[dict[str, str]]] = []

    def complete(
        self, messages: list[dict[str, str]], schema: dict[str, Any], max_tokens: int = 600
    ) -> Completion:
        self.calls.append([dict(m) for m in messages])
        r = self.responses.pop(0)
        text = r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)
        return Completion(text, 0, 0, 0.0)
