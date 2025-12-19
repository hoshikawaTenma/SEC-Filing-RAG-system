from __future__ import annotations

import json
from typing import Any

import requests

from .config import settings


class DashScopeClient:
    def __init__(self) -> None:
        self.api_key = settings.dashscope_api_key
        self.base_url = settings.dashscope_base_url.rstrip("/")
        self.chat_model = settings.dashscope_chat_model
        self.embed_model = settings.dashscope_embed_model

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat_text(self, system_prompt: str, user_prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set")
        url = f"{self.base_url}/api/v1/services/aigc/text-generation/generation"
        payload = {
            "model": self.chat_model,
            "input": {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            },
            "parameters": {"result_format": "message"},
        }
        response = requests.post(url, headers=self._headers(), json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["output"]["choices"][0]["message"]["content"]

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        message = self.chat_text(system_prompt, user_prompt)
        return json.loads(message)

    def chat_json_with_repair(
        self, system_prompt: str, user_prompt: str, max_retries: int = 2
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for _ in range(max_retries + 1):
            content = ""
            try:
                content = self.chat_text(system_prompt, user_prompt)
                return json.loads(content)
            except Exception as exc:
                last_error = exc
                user_prompt = (
                    "Fix the JSON output to be valid JSON only. "
                    "Return JSON only.\n\n"
                    f"{content}"
                )
        if last_error:
            raise last_error
        raise RuntimeError("Failed to parse JSON from DashScope.")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set")
        url = f"{self.base_url}/api/v1/services/embeddings/text-embedding/text-embedding"
        payload = {"model": self.embed_model, "input": {"texts": texts}}
        response = requests.post(url, headers=self._headers(), json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["output"]["embeddings"]]


def local_hash_embedding(text: str, dims: int = 128) -> list[float]:
    vec = [0.0] * dims
    for idx, ch in enumerate(text):
        vec[idx % dims] += float((ord(ch) % 13) - 6)
    return vec
