"""Ollama-backed LLM client for local generation, summarization, and extraction."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from memory_core.domain.enums import SourceType
from memory_core.interfaces.llm import LLMClient


class OllamaLLMClient(LLMClient):
    """Thin Ollama API client using the local chat endpoint."""

    def __init__(
        self,
        model_name: str = "llama3.1",
        *,
        base_url: str = "http://localhost:11434/api",
        timeout_seconds: float = 120.0,
        keep_alive: str | None = "5m",
        options: dict[str, Any] | None = None,
    ) -> None:
        if not model_name:
            raise ValueError("model_name must be non-empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.keep_alive = keep_alive
        self.options = dict(options or {})

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate a free-form assistant response."""
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be non-empty")

        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt.strip()})

        payload = self._chat(messages=messages, response_format=None)
        return self._extract_message_content(payload)

    def summarize(self, text: str, context: str | None = None) -> str:
        """Summarize text for session memory or document condensation."""
        if not text or not text.strip():
            raise ValueError("text must be non-empty")

        system_prompt = (
            "You create concise, faithful summaries for a local memory system. "
            "Preserve important facts, preferences, tasks, and provenance cues. "
            "Do not invent details."
        )
        prompt_parts = []
        if context and context.strip():
            prompt_parts.append(f"Context:\n{context.strip()}")
        prompt_parts.append("Summarize the following text in a compact, information-dense form:")
        prompt_parts.append(text.strip())
        return self.generate("\n\n".join(prompt_parts), system_prompt=system_prompt)

    def extract_structured_memory(self, text: str, source_type: SourceType) -> list[dict[str, Any]]:
        """Extract structured memory candidates as JSON objects."""
        if not text or not text.strip():
            raise ValueError("text must be non-empty")

        system_prompt = (
            "You extract reviewable memory candidates for an offline memory system. "
            "Return only valid JSON as an array of objects. "
            "Each object may contain: memory_type, content, summary, confidence, importance, "
            "tags, extraction_reason, suggested_action, existing_memory_id, metadata. "
            "Allowed memory_type values are: document, document_chunk, fact, episode, preference, "
            "task, workflow_rule, summary. "
            "Allowed suggested_action values are: create, merge, update. "
            "Do not include markdown fences or commentary."
        )
        prompt = (
            f"Source type: {source_type.value}\n\n"
            "Extract the most useful memory candidates from the text below. "
            "Prefer durable facts, tasks, preferences, procedural rules, and important document-level summaries. "
            "Use conservative confidence values when uncertain.\n\n"
            f"Text:\n{text.strip()}"
        )

        payload = self._chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format="json",
        )
        raw_content = self._extract_message_content(payload)
        parsed = self._parse_json_content(raw_content)

        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            if isinstance(parsed.get("items"), list):
                return [item for item in parsed["items"] if isinstance(item, dict)]
            return [parsed]
        raise ValueError("Ollama returned JSON that could not be interpreted as structured memory items")

    def _chat(self, messages: list[dict[str, str]], response_format: str | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
        }
        if response_format is not None:
            payload["format"] = response_format
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        if self.options:
            payload["options"] = self.options

        request = Request(
            url=f"{self.base_url}/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw_body = response.read().decode("utf-8")
        except HTTPError as exc:  # pragma: no cover - depends on external runtime
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama API HTTP error {exc.code}: {body}") from exc
        except URLError as exc:  # pragma: no cover - depends on external runtime
            raise RuntimeError(
                "Could not reach the Ollama API. Ensure Ollama is running locally "
                f"at {self.base_url}."
            ) from exc

        try:
            return json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned a non-JSON response") from exc

    @staticmethod
    def _extract_message_content(payload: dict[str, Any]) -> str:
        message = payload.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("Ollama chat response did not include a message object")

        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("Ollama chat response did not include message content")
        return content.strip()

    @staticmethod
    def _parse_json_content(raw_content: str) -> Any:
        content = raw_content.strip()
        if not content:
            return []

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            cleaned = OllamaLLMClient._strip_code_fences(content)
            return json.loads(cleaned)

    @staticmethod
    def _strip_code_fences(content: str) -> str:
        if content.startswith("```"):
            lines = content.splitlines()
            if len(lines) >= 3 and lines[-1].strip() == "```":
                return "\n".join(lines[1:-1]).strip()
        return content
