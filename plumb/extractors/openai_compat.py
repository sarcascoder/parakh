"""OpenAI-compatible extractor — covers Ollama, vLLM, llama.cpp server, and your
RunPod endpoint, since they all speak the /v1/chat/completions protocol.

The schema is rendered into a strict JSON instruction; the response is parsed
back into a dict. Kept dependency-light: uses urllib so there is nothing to
install. Network calls only happen when you actually run extraction.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Dict, List, Sequence

from ..metrics import FieldSpec, FieldType


def _schema_hint(specs: Sequence[FieldSpec]) -> str:
    lines: List[str] = []
    for s in specs:
        if s.type == FieldType.TABLE:
            cols = ", ".join(f'"{c.name}"' for c in s.columns)
            lines.append(f'  "{s.name}": [ {{ {cols} }} , ... ]')
        else:
            lines.append(f'  "{s.name}": <{s.type.value}>')
    return "{\n" + ",\n".join(lines) + "\n}"


class OpenAICompatExtractor:
    def __init__(self, base_url: str = "http://localhost:11434/v1",
                 model: str = "qwen2.5-vl", api_key: str = "not-needed",
                 temperature: float = 0.0, name: str | None = None,
                 few_shot_block: str = ""):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.name = name or model
        # Inject verified examples from the correction loop (see plumb.fewshot).
        self.few_shot_block = few_shot_block

    def extract(self, document: str, specs: Sequence[FieldSpec]) -> Dict:
        prefix = f"{self.few_shot_block}\n\n" if self.few_shot_block else ""
        prompt = (
            prefix +
            "Extract the following fields from the document and return ONLY valid "
            "JSON matching this shape:\n"
            f"{_schema_hint(specs)}\n\n"
            "Document:\n\"\"\"\n" + document + "\n\"\"\"\n"
            "Return only the JSON object."
        )
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        content = data["choices"][0]["message"]["content"]
        return _parse_json(content)


def _parse_json(text: str) -> Dict:
    text = text.strip()
    # Tolerate models that wrap JSON in markdown fences.
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            return json.loads(text[start:end + 1])
        raise
