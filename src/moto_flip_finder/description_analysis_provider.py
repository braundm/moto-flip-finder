from __future__ import annotations

from abc import ABC, abstractmethod
import json
import os

from dotenv import load_dotenv

from .models import DamageAnalysis


load_dotenv()


class DescriptionAnalysisProvider(ABC):
    @abstractmethod
    def analyze(self, text: str) -> DamageAnalysis:
        raise NotImplementedError


class OpenAIDescriptionAnalysisProvider(DescriptionAnalysisProvider):
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    def analyze(self, text: str) -> DamageAnalysis:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is not installed") from exc

        client = OpenAI(api_key=self.api_key)
        response = client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Analyze a motorcycle listing description and return JSON only. "
                                "Use this schema: "
                                '{"found_keywords": list[str], '
                                '"suspected_damage": list[str], '
                                '"hidden_risks": list[str], '
                                '"starts": true|false|null, '
                                '"severity": "low"|"medium"|"high"|"unknown"}.'
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            ],
        )
        payload = _extract_response_payload(response)
        return damage_analysis_from_payload(payload)


def damage_analysis_from_payload(payload: object) -> DamageAnalysis:
    if not isinstance(payload, dict):
        raise ValueError("OpenAI response payload must be a JSON object")

    found_keywords = _normalize_string_list(payload.get("found_keywords"))
    suspected_damage = _normalize_string_list(payload.get("suspected_damage"))
    hidden_risks = _normalize_string_list(payload.get("hidden_risks"))

    starts = payload.get("starts")
    if starts not in (True, False, None):
        starts = None

    severity = str(payload.get("severity", "unknown")).lower()
    if severity not in {"low", "medium", "high", "unknown"}:
        severity = "unknown"

    return DamageAnalysis(
        found_keywords=found_keywords,
        suspected_damage=suspected_damage,
        hidden_risks=hidden_risks,
        starts=starts,
        severity=severity,
    )


def _normalize_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []

    normalized = []
    for item in value:
        if isinstance(item, str) and item.strip():
            normalized.append(item.strip())

    return sorted(set(normalized))


def _extract_response_payload(response: object) -> dict:
    output_text = getattr(response, "output_text", None)
    if not isinstance(output_text, str) or not output_text.strip():
        raise ValueError("OpenAI response did not contain JSON text")

    try:
        payload = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise ValueError("OpenAI response was not valid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("OpenAI response JSON must be an object")

    return payload
