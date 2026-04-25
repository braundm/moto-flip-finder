from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()



from abc import ABC, abstractmethod
import json
import os
import sys

from .models import DamageAnalysis


ALLOWED_DAMAGE_NAMES = {
    "fairings",
    "lever",
    "mirror",
    "footpeg",
    "tank",
    "wheel",
    "exhaust",
    "forks",
    "frame",
    "swingarm",
}

NORMALIZED_DAMAGE_ALIASES = {
    "fairing": "fairings",
    "fairings": "fairings",
    "lever": "lever",
    "mirror": "mirror",
    "mirrors": "mirror",
    "footpeg": "footpeg",
    "footpegs": "footpeg",
    "tank": "tank",
    "rim": "wheel",
    "rims": "wheel",
    "wheel": "wheel",
    "wheels": "wheel",
    "exhaust": "exhaust",
    "front fork": "forks",
    "front forks": "forks",
    "fork": "forks",
    "forks": "forks",
    "chassis": "frame",
    "frame": "frame",
    "swing arm": "swingarm",
    "swingarm": "swingarm",
}


class DescriptionAnalysisProvider(ABC):
    @abstractmethod
    def analyze(self, text: str) -> DamageAnalysis:
        raise NotImplementedError


def damage_analysis_from_payload(payload: object) -> DamageAnalysis:
    if not isinstance(payload, dict):
        raise ValueError("Damage analysis payload must be a JSON object")

    found_keywords = _normalize_string_list(payload.get("found_keywords"))
    suspected_damage = _normalize_damage_list(payload.get("suspected_damage"))
    hidden_risks = _normalize_damage_list(payload.get("hidden_risks"))

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


def _normalize_damage_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []

    normalized = []
    for item in value:
        if not isinstance(item, str):
            continue
        key = item.strip().lower()
        if not key:
            continue
        mapped = NORMALIZED_DAMAGE_ALIASES.get(key)
        if mapped in ALLOWED_DAMAGE_NAMES:
            normalized.append(mapped)

    return sorted(set(normalized))


def _extract_response_payload(response: object) -> dict:
    output_text = getattr(response, "output_text", None)
    if not isinstance(output_text, str) or not output_text.strip():
        raise ValueError("Analysis response did not contain JSON text")

    try:
        payload = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Analysis response was not valid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("Analysis response JSON must be an object")

    if os.getenv("MOTO_FLIP_FINDER_ANALYSIS_DEBUG") == "1":
        print(
            "[damage_analysis] raw_json="
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            file=sys.stderr,
        )

    return payload
