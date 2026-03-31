from __future__ import annotations

from abc import ABC, abstractmethod
import json
import os
import re
import sys
from typing import Any

from dotenv import load_dotenv


load_dotenv()


class MotorcycleListingValidationProvider(ABC):
    @abstractmethod
    def validate(self, listing: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class OpenAIMotorcycleListingValidationProvider(MotorcycleListingValidationProvider):
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.1")
        self.timeout_seconds = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60"))

    def validate(self, listing: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is not installed") from exc

        client = OpenAI(api_key=self.api_key)
        response = client.with_options(
            timeout=self.timeout_seconds,
            max_retries=1,
        ).responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You validate a scraped motorcycle listing. "
                                "Use the structured fields, title, description, and attributes together. "
                                "Return JSON only with this schema: "
                                '{"vehicle_type": "motorcycle"|"scooter"|"quad"|"other"|null, '
                                '"resolved_brand": string|null, '
                                '"resolved_price_pln": integer|null, '
                                '"resolved_engine_cc": integer|null, '
                                '"resolved_year": integer|null, '
                                '"negotiable": true|false|null, '
                                '"mileage_km": integer|null, '
                                '"is_sensible_listing": true|false|null, '
                                '"reject_reason": string|null, '
                                '"validation_confidence": "low"|"medium"|"high"|null, '
                                '"validation_summary": string|null, '
                                '"data_issues": list[str]}. '
                                "Use null when the listing does not clearly support a value. "
                                "Give resolved_* fields only when you have high confidence that the inferred value "
                                "is better than the scraped one or when the scraped one is missing, generic, or inconsistent. "
                                "Infer brand, price, engine capacity, and year from description/title only when strongly supported. "
                                "If the listing is a quad/ATV, mark vehicle_type='quad' and set is_sensible_listing=false "
                                "because we only want motorcycles and scooters. "
                                "If the brand is generic/missing, for example 'Inna', try to infer the real brand from the title "
                                "and description; if you cannot infer it confidently, set is_sensible_listing=false. "
                                "Do not guess mileage or negotiable status from weak hints."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": json.dumps(listing, ensure_ascii=False, indent=2),
                        }
                    ],
                },
            ],
        )
        payload = _extract_response_payload(response)
        return motorcycle_listing_validation_from_payload(payload)


def get_motorcycle_listing_validation_provider() -> MotorcycleListingValidationProvider | None:
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIMotorcycleListingValidationProvider()
    return None


def validate_motorcycle_listing(
    listing: dict[str, Any],
    provider: MotorcycleListingValidationProvider | None = None,
) -> dict[str, Any]:
    selected_provider = provider if provider is not None else get_motorcycle_listing_validation_provider()
    if selected_provider is None:
        _emit_validation_debug("unvalidated")
        return empty_motorcycle_listing_validation()

    try:
        validation = selected_provider.validate(listing)
        _emit_validation_debug(_provider_mode_name(selected_provider))
        return validation
    except Exception as exc:
        if os.getenv("MOTO_FLIP_FINDER_ANALYSIS_DEBUG") == "1":
            print(f"[motorcycle_validation] error={exc}", file=sys.stderr)
        _emit_validation_debug("openai-error")
        return empty_motorcycle_listing_validation()


def empty_motorcycle_listing_validation() -> dict[str, Any]:
    return {
        "vehicle_type": None,
        "resolved_brand": None,
        "resolved_price_pln": None,
        "resolved_engine_cc": None,
        "resolved_year": None,
        "negotiable": None,
        "mileage_km": None,
        "is_sensible_listing": None,
        "reject_reason": None,
        "validation_confidence": None,
        "validation_summary": None,
        "data_issues": [],
    }


def motorcycle_listing_validation_from_payload(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("OpenAI listing validation payload must be a JSON object")

    vehicle_type = _normalize_optional_string(payload.get("vehicle_type"))
    if vehicle_type not in {"motorcycle", "scooter", "quad", "other", None}:
        vehicle_type = None

    resolved_brand = _normalize_optional_string(payload.get("resolved_brand"))

    resolved_price_pln = _normalize_positive_int(
        payload.get("resolved_price_pln"),
        minimum=2,
        maximum=10_000_000,
    )

    resolved_engine_cc = _normalize_positive_int(
        payload.get("resolved_engine_cc"),
        minimum=40,
        maximum=2500,
    )

    resolved_year = _normalize_positive_int(
        payload.get("resolved_year"),
        minimum=1900,
        maximum=2100,
    )

    negotiable = payload.get("negotiable")
    if negotiable not in (True, False, None):
        negotiable = None

    mileage_km = _normalize_positive_int(payload.get("mileage_km"), minimum=1, maximum=2_000_000)

    is_sensible_listing = payload.get("is_sensible_listing")
    if is_sensible_listing not in (True, False, None):
        is_sensible_listing = None

    reject_reason = _normalize_optional_string(payload.get("reject_reason"))
    validation_summary = _normalize_optional_string(payload.get("validation_summary"))

    validation_confidence = _normalize_optional_string(payload.get("validation_confidence"))
    if validation_confidence not in {"low", "medium", "high", None}:
        validation_confidence = None

    data_issues = _normalize_string_list(payload.get("data_issues"))

    return {
        "vehicle_type": vehicle_type,
        "resolved_brand": resolved_brand,
        "resolved_price_pln": resolved_price_pln,
        "resolved_engine_cc": resolved_engine_cc,
        "resolved_year": resolved_year,
        "negotiable": negotiable,
        "mileage_km": mileage_km,
        "is_sensible_listing": is_sensible_listing,
        "reject_reason": reject_reason,
        "validation_confidence": validation_confidence,
        "validation_summary": validation_summary,
        "data_issues": data_issues,
    }


def _normalize_positive_int(value: object, *, minimum: int, maximum: int) -> int | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        parsed = int(value)
    elif isinstance(value, str):
        match = re.search(r"\d(?:[\d ]*\d)?", value)
        if not match:
            return None
        digits = "".join(ch for ch in match.group(0) if ch.isdigit())
        parsed = int(digits)
    else:
        return None

    if minimum <= parsed <= maximum:
        return parsed
    return None


def _normalize_optional_string(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _normalize_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []

    normalized: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            normalized.append(item.strip())

    return normalized


def _extract_response_payload(response: object) -> dict[str, Any]:
    output_text = getattr(response, "output_text", None)
    if not isinstance(output_text, str) or not output_text.strip():
        raise ValueError("OpenAI response did not contain JSON text")

    try:
        payload = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise ValueError("OpenAI response was not valid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("OpenAI response JSON must be an object")

    if os.getenv("MOTO_FLIP_FINDER_ANALYSIS_DEBUG") == "1":
        print(
            "[motorcycle_validation] raw_json="
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            file=sys.stderr,
        )

    return payload


def _provider_mode_name(provider: MotorcycleListingValidationProvider) -> str:
    if isinstance(provider, OpenAIMotorcycleListingValidationProvider):
        return "openai"
    return provider.__class__.__name__


def _emit_validation_debug(mode: str) -> None:
    if os.getenv("MOTO_FLIP_FINDER_ANALYSIS_DEBUG") == "1":
        print(f"[motorcycle_validation] provider={mode}", file=sys.stderr)
