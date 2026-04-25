from __future__ import annotations

from abc import ABC, abstractmethod
import os
import re
import sys
from typing import Any

from dotenv import load_dotenv


load_dotenv()

HEURISTIC_BRAND_ALIASES = {
    "aprilia": "Aprilia",
    "barton": "Barton",
    "benelli": "Benelli",
    "beta": "Beta",
    "bmw": "BMW",
    "brixton": "Brixton",
    "can am": "Can-Am",
    "can-am": "Can-Am",
    "cfmoto": "CFMOTO",
    "ducati": "Ducati",
    "gasgas": "GasGas",
    "harley davidson": "Harley-Davidson",
    "harley-davidson": "Harley-Davidson",
    "honda": "Honda",
    "husqvarna": "Husqvarna",
    "indian": "Indian",
    "junak": "Junak",
    "kawasaki": "Kawasaki",
    "keeway": "Keeway",
    "ktm": "KTM",
    "kymco": "Kymco",
    "malaguti": "Malaguti",
    "moto guzzi": "Moto Guzzi",
    "mv agusta": "MV Agusta",
    "peugeot": "Peugeot",
    "piaggio": "Piaggio",
    "polaris": "Polaris",
    "qjmotor": "QJMotor",
    "romet": "Romet",
    "royal enfield": "Royal Enfield",
    "suzuki": "Suzuki",
    "triumph": "Triumph",
    "vespa": "Vespa",
    "yamaha": "Yamaha",
    "zontes": "Zontes",
}

GENERIC_BRANDS = {"inna", "inne", "other", "pozostałe", "pozostale", "unknown"}
QUAD_KEYWORDS = (
    "quad",
    "atv",
    "can-am outlander",
    "can am outlander",
    "renegade",
    "grizzly",
    "kingquad",
    "sportsman",
    "cforce",
)
SCOOTER_KEYWORDS = (
    "skuter",
    "scooter",
    "burgman",
    "xmax",
    "tmax",
    "nmax",
    "pcx",
    "forza",
    "vespa",
)
PARTS_ONLY_KEYWORDS = (
    "na części",
    "na czesci",
    "części",
    "czesci",
    "silnik do",
    "rama do",
    "owiewki",
    "felgi",
)
NEGOTIABLE_KEYWORDS = ("do negocjacji", "negocjacji", "negocjable")
MILEAGE_PATTERNS = [
    re.compile(r"\bprzebieg[^\d]{0,12}(\d[\d ]{2,8})\s*km\b", re.IGNORECASE),
    re.compile(r"\b(\d[\d ]{2,8})\s*km\b", re.IGNORECASE),
]
ENGINE_PATTERNS = [
    re.compile(r"\b(\d{2,4})\s*(?:cm3|cm³|cc)\b", re.IGNORECASE),
]
YEAR_PATTERNS = [
    re.compile(r"\b(19\d{2}|20\d{2})\b"),
]


class MotorcycleListingValidationProvider(ABC):
    @abstractmethod
    def validate(self, listing: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class HeuristicMotorcycleListingValidationProvider(MotorcycleListingValidationProvider):
    def validate(self, listing: dict[str, Any]) -> dict[str, Any]:
        title = _normalize_optional_string(listing.get("title"))
        short_description = _normalize_optional_string(listing.get("short_description"))
        full_description = _normalize_optional_string(listing.get("full_description"))
        attributes = listing.get("attributes") if isinstance(listing.get("attributes"), dict) else {}
        text = _joined_listing_text(title, short_description, full_description, attributes)

        brand = _resolve_brand_heuristically(listing, text)
        price_pln = _normalize_positive_int(
            listing.get("price_pln"),
            minimum=2,
            maximum=10_000_000,
        )
        engine_cc = _resolve_engine_cc_heuristically(listing, text)
        year = _resolve_year_heuristically(listing, text)
        mileage_km = _resolve_mileage_heuristically(listing, attributes, text)
        negotiable = _resolve_negotiable_heuristically(listing, text)
        vehicle_type = _resolve_vehicle_type_heuristically(listing, text, brand=brand, engine_cc=engine_cc)

        data_issues: list[str] = []
        if brand is None:
            data_issues.append("missing_brand")
        if price_pln is None:
            data_issues.append("missing_price")
        if year is None:
            data_issues.append("missing_year")
        if engine_cc is None:
            data_issues.append("missing_engine_cc")

        reject_reason: str | None = None
        is_sensible_listing: bool | None = None
        if _contains_any(text, PARTS_ONLY_KEYWORDS):
            reject_reason = "listing_looks_like_parts_only"
            is_sensible_listing = False
        elif vehicle_type == "quad":
            reject_reason = "listing_is_a_quad_or_atv"
            is_sensible_listing = False
        elif vehicle_type == "other":
            reject_reason = "listing_is_not_a_motorcycle"
            is_sensible_listing = False
        elif brand is None:
            reject_reason = "unable_to_resolve_brand"
            is_sensible_listing = False
        elif price_pln is None:
            reject_reason = "missing_price"
            is_sensible_listing = False
        elif vehicle_type in {"motorcycle", "scooter"}:
            is_sensible_listing = True

        validation_confidence = _heuristic_confidence(
            vehicle_type=vehicle_type,
            brand=brand,
            price_pln=price_pln,
            engine_cc=engine_cc,
            year=year,
            is_sensible_listing=is_sensible_listing,
            data_issues=data_issues,
        )

        validation_summary = _heuristic_summary(
            vehicle_type=vehicle_type,
            brand=brand,
            year=year,
            engine_cc=engine_cc,
            reject_reason=reject_reason,
            is_sensible_listing=is_sensible_listing,
        )

        return {
            "vehicle_type": vehicle_type,
            "resolved_brand": brand if _should_override_brand(listing.get("brand"), brand) else None,
            "resolved_price_pln": None,
            "resolved_engine_cc": engine_cc if listing.get("engine_cc") is None else None,
            "resolved_year": year if listing.get("year") is None else None,
            "negotiable": negotiable,
            "mileage_km": mileage_km,
            "is_sensible_listing": is_sensible_listing,
            "reject_reason": reject_reason,
            "validation_confidence": validation_confidence,
            "validation_summary": validation_summary,
            "data_issues": data_issues,
        }


def get_motorcycle_listing_validation_provider() -> MotorcycleListingValidationProvider | None:
    return HeuristicMotorcycleListingValidationProvider()


def validate_motorcycle_listing(
    listing: dict[str, Any],
    provider: MotorcycleListingValidationProvider | None = None,
) -> dict[str, Any]:
    selected_provider = provider if provider is not None else get_motorcycle_listing_validation_provider()
    if selected_provider is None:
        _emit_validation_debug("unvalidated")
        return HeuristicMotorcycleListingValidationProvider().validate(listing)

    try:
        validation = selected_provider.validate(listing)
        _emit_validation_debug(_provider_mode_name(selected_provider))
        return validation
    except Exception as exc:
        if os.getenv("MOTO_FLIP_FINDER_ANALYSIS_DEBUG") == "1":
            print(f"[motorcycle_validation] error={exc}", file=sys.stderr)
        fallback_provider = HeuristicMotorcycleListingValidationProvider()
        _emit_validation_debug("heuristic-fallback")
        return fallback_provider.validate(listing)


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
        raise ValueError("Listing validation payload must be a JSON object")

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


def _provider_mode_name(provider: MotorcycleListingValidationProvider) -> str:
    if isinstance(provider, HeuristicMotorcycleListingValidationProvider):
        return "heuristic"
    return provider.__class__.__name__


def _emit_validation_debug(mode: str) -> None:
    if os.getenv("MOTO_FLIP_FINDER_ANALYSIS_DEBUG") == "1":
        print(f"[motorcycle_validation] provider={mode}", file=sys.stderr)


def _joined_listing_text(
    title: str | None,
    short_description: str | None,
    full_description: str | None,
    attributes: dict[str, Any],
) -> str:
    parts = [title, short_description, full_description]
    for key, value in attributes.items():
        if isinstance(key, str):
            parts.append(key)
        if isinstance(value, str):
            parts.append(value)
    return " ".join(part for part in parts if isinstance(part, str) and part.strip()).casefold()


def _resolve_brand_heuristically(listing: dict[str, Any], text: str) -> str | None:
    raw_brand = _normalize_optional_string(listing.get("brand"))
    if raw_brand and raw_brand.casefold() not in GENERIC_BRANDS:
        return raw_brand

    for alias, canonical in sorted(HEURISTIC_BRAND_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if alias.casefold() in text:
            return canonical
    return None


def _resolve_vehicle_type_heuristically(
    listing: dict[str, Any],
    text: str,
    *,
    brand: str | None,
    engine_cc: int | None,
) -> str | None:
    existing_vehicle_type = _normalize_optional_string(listing.get("vehicle_type"))
    if existing_vehicle_type in {"motorcycle", "scooter", "quad", "other"}:
        return existing_vehicle_type

    if _contains_any(text, QUAD_KEYWORDS):
        return "quad"
    if _contains_any(text, SCOOTER_KEYWORDS):
        return "scooter"
    if brand is not None or engine_cc is not None:
        return "motorcycle"
    return None


def _resolve_engine_cc_heuristically(listing: dict[str, Any], text: str) -> int | None:
    resolved = _normalize_positive_int(listing.get("engine_cc"), minimum=40, maximum=2500)
    if resolved is not None:
        return resolved

    for pattern in ENGINE_PATTERNS:
        match = pattern.search(text)
        if match:
            return _normalize_positive_int(match.group(1), minimum=40, maximum=2500)
    return None


def _resolve_year_heuristically(listing: dict[str, Any], text: str) -> int | None:
    resolved = _normalize_positive_int(listing.get("year"), minimum=1900, maximum=2100)
    if resolved is not None:
        return resolved

    for pattern in YEAR_PATTERNS:
        match = pattern.search(text)
        if match:
            return _normalize_positive_int(match.group(1), minimum=1900, maximum=2100)
    return None


def _resolve_mileage_heuristically(
    listing: dict[str, Any],
    attributes: dict[str, Any],
    text: str,
) -> int | None:
    resolved = _normalize_positive_int(listing.get("mileage_km"), minimum=1, maximum=2_000_000)
    if resolved is not None:
        return resolved

    for key, value in attributes.items():
        if isinstance(key, str) and "przebieg" in key.casefold() and isinstance(value, str):
            parsed = _normalize_positive_int(value, minimum=1, maximum=2_000_000)
            if parsed is not None:
                return parsed

    for pattern in MILEAGE_PATTERNS:
        match = pattern.search(text)
        if match:
            return _normalize_positive_int(match.group(1), minimum=1, maximum=2_000_000)
    return None


def _resolve_negotiable_heuristically(listing: dict[str, Any], text: str) -> bool | None:
    negotiable = listing.get("negotiable")
    if negotiable in (True, False):
        return negotiable
    if _contains_any(text, NEGOTIABLE_KEYWORDS):
        return True
    return None


def _heuristic_confidence(
    *,
    vehicle_type: str | None,
    brand: str | None,
    price_pln: int | None,
    engine_cc: int | None,
    year: int | None,
    is_sensible_listing: bool | None,
    data_issues: list[str],
) -> str:
    if is_sensible_listing is False:
        return "medium"
    complete_fields = sum(value is not None for value in [vehicle_type, brand, price_pln, engine_cc, year])
    if complete_fields >= 5 and not data_issues:
        return "high"
    if complete_fields >= 3:
        return "medium"
    return "low"


def _heuristic_summary(
    *,
    vehicle_type: str | None,
    brand: str | None,
    year: int | None,
    engine_cc: int | None,
    reject_reason: str | None,
    is_sensible_listing: bool | None,
) -> str | None:
    if reject_reason is not None:
        return reject_reason.replace("_", " ")

    summary_parts: list[str] = []
    if is_sensible_listing is True:
        summary_parts.append("listing looks usable")
    if vehicle_type is not None:
        summary_parts.append(f"type={vehicle_type}")
    if brand is not None:
        summary_parts.append(f"brand={brand}")
    if year is not None:
        summary_parts.append(f"year={year}")
    if engine_cc is not None:
        summary_parts.append(f"engine={engine_cc}cc")
    if summary_parts:
        return ", ".join(summary_parts)
    return None


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _should_override_brand(raw_brand: object, resolved_brand: str | None) -> bool:
    normalized_brand = _normalize_optional_string(raw_brand)
    if resolved_brand is None:
        return False
    if normalized_brand is None:
        return True
    return normalized_brand.casefold() in GENERIC_BRANDS
