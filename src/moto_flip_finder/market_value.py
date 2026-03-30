from __future__ import annotations

import re
from statistics import median
from typing import Any


MIN_FILTERED_COMPARABLES = 3
TRIMMED_MEDIAN_THRESHOLD = 8


def estimate_healthy_market_value(
    record: dict[str, Any],
    healthy_comps: list[dict[str, Any]],
) -> int | None:
    valuation = build_market_valuation(record, healthy_comps)
    return valuation["healthy_market_value"]


def build_market_valuation(
    record: dict[str, Any],
    healthy_comps: list[dict[str, Any]],
) -> dict[str, Any]:
    generation_hint = detect_generation_hint(
        _string_or_none(record.get("title")),
        _string_or_none(record.get("full_description")),
    )
    comparables, strategy = select_comparables(record, healthy_comps)
    comparables_count = len(comparables)

    if comparables_count == 0:
        return {
            "healthy_market_value": None,
            "comparables_count": 0,
            "valuation_status": "insufficient_comparables",
            "valuation_confidence": "low",
            "generation_hint": generation_hint,
            "matched_same_generation": False,
            "selected_comparables_strategy": None,
            "selected_comparables": [],
        }

    prices = [_price_or_none(item.get("price_pln")) for item in comparables]
    normalized_prices = [price for price in prices if price is not None]
    if not normalized_prices:
        return {
            "healthy_market_value": None,
            "comparables_count": 0,
            "valuation_status": "insufficient_comparables",
            "valuation_confidence": "low",
            "generation_hint": generation_hint,
            "matched_same_generation": False,
            "selected_comparables_strategy": None,
            "selected_comparables": [],
        }

    market_value = _trimmed_median(normalized_prices)

    return {
        "healthy_market_value": market_value,
        "comparables_count": comparables_count,
        "valuation_status": strategy or "insufficient_comparables",
        "valuation_confidence": _valuation_confidence(strategy, comparables_count),
        "generation_hint": generation_hint,
        "matched_same_generation": strategy == "same_generation_same_model",
        "selected_comparables_strategy": strategy,
        "selected_comparables": comparables,
    }


def select_comparables(
    record: dict[str, Any],
    healthy_comps: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    normalized_candidates = _normalized_candidates(record, healthy_comps)
    if not normalized_candidates:
        return [], None

    generation_hint = detect_generation_hint(
        _string_or_none(record.get("title")),
        _string_or_none(record.get("full_description")),
    )
    if generation_hint is not None:
        same_generation_candidates = [
            item
            for item in normalized_candidates
            if detect_generation_hint(
                _string_or_none(item.get("title")),
                _string_or_none(item.get("full_description")),
            )
            == generation_hint
        ]
        if same_generation_candidates:
            return same_generation_candidates, "same_generation_same_model"

    year = _int_or_none(record.get("year"))
    if year is None:
        return normalized_candidates, "same_model_fallback"

    within_one_year = [
        item
        for item in normalized_candidates
        if (candidate_year := _int_or_none(item.get("year"))) is not None
        and abs(candidate_year - year) <= 1
    ]
    if len(within_one_year) >= MIN_FILTERED_COMPARABLES:
        return within_one_year, "same_model_year_window"

    within_two_years = [
        item
        for item in normalized_candidates
        if (candidate_year := _int_or_none(item.get("year"))) is not None
        and abs(candidate_year - year) <= 2
    ]
    if len(within_two_years) >= MIN_FILTERED_COMPARABLES:
        return within_two_years, "same_model_year_window"

    return normalized_candidates, "same_model_fallback"


def detect_generation_hint(title: str | None, description: str | None) -> str | None:
    haystack = " ".join(part for part in [title, description] if part).lower()
    if not haystack:
        return None

    match = re.search(r"(?<![a-z0-9])([klm]\d)(?![a-z0-9])", haystack)
    if not match:
        return None
    return match.group(1)


def _normalized_candidates(
    record: dict[str, Any],
    healthy_comps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized_model = _string_or_none(record.get("normalized_model"))
    if normalized_model is None:
        return []

    record_url = _string_or_none(record.get("url"))
    deduplicated: dict[str, dict[str, Any]] = {}
    url_less: list[dict[str, Any]] = []

    for item in healthy_comps:
        if _string_or_none(item.get("normalized_model")) != normalized_model:
            continue
        if _price_or_none(item.get("price_pln")) is None:
            continue

        comparable_url = _string_or_none(item.get("url"))
        if record_url and comparable_url == record_url:
            continue

        if comparable_url:
            deduplicated.setdefault(comparable_url, item)
        else:
            url_less.append(item)

    return list(deduplicated.values()) + url_less


def _valuation_confidence(selection_strategy: str | None, comparables_count: int) -> str:
    if selection_strategy is None or comparables_count == 0:
        return "low"
    if selection_strategy == "same_generation_same_model" and comparables_count >= 5:
        return "high"
    if selection_strategy == "same_model_year_window" and comparables_count >= 5:
        return "high"
    if comparables_count >= 3:
        return "medium"
    return "low"


def _trimmed_median(prices: list[int]) -> int:
    sorted_prices = sorted(prices)
    if len(sorted_prices) >= TRIMMED_MEDIAN_THRESHOLD:
        trim_count = max(1, int(len(sorted_prices) * 0.1))
        trimmed_prices = sorted_prices[trim_count:-trim_count]
        if trimmed_prices:
            sorted_prices = trimmed_prices
    return int(median(sorted_prices))


def _price_or_none(value: object) -> int | None:
    number = _int_or_none(value)
    if number is None or number <= 0:
        return None
    return number


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def _string_or_none(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None
