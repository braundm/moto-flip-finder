from moto_flip_finder.market_value import (
    build_market_valuation,
    detect_generation_hint,
    estimate_healthy_market_value,
    select_comparables,
)


def test_detect_generation_hint_reads_generation_from_title_or_description():
    assert detect_generation_hint("Suzuki GSX-R 600 K5", None) == "k5"
    assert detect_generation_hint("Suzuki GSX-R 600", "Swietna sztuka L4") == "l4"
    assert detect_generation_hint("Suzuki GSX-R 600", "Brak oznaczenia") is None


def test_select_comparables_prefers_same_generation_before_year_window():
    record = {
        "url": "https://example.com/self",
        "normalized_model": "gsxr_600",
        "year": 2012,
        "title": "Suzuki GSX-R 600 K5",
    }
    healthy_comps = [
        {"url": "https://example.com/self", "normalized_model": "gsxr_600", "year": 2012, "price_pln": 20000, "title": "GSX-R 600 K5"},
        {"url": "https://example.com/a", "normalized_model": "gsxr_600", "year": 2011, "price_pln": 19000, "title": "GSX-R 600 K5"},
        {"url": "https://example.com/b", "normalized_model": "gsxr_600", "year": 2012, "price_pln": 20000, "title": "GSX-R 600 L1"},
        {"url": "https://example.com/c", "normalized_model": "gsxr_600", "year": 2013, "price_pln": 21000, "title": "GSX-R 600 K5"},
        {"url": "https://example.com/d", "normalized_model": "gsxr_600", "year": 2012, "price_pln": 20500, "title": "GSX-R 600 L2"},
        {"url": "https://example.com/e", "normalized_model": "gsxr_1000", "year": 2012, "price_pln": 32000, "title": "GSX-R 1000 K5"},
    ]

    comparables, strategy = select_comparables(record, healthy_comps)

    assert [item["url"] for item in comparables] == [
        "https://example.com/a",
        "https://example.com/c",
    ]
    assert strategy == "same_generation_same_model"


def test_select_comparables_deduplicates_and_ignores_missing_price_or_model():
    record = {"normalized_model": "gsxr_600", "year": 2012}
    healthy_comps = [
        {"url": "https://example.com/a", "normalized_model": "gsxr_600", "year": 2011, "price_pln": 19000},
        {"url": "https://example.com/a", "normalized_model": "gsxr_600", "year": 2011, "price_pln": 19500},
        {"url": "https://example.com/b", "normalized_model": "gsxr_600", "year": 2012, "price_pln": None},
        {"url": "https://example.com/c", "normalized_model": None, "year": 2012, "price_pln": 20000},
        {"url": "https://example.com/d", "normalized_model": "gsxr_600", "year": 2013, "price_pln": 21000},
        {"normalized_model": "gsxr_600", "year": 2013, "price_pln": 20500},
    ]

    comparables, strategy = select_comparables(record, healthy_comps)

    assert len(comparables) == 3
    assert comparables[0]["price_pln"] == 19000
    assert strategy == "same_model_year_window"


def test_estimate_healthy_market_value_falls_back_to_model_only_when_year_filters_are_too_small():
    record = {"normalized_model": "gsxr_600", "year": 2012}
    healthy_comps = [
        {"url": "https://example.com/a", "normalized_model": "gsxr_600", "year": 2011, "price_pln": 19000},
        {"url": "https://example.com/b", "normalized_model": "gsxr_600", "year": 2006, "price_pln": 15000},
        {"url": "https://example.com/c", "normalized_model": "gsxr_600", "year": 2007, "price_pln": 16000},
        {"url": "https://example.com/d", "normalized_model": "gsxr_600", "year": 2008, "price_pln": 17000},
    ]

    market_value = estimate_healthy_market_value(record, healthy_comps)

    assert market_value == 16500


def test_build_market_valuation_uses_trimmed_median_and_sets_confidence():
    record = {"normalized_model": "gsxr_600", "year": 2012}
    healthy_comps = [
        {"url": f"https://example.com/{idx}", "normalized_model": "gsxr_600", "year": 2012, "price_pln": price}
        for idx, price in enumerate([10000, 18000, 19000, 19500, 20000, 20500, 21000, 22000, 50000], start=1)
    ]

    valuation = build_market_valuation(record, healthy_comps)

    assert valuation["healthy_market_value"] == 20000
    assert valuation["comparables_count"] == 9
    assert valuation["valuation_status"] == "same_model_year_window"
    assert valuation["valuation_confidence"] == "high"


def test_build_market_valuation_returns_insufficient_comparables_when_model_missing():
    valuation = build_market_valuation({"year": 2012}, [])

    assert valuation["healthy_market_value"] is None
    assert valuation["comparables_count"] == 0
    assert valuation["valuation_status"] == "insufficient_comparables"
    assert valuation["valuation_confidence"] == "low"


def test_build_market_valuation_marks_same_generation_match():
    record = {"normalized_model": "gsxr_600", "title": "Suzuki GSX-R 600 K5"}
    healthy_comps = [
        {"url": "https://example.com/1", "normalized_model": "gsxr_600", "price_pln": 18000, "title": "GSX-R 600 K5"},
        {"url": "https://example.com/2", "normalized_model": "gsxr_600", "price_pln": 20000, "title": "GSX-R 600 K5"},
    ]

    valuation = build_market_valuation(record, healthy_comps)

    assert valuation["generation_hint"] == "k5"
    assert valuation["matched_same_generation"] is True
    assert valuation["selected_comparables_strategy"] == "same_generation_same_model"
