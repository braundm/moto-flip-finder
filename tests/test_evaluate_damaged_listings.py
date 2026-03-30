from pathlib import Path

from moto_flip_finder.models import DamageAnalysis
from moto_flip_finder.evaluate_damaged_listings import (
    evaluate_damaged_listings,
    save_top_profit_report,
    sort_top_deals,
    sort_top_profit_deals,
    summarize_evaluations,
)


def test_evaluate_damaged_listings_builds_expected_output(monkeypatch):
    def fake_analyze_description(text: str) -> DamageAnalysis:
        assert text == "Motocykl po szlifie. Do wymiany owiewki i klamka. Silnik odpala."
        return DamageAnalysis(
            found_keywords=["po szlifie", "owiewki", "klamka", "silnik odpala"],
            suspected_damage=["fairings", "lever"],
            hidden_risks=["forks", "frame", "swingarm"],
            starts=True,
            severity="medium",
        )

    monkeypatch.setattr(
        "moto_flip_finder.evaluate_damaged_listings.analyze_description",
        fake_analyze_description,
    )

    damaged_candidates = [
        {
            "source": "olx",
            "url": "https://www.olx.pl/d/oferta/test-damaged-CID5-ID123.html",
            "title": "Suzuki GSX-R 600 K5 po szlifie",
            "price_pln": 12000,
            "full_description": "Motocykl po szlifie. Do wymiany owiewki i klamka. Silnik odpala.",
            "normalized_model": "gsxr_600",
            "year": 2011,
            "brand": "Suzuki",
        }
    ]
    healthy_comps = [
        {"url": "https://example.com/1", "normalized_model": "gsxr_600", "year": 2010, "price_pln": 20000, "title": "GSX-R 600 K5"},
        {"url": "https://example.com/2", "normalized_model": "gsxr_600", "year": 2011, "price_pln": 22000, "title": "GSX-R 600 K5"},
        {"url": "https://example.com/3", "normalized_model": "gsxr_600", "year": 2012, "price_pln": 24000, "title": "GSX-R 600 L2"},
    ]

    evaluations = evaluate_damaged_listings(damaged_candidates, healthy_comps)

    assert len(evaluations) == 1
    evaluation = evaluations[0]
    assert evaluation["healthy_market_value"] == 21000
    assert evaluation["comparables_count"] == 2
    assert evaluation["valuation_status"] == "same_generation_same_model"
    assert evaluation["valuation_confidence"] == "low"
    assert evaluation["generation_hint"] == "k5"
    assert evaluation["matched_same_generation"] is True
    assert evaluation["selected_comparables_strategy"] == "same_generation_same_model"
    assert evaluation["repair_estimate"]["total_cost_pln"] == 5930
    assert evaluation["expected_profit"] == 3070
    assert evaluation["roi_percent"] == 25.58
    assert evaluation["deal_score"] == 48
    assert evaluation["original_listing"]["title"] == "Suzuki GSX-R 600 K5 po szlifie"
    assert evaluation["damage_analysis"]["suspected_damage"] == ["fairings", "lever"]


def test_evaluate_damaged_listings_marks_insufficient_comparables(monkeypatch):
    monkeypatch.setattr(
        "moto_flip_finder.evaluate_damaged_listings.analyze_description",
        lambda text: DamageAnalysis(severity="high", starts=False),
    )

    evaluations = evaluate_damaged_listings(
        [
            {
                "source": "olx",
                "url": "https://www.olx.pl/d/oferta/test-damaged-CID5-ID123.html",
                "title": "Suzuki GSX-R 600",
                "price_pln": 10000,
                "full_description": "Opis",
                "normalized_model": None,
            }
        ],
        [],
    )

    evaluation = evaluations[0]
    assert evaluation["healthy_market_value"] is None
    assert evaluation["comparables_count"] == 0
    assert evaluation["valuation_status"] == "insufficient_comparables"
    assert evaluation["valuation_confidence"] == "low"
    assert evaluation["generation_hint"] is None
    assert evaluation["matched_same_generation"] is False
    assert evaluation["selected_comparables_strategy"] is None
    assert evaluation["expected_profit"] is None
    assert evaluation["deal_score"] == 0


def test_sort_top_deals_orders_by_deal_score_then_expected_profit():
    evaluations = [
        {"deal_score": 70, "expected_profit": 3000, "original_listing": {"title": "A"}},
        {"deal_score": 80, "expected_profit": 1000, "original_listing": {"title": "B"}},
        {"deal_score": 70, "expected_profit": 5000, "original_listing": {"title": "C"}},
    ]

    sorted_records = sort_top_deals(evaluations)

    assert [item["original_listing"]["title"] for item in sorted_records] == ["B", "C", "A"]


def test_sort_top_profit_deals_orders_by_expected_profit():
    evaluations = [
        {"deal_score": 10, "expected_profit": 3000, "original_listing": {"title": "A"}},
        {"deal_score": 80, "expected_profit": 1000, "original_listing": {"title": "B"}},
        {"deal_score": 70, "expected_profit": 5000, "original_listing": {"title": "C"}},
    ]

    sorted_records = sort_top_profit_deals(evaluations, limit=2)

    assert [item["original_listing"]["title"] for item in sorted_records] == ["C", "A"]


def test_summarize_evaluations_returns_expected_profit_and_deal_score_views():
    evaluations = [
        {
            "expected_profit": 1000,
            "deal_score": 60,
            "valuation_status": "year_window_1",
            "original_listing": {"title": "A"},
        },
        {
            "expected_profit": 5000,
            "deal_score": 55,
            "valuation_status": "insufficient_comparables",
            "original_listing": {"title": "B"},
        },
        {
            "expected_profit": None,
            "deal_score": 80,
            "valuation_status": "year_window_2",
            "original_listing": {"title": "C"},
        },
    ]

    summary = summarize_evaluations(evaluations)

    assert summary["evaluated_count"] == 3
    assert summary["insufficient_comparables_count"] == 1
    assert [item["original_listing"]["title"] for item in summary["top_5_highest_expected_profit"]] == [
        "B",
        "A",
    ]
    assert [item["original_listing"]["title"] for item in summary["top_5_highest_deal_score"]] == [
        "C",
        "A",
        "B",
    ]


def test_save_top_profit_report_writes_html_with_key_fields(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "moto_flip_finder.evaluate_damaged_listings._download_primary_image",
        lambda evaluation, assets_dir, report_dir, index: "assets/top_profit_01.jpg",
    )

    report_path = save_top_profit_report(
        [
            {
                "expected_profit": 5000,
                "healthy_market_value": 22000,
                "valuation_confidence": "medium",
                "damage_analysis": {
                    "severity": "medium",
                    "suspected_damage": ["fairings", "lever"],
                },
                "repair_estimate": {"total_cost_pln": 5930},
                "original_listing": {
                    "title": "Suzuki GSX-R 600 K5 po szlifie",
                    "url": "https://example.com/oferta",
                    "location": "Warszawa",
                    "price_pln": 12000,
                    "image_urls": ["https://example.com/image.jpg"],
                },
            }
        ],
        tmp_path / "top_profit_report.html",
        assets_dir=tmp_path / "assets",
        limit=1,
    )

    html = report_path.read_text(encoding="utf-8")

    assert "Suzuki GSX-R 600 K5 po szlifie" in html
    assert "Szacowany profit:" in html
    assert "Warszawa" in html
    assert "assets/top_profit_01.jpg" in html
