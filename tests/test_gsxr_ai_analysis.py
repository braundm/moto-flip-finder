from moto_flip_finder.gsxr_ai_analysis import (
    analyze_gsxr_records,
    filter_gsxr_records,
    rank_records_for_review,
    summarize_analysis,
)
from moto_flip_finder.models import DamageAnalysis


def test_filter_gsxr_records_applies_engine_year_filters_and_limit():
    records = [
        {"normalized_model": "gsxr_600", "engine_cc": 600, "year": 2005, "title": "A"},
        {"normalized_model": "r6", "title": "B"},
        {"normalized_model": "gsxr_125", "engine_cc": 125, "year": 2018, "title": "C"},
        {"normalized_model": "gsxr_600", "engine_cc": 600, "year": 2008, "title": "D"},
    ]

    filtered = filter_gsxr_records(
        records,
        max_records=1,
        engine_cc=600,
        year_from=2004,
        year_to=2006,
    )

    assert filtered == [{"normalized_model": "gsxr_600", "engine_cc": 600, "year": 2005, "title": "A"}]


def test_analyze_gsxr_records_builds_expected_output(monkeypatch):
    def fake_analyze_description(text: str) -> DamageAnalysis:
        assert "Tytul: Suzuki GSX-R 600 K5 po szlifie" in text
        assert "Pelny opis: Motocykl po szlifie." in text
        assert "Stan techniczny: Uszkodzony" in text
        return DamageAnalysis(
            found_keywords=["po szlifie", "owiewki"],
            suspected_damage=["fairings"],
            hidden_risks=["frame", "forks"],
            starts=False,
            severity="high",
        )

    monkeypatch.setattr(
        "moto_flip_finder.gsxr_ai_analysis.analyze_description",
        fake_analyze_description,
    )

    results = analyze_gsxr_records(
        [
            {
                "title": "Suzuki GSX-R 600 K5 po szlifie",
                "full_description": "Motocykl po szlifie.",
                "short_description": "Cena do negocjacji",
                "attributes": {"Stan techniczny": "Uszkodzony"},
                "normalized_model": "gsxr_600",
            }
        ]
    )

    assert len(results) == 1
    item = results[0]
    assert item["severity"] == "high"
    assert item["starts"] is False
    assert item["detected_damage_signals"] == ["po szlifie", "owiewki"]
    assert item["hidden_risks"] == ["frame", "forks"]
    assert item["has_meaningful_description"] is True
    assert item["review_priority_score"] > 0
    assert "Opis sugeruje problem z odpalaniem" in item["analysis_summary"]


def test_analyze_gsxr_records_marks_sparse_description(monkeypatch):
    monkeypatch.setattr(
        "moto_flip_finder.gsxr_ai_analysis.analyze_description",
        lambda text: DamageAnalysis(severity="unknown"),
    )

    results = analyze_gsxr_records(
        [
            {
                "title": "Suzuki GSX-R 600",
                "short_description": "OK",
                "normalized_model": "gsxr_600",
            }
        ]
    )

    assert results[0]["has_meaningful_description"] is False
    assert "zbyt skapy" in results[0]["analysis_summary"]


def test_summarize_analysis_counts_severity_and_starts():
    summary = summarize_analysis(
        [
            {
                "severity": "high",
                "starts": False,
                "hidden_risks": ["frame"],
                "damage_analysis": {"suspected_damage": ["fairings", "lever"]},
                "original_listing": {"title": "A"},
            },
            {
                "severity": "medium",
                "starts": True,
                "hidden_risks": [],
                "damage_analysis": {"suspected_damage": ["mirror"]},
                "original_listing": {"title": "B"},
            },
        ]
    )

    assert summary["analyzed_count"] == 2
    assert summary["severity_counts"]["high"] == 1
    assert summary["severity_counts"]["medium"] == 1
    assert summary["starts_true"] == 1
    assert summary["starts_false"] == 1
    assert summary["top_5_most_concerning"][0]["original_listing"]["title"] == "A"


def test_rank_records_for_review_uses_priority_and_top_n():
    ranked = rank_records_for_review(
        [
            {
                "review_priority_score": 30,
                "severity": "medium",
                "starts": None,
                "hidden_risks": [],
                "damage_analysis": {"suspected_damage": ["mirror"]},
                "original_listing": {"title": "A"},
            },
            {
                "review_priority_score": 50,
                "severity": "low",
                "starts": False,
                "hidden_risks": ["frame"],
                "damage_analysis": {"suspected_damage": []},
                "original_listing": {"title": "B"},
            },
        ],
        top_n=1,
    )

    assert [item["original_listing"]["title"] for item in ranked] == ["B"]
