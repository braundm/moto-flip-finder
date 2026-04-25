from pathlib import Path

from moto_flip_finder.ready_price_report import (
    rank_ready_price_predictions,
    save_ready_price_report,
    summarize_ready_price_predictions,
)


def _prediction(
    title: str,
    *,
    listed: int,
    predicted: int,
    delta: int,
    url: str = "https://example.com/oferta",
    image_url: str = "https://example.com/image.jpg",
) -> dict:
    return {
        "original_listing": {
            "title": title,
            "url": url,
            "location": "Warszawa",
            "price_pln": listed,
            "year": 2015,
            "engine_cc": 650,
            "mileage_km": 22000,
            "validation_summary": "Looks consistent for a healthy street bike.",
            "image_urls": [image_url],
        },
        "listed_price_pln": listed,
        "predicted_price_pln": predicted,
        "price_delta_pln": delta,
    }


def test_rank_ready_price_predictions_prefers_positive_delta():
    ranked = rank_ready_price_predictions(
        [
            _prediction("A", listed=20000, predicted=22000, delta=2000),
            _prediction("B", listed=20000, predicted=25000, delta=5000),
            _prediction("C", listed=20000, predicted=19000, delta=-1000),
        ],
        limit=2,
    )

    assert [item["original_listing"]["title"] for item in ranked] == ["B", "A"]


def test_summarize_ready_price_predictions_counts_positive_and_negative():
    summary = summarize_ready_price_predictions(
        [
            _prediction("A", listed=20000, predicted=22000, delta=2000),
            _prediction("B", listed=20000, predicted=18000, delta=-2000),
            {"original_listing": {"title": "C"}},
        ]
    )

    assert summary["total_predictions"] == 3
    assert summary["predictions_with_delta"] == 2
    assert summary["positive_delta_count"] == 1
    assert summary["negative_delta_count"] == 1
    assert summary["best_positive_delta_pln"] == 2000


def test_save_ready_price_report_writes_html_with_links_and_image(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        "moto_flip_finder.ready_price_report._download_primary_image",
        lambda prediction, assets_dir, report_dir, index: "assets/ready_price_01.jpg",
    )

    report_path = save_ready_price_report(
        [
            _prediction(
                "Kawasaki Ninja 650",
                listed=21900,
                predicted=24500,
                delta=2600,
            )
        ],
        tmp_path / "ready_price_report.html",
        assets_dir=tmp_path / "assets",
        brand="Kawasaki",
        backend="torch",
        dataset_label="kawasaki_all_pages",
    )

    html = report_path.read_text(encoding="utf-8")

    assert "Ready Price Opportunities: Kawasaki" in html
    assert "backend=torch" in html
    assert "Kawasaki Ninja 650" in html
    assert "assets/ready_price_01.jpg" in html
    assert "https://example.com/oferta" in html
    assert "Price delta:" in html


def test_primary_image_url_prefers_listing_photo_over_avatar(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "moto_flip_finder.ready_price_report._download_primary_image",
        lambda prediction, assets_dir, report_dir, index: None,
    )

    report_path = save_ready_price_report(
        [
            {
                "original_listing": {
                    "title": "Kawasaki Z750",
                    "url": "https://example.com/oferta",
                    "location": "Warszawa",
                    "price_pln": 18000,
                    "image_urls": [
                        "https://example.com/profile-avatar.jpg",
                        "https://ireland.apollo.olxcdn.com/v1/files/moto-photo",
                    ],
                },
                "listed_price_pln": 18000,
                "predicted_price_pln": 20500,
                "price_delta_pln": 2500,
            }
        ],
        tmp_path / "ready_price_report.html",
        assets_dir=tmp_path / "assets",
    )

    html = report_path.read_text(encoding="utf-8")

    assert "https://ireland.apollo.olxcdn.com/v1/files/moto-photo" in html
    assert "profile-avatar.jpg" not in html
