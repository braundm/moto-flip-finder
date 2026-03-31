import json
from pathlib import Path

from moto_flip_finder.build_ready_motorcycles_dataset import (
    build_ready_records,
    build_ready_records_with_brand_filter,
    filter_records_by_keyword,
    save_ready_dataset,
    save_filtered_records,
    summarize_ready_records,
)


def test_build_ready_records_filters_unsensible_and_deduplicates():
    details = [
        {
            "url": "https://example.com/a",
            "title": "A",
            "price_pln": 10000,
            "is_sensible_listing": True,
            "vehicle_type": "motorcycle",
            "brand": "Honda",
        },
        {
            "url": "https://example.com/a",
            "title": "A duplicate",
            "price_pln": 11000,
            "is_sensible_listing": True,
            "vehicle_type": "motorcycle",
            "brand": "Honda",
        },
        {
            "url": "https://example.com/b",
            "title": "B quad",
            "price_pln": 9000,
            "is_sensible_listing": False,
            "vehicle_type": "quad",
            "brand": "Yamaha",
        },
        {
            "url": "https://example.com/c",
            "title": "C",
            "price_pln": 12000,
            "is_sensible_listing": True,
            "vehicle_type": "scooter",
            "brand": "Vespa",
        },
    ]

    ready = build_ready_records(details)

    assert [item["url"] for item in ready] == [
        "https://example.com/a",
        "https://example.com/c",
    ]


def test_summarize_ready_records_counts_vehicle_types_and_brands():
    ready = [
        {"vehicle_type": "motorcycle", "brand": "Honda"},
        {"vehicle_type": "motorcycle", "brand": "Honda"},
        {"vehicle_type": "scooter", "brand": "Vespa"},
    ]
    details = ready + [{"vehicle_type": "quad", "brand": "Yamaha"}]

    summary = summarize_ready_records(ready, details)

    assert summary["detail_count"] == 4
    assert summary["ready_count"] == 3
    assert summary["rejected_count"] == 1
    assert summary["vehicle_type_counts"] == {"motorcycle": 2, "scooter": 1}
    assert summary["top_brands"] == {"Honda": 2, "Vespa": 1}


def test_filter_records_by_keyword_matches_title_and_url():
    records = [
        {"title": "Kawasaki Z750", "url": "https://example.com/a"},
        {"title": "Honda CBR", "url": "https://example.com/b"},
        {"title": "ZX6R", "url": "https://example.com/kawasaki-c"},
    ]

    filtered = filter_records_by_keyword(records, "kawasaki")

    assert [item["url"] for item in filtered] == [
        "https://example.com/a",
        "https://example.com/kawasaki-c",
    ]


def test_build_ready_records_with_brand_filter_keeps_only_required_brand():
    details = [
        {
            "url": "https://example.com/a",
            "title": "A",
            "price_pln": 10000,
            "is_sensible_listing": True,
            "vehicle_type": "motorcycle",
            "brand": "Kawasaki",
        },
        {
            "url": "https://example.com/b",
            "title": "B",
            "price_pln": 11000,
            "is_sensible_listing": True,
            "vehicle_type": "motorcycle",
            "brand": "Honda",
        },
    ]

    ready = build_ready_records_with_brand_filter(details, required_brand="Kawasaki")

    assert [item["url"] for item in ready] == ["https://example.com/a"]


def test_save_filtered_records_writes_keyword_metadata(tmp_path: Path):
    output_path = save_filtered_records(
        records=[{"title": "Kawasaki Z750", "url": "https://example.com/a"}],
        source_records_path=Path("data/raw/all.json"),
        keyword="kawasaki",
        output_dir=tmp_path,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path.name.startswith("olx_motorcycles_kawasaki_")
    assert payload["keyword_filter"] == "kawasaki"
    assert payload["record_count"] == 1


def test_save_ready_dataset_writes_expected_payload(tmp_path: Path):
    ready_records = [
        {
            "source": "olx",
            "url": "https://example.com/a",
            "title": "Honda CBR",
            "price_pln": 10000,
            "vehicle_type": "motorcycle",
            "brand": "Honda",
        }
    ]
    detail_records = ready_records.copy()

    output_path = save_ready_dataset(
        search_url="https://www.olx.pl/motoryzacja/motocykle-skutery/",
        search_records_path=Path("data/raw/records.json"),
        details_path=Path("data/raw/details.json"),
        ready_records=ready_records,
        detail_records=detail_records,
        output_dir=tmp_path,
        dataset_label="kawasaki",
        keyword_filter="kawasaki",
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path.name.startswith("olx_kawasaki_ready_")
    assert payload["search_url"] == "https://www.olx.pl/motoryzacja/motocykle-skutery/"
    assert payload["keyword_filter"] == "kawasaki"
    assert payload["category"] == "kawasaki"
    assert payload["summary"]["ready_count"] == 1
    assert payload["records"] == ready_records
