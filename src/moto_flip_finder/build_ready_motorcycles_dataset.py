from __future__ import annotations

from argparse import ArgumentParser
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from moto_flip_finder.sources.olx.import_motorcycles_details import (
    import_olx_motorcycles_details,
)
from moto_flip_finder.sources.olx.import_motorcycles_search import (
    import_olx_motorcycles_search,
)


def load_records(records_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(records_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError("Records file must contain a 'records' list")
    return [item for item in records if isinstance(item, dict)]


def filter_records_by_keyword(records: list[dict[str, Any]], keyword: str | None) -> list[dict[str, Any]]:
    if not keyword:
        return records

    normalized_keyword = keyword.strip().casefold()
    if not normalized_keyword:
        return records

    filtered: list[dict[str, Any]] = []
    for record in records:
        haystack = " ".join(
            str(record.get(field, "")).casefold()
            for field in ("title", "short_description", "url")
        )
        if normalized_keyword in haystack:
            filtered.append(record)
    return filtered


def save_filtered_records(
    *,
    records: list[dict[str, Any]],
    source_records_path: Path,
    keyword: str,
    output_dir: Path | None = None,
) -> Path:
    base_dir = output_dir or Path("data/raw")
    base_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_keyword = "_".join(keyword.strip().casefold().split())
    output_path = base_dir / f"olx_motorcycles_{safe_keyword}_{stamp}_records.json"
    output_path.write_text(
        json.dumps(
            {
                "source": "olx",
                "category": "motorcycles",
                "created_at": stamp,
                "keyword_filter": keyword,
                "source_records_file": str(source_records_path),
                "record_count": len(records),
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path


def load_details(details_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(details_path.read_text(encoding="utf-8"))
    details = payload.get("details", [])
    if not isinstance(details, list):
        raise ValueError("Details file must contain a 'details' list")
    return [item for item in details if isinstance(item, dict)]


def build_ready_records(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return build_ready_records_with_brand_filter(details, required_brand=None)


def build_ready_records_with_brand_filter(
    details: list[dict[str, Any]],
    *,
    required_brand: str | None,
) -> list[dict[str, Any]]:
    ready: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    normalized_required_brand = required_brand.casefold() if isinstance(required_brand, str) and required_brand.strip() else None

    for item in details:
        url = item.get("url")
        if not isinstance(url, str) or not url or url in seen_urls:
            continue

        if not _is_ready_listing_candidate(item):
            continue

        brand = item.get("brand")
        if normalized_required_brand is not None:
            if not isinstance(brand, str) or brand.casefold() != normalized_required_brand:
                continue

        seen_urls.add(url)
        ready.append(
            {
                "source": "olx",
                "url": url,
                "title": item.get("title"),
                "price_pln": item.get("price_pln"),
                "negotiable": item.get("negotiable"),
                "location": item.get("location"),
                "short_description": item.get("short_description"),
                "full_description": item.get("full_description"),
                "image_urls": item.get("image_urls", []),
                "brand": brand,
                "year": item.get("year"),
                "technical_state": item.get("technical_state"),
                "origin_country": item.get("origin_country"),
                "seller_type": item.get("seller_type"),
                "engine_cc": item.get("engine_cc"),
                "vehicle_type": item.get("vehicle_type"),
                "mileage_km": item.get("mileage_km"),
                "validation_confidence": item.get("validation_confidence"),
                "validation_summary": item.get("validation_summary"),
                "data_issues": item.get("data_issues", []),
            }
        )

    return ready


def _is_ready_listing_candidate(item: dict[str, Any]) -> bool:
    if item.get("is_sensible_listing") is False:
        return False

    vehicle_type = item.get("vehicle_type")
    if vehicle_type not in {"motorcycle", "scooter"}:
        return False

    if not isinstance(item.get("title"), str) or not item["title"].strip():
        return False

    if not isinstance(item.get("brand"), str) or not item["brand"].strip():
        return False

    price_pln = item.get("price_pln")
    if not isinstance(price_pln, int) or price_pln <= 0:
        return False

    return item.get("is_sensible_listing") is True or vehicle_type in {"motorcycle", "scooter"}


def summarize_ready_records(ready_records: list[dict[str, Any]], detail_records: list[dict[str, Any]]) -> dict[str, Any]:
    vehicle_type_counts: dict[str, int] = {}
    brand_counts: dict[str, int] = {}

    for item in ready_records:
        vehicle_type = item.get("vehicle_type") or "unknown"
        vehicle_type_counts[vehicle_type] = vehicle_type_counts.get(vehicle_type, 0) + 1

        brand = item.get("brand") or "unknown"
        brand_counts[brand] = brand_counts.get(brand, 0) + 1

    return {
        "detail_count": len(detail_records),
        "ready_count": len(ready_records),
        "rejected_count": len(detail_records) - len(ready_records),
        "vehicle_type_counts": dict(sorted(vehicle_type_counts.items())),
        "top_brands": dict(
            sorted(
                brand_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:10]
        ),
    }


def save_ready_dataset(
    *,
    search_url: str,
    search_records_path: Path,
    details_path: Path,
    ready_records: list[dict[str, Any]],
    detail_records: list[dict[str, Any]],
    output_dir: Path | None = None,
    dataset_label: str = "motorcycles",
    keyword_filter: str | None = None,
) -> Path:
    base_dir = output_dir or Path("data/ready")
    base_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_label = "_".join(dataset_label.strip().casefold().split()) or "motorcycles"
    output_path = base_dir / f"olx_{safe_label}_ready_{stamp}.json"
    summary = summarize_ready_records(ready_records, detail_records)

    output_path.write_text(
        json.dumps(
            {
                "source": "olx",
                "category": dataset_label,
                "created_at": stamp,
                "search_url": search_url,
                "keyword_filter": keyword_filter,
                "source_records_file": str(search_records_path),
                "source_details_file": str(details_path),
                "summary": summary,
                "records": ready_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path


def build_ready_motorcycles_dataset(
    *,
    search_url: str,
    max_pages: int,
    max_records: int,
    raw_output_dir: Path | None = None,
    ready_output_dir: Path | None = None,
    search_delay_seconds: float = 0.0,
    detail_delay_seconds: float = 0.0,
    detail_max_workers: int = 1,
    dataset_label: str = "motorcycles",
    keyword_filter: str | None = None,
    required_brand: str | None = None,
) -> tuple[Path, Path, Path]:
    _, imported_records_path = import_olx_motorcycles_search(
        search_url,
        output_dir=raw_output_dir,
        max_pages=max_pages,
        delay_seconds=search_delay_seconds,
    )
    records_path = imported_records_path
    if keyword_filter:
        filtered_records = filter_records_by_keyword(
            load_records(imported_records_path),
            keyword_filter,
        )
        records_path = save_filtered_records(
            records=filtered_records,
            source_records_path=imported_records_path,
            keyword=keyword_filter,
            output_dir=raw_output_dir,
        )
    details_path = import_olx_motorcycles_details(
        records_path,
        output_dir=raw_output_dir,
        max_records=max_records,
        delay_seconds=detail_delay_seconds,
        max_workers=detail_max_workers,
    )
    detail_records = load_details(details_path)
    ready_records = build_ready_records_with_brand_filter(
        detail_records,
        required_brand=required_brand,
    )
    ready_path = save_ready_dataset(
        search_url=search_url,
        search_records_path=records_path,
        details_path=details_path,
        ready_records=ready_records,
        detail_records=detail_records,
        output_dir=ready_output_dir,
        dataset_label=dataset_label,
        keyword_filter=keyword_filter,
    )
    return records_path, details_path, ready_path


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build final ready motorcycle data from OLX with heuristic validation")
    parser.add_argument(
        "--url",
        default="https://www.olx.pl/motoryzacja/motocykle-skutery/",
        help="OLX motorcycles category or search URL",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="Maximum number of search result pages to fetch",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=500,
        help="Maximum number of detail records to fetch and validate",
    )
    parser.add_argument(
        "--search-delay-seconds",
        type=float,
        default=0.5,
        help="Delay between search page requests",
    )
    parser.add_argument(
        "--detail-delay-seconds",
        type=float,
        default=0.5,
        help="Delay between detail page requests",
    )
    parser.add_argument(
        "--detail-max-workers",
        type=int,
        default=6,
        help="Number of concurrent workers for detail scraping and validation",
    )
    parser.add_argument(
        "--raw-output-dir",
        default="data/raw",
        help="Directory for raw search/detail JSON output",
    )
    parser.add_argument(
        "--ready-output-dir",
        default="data/ready",
        help="Directory for final ready dataset JSON output",
    )
    parser.add_argument(
        "--dataset-label",
        default="motorcycles",
        help="Label used in the ready dataset filename and metadata",
    )
    parser.add_argument(
        "--keyword-filter",
        default=None,
        help="Optional cheap keyword filter on search records before detail scraping, e.g. kawasaki",
    )
    parser.add_argument(
        "--required-brand",
        default=None,
        help="Optional exact brand required in the final ready dataset, e.g. Kawasaki",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    records_path, details_path, ready_path = build_ready_motorcycles_dataset(
        search_url=args.url,
        max_pages=args.max_pages,
        max_records=args.max_records,
        raw_output_dir=Path(args.raw_output_dir),
        ready_output_dir=Path(args.ready_output_dir),
        search_delay_seconds=args.search_delay_seconds,
        detail_delay_seconds=args.detail_delay_seconds,
        detail_max_workers=max(1, args.detail_max_workers),
        dataset_label=args.dataset_label,
        keyword_filter=args.keyword_filter,
        required_brand=args.required_brand,
    )
    detail_records = load_details(details_path)
    ready_records = build_ready_records_with_brand_filter(
        detail_records,
        required_brand=args.required_brand,
    )
    summary = summarize_ready_records(ready_records, detail_records)

    print(f"Saved search records to {records_path}")
    print(f"Saved detail records to {details_path}")
    print(f"Saved ready dataset to {ready_path}")
    print(
        "Ready summary: "
        f"details={summary['detail_count']} "
        f"ready={summary['ready_count']} "
        f"rejected={summary['rejected_count']}"
    )


if __name__ == "__main__":
    main()
