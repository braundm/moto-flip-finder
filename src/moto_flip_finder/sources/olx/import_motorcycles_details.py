from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import sys
from time import sleep
from typing import Any
from urllib.error import HTTPError, URLError

from bs4 import BeautifulSoup

from moto_flip_finder.motorcycle_listing_validation import (
    MotorcycleListingValidationProvider,
    validate_motorcycle_listing,
)

from .import_details import (
    _attribute_value,
    _detail_attributes,
    _detail_city_region,
    _detail_description,
    _detail_image_urls,
    _detail_title,
    _extract_price_from_short_description,
    _join_location,
    _seed_string,
)
from .import_search import (
    _extract_price,
    _pick_string,
    _walk_dicts,
    _walk_json_ld_items,
    fetch_olx_search_page,
)

ENGINE_CC_PATTERN = re.compile(r"(?<!\d)(\d{2,4})\s*(?:cm3|cm³|cc)\b", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")
BRAND_NORMALIZATION = {
    "aprilia": "Aprilia",
    "barton": "Barton",
    "benelli": "Benelli",
    "beta": "Beta",
    "bmw": "BMW",
    "brixton": "Brixton",
    "cfmoto": "CFMOTO",
    "ducati": "Ducati",
    "fantic": "Fantic",
    "gasgas": "GasGas",
    "generic": "Generic",
    "harley-davidson": "Harley-Davidson",
    "harley davidson": "Harley-Davidson",
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
    "qjmotor": "QJMotor",
    "romet": "Romet",
    "royal enfield": "Royal Enfield",
    "sherco": "Sherco",
    "suzuki": "Suzuki",
    "triumph": "Triumph",
    "vespa": "Vespa",
    "yamaha": "Yamaha",
    "zontes": "Zontes",
}


@dataclass
class OlxMotorcycleDetailRecord:
    url: str
    title: str | None
    price_pln: int | None
    negotiable: bool | None
    location: str | None
    short_description: str | None
    full_description: str | None
    image_urls: list[str]
    brand: str | None
    year: int | None
    technical_state: str | None
    origin_country: str | None
    seller_type: str | None
    engine_cc: int | None
    vehicle_type: str | None
    mileage_km: int | None
    is_sensible_listing: bool | None
    reject_reason: str | None
    validation_confidence: str | None
    validation_summary: str | None
    data_issues: list[str]
    attributes: dict[str, str]


def load_records(records_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(records_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError("Records file must contain a 'records' list")
    return [record for record in records if isinstance(record, dict)]


def select_detail_targets(
    records: list[dict[str, Any]],
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for record in records:
        url = record.get("url")
        if not isinstance(url, str) or not url or url in seen_urls:
            continue

        seen_urls.add(url)
        selected.append(record)

        if max_records is not None and len(selected) >= max_records:
            break

    return selected


def fetch_detail_payloads(
    url: str,
    retry_attempts: int = 3,
    retry_delay_seconds: float = 1.0,
) -> dict[str, Any]:
    from .import_search import _extract_json_ld_blocks, _extract_next_data_script

    last_error: Exception | None = None
    for attempt in range(1, retry_attempts + 1):
        try:
            html = fetch_olx_search_page(url)
            return {
                "html": html,
                "next_data": _extract_next_data_script(html),
                "json_ld": _extract_json_ld_blocks(html),
            }
        except (HTTPError, URLError) as exc:
            last_error = exc
            if attempt < retry_attempts:
                sleep(retry_delay_seconds)

    assert last_error is not None
    raise last_error


def build_detail_record(
    seed_record: dict[str, Any],
    raw_payloads: dict[str, Any],
    validation_provider: MotorcycleListingValidationProvider | None = None,
) -> dict[str, Any]:
    url = str(seed_record.get("url", ""))
    html = raw_payloads.get("html")
    soup = BeautifulSoup(html, "html.parser") if isinstance(html, str) else None
    next_data = raw_payloads.get("next_data")
    json_ld_blocks = raw_payloads.get("json_ld", [])

    title = _detail_title(seed_record, next_data, json_ld_blocks, soup)
    short_description = _seed_string(seed_record.get("short_description"))
    full_description = _detail_description(next_data, json_ld_blocks, soup, short_description)
    attributes = _detail_attributes(next_data, soup)
    city, region = _detail_city_region(soup, json_ld_blocks)
    location = _join_location(city, region)
    price_pln = _detail_price(seed_record, next_data, json_ld_blocks, short_description)
    image_urls = _detail_image_urls(seed_record, next_data, json_ld_blocks, soup, url)
    brand = _detail_brand(attributes, title)
    year = _detail_year(attributes, title)
    technical_state = _attribute_value(attributes, "Stan techniczny")
    origin_country = _attribute_value(attributes, "Kraj pochodzenia")
    seller_type = _detail_seller_type(next_data, attributes)
    engine_cc = _detail_engine_cc(attributes, title)
    ai_validation = _detail_ai_validation(
        title=title,
        price_pln=price_pln,
        location=location,
        short_description=short_description,
        full_description=full_description,
        brand=brand,
        year=year,
        technical_state=technical_state,
        origin_country=origin_country,
        seller_type=seller_type,
        engine_cc=engine_cc,
        attributes=attributes,
        validation_provider=validation_provider,
    )
    price_pln = ai_validation["resolved_price_pln"] or price_pln
    brand = ai_validation["resolved_brand"] or brand
    year = ai_validation["resolved_year"] or year
    engine_cc = ai_validation["resolved_engine_cc"] or engine_cc

    record = OlxMotorcycleDetailRecord(
        url=url,
        title=title,
        price_pln=price_pln,
        negotiable=ai_validation["negotiable"],
        location=location,
        short_description=short_description,
        full_description=full_description or short_description,
        image_urls=image_urls,
        brand=brand,
        year=year,
        technical_state=technical_state,
        origin_country=origin_country,
        seller_type=seller_type,
        engine_cc=engine_cc,
        vehicle_type=ai_validation["vehicle_type"],
        mileage_km=ai_validation["mileage_km"],
        is_sensible_listing=ai_validation["is_sensible_listing"],
        reject_reason=ai_validation["reject_reason"],
        validation_confidence=ai_validation["validation_confidence"],
        validation_summary=ai_validation["validation_summary"],
        data_issues=ai_validation["data_issues"],
        attributes=attributes,
    )
    return asdict(record)


def save_detail_results(
    source_records_path: Path,
    details: list[dict[str, Any]],
    output_dir: Path | None = None,
) -> Path:
    base_dir = output_dir or Path("data/raw")
    base_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    details_path = base_dir / f"olx_motorcycles_{stamp}_details.json"
    details_path.write_text(
        json.dumps(
            {
                "source": "olx",
                "category": "motorcycles",
                "fetched_at": stamp,
                "source_records_file": str(source_records_path),
                "detail_count": len(details),
                "details": details,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return details_path


def import_olx_motorcycles_details(
    records_path: Path,
    output_dir: Path | None = None,
    max_records: int | None = None,
    delay_seconds: float = 0.0,
    validation_provider: MotorcycleListingValidationProvider | None = None,
    max_workers: int = 1,
) -> Path:
    records = load_records(records_path)
    targets = select_detail_targets(records, max_records=max_records)

    if max_workers <= 1:
        details = _import_olx_motorcycles_details_sequential(
            targets,
            delay_seconds=delay_seconds,
            validation_provider=validation_provider,
        )
    else:
        details = _import_olx_motorcycles_details_parallel(
            targets,
            delay_seconds=delay_seconds,
            validation_provider=validation_provider,
            max_workers=max_workers,
        )

    return save_detail_results(records_path, details, output_dir=output_dir)


def _import_olx_motorcycles_details_sequential(
    targets: list[dict[str, Any]],
    *,
    delay_seconds: float,
    validation_provider: MotorcycleListingValidationProvider | None,
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for index, record in enumerate(targets):
        built_record = _fetch_and_build_detail_record(
            record,
            delay_seconds=delay_seconds,
            validation_provider=validation_provider,
        )
        if built_record is not None:
            details.append(built_record)

        processed_count = index + 1
        if processed_count % 10 == 0 or processed_count == len(targets):
            print(
                f"[olx.import_motorcycles_details] processed={processed_count}/{len(targets)} "
                f"saved_in_memory={len(details)}",
                file=sys.stderr,
            )

        if delay_seconds > 0 and index < len(targets) - 1:
            sleep(delay_seconds)

    return details


def _import_olx_motorcycles_details_parallel(
    targets: list[dict[str, Any]],
    *,
    delay_seconds: float,
    validation_provider: MotorcycleListingValidationProvider | None,
    max_workers: int,
) -> list[dict[str, Any]]:
    indexed_results: list[tuple[int, dict[str, Any]]] = []
    processed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(
                _fetch_and_build_detail_record,
                record,
                delay_seconds=delay_seconds,
                validation_provider=validation_provider,
            ): index
            for index, record in enumerate(targets)
        }

        for future in as_completed(future_to_index):
            processed_count += 1
            index = future_to_index[future]
            built_record = future.result()
            if built_record is not None:
                indexed_results.append((index, built_record))

            if processed_count % 10 == 0 or processed_count == len(targets):
                print(
                    f"[olx.import_motorcycles_details] processed={processed_count}/{len(targets)} "
                    f"saved_in_memory={len(indexed_results)}",
                    file=sys.stderr,
                )

    indexed_results.sort(key=lambda item: item[0])
    return [record for _, record in indexed_results]


def _fetch_and_build_detail_record(
    record: dict[str, Any],
    *,
    delay_seconds: float,
    validation_provider: MotorcycleListingValidationProvider | None,
) -> dict[str, Any] | None:
    try:
        payloads = fetch_detail_payloads(
            record["url"],
            retry_attempts=3,
            retry_delay_seconds=max(delay_seconds, 1.0),
        )
    except (HTTPError, URLError) as exc:
        print(
            f"[olx.import_motorcycles_details] skipped url={record['url']} error={exc}",
            file=sys.stderr,
        )
        return None

    return build_detail_record(
        record,
        payloads,
        validation_provider=validation_provider,
    )


def _detail_price(
    seed_record: dict[str, Any],
    next_data: Any,
    json_ld_blocks: list[Any],
    short_description: str | None,
) -> int | None:
    for item in _walk_json_ld_items(json_ld_blocks):
        price = _extract_price(item)
        if price is not None:
            return price

    if next_data:
        for item in _walk_dicts(next_data):
            price = _extract_price(item)
            if price is not None:
                return price

    seed_price = seed_record.get("price_pln")
    if isinstance(seed_price, int) and seed_price > 1:
        return seed_price

    if short_description:
        return _extract_price_from_short_description(short_description)

    return None


def _detail_brand(attributes: dict[str, str], title: str | None) -> str | None:
    brand = _attribute_value(attributes, "Marka")
    if brand:
        return _normalize_brand(brand)

    if not title:
        return None

    lowered_title = title.lower()
    for raw_brand, normalized in BRAND_NORMALIZATION.items():
        if re.search(rf"\b{re.escape(raw_brand)}\b", lowered_title):
            return normalized

    return None


def _detail_year(attributes: dict[str, str], title: str | None) -> int | None:
    for value in (
        _attribute_value(attributes, "Rok produkcji"),
        title,
    ):
        parsed = _parse_year(value)
        if parsed is not None:
            return parsed
    return None


def _detail_seller_type(next_data: Any, attributes: dict[str, str]) -> str | None:
    seller_type = _attribute_value(attributes, "Typ ogłoszeniodawcy", "Typ ogloszeniodawcy")
    if seller_type:
        return seller_type

    if next_data:
        for item in _walk_dicts(next_data):
            seller_type = _pick_string(item, "sellerType", "userType", "accountType")
            if seller_type:
                return seller_type

    return None


def _detail_engine_cc(attributes: dict[str, str], title: str | None) -> int | None:
    for value in (
        _attribute_value(attributes, "Poj. silnika", "Pojemnosc skokowa", "Pojemność skokowa"),
        title,
    ):
        parsed = _parse_engine_cc(value)
        if parsed is not None:
            return parsed
    return None


def _detail_ai_validation(
    *,
    title: str | None,
    price_pln: int | None,
    location: str | None,
    short_description: str | None,
    full_description: str | None,
    brand: str | None,
    year: int | None,
    technical_state: str | None,
    origin_country: str | None,
    seller_type: str | None,
    engine_cc: int | None,
    attributes: dict[str, str],
    validation_provider: MotorcycleListingValidationProvider | None = None,
) -> dict[str, Any]:
    return validate_motorcycle_listing(
        {
            "title": title,
            "price_pln": price_pln,
            "location": location,
            "short_description": short_description,
            "full_description": full_description,
            "brand": brand,
            "year": year,
            "technical_state": technical_state,
            "origin_country": origin_country,
            "seller_type": seller_type,
            "engine_cc": engine_cc,
            "structured_brand_missing_or_generic": brand is None or str(brand).strip().lower() in {"inna", "generic"},
            "attributes": attributes,
        },
        provider=validation_provider,
    )


def _normalize_brand(value: str) -> str:
    lowered = value.strip().lower()
    return BRAND_NORMALIZATION.get(lowered, value.strip())


def _parse_engine_cc(value: str | None) -> int | None:
    if not value:
        return None

    match = ENGINE_CC_PATTERN.search(value)
    if not match:
        return None

    parsed = int(match.group(1))
    if 40 <= parsed <= 2500:
        return parsed
    return None


def _parse_year(value: str | None) -> int | None:
    if not value:
        return None

    match = YEAR_PATTERN.search(value)
    if match:
        return int(match.group(1))
    return None


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Import OLX motorcycle listing details into data/raw/")
    parser.add_argument("--records-file", required=True, help="Path to saved motorcycle search records JSON")
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Directory for detail JSON output",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap on the number of detail pages to fetch",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
        help="Optional delay between detail page requests",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of concurrent detail workers",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    details_path = import_olx_motorcycles_details(
        Path(args.records_file),
        output_dir=Path(args.output_dir),
        max_records=args.max_records,
        delay_seconds=args.delay_seconds,
        max_workers=max(1, args.max_workers),
    )
    print(f"Saved detail records to {details_path}")


if __name__ == "__main__":
    main()
