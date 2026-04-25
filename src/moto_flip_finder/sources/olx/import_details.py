from __future__ import annotations

from argparse import ArgumentParser
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

from .import_search import (
    _extract_json_ld_blocks,
    _extract_next_data_script,
    _extract_price,
    _normalize_image_urls,
    _normalize_url,
    _pick_string,
    _walk_dicts,
    _walk_json_ld_items,
    detect_damaged_listing,
    detect_gsxr_600,
    fetch_olx_search_page,
    infer_normalized_model,
)

PRICE_IN_LINE_PATTERN = re.compile(r"(\d(?:[\d ]{0,20}\d)?)\s*zł\b", re.IGNORECASE)
UI_IMAGE_MARKERS = [".svg", "app_store", "google_play", "full-screen", "fb."]
NON_LISTING_IMAGE_MARKERS = [
    "avatar",
    "profile",
    "user-photo",
    "user_photo",
    "seller-logo",
    "seller_logo",
    "default-user",
    "default_user",
    "no-photo",
    "no_photo",
    "placeholder",
]


@dataclass
class OlxDetailRecord:
    url: str
    title: str | None
    price_pln: int | None
    location: str | None
    city: str | None
    region: str | None
    short_description: str | None
    full_description: str | None
    image_urls: list[str]
    brand: str | None
    model_family: str | None
    engine_cc: int | None
    normalized_model: str | None
    year: int | None
    technical_state: str | None
    origin_country: str | None
    seller_type: str | None
    mentions_600: bool
    looks_damaged: bool
    attributes: dict[str, str]


def load_records(records_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(records_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError("Records file must contain a 'records' list")
    return [record for record in records if isinstance(record, dict)]


def select_detail_targets(
    records: list[dict[str, Any]],
    only_mentions_600: bool = True,
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for record in records:
        if only_mentions_600 and record.get("mentions_600") is not True:
            continue

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


def build_detail_record(seed_record: dict[str, Any], raw_payloads: dict[str, Any]) -> dict[str, Any]:
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
    location = _detail_location(city, region)
    price_pln = _detail_price(seed_record, next_data, json_ld_blocks, short_description)
    image_urls = _detail_image_urls(seed_record, next_data, json_ld_blocks, soup, url)
    brand = _detail_brand(seed_record, title, attributes)
    engine_cc = _detail_engine_cc(seed_record, title, full_description, short_description, attributes)
    model_family = _detail_model_family(title, full_description, short_description)
    normalized_model = _detail_normalized_model(brand, model_family, engine_cc)
    year = _detail_year(attributes)
    technical_state = _detail_technical_state(attributes)
    origin_country = _detail_origin_country(attributes)
    seller_type = _detail_seller_type(next_data, attributes)

    combined_description = full_description or short_description

    record = OlxDetailRecord(
        url=url,
        title=title,
        price_pln=price_pln,
        location=location,
        city=city,
        region=region,
        short_description=short_description,
        full_description=combined_description,
        image_urls=image_urls,
        brand=brand,
        model_family=model_family,
        engine_cc=engine_cc,
        normalized_model=normalized_model,
        year=year,
        technical_state=technical_state,
        origin_country=origin_country,
        seller_type=seller_type,
        mentions_600=detect_gsxr_600(title, combined_description),
        looks_damaged=detect_damaged_listing(title, combined_description),
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
    details_path = base_dir / f"olx_gsxr_{stamp}_details.json"
    details_path.write_text(
        json.dumps(
            {
                "source": "olx",
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


def import_olx_details(
    records_path: Path,
    output_dir: Path | None = None,
    only_mentions_600: bool = True,
    max_records: int | None = None,
    delay_seconds: float = 0.0,
) -> Path:
    records = load_records(records_path)
    targets = select_detail_targets(
        records,
        only_mentions_600=only_mentions_600,
        max_records=max_records,
    )

    details: list[dict[str, Any]] = []
    for index, record in enumerate(targets):
        try:
            payloads = fetch_detail_payloads(
                record["url"],
                retry_attempts=3,
                retry_delay_seconds=max(delay_seconds, 1.0),
            )
        except (HTTPError, URLError) as exc:
            print(
                f"[olx.import_details] skipped url={record['url']} error={exc}",
                file=sys.stderr,
            )
            continue

        details.append(build_detail_record(record, payloads))

        if delay_seconds > 0 and index < len(targets) - 1:
            sleep(delay_seconds)

    return save_detail_results(records_path, details, output_dir=output_dir)


def _detail_title(
    seed_record: dict[str, Any],
    next_data: Any,
    json_ld_blocks: list[Any],
    soup: BeautifulSoup | None,
) -> str | None:
    title = _seed_string(seed_record.get("title"))
    if title:
        return title

    for item in _walk_json_ld_items(json_ld_blocks):
        title = _pick_string(item, "name", "title")
        if title:
            return title

    if next_data:
        for item in _walk_dicts(next_data):
            title = _pick_string(item, "title", "name")
            if title:
                return title

    if soup:
        meta = soup.find("meta", attrs={"property": "og:title"})
        if meta and meta.get("content"):
            return meta["content"].strip()

    return None


def _detail_description(
    next_data: Any,
    json_ld_blocks: list[Any],
    soup: BeautifulSoup | None,
    fallback: str | None,
) -> str | None:
    if next_data:
        for item in _walk_dicts(next_data):
            description = _pick_string(
                item,
                "description",
                "descriptionShort",
                "description_short",
                "seoDescription",
            )
            if description:
                return description

    if soup:
        meta = soup.find("meta", attrs={"property": "og:description"})
        if meta and meta.get("content"):
            return meta["content"].strip()

        for tag in soup.find_all(["div", "section", "p"]):
            text = " ".join(tag.stripped_strings).strip()
            if text.lower().startswith("opis"):
                return text

    for item in _walk_json_ld_items(json_ld_blocks):
        description = _pick_string(item, "description")
        if description:
            return description

    return fallback


def _detail_location(city: str | None, region: str | None) -> str | None:
    return _join_location(city, region)


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
    if isinstance(seed_price, int):
        return seed_price

    if short_description:
        short_description_price = _extract_price_from_short_description(short_description)
        if short_description_price is not None:
            return short_description_price

    return None


def _detail_image_urls(
    seed_record: dict[str, Any],
    next_data: Any,
    json_ld_blocks: list[Any],
    soup: BeautifulSoup | None,
    detail_url: str,
) -> list[str]:
    image_urls: list[str] = []

    seed_images = seed_record.get("image_urls")
    if isinstance(seed_images, list):
        image_urls.extend(str(url) for url in seed_images if isinstance(url, str) and url)

    for item in _walk_json_ld_items(json_ld_blocks):
        image_urls.extend(_normalize_image_urls(item.get("image"), detail_url))

    if next_data:
        for item in _walk_dicts(next_data):
            image_urls.extend(
                _normalize_image_urls(
                    item.get("images") or item.get("photos") or item.get("image"),
                    detail_url,
                )
            )

    if soup:
        for meta in soup.find_all("meta", attrs={"property": "og:image"}):
            content = meta.get("content", "").strip()
            if content:
                image_urls.append(_normalize_url(content, detail_url) or content)

        for img in soup.find_all("img", src=True):
            src = img.get("src", "").strip()
            if src:
                image_urls.append(_normalize_url(src, detail_url) or src)

    filtered_urls = [
        url
        for url in image_urls
        if isinstance(url, str) and _is_listing_image_url(url)
    ]
    unique_urls = list(dict.fromkeys(filtered_urls))
    return sorted(unique_urls, key=_image_url_sort_key)


def _detail_attributes(next_data: Any, soup: BeautifulSoup | None) -> dict[str, str]:
    attributes: dict[str, str] = {}

    if soup:
        attributes.update(_extract_attributes_from_parameters_container(soup))

    if next_data:
        for item in _walk_dicts(next_data):
            label = _pick_string(item, "label", "key", "name")
            value = _pick_string(item, "value", "valueLabel", "localizedValue", "text")
            if label and value and label not in attributes and label != value:
                attributes[label] = value

    return attributes


def _detail_city_region(
    soup: BeautifulSoup | None,
    json_ld_blocks: list[Any],
) -> tuple[str | None, str | None]:
    if soup is None:
        return _location_from_json_ld(json_ld_blocks)

    location_section = soup.find(attrs={"data-testid": "map-aside-section"})
    if location_section is None:
        return _location_from_json_ld(json_ld_blocks)

    paragraphs = [
        tag.get_text(strip=True)
        for tag in location_section.find_all("p")
        if tag.get_text(strip=True)
    ]
    paragraphs = [value for value in paragraphs if value.lower() != "lokalizacja"]

    city = paragraphs[0] if len(paragraphs) >= 1 else None
    region = paragraphs[1] if len(paragraphs) >= 2 else None

    if city is None and region is None:
        return _location_from_json_ld(json_ld_blocks)

    return city or None, region or None


def _detail_brand(seed_record: dict[str, Any], title: str | None, attributes: dict[str, str]) -> str | None:
    brand = _attribute_value(attributes, "Marka")
    if brand:
        return brand

    for source in (title, _seed_string(seed_record.get("title"))):
        if not source:
            continue
        lowered = source.lower()
        if "suzuki" in lowered:
            return "Suzuki"

    return None


def _detail_model_family(
    title: str | None,
    full_description: str | None,
    short_description: str | None,
) -> str | None:
    combined = " ".join(part for part in [title, full_description, short_description] if part)
    if infer_normalized_model(title, combined) == "gsxr":
        return "gsxr"
    return None


def _detail_engine_cc(
    seed_record: dict[str, Any],
    title: str | None,
    full_description: str | None,
    short_description: str | None,
    attributes: dict[str, str],
) -> int | None:
    for value in (
        _attribute_value(attributes, "Poj. silnika", "Pojemnosc skokowa", "Pojemność skokowa"),
        title,
        full_description,
        short_description,
        _seed_string(seed_record.get("title")),
    ):
        parsed = _parse_engine_cc(value)
        if parsed is not None:
            return parsed
    return None


def _detail_normalized_model(
    brand: str | None,
    model_family: str | None,
    engine_cc: int | None,
) -> str | None:
    if brand != "Suzuki" or model_family != "gsxr":
        return model_family

    if engine_cc is not None:
        return f"gsxr_{engine_cc}"

    return "gsxr"


def _detail_year(attributes: dict[str, str]) -> int | None:
    return _parse_year(_attribute_value(attributes, "Rok produkcji"))


def _detail_technical_state(attributes: dict[str, str]) -> str | None:
    return _attribute_value(attributes, "Stan techniczny")


def _detail_origin_country(attributes: dict[str, str]) -> str | None:
    return _attribute_value(attributes, "Kraj pochodzenia")


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


def _extract_attributes_from_parameters_container(soup: BeautifulSoup) -> dict[str, str]:
    container = soup.find(attrs={"data-testid": "ad-parameters-container"})
    if container is None:
        return {}

    values = [
        text.strip()
        for text in container.stripped_strings
        if text and text.strip()
    ]

    attributes: dict[str, str] = {}
    pending_key: str | None = None

    for value in values:
        if value.lower() == "lokalizacja":
            continue

        if ":" in value:
            key, raw_value = value.split(":", 1)
            key = key.strip()
            raw_value = raw_value.strip()
            if key and raw_value:
                attributes[key] = raw_value
                pending_key = None
                continue
            if key:
                pending_key = key
                continue

        if pending_key:
            attributes[pending_key] = value
            pending_key = None
            continue

        if value in {"Prywatne", "Firma"} and "Typ ogloszeniodawcy" not in attributes:
            attributes["Typ ogloszeniodawcy"] = value

    return attributes


def _seed_string(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _attribute_value(attributes: dict[str, str], *labels: str) -> str | None:
    normalized = {_normalize_label(key): value for key, value in attributes.items()}
    for label in labels:
        value = normalized.get(_normalize_label(label))
        if value:
            return value
    return None


def _normalize_label(label: str) -> str:
    return (
        label.lower()
        .replace("ą", "a")
        .replace("ć", "c")
        .replace("ę", "e")
        .replace("ł", "l")
        .replace("ń", "n")
        .replace("ó", "o")
        .replace("ś", "s")
        .replace("ż", "z")
        .replace("ź", "z")
        .strip()
    )


def _parse_engine_cc(value: str | None) -> int | None:
    if not value:
        return None

    match = re.search(r"(?<!\d)(600|750|1000)(?!\d)", value)
    if match:
        return int(match.group(1))

    match = re.search(r"(\d{2,4})\s*cm3", value.lower())
    if match:
        return int(match.group(1))

    return None


def _parse_year(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"(19\d{2}|20\d{2})", value)
    if match:
        return int(match.group(1))
    return None


def _join_location(city: str | None, region: str | None) -> str | None:
    parts = [part for part in [city, region] if part]
    if parts:
        return ", ".join(parts)
    return None


def _extract_price_from_short_description(text: str) -> int | None:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = PRICE_IN_LINE_PATTERN.search(line)
        if not match:
            continue

        digits = re.sub(r"[^\d]", "", match.group(1))
        if digits:
            return int(digits)

    return None


def _is_listing_image_url(url: str) -> bool:
    lowered = url.lower()
    if any(marker in lowered for marker in UI_IMAGE_MARKERS):
        return False
    if any(marker in lowered for marker in NON_LISTING_IMAGE_MARKERS):
        return False
    return True


def _image_url_sort_key(url: str) -> tuple[int, str]:
    lowered = url.lower()
    score = 0
    if "apollo.olxcdn.com" in lowered or "/v1/files/" in lowered:
        score += 4
    if "img.next" in lowered or "img.jsonld" in lowered or "img.seed" in lowered:
        score += 3
    if "meta" in lowered:
        score -= 1
    return (-score, lowered)


def _location_from_json_ld(json_ld_blocks: list[Any]) -> tuple[str | None, str | None]:
    for item in _walk_json_ld_items(json_ld_blocks):
        area_served = item.get("areaServed")
        if isinstance(area_served, dict):
            name = _pick_string(area_served, "name")
            if name:
                return _split_location_name(name)
        if isinstance(area_served, list):
            for entry in area_served:
                if isinstance(entry, dict):
                    name = _pick_string(entry, "name")
                    if name:
                        return _split_location_name(name)
    return None, None


def _split_location_name(value: str) -> tuple[str | None, str | None]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) >= 2:
        return parts[0], parts[1]
    if len(parts) == 1:
        return parts[0], None
    return None, None


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Import OLX listing details into data/raw/")
    parser.add_argument("--records-file", required=True, help="Path to saved search records JSON")
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Directory for detail JSON output",
    )
    parser.add_argument(
        "--all-records",
        action="store_true",
        help="Import details for all records, not only mentions_600 == true",
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
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    details_path = import_olx_details(
        Path(args.records_file),
        output_dir=Path(args.output_dir),
        only_mentions_600=not args.all_records,
        max_records=args.max_records,
        delay_seconds=args.delay_seconds,
    )
    print(f"Saved detail records to {details_path}")


if __name__ == "__main__":
    main()
