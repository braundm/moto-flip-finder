from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from time import sleep
from typing import Any

from .import_search import (
    _build_short_description,
    _collapse_text,
    _extract_image_urls_from_card,
    _extract_json_ld_blocks,
    _extract_location,
    _extract_location_from_text,
    _extract_next_data_script,
    _extract_price,
    _extract_price_from_text,
    _is_olx_offer_url,
    _normalize_image_urls,
    _normalize_url,
    _pick_string,
    _walk_dicts,
    _walk_json_ld_items,
    build_search_page_url,
    fetch_olx_search_page,
)

@dataclass
class OlxMotorcycleSearchRecord:
    title: str
    price_pln: int | None
    negotiable: bool | None
    url: str
    location: str | None
    image_urls: list[str]
    short_description: str | None



def extract_raw_payloads(html: str) -> dict[str, Any]:
    return {
        "html": html,
        "next_data": _extract_next_data_script(html),
        "json_ld": _extract_json_ld_blocks(html),
    }


def build_structured_records(raw_payloads: dict[str, Any], search_url: str) -> list[dict[str, Any]]:
    records: list[OlxMotorcycleSearchRecord] = []
    seen_urls: set[str] = set()

    html = raw_payloads.get("html")
    if isinstance(html, str):
        for item in _extract_from_html_cards(html, search_url):
            if item.url in seen_urls:
                continue
            seen_urls.add(item.url)
            records.append(item)

    for item in _extract_from_json_ld(raw_payloads.get("json_ld", []), search_url):
        if item.url in seen_urls:
            continue
        seen_urls.add(item.url)
        records.append(item)

    for item in _extract_from_next_data(raw_payloads.get("next_data"), search_url):
        if item.url in seen_urls:
            continue
        seen_urls.add(item.url)
        records.append(item)

    return [asdict(record) for record in records]


def save_import_results(
    search_url: str,
    raw_payloads: dict[str, Any],
    records: list[dict[str, Any]],
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    base_dir = output_dir or Path("data/raw")
    base_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    raw_path = base_dir / f"olx_motorcycles_{stamp}_raw.json"
    records_path = base_dir / f"olx_motorcycles_{stamp}_records.json"

    raw_path.write_text(
        json.dumps(
            {
                "source": "olx",
                "category": "motorcycles",
                "search_url": search_url,
                "fetched_at": stamp,
                "record_count": len(records),
                "raw_payloads": raw_payloads,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    records_path.write_text(
        json.dumps(
            {
                "source": "olx",
                "category": "motorcycles",
                "search_url": search_url,
                "fetched_at": stamp,
                "record_count": len(records),
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return raw_path, records_path


def import_olx_motorcycles_search(
    search_url: str,
    output_dir: Path | None = None,
    max_pages: int = 1,
    delay_seconds: float = 0.0,
) -> tuple[Path, Path]:
    page_payloads: list[dict[str, Any]] = []
    records_by_url: dict[str, dict[str, Any]] = {}

    for page_number in range(1, max_pages + 1):
        page_url = build_search_page_url(search_url, page_number)
        html = fetch_olx_search_page(page_url)
        raw_payloads = extract_raw_payloads(html)
        page_records = build_structured_records(raw_payloads, page_url)

        new_records = 0
        for record in page_records:
            url = record.get("url")
            if not isinstance(url, str) or not url:
                continue
            if url not in records_by_url:
                records_by_url[url] = record
                new_records += 1

        page_payloads.append(
            {
                "page_number": page_number,
                "page_url": page_url,
                "record_count": len(page_records),
                "new_record_count": new_records,
                "raw_payloads": raw_payloads,
            }
        )

        if not page_records or new_records == 0:
            break

        if delay_seconds > 0 and page_number < max_pages:
            sleep(delay_seconds)

    aggregated_raw_payloads = {
        "pages": page_payloads,
        "pages_fetched": len(page_payloads),
    }
    records = list(records_by_url.values())
    return save_import_results(search_url, aggregated_raw_payloads, records, output_dir=output_dir)


def _extract_from_html_cards(html: str, search_url: str) -> list[OlxMotorcycleSearchRecord]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    records: list[OlxMotorcycleSearchRecord] = []
    seen_urls: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "").strip()
        title = " ".join(anchor.stripped_strings).strip()
        if not title:
            continue

        full_url = _normalize_url(href, search_url)
        if not full_url or not _is_olx_offer_url(full_url) or full_url in seen_urls:
            continue

        card = anchor.find_parent(["li", "article", "div"])
        if card is None:
            card = anchor.parent

        card_text = _collapse_text(card.get_text("\n", strip=True) if card else title)
        price_pln = _extract_price_from_text(card_text)
        location = _extract_location_from_text(card_text)
        image_urls = _extract_image_urls_from_card(card, search_url)
        short_description = _build_short_description(card_text, title, location, price_pln)
        records.append(
            OlxMotorcycleSearchRecord(
                title=title,
                price_pln=price_pln,
                negotiable=None,
                url=full_url,
                location=location,
                image_urls=image_urls,
                short_description=short_description,
            )
        )
        seen_urls.add(full_url)

    return records


def _extract_from_json_ld(blocks: list[Any], search_url: str) -> list[OlxMotorcycleSearchRecord]:
    records: list[OlxMotorcycleSearchRecord] = []

    for block in blocks:
        for item in _walk_json_ld_items(block):
            title = _pick_string(item, "name", "title")
            item_url = _normalize_url(_pick_string(item, "url"), search_url)
            if not title or not item_url or not _is_olx_offer_url(item_url):
                continue

            description = _pick_string(item, "description")
            image_urls = _normalize_image_urls(item.get("image"), search_url)
            location = _extract_location(item)
            price_pln = _extract_price(item)
            records.append(
                OlxMotorcycleSearchRecord(
                    title=title,
                    price_pln=price_pln,
                    negotiable=None,
                    url=item_url,
                    location=location,
                    image_urls=image_urls,
                    short_description=description,
                )
            )

    return records


def _extract_from_next_data(next_data: Any, search_url: str) -> list[OlxMotorcycleSearchRecord]:
    if not next_data:
        return []

    records: list[OlxMotorcycleSearchRecord] = []
    for item in _walk_dicts(next_data):
        title = _pick_string(item, "title", "name")
        item_url = _normalize_url(
            _pick_string(item, "url", "href", "absolute_url", "full_url"),
            search_url,
        )
        if not title or not item_url or not _is_olx_offer_url(item_url):
            continue

        description = _pick_string(
            item,
            "description",
            "descriptionShort",
            "description_short",
            "snippet",
        )
        price_pln = _extract_price(item)
        location = _extract_location(item)
        image_urls = _normalize_image_urls(
            item.get("images") or item.get("photos") or item.get("image"),
            search_url,
        )
        records.append(
            OlxMotorcycleSearchRecord(
                title=title,
                price_pln=price_pln,
                negotiable=None,
                url=item_url,
                location=location,
                image_urls=image_urls,
                short_description=description,
            )
        )

    return records
def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Import OLX motorcycle search results into data/raw/")
    parser.add_argument("--url", required=True, help="OLX motorcycles category or search URL")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help="Maximum number of OLX result pages to fetch",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
        help="Optional delay between page requests",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Directory for raw and structured JSON output",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    raw_path, records_path = import_olx_motorcycles_search(
        args.url,
        Path(args.output_dir),
        max_pages=args.max_pages,
        delay_seconds=args.delay_seconds,
    )
    print(f"Saved raw payloads to {raw_path}")
    print(f"Saved structured records to {records_path}")


if __name__ == "__main__":
    main()
