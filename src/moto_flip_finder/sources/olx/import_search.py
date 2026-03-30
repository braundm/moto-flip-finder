from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from html import unescape
import json
from pathlib import Path
import re
from time import sleep
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup


GSXR_PATTERN = re.compile(r"\bgsx[\s-]?r\b", re.IGNORECASE)
MODEL_600_PATTERN = re.compile(r"(?<!\d)600(?!\d)")
PRICE_PATTERN = re.compile(r"^(\d(?:[\d ]{0,20}\d)?)\s*zł\b", re.IGNORECASE)

DAMAGED_KEYWORDS = [
    "uszkodz",
    "po szlifie",
    "po dzwonie",
    "rozbit",
    "do naprawy",
    "nie odpala",
    "krzywy",
    "bez prawa",
]


@dataclass
class OlxSearchRecord:
    title: str
    price_pln: int | None
    url: str
    location: str | None
    image_urls: list[str]
    short_description: str | None
    normalized_model: str | None
    mentions_600: bool
    looks_damaged: bool


def normalize_gsxr_text(text: str) -> str:
    return GSXR_PATTERN.sub("gsxr", text.lower())


def detect_gsxr_600(title: str | None, description: str | None) -> bool:
    combined = normalize_gsxr_text(" ".join(part for part in [title, description] if part))
    return "gsxr" in combined and bool(MODEL_600_PATTERN.search(combined))


def detect_damaged_listing(title: str | None, description: str | None) -> bool:
    combined = " ".join(part for part in [title, description] if part).lower()
    return any(keyword in combined for keyword in DAMAGED_KEYWORDS)


def infer_normalized_model(title: str | None, description: str | None) -> str | None:
    combined = normalize_gsxr_text(" ".join(part for part in [title, description] if part))
    if "gsxr" in combined:
        return "gsxr"
    return None


def fetch_olx_search_page(search_url: str, timeout: int = 30) -> str:
    request = Request(
        search_url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0 Safari/537.36"
            )
        },
    )
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def extract_raw_payloads(html: str) -> dict[str, Any]:
    next_data = _extract_next_data_script(html)
    json_ld_blocks = _extract_json_ld_blocks(html)

    return {
        "html": html,
        "next_data": next_data,
        "json_ld": json_ld_blocks,
    }


def build_structured_records(raw_payloads: dict[str, Any], search_url: str) -> list[dict[str, Any]]:
    records: list[OlxSearchRecord] = []
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
    raw_path = base_dir / f"olx_gsxr_{stamp}_raw.json"
    records_path = base_dir / f"olx_gsxr_{stamp}_records.json"

    raw_path.write_text(
        json.dumps(
            {
                "source": "olx",
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


def build_search_page_url(search_url: str, page_number: int) -> str:
    if page_number <= 1:
        return search_url

    parsed = urlparse(search_url)
    query_items = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query_items["page"] = str(page_number)
    return urlunparse(parsed._replace(query=urlencode(query_items)))


def import_olx_search(
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


def _extract_from_html_cards(html: str, search_url: str) -> list[OlxSearchRecord]:
    soup = BeautifulSoup(html, "html.parser")
    records: list[OlxSearchRecord] = []
    seen_urls: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "").strip()
        title = " ".join(anchor.stripped_strings).strip()

        if not title:
            continue

        full_url = _normalize_url(href, search_url)
        if not full_url:
            continue

        if not _is_olx_offer_url(full_url):
            continue

        if full_url in seen_urls:
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
            _build_record(
                title=title,
                price_pln=price_pln,
                item_url=full_url,
                location=location,
                image_urls=image_urls,
                short_description=short_description,
            )
        )
        seen_urls.add(full_url)

    return records


def _is_olx_offer_url(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()

    if "olx.pl" not in host:
        return False

    return "/d/oferta/" in path or "/oferta/" in path


def _extract_price_from_text(text: str) -> int | None:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = PRICE_PATTERN.match(line)
        if not match:
            continue

        digits = re.sub(r"[^\d]", "", match.group(1))
        if digits:
            return int(digits)

    return None


def _extract_location_from_text(text: str) -> str | None:
    for line in [line.strip() for line in text.split("\n") if line.strip()]:
        if " - " in line and (
            "dzisiaj" in line.lower()
            or "odświeżono" in line.lower()
            or re.search(r"\d{4}", line)
        ):
            return line.split(" - ", 1)[0].strip()

    return None


def _extract_image_urls_from_card(card: Any, search_url: str) -> list[str]:
    if card is None:
        return []

    image_urls: list[str] = []

    for img in card.find_all("img", src=True):
        src = img.get("src", "").strip()
        if src:
            image_urls.append(_normalize_url(src, search_url) or src)

    for img in card.find_all("img", attrs={"data-src": True}):
        src = img.get("data-src", "").strip()
        if src:
            image_urls.append(_normalize_url(src, search_url) or src)

    return sorted(set(url for url in image_urls if url))


def _build_short_description(
    card_text: str,
    title: str,
    location: str | None,
    price_pln: int | None,
) -> str | None:
    lines = [line.strip() for line in card_text.split("\n") if line.strip()]
    cleaned: list[str] = []

    for line in lines:
        if line == title:
            continue
        if location and line.startswith(location):
            continue
        if price_pln is not None and str(price_pln) in re.sub(r"[^\d]", "", line):
            continue
        if line.lower() == "obserwuj":
            continue
        cleaned.append(line)

    if not cleaned:
        return None

    return " | ".join(cleaned[:4])


def _collapse_text(text: str) -> str:
    return re.sub(r"\n{2,}", "\n", text).strip()


def _extract_next_data_script(html: str) -> dict[str, Any] | None:
    match = re.search(
        r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>',
        html,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return None

    raw = unescape(match.group(1).strip())
    if not raw:
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if isinstance(payload, dict):
        return payload
    return None


def _extract_json_ld_blocks(html: str) -> list[Any]:
    blocks: list[Any] = []
    for raw in re.findall(
        r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>',
        html,
        re.DOTALL | re.IGNORECASE,
    ):
        text = unescape(raw.strip())
        if not text:
            continue
        try:
            blocks.append(json.loads(text))
        except json.JSONDecodeError:
            continue
    return blocks


def _extract_from_json_ld(blocks: list[Any], search_url: str) -> list[OlxSearchRecord]:
    records: list[OlxSearchRecord] = []

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
                _build_record(
                    title=title,
                    price_pln=price_pln,
                    item_url=item_url,
                    location=location,
                    image_urls=image_urls,
                    short_description=description,
                )
            )

    return records


def _extract_from_next_data(next_data: Any, search_url: str) -> list[OlxSearchRecord]:
    if not next_data:
        return []

    records: list[OlxSearchRecord] = []
    for item in _walk_dicts(next_data):
        title = _pick_string(item, "title", "name")
        item_url = _normalize_url(
            _pick_string(item, "url", "href", "absolute_url", "full_url"),
            search_url,
        )
        price_pln = _extract_price(item)

        if not title or not item_url or price_pln is None or not _is_olx_offer_url(item_url):
            continue

        description = _pick_string(
            item,
            "description",
            "descriptionShort",
            "description_short",
            "snippet",
        )
        location = _extract_location(item)
        image_urls = _normalize_image_urls(
            item.get("images") or item.get("photos") or item.get("image"),
            search_url,
        )

        records.append(
            _build_record(
                title=title,
                price_pln=price_pln,
                item_url=item_url,
                location=location,
                image_urls=image_urls,
                short_description=description,
            )
        )

    return records


def _build_record(
    *,
    title: str,
    price_pln: int | None,
    item_url: str,
    location: str | None,
    image_urls: list[str],
    short_description: str | None,
) -> OlxSearchRecord:
    return OlxSearchRecord(
        title=title,
        price_pln=price_pln,
        url=item_url,
        location=location,
        image_urls=image_urls,
        short_description=short_description,
        normalized_model=infer_normalized_model(title, short_description),
        mentions_600=detect_gsxr_600(title, short_description),
        looks_damaged=detect_damaged_listing(title, short_description),
    )


def _walk_json_ld_items(value: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if value.get("@type") in {"ListItem", "Product", "Offer"}:
            items.append(value)
        for nested in value.values():
            items.extend(_walk_json_ld_items(nested))
    elif isinstance(value, list):
        for nested in value:
            items.extend(_walk_json_ld_items(nested))
    return items


def _walk_dicts(value: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if isinstance(value, dict):
        items.append(value)
        for nested in value.values():
            items.extend(_walk_dicts(nested))
    elif isinstance(value, list):
        for nested in value:
            items.extend(_walk_dicts(nested))
    return items


def _pick_string(data: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_location(data: dict[str, Any]) -> str | None:
    direct = _pick_string(data, "location", "city", "region")
    if direct:
        return direct

    location_data = data.get("address") or data.get("location")
    if isinstance(location_data, dict):
        return _pick_string(
            location_data,
            "addressLocality",
            "city",
            "district",
            "region",
            "municipality",
        )

    return None


def _extract_price(data: dict[str, Any]) -> int | None:
    candidate_values = [
        data.get("price"),
        data.get("priceValue"),
        data.get("regularPrice"),
        data.get("amount"),
    ]

    offers = data.get("offers")
    if isinstance(offers, dict):
        candidate_values.extend(
            [
                offers.get("price"),
                offers.get("priceValue"),
            ]
        )

    for value in candidate_values:
        parsed = _parse_price_pln(value)
        if parsed is not None:
            return parsed

    return None


def _parse_price_pln(value: Any) -> int | None:
    if isinstance(value, (int, float)):
        return int(value)
    if not isinstance(value, str):
        return None

    digits = re.sub(r"[^\d]", "", value)
    if not digits:
        return None

    return int(digits)


def _normalize_image_urls(value: Any, search_url: str) -> list[str]:
    if isinstance(value, str) and value.strip():
        normalized = _normalize_url(value.strip(), search_url) or value.strip()
        return [normalized]

    if not isinstance(value, list):
        return []

    image_urls: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            normalized = _normalize_url(item.strip(), search_url) or item.strip()
            image_urls.append(normalized)
        elif isinstance(item, dict):
            candidate = _pick_string(item, "url", "src", "link")
            if candidate:
                normalized = _normalize_url(candidate, search_url) or candidate
                image_urls.append(normalized)

    return sorted(set(image_urls))


def _normalize_url(value: str | None, search_url: str) -> str | None:
    if not value:
        return None
    return urljoin(search_url, value)


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Import OLX search results into data/raw/")
    parser.add_argument("--url", required=True, help="OLX search URL")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help="Maximum number of OLX search result pages to fetch",
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
    raw_path, records_path = import_olx_search(
        args.url,
        Path(args.output_dir),
        max_pages=args.max_pages,
        delay_seconds=args.delay_seconds,
    )
    print(f"Saved raw payloads to {raw_path}")
    print(f"Saved structured records to {records_path}")


if __name__ == "__main__":
    main()
