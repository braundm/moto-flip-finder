from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict
from html import escape
import json
import os
from pathlib import Path
from typing import Any
from urllib.request import urlopen
from urllib.parse import urlparse

from .damage_analysis import analyze_description
from .market_value import build_market_valuation
from .models import Listing
from .repair_estimator import estimate_repair_cost


def load_processed_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Processed dataset must be a JSON list: {path}")
    return [item for item in payload if isinstance(item, dict)]


def evaluate_damaged_listings(
    damaged_candidates: list[dict[str, Any]],
    healthy_comps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evaluations = []

    for record in damaged_candidates:
        valuation = build_market_valuation(record, healthy_comps)
        listing = _listing_from_record(record)
        damage_analysis = analyze_description(listing.description)
        repair_estimate = estimate_repair_cost(damage_analysis)

        purchase_price = _int_or_none(record.get("price_pln"))
        expected_profit = None
        if valuation["healthy_market_value"] is not None and purchase_price is not None:
            expected_profit = (
                valuation["healthy_market_value"] - repair_estimate.total_cost_pln - purchase_price
            )

        roi_percent = _roi_percent(expected_profit, purchase_price)
        damage_analysis_payload = asdict(damage_analysis)
        evaluation = {
            "original_listing": record,
            "healthy_market_value": valuation["healthy_market_value"],
            "comparables_count": valuation["comparables_count"],
            "valuation_status": valuation["valuation_status"],
            "valuation_confidence": valuation["valuation_confidence"],
            "generation_hint": valuation["generation_hint"],
            "matched_same_generation": valuation["matched_same_generation"],
            "selected_comparables_strategy": valuation["selected_comparables_strategy"],
            "damage_analysis": damage_analysis_payload,
            "repair_estimate": asdict(repair_estimate),
            "expected_profit": expected_profit,
            "roi_percent": roi_percent,
            "deal_score": _deal_score(
                expected_profit=expected_profit,
                valuation_confidence=valuation["valuation_confidence"],
                valuation_status=valuation["valuation_status"],
                severity=_string_or_none(damage_analysis_payload.get("severity")),
                starts=damage_analysis_payload.get("starts"),
                roi_percent=roi_percent,
            ),
        }
        evaluations.append(evaluation)

    return evaluations


def sort_top_deals(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        evaluations,
        key=lambda item: (
            _sortable_number(item.get("deal_score")),
            _sortable_number(item.get("expected_profit")),
        ),
        reverse=True,
    )


def sort_top_profit_deals(
    evaluations: list[dict[str, Any]],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    ranked = sorted(
        evaluations,
        key=lambda item: _sortable_number(item.get("expected_profit")),
        reverse=True,
    )
    if limit is not None:
        return ranked[:limit]
    return ranked


def save_damaged_evaluations(
    evaluations: list[dict[str, Any]],
    output_path: Path | None = None,
) -> Path:
    path = output_path or Path("data/processed/damaged_evaluations.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evaluations, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_top_deals(
    evaluations: list[dict[str, Any]],
    output_path: Path | None = None,
) -> Path:
    path = output_path or Path("data/processed/top_deals.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sort_top_deals(evaluations), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def save_top_profit_report(
    evaluations: list[dict[str, Any]],
    output_path: Path | None = None,
    *,
    assets_dir: Path | None = None,
    limit: int = 3,
) -> Path:
    path = output_path or Path("data/processed/top_profit_report.html")
    path.parent.mkdir(parents=True, exist_ok=True)
    report_assets_dir = assets_dir or path.parent / "top_profit_assets"
    report_assets_dir.mkdir(parents=True, exist_ok=True)

    top_profit_records = sort_top_profit_deals(evaluations, limit)
    rendered_records = []
    for index, evaluation in enumerate(top_profit_records, start=1):
        local_image = _download_primary_image(
            evaluation,
            report_assets_dir,
            report_dir=path.parent,
            index=index,
        )
        rendered_records.append(
            {
                "evaluation": evaluation,
                "image_src": local_image or _primary_image_url(evaluation),
                "why_this_bike": _why_this_bike(evaluation),
            }
        )

    path.write_text(_render_top_profit_report(rendered_records), encoding="utf-8")
    return path


def summarize_evaluations(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    by_expected_profit = [
        item for item in evaluations if isinstance(item.get("expected_profit"), int)
    ]
    top_expected_profit = sorted(
        by_expected_profit,
        key=lambda item: item["expected_profit"],
        reverse=True,
    )[:5]
    top_deal_score = sort_top_deals(evaluations)[:5]
    insufficient_count = sum(
        1 for item in evaluations if item.get("valuation_status") == "insufficient_comparables"
    )
    return {
        "evaluated_count": len(evaluations),
        "insufficient_comparables_count": insufficient_count,
        "top_5_highest_expected_profit": top_expected_profit,
        "top_5_highest_deal_score": top_deal_score,
    }


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Evaluate damaged OLX listings against healthy comps")
    parser.add_argument(
        "--damaged-file",
        default="data/processed/damaged_candidates.json",
        help="Path to damaged_candidates.json",
    )
    parser.add_argument(
        "--healthy-file",
        default="data/processed/healthy_comps.json",
        help="Path to healthy_comps.json",
    )
    parser.add_argument(
        "--output-file",
        default="data/processed/damaged_evaluations.json",
        help="Where to save the evaluation results",
    )
    parser.add_argument(
        "--top-deals-file",
        default="data/processed/top_deals.json",
        help="Where to save the ranked deals list",
    )
    parser.add_argument(
        "--top-profit-report-file",
        default="data/processed/top_profit_report.html",
        help="Where to save the HTML report for top expected profit deals",
    )
    parser.add_argument(
        "--top-profit-assets-dir",
        default="data/processed/top_profit_assets",
        help="Directory for downloaded images used in the HTML report",
    )
    parser.add_argument(
        "--top-profit-limit",
        type=int,
        default=3,
        help="How many top expected profit deals to include in the HTML report",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    damaged_records = load_processed_records(Path(args.damaged_file))
    healthy_records = load_processed_records(Path(args.healthy_file))
    evaluations = evaluate_damaged_listings(damaged_records, healthy_records)
    output_path = save_damaged_evaluations(evaluations, Path(args.output_file))
    top_deals_path = save_top_deals(evaluations, Path(args.top_deals_file))
    top_profit_report_path = save_top_profit_report(
        evaluations,
        Path(args.top_profit_report_file),
        assets_dir=Path(args.top_profit_assets_dir),
        limit=args.top_profit_limit,
    )
    summary = summarize_evaluations(evaluations)

    print(f"Evaluated records: {summary['evaluated_count']}")
    print(
        "Records with insufficient comparables: "
        f"{summary['insufficient_comparables_count']}"
    )
    print("Top 5 highest expected_profit:")
    for item in summary["top_5_highest_expected_profit"]:
        listing = item["original_listing"]
        print(
            f"  {listing.get('title')} | expected_profit={item.get('expected_profit')} PLN"
        )
    print("Top 5 highest deal_score:")
    for item in summary["top_5_highest_deal_score"]:
        listing = item["original_listing"]
        print(f"  {listing.get('title')} | deal_score={item.get('deal_score')}")
    print(f"Saved evaluations to {output_path}")
    print(f"Saved ranked deals to {top_deals_path}")
    print(f"Saved top profit report to {top_profit_report_path}")


def _deal_score(
    *,
    expected_profit: int | None,
    valuation_confidence: str | None,
    valuation_status: str | None,
    severity: str | None,
    starts: object,
    roi_percent: float | None,
) -> int:
    score = 50

    if isinstance(expected_profit, int):
        score += max(-40, min(40, expected_profit // 500))
    if isinstance(roi_percent, float):
        score += max(-10, min(15, int(roi_percent // 10)))

    if valuation_confidence == "high":
        score += 20
    elif valuation_confidence == "medium":
        score += 10

    if valuation_status == "insufficient_comparables":
        score -= 20

    if severity == "high":
        score -= 20
    elif severity == "medium":
        score -= 10
    elif severity == "unknown":
        score -= 5

    if starts is False:
        score -= 15

    return max(0, min(score, 100))


def _roi_percent(expected_profit: int | None, purchase_price: int | None) -> float | None:
    if expected_profit is None or purchase_price is None or purchase_price <= 0:
        return None
    return round((expected_profit / purchase_price) * 100, 2)


def _listing_from_record(record: dict[str, Any]) -> Listing:
    return Listing(
        source=str(record.get("source") or "olx"),
        listing_id=_listing_id_from_url(record.get("url")),
        url=str(record.get("url") or ""),
        title=str(record.get("title") or ""),
        description=_description_from_record(record),
        price_pln=_int_or_none(record.get("price_pln")) or 0,
        image_urls=_image_urls(record.get("image_urls")),
        brand=_string_or_none(record.get("brand")),
        model=_string_or_none(record.get("normalized_model")),
        year=_int_or_none(record.get("year")),
    )


def _description_from_record(record: dict[str, Any]) -> str:
    return (
        _string_or_none(record.get("full_description"))
        or _string_or_none(record.get("short_description"))
        or _string_or_none(record.get("title"))
        or ""
    )


def _image_urls(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _listing_id_from_url(value: object) -> str:
    if not isinstance(value, str) or not value:
        return ""
    path = urlparse(value).path.rstrip("/")
    if not path:
        return value
    return path.split("/")[-1]


def _string_or_none(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def _sortable_number(value: object) -> int:
    if isinstance(value, int):
        return value
    return -10**9


def _primary_image_url(evaluation: dict[str, Any]) -> str | None:
    listing = evaluation.get("original_listing")
    if not isinstance(listing, dict):
        return None
    images = listing.get("image_urls")
    if not isinstance(images, list):
        return None
    for item in images:
        if isinstance(item, str) and item.strip():
            return item
    return None


def _download_primary_image(
    evaluation: dict[str, Any],
    assets_dir: Path,
    *,
    report_dir: Path,
    index: int,
) -> str | None:
    image_url = _primary_image_url(evaluation)
    if image_url is None:
        return None

    extension = _image_extension_from_url(image_url)
    target_path = assets_dir / f"top_profit_{index:02d}{extension}"
    if target_path.exists():
        return os.path.relpath(target_path, start=report_dir)

    try:
        with urlopen(image_url, timeout=20) as response:
            target_path.write_bytes(response.read())
    except Exception:
        return None
    return os.path.relpath(target_path, start=report_dir)


def _image_extension_from_url(url: str) -> str:
    path = urlparse(url).path.lower()
    if path.endswith(".png"):
        return ".png"
    if path.endswith(".webp"):
        return ".webp"
    return ".jpg"


def _why_this_bike(evaluation: dict[str, Any]) -> str:
    listing = evaluation.get("original_listing")
    if not isinstance(listing, dict):
        return "Brak danych listingu."

    reasons = []
    expected_profit = evaluation.get("expected_profit")
    if isinstance(expected_profit, int):
        reasons.append(f"Szacowany profit: {expected_profit} PLN.")

    market_value = evaluation.get("healthy_market_value")
    if isinstance(market_value, int):
        reasons.append(f"Szacowana wartosc zdrowej sztuki: {market_value} PLN.")

    repair_estimate = evaluation.get("repair_estimate")
    if isinstance(repair_estimate, dict):
        total_cost = repair_estimate.get("total_cost_pln")
        if isinstance(total_cost, int):
            reasons.append(f"Szacowany koszt naprawy: {total_cost} PLN.")

    severity = None
    damage_analysis = evaluation.get("damage_analysis")
    if isinstance(damage_analysis, dict):
        severity = damage_analysis.get("severity")
        suspected = damage_analysis.get("suspected_damage")
        if isinstance(suspected, list) and suspected:
            reasons.append("Podejrzane uszkodzenia: " + ", ".join(str(item) for item in suspected) + ".")

    if isinstance(severity, str):
        reasons.append(f"Severity: {severity}.")

    valuation_confidence = evaluation.get("valuation_confidence")
    if isinstance(valuation_confidence, str):
        reasons.append(f"Jakosc wyceny: {valuation_confidence}.")

    if not reasons:
        reasons.append("Oferta trafila do top listy przez ranking expected_profit.")
    return " ".join(reasons)


def _render_top_profit_report(rendered_records: list[dict[str, Any]]) -> str:
    cards = []
    for item in rendered_records:
        evaluation = item["evaluation"]
        listing = evaluation.get("original_listing", {})
        title = escape(str(listing.get("title") or "Bez tytulu"))
        url = escape(str(listing.get("url") or ""))
        location = escape(str(listing.get("location") or listing.get("city") or "Brak lokalizacji"))
        price = listing.get("price_pln")
        expected_profit = evaluation.get("expected_profit")
        why_this_bike = escape(str(item.get("why_this_bike") or ""))
        image_src = item.get("image_src")
        image_html = ""
        if isinstance(image_src, str) and image_src:
            image_html = f'<img src="{escape(image_src)}" alt="{title}">'

        cards.append(
            f"""
            <article class="card">
              <div class="image">{image_html}</div>
              <div class="content">
                <h2>{title}</h2>
                <p><strong>Cena:</strong> {escape(str(price))} PLN</p>
                <p><strong>Szacowany profit:</strong> {escape(str(expected_profit))} PLN</p>
                <p><strong>Lokalizacja:</strong> {location}</p>
                <p><strong>Link:</strong> <a href="{url}">{url}</a></p>
                <p><strong>Czemu ten motocykl:</strong> {why_this_bike}</p>
              </div>
            </article>
            """
        )

    body = "\n".join(cards) if cards else "<p>Brak ofert do pokazania.</p>"
    return f"""<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Top Profit Deals</title>
  <style>
    :root {{
      --bg: #f3efe6;
      --panel: #fffaf0;
      --ink: #1e1a16;
      --accent: #a63d1f;
      --line: #d8cdbd;
    }}
    body {{
      margin: 0;
      background: linear-gradient(180deg, #efe8da 0%, var(--bg) 100%);
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
    }}
    main {{
      max-width: 980px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    h1 {{
      font-size: 2.2rem;
      margin-bottom: 8px;
    }}
    .lead {{
      color: #564a3f;
      margin-bottom: 28px;
    }}
    .card {{
      display: grid;
      grid-template-columns: 280px 1fr;
      gap: 20px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      margin-bottom: 20px;
      box-shadow: 0 14px 30px rgba(30, 26, 22, 0.08);
    }}
    .image img {{
      width: 100%;
      height: 220px;
      object-fit: cover;
      border-radius: 12px;
      background: #ddd3c2;
    }}
    .content h2 {{
      margin-top: 0;
      margin-bottom: 12px;
      font-size: 1.4rem;
    }}
    .content p {{
      margin: 8px 0;
      line-height: 1.45;
    }}
    a {{
      color: var(--accent);
      word-break: break-all;
    }}
    @media (max-width: 720px) {{
      .card {{
        grid-template-columns: 1fr;
      }}
      .image img {{
        height: 200px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Top oferty po expected profit</h1>
    <p class="lead">Raport automatyczny: cena zakupu, estymowany profit, lokalizacja, link i szybkie uzasadnienie wyboru.</p>
    {body}
  </main>
</body>
</html>
"""


if __name__ == "__main__":
    main()
