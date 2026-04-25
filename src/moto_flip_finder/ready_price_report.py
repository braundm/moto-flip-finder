from __future__ import annotations

from html import escape
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen


def rank_ready_price_predictions(
    predictions: list[dict[str, Any]],
    *,
    limit: int | None = None,
    only_positive_delta: bool = True,
) -> list[dict[str, Any]]:
    ranked = []
    for item in predictions:
        delta = item.get("price_delta_pln")
        if only_positive_delta and not isinstance(delta, int):
            continue
        if only_positive_delta and delta <= 0:
            continue
        ranked.append(item)

    if not ranked and only_positive_delta:
        ranked = list(predictions)

    ranked = sorted(
        ranked,
        key=lambda item: (
            _sortable_number(item.get("price_delta_pln")),
            _sortable_number(item.get("predicted_price_pln")),
            -_sortable_number(item.get("listed_price_pln")),
        ),
        reverse=True,
    )
    if limit is not None:
        return ranked[:limit]
    return ranked


def summarize_ready_price_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    with_delta = [
        item for item in predictions
        if isinstance(item.get("price_delta_pln"), int)
    ]
    positive = [item for item in with_delta if item["price_delta_pln"] > 0]
    negative = [item for item in with_delta if item["price_delta_pln"] < 0]
    best_delta = max((item["price_delta_pln"] for item in positive), default=None)

    return {
        "total_predictions": len(predictions),
        "predictions_with_delta": len(with_delta),
        "positive_delta_count": len(positive),
        "negative_delta_count": len(negative),
        "best_positive_delta_pln": best_delta,
    }


def save_ready_price_report(
    predictions: list[dict[str, Any]],
    output_path: Path | None = None,
    *,
    assets_dir: Path | None = None,
    limit: int = 20,
    brand: str | None = None,
    backend: str | None = None,
    dataset_label: str | None = None,
    only_positive_delta: bool = True,
) -> Path:
    path = output_path or Path("data/processed/ready_price_report.html")
    path.parent.mkdir(parents=True, exist_ok=True)
    report_assets_dir = assets_dir or path.parent / f"{path.stem}_assets"
    report_assets_dir.mkdir(parents=True, exist_ok=True)

    ranked = rank_ready_price_predictions(
        predictions,
        limit=limit,
        only_positive_delta=only_positive_delta,
    )
    rendered_cards = []
    for index, item in enumerate(ranked, start=1):
        image_src = _download_primary_image(
            item,
            report_assets_dir,
            report_dir=path.parent,
            index=index,
        ) or _primary_image_url(item)
        rendered_cards.append(
            {
                "prediction": item,
                "image_src": image_src,
                "reason": _why_this_listing(item),
            }
        )

    summary = summarize_ready_price_predictions(predictions)
    path.write_text(
        _render_ready_price_report(
            rendered_cards,
            summary=summary,
            brand=brand,
            backend=backend,
            dataset_label=dataset_label,
            only_positive_delta=only_positive_delta,
        ),
        encoding="utf-8",
    )
    return path


def _primary_image_url(prediction: dict[str, Any]) -> str | None:
    listing = prediction.get("original_listing")
    if not isinstance(listing, dict):
        return None
    images = listing.get("image_urls")
    if not isinstance(images, list):
        return None
    candidates = [
        item.strip()
        for item in images
        if isinstance(item, str) and item.strip()
    ]
    if not candidates:
        return None
    ranked = sorted(candidates, key=_report_image_sort_key)
    return ranked[0]


def _download_primary_image(
    prediction: dict[str, Any],
    assets_dir: Path,
    *,
    report_dir: Path,
    index: int,
) -> str | None:
    image_url = _primary_image_url(prediction)
    if image_url is None:
        return None

    extension = _image_extension_from_url(image_url)
    target_path = assets_dir / f"ready_price_{index:02d}{extension}"
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


def _why_this_listing(prediction: dict[str, Any]) -> str:
    listing = prediction.get("original_listing")
    if not isinstance(listing, dict):
        return "No listing details available."

    reasons = []
    listed_price = prediction.get("listed_price_pln")
    predicted_price = prediction.get("predicted_price_pln")
    delta = prediction.get("price_delta_pln")
    if isinstance(delta, int):
        reasons.append(f"Model delta: {delta} PLN.")
    if isinstance(predicted_price, int):
        reasons.append(f"Predicted market price: {predicted_price} PLN.")
    if isinstance(listed_price, int):
        reasons.append(f"Listed price: {listed_price} PLN.")

    year = listing.get("year")
    if isinstance(year, int):
        reasons.append(f"Year: {year}.")

    engine_cc = listing.get("engine_cc")
    if isinstance(engine_cc, int):
        reasons.append(f"Engine: {engine_cc} cc.")

    mileage = listing.get("mileage_km")
    if isinstance(mileage, int):
        reasons.append(f"Mileage: {mileage} km.")

    validation_summary = listing.get("validation_summary")
    if isinstance(validation_summary, str) and validation_summary.strip():
        reasons.append(validation_summary.strip())

    if not reasons:
        reasons.append("Selected by the price-delta ranking.")
    return " ".join(reasons)


def _render_ready_price_report(
    rendered_cards: list[dict[str, Any]],
    *,
    summary: dict[str, Any],
    brand: str | None,
    backend: str | None,
    dataset_label: str | None,
    only_positive_delta: bool,
) -> str:
    heading = "Ready Price Opportunities"
    if isinstance(brand, str) and brand.strip():
        heading = f"{heading}: {escape(brand.strip())}"

    subheading_parts = []
    if isinstance(dataset_label, str) and dataset_label.strip():
        subheading_parts.append(f"dataset={escape(dataset_label.strip())}")
    if isinstance(backend, str) and backend.strip():
        subheading_parts.append(f"backend={escape(backend.strip())}")
    if only_positive_delta:
        subheading_parts.append("ranking=positive_price_delta")
    subheading = " | ".join(subheading_parts)

    cards = []
    for item in rendered_cards:
        prediction = item["prediction"]
        listing = prediction.get("original_listing", {})
        title = escape(str(listing.get("title") or "Untitled listing"))
        url = escape(str(listing.get("url") or ""))
        location = escape(str(listing.get("location") or "No location"))
        listed_price = escape(str(prediction.get("listed_price_pln")))
        predicted_price = escape(str(prediction.get("predicted_price_pln")))
        delta = prediction.get("price_delta_pln")
        delta_class = "delta positive" if isinstance(delta, int) and delta > 0 else "delta"
        image_src = item.get("image_src")
        image_html = ""
        if isinstance(image_src, str) and image_src:
            image_html = f'<img src="{escape(image_src)}" alt="{title}">'
        reason = escape(str(item.get("reason") or ""))

        cards.append(
            f"""
            <article class="card">
              <div class="image">{image_html}</div>
              <div class="content">
                <h2>{title}</h2>
                <p class="{delta_class}"><strong>Price delta:</strong> {escape(str(delta))} PLN</p>
                <p><strong>Listed:</strong> {listed_price} PLN</p>
                <p><strong>Predicted:</strong> {predicted_price} PLN</p>
                <p><strong>Location:</strong> {location}</p>
                <p><strong>Link:</strong> <a href="{url}">{url}</a></p>
                <p><strong>Why it stands out:</strong> {reason}</p>
              </div>
            </article>
            """
        )

    summary_html = (
        f"<p><strong>Total predictions:</strong> {summary['total_predictions']} | "
        f"<strong>With delta:</strong> {summary['predictions_with_delta']} | "
        f"<strong>Positive delta:</strong> {summary['positive_delta_count']} | "
        f"<strong>Negative delta:</strong> {summary['negative_delta_count']} | "
        f"<strong>Best delta:</strong> {escape(str(summary['best_positive_delta_pln']))} PLN</p>"
    )
    body = "\n".join(cards) if cards else "<p>No listings available for this report.</p>"

    subheading_html = f'<p class="lead">{subheading}</p>' if subheading else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{heading}</title>
  <style>
    :root {{
      --bg: #f2efe8;
      --panel: #fffdf8;
      --ink: #1c1916;
      --accent: #0f766e;
      --accent-soft: #dff3ef;
      --line: #ddd5c7;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.08), transparent 30%),
        linear-gradient(180deg, #ece6d8 0%, var(--bg) 100%);
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
    }}
    main {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 2.3rem;
    }}
    .lead {{
      margin: 0 0 24px;
      color: #4c463c;
    }}
    .summary {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px 18px;
      margin-bottom: 24px;
      box-shadow: 0 10px 24px rgba(28, 25, 22, 0.06);
    }}
    .cards {{
      display: grid;
      gap: 18px;
    }}
    .card {{
      display: grid;
      grid-template-columns: 280px 1fr;
      gap: 18px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 14px 30px rgba(28, 25, 22, 0.08);
      break-inside: avoid;
    }}
    .image img {{
      width: 100%;
      height: 220px;
      object-fit: cover;
      border-radius: 12px;
      background: #ddd6c8;
    }}
    .content h2 {{
      margin: 0 0 12px;
      font-size: 1.45rem;
    }}
    .content p {{
      margin: 8px 0;
      line-height: 1.45;
    }}
    .delta {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: #ece7db;
    }}
    .delta.positive {{
      background: var(--accent-soft);
      color: #0b4f49;
    }}
    a {{
      color: var(--accent);
      word-break: break-all;
    }}
    @media (max-width: 760px) {{
      .card {{
        grid-template-columns: 1fr;
      }}
      .image img {{
        height: 200px;
      }}
    }}
    @media print {{
      body {{
        background: white;
      }}
      main {{
        max-width: none;
        padding: 12mm;
      }}
      .card {{
        box-shadow: none;
      }}
      a {{
        color: inherit;
        text-decoration: none;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>{heading}</h1>
    {subheading_html}
    <section class="summary">
      {summary_html}
      <p>This HTML report is print-friendly, so it can be saved as PDF from the browser.</p>
    </section>
    <section class="cards">
      {body}
    </section>
  </main>
</body>
</html>
"""


def _sortable_number(value: object) -> int:
    if isinstance(value, int):
        return value
    return -10**9


def _report_image_sort_key(url: str) -> tuple[int, str]:
    lowered = url.lower()
    score = 0
    if any(
        marker in lowered
        for marker in [
            "avatar",
            "profile",
            "user-photo",
            "user_photo",
            "seller-logo",
            "seller_logo",
            "default-user",
            "default_user",
            "placeholder",
        ]
    ):
        score -= 10
    if "apollo.olxcdn.com" in lowered or "/v1/files/" in lowered:
        score += 4
    if "img.next" in lowered or "img.jsonld" in lowered or "img.seed" in lowered:
        score += 3
    if "meta" in lowered:
        score -= 1
    return (-score, lowered)
