from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from .evaluate_damaged_listings import load_processed_records
from .price_model import detect_model_family, score_records_with_price_model, train_price_model
from .ready_price_report import save_ready_price_report
from .train_price_model import save_price_model, save_price_model_report


SUSPICIOUS_TRAINING_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"a2\s+w\s+dowodzie",
        r"zarejestrowan\w*\s+jako\s+125",
        r"kat\.?\s*[ab1]",
        r"stunt",
        r"bobber",
        r"projekt",
        r"custom",
        r"skuter",
        r"cesja\s+leasingu",
        r"w\s+leasingu",
        r"odstępn[eao]",
    ]
]


def filter_training_records(
    records: list[dict[str, Any]],
    *,
    required_brand: str | None = None,
    only_healthy: bool = True,
    only_motorcycles: bool = True,
    min_family_records: int = 3,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    normalized_brand = required_brand.casefold() if isinstance(required_brand, str) and required_brand.strip() else None

    for record in records:
        if normalized_brand is not None:
            brand = record.get("brand")
            if not isinstance(brand, str) or brand.casefold() != normalized_brand:
                continue

        if only_healthy and record.get("technical_state") != "Nieuszkodzony":
            continue

        if only_motorcycles and record.get("vehicle_type") != "motorcycle":
            continue

        if _looks_suspicious_for_price_training(record):
            continue

        filtered.append(record)

    family_counts = Counter(_family_key(record) for record in filtered)
    family_filtered = [
        record
        for record in filtered
        if _family_key(record) != "unknown" and family_counts[_family_key(record)] >= min_family_records
    ]

    return _trim_family_outliers(family_filtered)


def save_ready_price_predictions(
    predictions: list[dict[str, Any]],
    output_path: Path | None = None,
) -> Path:
    path = output_path or Path("data/processed/ready_price_predictions.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_enriched_ready_price_predictions(
    ready_records: list[dict[str, Any]],
    model_bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    predictions = score_records_with_price_model(ready_records, model_bundle)
    enriched_predictions = []
    for item in predictions:
        listing = item["original_listing"]
        listed_price = listing.get("price_pln")
        predicted_price = item.get("predicted_price_pln")
        price_delta = None
        if isinstance(listed_price, int) and isinstance(predicted_price, int):
            price_delta = predicted_price - listed_price
        enriched_predictions.append(
            {
                **item,
                "listed_price_pln": listed_price,
                "price_delta_pln": price_delta,
            }
        )
    return enriched_predictions


def run_ready_price_training(
    ready_records: list[dict[str, Any]],
    *,
    backend: str = "auto",
    required_brand: str | None,
    include_damaged: bool = False,
    include_scooters: bool = False,
    min_family_records: int = 3,
    validation_fraction: float = 0.2,
    random_seed: int = 42,
    cv_folds: int = 5,
    search_iterations: int = 8,
) -> dict[str, Any]:
    training_records = filter_training_records(
        ready_records,
        required_brand=required_brand,
        only_healthy=not include_damaged,
        only_motorcycles=not include_scooters,
        min_family_records=min_family_records,
    )

    model_bundle = train_price_model(
        training_records,
        backend=backend,
        validation_fraction=validation_fraction,
        random_seed=random_seed,
        cv_folds=cv_folds,
        search_iterations=search_iterations,
    )

    report_payload: dict[str, Any] = {
        key: value
        for key, value in model_bundle.items()
        if key != "estimator"
    }
    report_payload["required_brand"] = required_brand
    report_payload["input_record_count"] = len(ready_records)
    report_payload["training_record_count"] = len(training_records)
    report_payload["min_family_records"] = min_family_records

    predictions = build_enriched_ready_price_predictions(ready_records, model_bundle)

    return {
        "training_records": training_records,
        "model_bundle": model_bundle,
        "report_payload": report_payload,
        "predictions": predictions,
    }


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Train price model directly from ready motorcycle data")
    parser.add_argument(
        "--ready-file",
        default="data/ready/olx_kawasaki_all_pages_ready_20260331T172001Z.json",
        help="Path to ready JSON dataset",
    )
    parser.add_argument(
        "--required-brand",
        default="Kawasaki",
        help="Only train on this exact brand",
    )
    parser.add_argument(
        "--include-damaged",
        action="store_true",
        help="Include records whose technical_state is not 'Nieuszkodzony'",
    )
    parser.add_argument(
        "--include-scooters",
        action="store_true",
        help="Include scooter records in training",
    )
    parser.add_argument(
        "--min-family-records",
        type=int,
        default=3,
        help="Minimum records required per model family to keep it in training",
    )
    parser.add_argument(
        "--model-output",
        default=None,
        help="Where to save the trained model. Defaults depend on --backend and --required-brand.",
    )
    parser.add_argument(
        "--report-output",
        default=None,
        help="Where to save the training report. Defaults depend on --backend and --required-brand.",
    )
    parser.add_argument(
        "--predictions-output",
        default=None,
        help="Where to save model predictions for the same ready dataset. Defaults depend on --backend and --required-brand.",
    )
    parser.add_argument(
        "--html-report-output",
        default=None,
        help="Where to save the HTML opportunity report. Defaults depend on --backend and --required-brand.",
    )
    parser.add_argument(
        "--html-report-assets-dir",
        default=None,
        help="Directory for downloaded images used in the HTML report. Defaults next to the report file.",
    )
    parser.add_argument(
        "--report-limit",
        type=int,
        default=20,
        help="How many ranked listings to include in the HTML report",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "comparable", "sklearn", "torch"],
        default="auto",
        help="Price-model backend to train and use for scoring",
    )
    parser.add_argument(
        "--search-iterations",
        type=int,
        default=8,
        help="Randomized-search iterations per estimator",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Cross-validation folds",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Holdout fraction for final validation metrics",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for the split and parameter search",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    ready_records = load_processed_records(Path(args.ready_file))
    brand_value = args.required_brand.strip() if isinstance(args.required_brand, str) else ""
    brand_slug = "_".join(brand_value.casefold().split()) or "motorcycles"
    training_result = run_ready_price_training(
        ready_records,
        backend=args.backend,
        required_brand=args.required_brand,
        include_damaged=args.include_damaged,
        include_scooters=args.include_scooters,
        min_family_records=args.min_family_records,
        validation_fraction=args.validation_fraction,
        random_seed=args.random_seed,
        cv_folds=args.cv_folds,
        search_iterations=args.search_iterations,
    )
    training_records = training_result["training_records"]
    model_bundle = training_result["model_bundle"]
    actual_backend = model_bundle["backend"]
    default_stem = (
        f"{brand_slug}_ready_price_model_v1"
        if actual_backend == "sklearn"
        else f"{brand_slug}_ready_price_model_{actual_backend}_v1"
    )
    model_output = Path(args.model_output) if args.model_output else Path("data/models") / f"{default_stem}.joblib"
    report_output = (
        Path(args.report_output)
        if args.report_output
        else Path("data/processed") / f"{default_stem}_report.json"
    )
    predictions_output = (
        Path(args.predictions_output)
        if args.predictions_output
        else Path("data/processed") / f"{default_stem}_predictions.json"
    )
    html_report_output = (
        Path(args.html_report_output)
        if args.html_report_output
        else Path("data/processed") / f"{default_stem}_report.html"
    )

    model_path = save_price_model(model_bundle, model_output)
    report_payload = training_result["report_payload"]
    report_payload["ready_file"] = args.ready_file
    report_path = save_price_model_report(report_payload, report_output)

    predictions_path = save_ready_price_predictions(
        training_result["predictions"],
        predictions_output,
    )
    html_report_path = save_ready_price_report(
        training_result["predictions"],
        html_report_output,
        assets_dir=Path(args.html_report_assets_dir) if args.html_report_assets_dir else None,
        limit=max(1, args.report_limit),
        brand=args.required_brand,
        backend=model_bundle["backend"],
        dataset_label=brand_slug,
    )

    print(f"Backend: {model_bundle['backend']}")
    print(f"Training records: {len(training_records)}")
    print(f"Best candidate: {model_bundle['best_candidate_name']}")
    print(f"Validation MAE: {model_bundle['validation_metrics']['mae_pln']} PLN")
    print(f"Validation RMSE: {model_bundle['validation_metrics']['rmse_pln']} PLN")
    print(f"Saved model to {model_path}")
    print(f"Saved report to {report_path}")
    print(f"Saved ready predictions to {predictions_path}")
    print(f"Saved HTML report to {html_report_path}")


def _family_key(record: dict[str, Any]) -> str:
    family = detect_model_family(
        record.get("title"),
        record.get("full_description"),
        record.get("brand"),
        normalized_model=record.get("normalized_model"),
    )
    return family or "unknown"


def _looks_suspicious_for_price_training(record: dict[str, Any]) -> bool:
    text = " ".join(
        part
        for part in [record.get("title"), record.get("full_description"), record.get("short_description")]
        if isinstance(part, str) and part.strip()
    )
    if not text:
        return False
    return any(pattern.search(text) for pattern in SUSPICIOUS_TRAINING_PATTERNS)


def _trim_family_outliers(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for record in records:
        price = record.get("price_pln")
        if isinstance(price, int):
            grouped[_family_key(record)].append(price)

    bounds: dict[str, tuple[float, float]] = {}
    for family, prices in grouped.items():
        if len(prices) < 6:
            continue
        q1, q3 = np.percentile(prices, [25, 75])
        iqr = q3 - q1
        lower = max(0.0, q1 - 1.5 * iqr)
        upper = q3 + 1.5 * iqr
        bounds[family] = (lower, upper)

    trimmed: list[dict[str, Any]] = []
    for record in records:
        price = record.get("price_pln")
        if not isinstance(price, int):
            continue
        family = _family_key(record)
        family_bounds = bounds.get(family)
        if family_bounds is None:
            trimmed.append(record)
            continue
        lower, upper = family_bounds
        if lower <= price <= upper:
            trimmed.append(record)
    return trimmed


if __name__ == "__main__":
    main()
