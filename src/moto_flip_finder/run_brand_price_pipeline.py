from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from .build_ready_motorcycles_dataset import build_ready_motorcycles_dataset
from .evaluate_damaged_listings import load_processed_records
from .ready_price_report import save_ready_price_report
from .train_price_model import save_price_model, save_price_model_report
from .train_ready_price_model import run_ready_price_training, save_ready_price_predictions


def build_brand_slug(brand: str) -> str:
    slug = "_".join(brand.strip().casefold().split())
    if not slug:
        raise ValueError("Brand must not be empty")
    return slug


def build_default_brand_search_url(brand: str) -> str:
    search_keyword = "-".join(brand.strip().casefold().split())
    if not search_keyword:
        raise ValueError("Brand must not be empty")
    return f"https://www.olx.pl/motoryzacja/motocykle-skutery/q-{search_keyword}/"


def run_brand_price_pipeline(
    *,
    brand: str,
    backend: str = "auto",
    search_url: str | None = None,
    max_pages: int = 50,
    max_records: int = 500,
    raw_output_dir: Path | None = None,
    ready_output_dir: Path | None = None,
    search_delay_seconds: float = 0.5,
    detail_delay_seconds: float = 0.2,
    detail_max_workers: int = 6,
    dataset_label: str | None = None,
    keyword_filter: str | None = None,
    include_damaged: bool = False,
    include_scooters: bool = False,
    min_family_records: int = 3,
    model_output: Path | None = None,
    report_output: Path | None = None,
    predictions_output: Path | None = None,
    html_report_output: Path | None = None,
    html_report_assets_dir: Path | None = None,
    report_limit: int = 20,
    search_iterations: int = 8,
    cv_folds: int = 5,
    validation_fraction: float = 0.2,
    random_seed: int = 42,
) -> dict[str, Any]:
    normalized_brand = brand.strip()
    if not normalized_brand:
        raise ValueError("Brand must not be empty")

    brand_slug = build_brand_slug(normalized_brand)
    effective_search_url = search_url or build_default_brand_search_url(normalized_brand)
    effective_dataset_label = dataset_label or f"{brand_slug}_all_pages"
    effective_keyword_filter = normalized_brand if keyword_filter is None else keyword_filter

    records_path, details_path, ready_path = build_ready_motorcycles_dataset(
        search_url=effective_search_url,
        max_pages=max_pages,
        max_records=max_records,
        raw_output_dir=raw_output_dir,
        ready_output_dir=ready_output_dir,
        search_delay_seconds=search_delay_seconds,
        detail_delay_seconds=detail_delay_seconds,
        detail_max_workers=max(1, detail_max_workers),
        dataset_label=effective_dataset_label,
        keyword_filter=effective_keyword_filter,
        required_brand=normalized_brand,
    )

    ready_records = load_processed_records(ready_path)
    training_result = run_ready_price_training(
        ready_records,
        backend=backend,
        required_brand=normalized_brand,
        include_damaged=include_damaged,
        include_scooters=include_scooters,
        min_family_records=min_family_records,
        validation_fraction=validation_fraction,
        random_seed=random_seed,
        cv_folds=cv_folds,
        search_iterations=search_iterations,
    )

    model_bundle = training_result["model_bundle"]
    actual_backend = model_bundle["backend"]
    model_stem = (
        f"{brand_slug}_ready_price_model_v1"
        if actual_backend == "sklearn"
        else f"{brand_slug}_ready_price_model_{actual_backend}_v1"
    )
    model_output_path = model_output or Path("data/models") / f"{model_stem}.joblib"
    report_output_path = report_output or Path("data/processed") / f"{model_stem}_report.json"
    predictions_output_path = (
        predictions_output
        or Path("data/processed") / f"{model_stem}_predictions.json"
    )
    html_report_output_path = (
        html_report_output
        or Path("data/processed") / f"{model_stem}_report.html"
    )

    report_payload = dict(training_result["report_payload"])
    report_payload["ready_file"] = str(ready_path)
    report_payload["search_url"] = effective_search_url
    report_payload["dataset_label"] = effective_dataset_label
    report_payload["keyword_filter"] = effective_keyword_filter

    model_path = save_price_model(model_bundle, model_output_path)
    report_path = save_price_model_report(report_payload, report_output_path)
    predictions_path = save_ready_price_predictions(
        training_result["predictions"],
        predictions_output_path,
    )
    html_report_path = save_ready_price_report(
        training_result["predictions"],
        html_report_output_path,
        assets_dir=html_report_assets_dir,
        limit=max(1, report_limit),
        brand=normalized_brand,
        backend=actual_backend,
        dataset_label=effective_dataset_label,
    )

    return {
        "brand": normalized_brand,
        "requested_backend": backend,
        "backend": actual_backend,
        "search_url": effective_search_url,
        "dataset_label": effective_dataset_label,
        "keyword_filter": effective_keyword_filter,
        "records_path": records_path,
        "details_path": details_path,
        "ready_path": ready_path,
        "model_path": model_path,
        "report_path": report_path,
        "predictions_path": predictions_path,
        "html_report_path": html_report_path,
        "training_records_count": len(training_result["training_records"]),
        "best_candidate_name": model_bundle["best_candidate_name"],
        "validation_metrics": model_bundle["validation_metrics"],
    }


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Run an end-to-end OLX brand pipeline from scraping to ready price model"
    )
    parser.add_argument(
        "--brand",
        required=True,
        help="Exact brand used for the final ready dataset and model training, e.g. Kawasaki",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Optional OLX search URL override. Defaults to a brand search URL inferred from --brand.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "comparable", "sklearn", "torch"],
        default="auto",
        help="Price-model backend to train after the ready dataset is built",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=50,
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
        default=0.2,
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
        default=None,
        help="Optional dataset label override. Defaults to <brand>_all_pages.",
    )
    parser.add_argument(
        "--keyword-filter",
        default=None,
        help="Optional keyword filter override. Defaults to the brand name.",
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
        help="Where to save the trained model. Defaults depend on --backend.",
    )
    parser.add_argument(
        "--report-output",
        default=None,
        help="Where to save the training report. Defaults depend on --backend.",
    )
    parser.add_argument(
        "--predictions-output",
        default=None,
        help="Where to save model predictions for the ready dataset. Defaults depend on --backend.",
    )
    parser.add_argument(
        "--html-report-output",
        default=None,
        help="Where to save the HTML opportunity report. Defaults depend on --backend.",
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

    result = run_brand_price_pipeline(
        brand=args.brand,
        backend=args.backend,
        search_url=args.url,
        max_pages=args.max_pages,
        max_records=args.max_records,
        raw_output_dir=Path(args.raw_output_dir),
        ready_output_dir=Path(args.ready_output_dir),
        search_delay_seconds=args.search_delay_seconds,
        detail_delay_seconds=args.detail_delay_seconds,
        detail_max_workers=args.detail_max_workers,
        dataset_label=args.dataset_label,
        keyword_filter=args.keyword_filter,
        include_damaged=args.include_damaged,
        include_scooters=args.include_scooters,
        min_family_records=args.min_family_records,
        model_output=Path(args.model_output) if args.model_output else None,
        report_output=Path(args.report_output) if args.report_output else None,
        predictions_output=Path(args.predictions_output) if args.predictions_output else None,
        html_report_output=Path(args.html_report_output) if args.html_report_output else None,
        html_report_assets_dir=Path(args.html_report_assets_dir) if args.html_report_assets_dir else None,
        report_limit=args.report_limit,
        search_iterations=args.search_iterations,
        cv_folds=args.cv_folds,
        validation_fraction=args.validation_fraction,
        random_seed=args.random_seed,
    )

    print(f"Brand: {result['brand']}")
    if result["requested_backend"] != result["backend"]:
        print(
            f"Requested backend: {result['requested_backend']} "
            f"(resolved to {result['backend']})"
        )
    print(f"Backend: {result['backend']}")
    print(f"Search URL: {result['search_url']}")
    print(f"Saved search records to {result['records_path']}")
    print(f"Saved detail records to {result['details_path']}")
    print(f"Saved ready dataset to {result['ready_path']}")
    print(f"Training records: {result['training_records_count']}")
    print(f"Best candidate: {result['best_candidate_name']}")
    print(f"Validation MAE: {result['validation_metrics']['mae_pln']} PLN")
    print(f"Validation RMSE: {result['validation_metrics']['rmse_pln']} PLN")
    print(f"Saved model to {result['model_path']}")
    print(f"Saved report to {result['report_path']}")
    print(f"Saved ready predictions to {result['predictions_path']}")
    print(f"Saved HTML report to {result['html_report_path']}")


if __name__ == "__main__":
    main()
