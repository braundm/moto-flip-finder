from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any

import joblib

from .evaluate_damaged_listings import load_processed_records
from .price_model import score_records_with_price_model, train_price_model


def save_price_model(model_bundle: dict[str, Any], output_path: Path | None = None) -> Path:
    path = output_path or Path("data/models/healthy_price_model_v2.joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable_bundle = {
        key: value
        for key, value in model_bundle.items()
        if not key.startswith("_")
    }
    joblib.dump(serializable_bundle, path)
    return path


def save_price_model_report(
    model_bundle: dict[str, Any],
    output_path: Path | None = None,
) -> Path:
    path = output_path or Path("data/processed/price_model_training_report.json")
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        key: value
        for key, value in model_bundle.items()
        if key != "estimator" and not key.startswith("_")
    }
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_damaged_price_predictions(
    predictions: list[dict[str, Any]],
    output_path: Path | None = None,
) -> Path:
    path = output_path or Path("data/processed/damaged_ml_price_predictions.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Train price model on healthy motorcycle comps")
    parser.add_argument(
        "--healthy-file",
        default="data/processed/healthy_comps.json",
        help="Path to healthy comparable listings JSON",
    )
    parser.add_argument(
        "--damaged-file",
        default="data/processed/damaged_candidates.json",
        help="Path to damaged candidates JSON for scoring after training",
    )
    parser.add_argument(
        "--model-output",
        default=None,
        help="Where to save the trained model. Defaults depend on --backend.",
    )
    parser.add_argument(
        "--report-output",
        default=None,
        help="Where to save the training metrics report. Defaults depend on --backend.",
    )
    parser.add_argument(
        "--predictions-output",
        default=None,
        help="Where to save scored damaged candidate prices. Defaults depend on --backend.",
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
        default=12,
        help="How many randomized-search iterations to run per estimator",
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

    healthy_records = load_processed_records(Path(args.healthy_file))
    damaged_records = load_processed_records(Path(args.damaged_file))

    model_bundle = train_price_model(
        healthy_records,
        backend=args.backend,
        validation_fraction=args.validation_fraction,
        random_seed=args.random_seed,
        cv_folds=args.cv_folds,
        search_iterations=args.search_iterations,
    )
    model_bundle["healthy_file"] = args.healthy_file
    model_bundle["damaged_file"] = args.damaged_file
    actual_backend = model_bundle["backend"]
    default_stem = (
        "healthy_price_model_v2"
        if actual_backend == "sklearn"
        else f"healthy_price_model_{actual_backend}_v1"
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

    model_path = save_price_model(model_bundle, model_output)
    report_path = save_price_model_report(model_bundle, report_output)
    predictions = score_records_with_price_model(damaged_records, model_bundle)
    predictions_path = save_damaged_price_predictions(predictions, predictions_output)

    print(f"Backend: {model_bundle['backend']}")
    print(f"Best candidate: {model_bundle['best_candidate_name']}")
    print(f"Training size: {model_bundle['training_size']}")
    print(f"Validation size: {model_bundle['validation_size']}")
    print(
        "Validation MAE: "
        f"{model_bundle['validation_metrics']['mae_pln']} PLN"
    )
    print(
        "Validation RMSE: "
        f"{model_bundle['validation_metrics']['rmse_pln']} PLN"
    )
    print(f"Saved model to {model_path}")
    print(f"Saved report to {report_path}")
    print(f"Saved damaged price predictions to {predictions_path}")


if __name__ == "__main__":
    main()
