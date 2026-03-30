from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
from typing import Any

from moto_flip_finder.sources.olx.import_search import find_damage_keyword


DAMAGED_KEYWORDS = [
    "uszkodz",
    "po szlifie",
    "po dzwonie",
    "do naprawy",
    "nie odpala",
    "rozbit",
    "krzywy",
]


def find_latest_details_file(raw_dir: Path | None = None) -> Path:
    base_dir = raw_dir or Path("data/raw")
    candidates = list(base_dir.glob("*_details.json"))
    if not candidates:
        raise FileNotFoundError(f"No details JSON files found in {base_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_detail_records(details_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(details_path.read_text(encoding="utf-8"))
    details = payload.get("details", [])
    if not isinstance(details, list):
        raise ValueError("Details file must contain a 'details' list")
    return [item for item in details if isinstance(item, dict)]


def build_processed_datasets(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    classified_records = [_with_classification(record) for record in records]
    all_gsxr = classified_records
    healthy_comps = [
        record for record in classified_records if record.get("dataset_class") == "healthy"
    ]
    damaged_candidates = [
        record for record in classified_records if record.get("dataset_class") == "damaged"
    ]

    return {
        "all_gsxr": all_gsxr,
        "healthy_comps": healthy_comps,
        "damaged_candidates": damaged_candidates,
    }


def is_healthy_record(record: dict[str, Any]) -> bool:
    return classify_record(record)[0] == "healthy"


def is_damaged_record(record: dict[str, Any]) -> bool:
    return classify_record(record)[0] == "damaged"


def classify_record(record: dict[str, Any]) -> tuple[str | None, str | None]:
    technical_state = _normalized_text(record.get("technical_state"))
    looks_damaged = record.get("looks_damaged") is True
    matched_keyword = _find_damage_keyword(record)

    if technical_state:
        if technical_state == "nieuszkodzony":
            if matched_keyword:
                return "damaged", f"damaged: keyword={matched_keyword}"
            return "healthy", "healthy: technical_state=Nieuszkodzony"
        return "damaged", f"damaged: technical_state={record.get('technical_state')}"

    if looks_damaged:
        return "damaged", "damaged: looks_damaged=true"
    if matched_keyword:
        return "damaged", f"damaged: keyword={matched_keyword}"

    return None, None


def save_processed_datasets(
    datasets: dict[str, list[dict[str, Any]]],
    output_dir: Path | None = None,
) -> dict[str, Path]:
    base_dir = output_dir or Path("data/processed")
    base_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "all_gsxr": base_dir / "all_gsxr.json",
        "healthy_comps": base_dir / "healthy_comps.json",
        "damaged_candidates": base_dir / "damaged_candidates.json",
    }

    for name, path in paths.items():
        path.write_text(
            json.dumps(datasets[name], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return paths


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    normalized_model_counts = Counter(
        str(record.get("normalized_model") or "unknown") for record in records
    )
    datasets = build_processed_datasets(records)
    return {
        "all_records": len(records),
        "healthy_records": len(datasets["healthy_comps"]),
        "damaged_records": len(datasets["damaged_candidates"]),
        "normalized_model_counts": dict(sorted(normalized_model_counts.items())),
    }


def _normalized_text(value: object) -> str:
    if isinstance(value, str):
        return value.lower()
    return ""


def _find_damage_keyword(record: dict[str, Any]) -> str | None:
    return find_damage_keyword(
        _string_or_none(record.get("title")),
        " ".join(
            part
            for part in [
                _string_or_none(record.get("full_description")),
                _string_or_none(record.get("short_description")),
            ]
            if part
        )
        or None,
    )


def _string_or_none(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _with_classification(record: dict[str, Any]) -> dict[str, Any]:
    dataset_class, reason = classify_record(record)
    enriched = dict(record)
    enriched["classification_reason"] = reason
    if dataset_class is not None:
        enriched["dataset_class"] = dataset_class
    return enriched


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build processed datasets from OLX detail records")
    parser.add_argument(
        "--details-file",
        default=None,
        help="Path to a *_details.json file. Defaults to the newest file in data/raw/",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory used to find the newest *_details.json when --details-file is omitted",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for processed dataset JSON files",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    details_path = (
        Path(args.details_file)
        if args.details_file
        else find_latest_details_file(Path(args.raw_dir))
    )
    records = load_detail_records(details_path)
    datasets = build_processed_datasets(records)
    output_paths = save_processed_datasets(datasets, Path(args.output_dir))
    summary = summarize_records(records)

    print(f"Source details file: {details_path}")
    print(f"All records: {summary['all_records']}")
    print(f"Healthy records: {summary['healthy_records']}")
    print(f"Damaged records: {summary['damaged_records']}")
    print("Counts by normalized_model:")
    for model, count in summary["normalized_model_counts"].items():
        print(f"  {model}: {count}")
    print("Saved datasets:")
    for name, path in output_paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
