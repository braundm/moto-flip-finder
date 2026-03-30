from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
from typing import Any

from .build_datasets import find_latest_details_file, load_detail_records
from .damage_analysis import analyze_description
from .models import DamageAnalysis


def analyze_gsxr_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []

    for record in records:
        analysis_text, has_meaningful_description = _analysis_text(record)
        damage_analysis = analyze_description(analysis_text)
        results.append(
            {
                "original_listing": record,
                "damage_analysis": _damage_analysis_payload(damage_analysis),
                "detected_damage_signals": damage_analysis.found_keywords,
                "severity": damage_analysis.severity,
                "starts": damage_analysis.starts,
                "hidden_risks": damage_analysis.hidden_risks,
                "has_meaningful_description": has_meaningful_description,
                "review_priority_score": _review_priority_score(
                    damage_analysis,
                    has_meaningful_description=has_meaningful_description,
                ),
                "analysis_summary": _human_summary(
                    record,
                    damage_analysis,
                    has_meaningful_description=has_meaningful_description,
                ),
            }
        )

    return results


def filter_gsxr_records(
    records: list[dict[str, Any]],
    max_records: int | None = None,
    *,
    engine_cc: int | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
) -> list[dict[str, Any]]:
    filtered = [
        record
        for record in records
        if isinstance(record.get("normalized_model"), str)
        and str(record.get("normalized_model")).startswith("gsxr")
    ]

    if engine_cc is not None:
        filtered = [
            record
            for record in filtered
            if _int_or_none(record.get("engine_cc")) == engine_cc
        ]

    if year_from is not None:
        filtered = [
            record
            for record in filtered
            if (year := _int_or_none(record.get("year"))) is not None and year >= year_from
        ]

    if year_to is not None:
        filtered = [
            record
            for record in filtered
            if (year := _int_or_none(record.get("year"))) is not None and year <= year_to
        ]

    if max_records is not None:
        return filtered[:max_records]
    return filtered


def save_gsxr_ai_analysis(
    analysis_results: list[dict[str, Any]],
    output_path: Path | None = None,
) -> Path:
    path = output_path or Path("data/processed/gsxr_ai_analysis.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(analysis_results, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_top_candidates(
    analysis_results: list[dict[str, Any]],
    output_path: Path | None = None,
) -> Path:
    path = output_path or Path("data/processed/gsxr_ai_top_candidates.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    ranked = rank_records_for_review(analysis_results)
    path.write_text(json.dumps(ranked, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def summarize_analysis(analysis_results: list[dict[str, Any]]) -> dict[str, Any]:
    severity_counts = Counter(item.get("severity") or "unknown" for item in analysis_results)
    starts_true = sum(1 for item in analysis_results if item.get("starts") is True)
    starts_false = sum(1 for item in analysis_results if item.get("starts") is False)
    ranked = sorted(
        analysis_results,
        key=lambda item: (
            _severity_rank(item.get("severity")),
            len(item.get("damage_analysis", {}).get("suspected_damage", [])),
            len(item.get("hidden_risks", [])),
        ),
        reverse=True,
    )[:5]
    return {
        "analyzed_count": len(analysis_results),
        "severity_counts": dict(severity_counts),
        "starts_true": starts_true,
        "starts_false": starts_false,
        "top_5_most_concerning": ranked,
    }


def rank_records_for_review(
    analysis_results: list[dict[str, Any]],
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    ranked = sorted(
        analysis_results,
        key=lambda item: (
            _sortable_number(item.get("review_priority_score")),
            _severity_rank(item.get("severity")),
            len(item.get("damage_analysis", {}).get("suspected_damage", [])),
            len(item.get("hidden_risks", [])),
            1 if item.get("starts") is False else 0,
        ),
        reverse=True,
    )
    if top_n is not None:
        return ranked[:top_n]
    return ranked


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run AI damage analysis for GSX-R detail records")
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
        "--output-file",
        default="data/processed/gsxr_ai_analysis.json",
        help="Where to save the AI analysis output",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=10,
        help="Limit the number of GSX-R records analyzed",
    )
    parser.add_argument(
        "--engine-cc",
        type=int,
        default=None,
        help="Filter GSX-R records by engine capacity, for example 600 or 125",
    )
    parser.add_argument(
        "--year-from",
        type=int,
        default=None,
        help="Filter records with year >= this value",
    )
    parser.add_argument(
        "--year-to",
        type=int,
        default=None,
        help="Filter records with year <= this value",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="How many top records to save for manual review",
    )
    parser.add_argument(
        "--top-output-file",
        default="data/processed/gsxr_ai_top_candidates.json",
        help="Where to save the top records selected for review",
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
    gsxr_records = filter_gsxr_records(
        records,
        args.max_records,
        engine_cc=args.engine_cc,
        year_from=args.year_from,
        year_to=args.year_to,
    )
    analysis_results = analyze_gsxr_records(gsxr_records)
    output_path = save_gsxr_ai_analysis(analysis_results, Path(args.output_file))
    top_candidates = rank_records_for_review(analysis_results, args.top_n)
    top_output_path = save_top_candidates(top_candidates, Path(args.top_output_file))
    summary = summarize_analysis(analysis_results)

    print(f"Source details file: {details_path}")
    print(f"Analyzed records: {summary['analyzed_count']}")
    if args.engine_cc is not None:
        print(f"Filtered engine_cc: {args.engine_cc}")
    if args.year_from is not None or args.year_to is not None:
        print(f"Filtered years: {args.year_from or '-inf'} to {args.year_to or '+inf'}")
    print(f"low: {summary['severity_counts'].get('low', 0)}")
    print(f"medium: {summary['severity_counts'].get('medium', 0)}")
    print(f"high: {summary['severity_counts'].get('high', 0)}")
    print(f"starts == true: {summary['starts_true']}")
    print(f"starts == false: {summary['starts_false']}")
    print("Top 5 most concerning records:")
    for item in summary["top_5_most_concerning"]:
        listing = item["original_listing"]
        print(
            f"  {listing.get('title')} | severity={item.get('severity')} "
            f"| suspected_damage={len(item.get('damage_analysis', {}).get('suspected_damage', []))}"
        )
    print(f"Top records selected for review: {len(top_candidates)}")
    print(f"Saved AI analysis to {output_path}")
    print(f"Saved top candidates to {top_output_path}")


def _analysis_text(record: dict[str, Any]) -> tuple[str, bool]:
    title = _string_or_none(record.get("title"))
    full_description = _string_or_none(record.get("full_description"))
    short_description = _string_or_none(record.get("short_description"))
    attributes = record.get("attributes")

    attribute_lines: list[str] = []
    if isinstance(attributes, dict):
        for key, value in attributes.items():
            if isinstance(key, str) and isinstance(value, str) and key.strip() and value.strip():
                attribute_lines.append(f"{key}: {value}")

    parts = []
    if title:
        parts.append(f"Tytul: {title}")
    if full_description:
        parts.append(f"Pelny opis: {full_description}")
    elif short_description:
        parts.append(f"Krotki opis: {short_description}")
    if attribute_lines:
        parts.append("Atrybuty:\n" + "\n".join(attribute_lines))

    meaningful_description = _has_meaningful_description(full_description, short_description)

    return "\n\n".join(parts), meaningful_description


def _has_meaningful_description(
    full_description: str | None,
    short_description: str | None,
) -> bool:
    candidate = full_description or short_description
    if not candidate:
        return False

    cleaned = candidate.strip()
    if len(cleaned) < 20:
        return False

    letter_count = sum(1 for char in cleaned if char.isalpha())
    return letter_count >= 15


def _damage_analysis_payload(analysis: DamageAnalysis) -> dict[str, Any]:
    return {
        "found_keywords": analysis.found_keywords,
        "suspected_damage": analysis.suspected_damage,
        "hidden_risks": analysis.hidden_risks,
        "starts": analysis.starts,
        "severity": analysis.severity,
    }


def _human_summary(
    record: dict[str, Any],
    analysis: DamageAnalysis,
    *,
    has_meaningful_description: bool,
) -> str:
    if not has_meaningful_description:
        return "Opis ogloszenia jest zbyt skapy, wiec wynik analizy ma niska wartosc informacyjna."

    parts = []
    if analysis.suspected_damage:
        parts.append(
            "W opisie widac sygnaly problemow z: "
            + ", ".join(analysis.suspected_damage)
            + "."
        )
    else:
        parts.append("W opisie nie ma jednoznacznych sygnalow konkretnych uszkodzen.")

    if analysis.hidden_risks:
        parts.append(
            "Ukryte ryzyka dotycza: " + ", ".join(analysis.hidden_risks) + "."
        )

    if analysis.starts is True:
        parts.append("Opis sugeruje, ze motocykl odpala.")
    elif analysis.starts is False:
        parts.append("Opis sugeruje problem z odpalaniem lub praca silnika.")
    else:
        parts.append("Opis nie daje pewnej informacji, czy motocykl odpala.")

    severity_label = analysis.severity or "unknown"
    parts.append(f"Ocena ogolna ryzyka: {severity_label}.")
    return " ".join(parts[:4])


def _severity_rank(value: object) -> int:
    if value == "high":
        return 3
    if value == "medium":
        return 2
    if value == "low":
        return 1
    return 0


def _review_priority_score(
    analysis: DamageAnalysis,
    *,
    has_meaningful_description: bool,
) -> int:
    score = 0
    score += _severity_rank(analysis.severity) * 30
    score += len(analysis.suspected_damage) * 10
    score += len(analysis.hidden_risks) * 6
    if analysis.starts is False:
        score += 12
    elif analysis.starts is True:
        score -= 4
    if not has_meaningful_description:
        score -= 15
    return score


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


def _string_or_none(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


if __name__ == "__main__":
    main()
