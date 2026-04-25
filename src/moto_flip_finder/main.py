from __future__ import annotations

import sys

from . import __version__
from .build_ready_motorcycles_dataset import main as build_ready_dataset_main
from .evaluate_damaged_listings import main as evaluate_damaged_main
from .run_brand_price_pipeline import main as run_brand_price_pipeline_main
from .train_ready_price_model import main as train_ready_price_model_main


COMMANDS = {
    "brand-pipeline": run_brand_price_pipeline_main,
    "build-ready-dataset": build_ready_dataset_main,
    "train-ready-price": train_ready_price_model_main,
    "evaluate-damaged": evaluate_damaged_main,
}


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        _print_help()
        return

    if args[0] in {"--version", "version"}:
        print(__version__)
        return

    command = args[0]
    handler = COMMANDS.get(command)
    if handler is None:
        supported = ", ".join(sorted(COMMANDS))
        raise SystemExit(f"Unknown command: {command}. Expected one of: {supported}")

    original_argv = sys.argv[:]
    try:
        sys.argv = [f"{sys.argv[0]} {command}", *args[1:]]
        handler()
    finally:
        sys.argv = original_argv


def _print_help() -> None:
    print(f"moto-flip-finder v{__version__}")
    print("Usage: moto-flip-finder <command> [options]")
    print("")
    print("Commands:")
    print("  brand-pipeline      Run scrape -> ready dataset -> robust price report for one brand")
    print("  build-ready-dataset Build a ready dataset from OLX with validation")
    print("  train-ready-price   Train a ready-dataset price model with auto/sklearn/torch/comparable")
    print("  evaluate-damaged    Score damaged candidates with valuation and repair logic")
    print("")
    print("Use '<command> --help' for command-specific options.")


if __name__ == "__main__":
    main()
