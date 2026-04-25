# moto-flip-finder

`moto-flip-finder` is a machine-learning oriented Python project for scraping motorcycle listings from OLX, building a clean tabular dataset, estimating fair prices, and exporting ranked HTML reports with links and photos.

The project started around damaged Suzuki GSX-R listings, but the current main direction is broader:

1. scrape OLX search results for motorcycles
2. fetch listing details
3. validate and normalize listings with deterministic heuristics
4. reject bad listings such as quads, generic junk, or inconsistent records
5. save a clean `ready` dataset
6. train the most sensible price backend for the cleaned data
7. export a report of the best opportunities

## Release v1

`v1.0.0` is the first release-focused version of the repo. The practical scope is:

- one-command brand pipeline from OLX scrape to saved model, predictions, and report
- stable `auto` price-model default that prefers `scikit-learn` on larger tabular datasets
- built-in comparable-based fallback for smaller datasets
- optional `PyTorch` backend for tabular price prediction experiments
- tested legacy GSX-R flow for damage and profitability analysis

## Current State

What is working today:

- OLX search scraping with pagination and deduplication
- OLX detail scraping with structured field extraction
- heuristic listing validation for dataset cleaning and normalization
- ready-dataset builder for brand-focused subsets such as `Kawasaki`
- one-command brand pipeline for `ready` dataset plus model outputs and HTML report
- price model training on `ready` datasets with `auto`, `scikit-learn`, `comparable`, or `PyTorch`
- legacy GSX-R flow for damage analysis, repair estimation, and deal ranking

There are effectively two pipelines in the repo:

1. `all motorcycles / ready-data pipeline`
   This is the current direction for scaling to many motorcycle brands.
2. `GSX-R damaged flip pipeline`
   This is still useful and tested, but now acts more like a specialized legacy flow.

## Install

Create a virtual environment and install the project:

```bash
python -m venv .venv
.venv/bin/pip install -e .
```

Install the optional `PyTorch` backend:

```bash
.venv/bin/pip install -e ".[torch]"
```

Note:

- on Linux, a plain `torch` install may pull a large CUDA-enabled wheel
- if you want a lighter CPU-only setup, install the appropriate CPU build of `PyTorch` first, then install this project

The project currently depends on:

- `python-dotenv`
- `beautifulsoup4`
- `joblib`
- `pandas`
- `scikit-learn`
- optional `torch`

After installation, the release CLI is available as:

```bash
.venv/bin/moto-flip-finder --help
```

## Environment

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Typical variables:

```env
MOTO_FLIP_FINDER_ANALYSIS_DEBUG=0
```

Notes:

- when `MOTO_FLIP_FINDER_ANALYSIS_DEBUG=1`, the heuristics layer prints provider/debug information to `stderr`
- the main pipeline is intentionally local and deterministic, so the same inputs and seeds produce reproducible artifacts

## Main Pipeline

### 1. Search all motorcycles on OLX

```bash
cd /home/admin/Desktop/projekty/moto-flip-finder
PYTHONPATH=src .venv/bin/python -m moto_flip_finder.sources.olx.import_motorcycles_search \
  --url "https://www.olx.pl/motoryzacja/motocykle-skutery/" \
  --max-pages 10 \
  --delay-seconds 0.5
```

Outputs:

- `data/raw/olx_motorcycles_<timestamp>_raw.json`
- `data/raw/olx_motorcycles_<timestamp>_records.json`

### 2. Fetch details and validate listings

```bash
cd /home/admin/Desktop/projekty/moto-flip-finder
LATEST="$(ls -t data/raw/olx_motorcycles_*_records.json | head -n 1)"
MOTO_FLIP_FINDER_ANALYSIS_DEBUG=1 PYTHONPATH=src .venv/bin/python -m moto_flip_finder.sources.olx.import_motorcycles_details \
  --records-file "$LATEST" \
  --max-records 50 \
  --delay-seconds 0.5 \
  --max-workers 6
```

Outputs:

- `data/raw/olx_motorcycles_<timestamp>_details.json`

The detail stage extracts hard fields from OLX and then applies deterministic heuristics to resolve or validate fields such as:

- `vehicle_type`
- `resolved_brand`
- `resolved_price_pln`
- `resolved_engine_cc`
- `resolved_year`
- `negotiable`
- `mileage_km`
- `is_sensible_listing`
- `reject_reason`
- `validation_confidence`
- `validation_summary`
- `data_issues`

### 3. Build a ready dataset

This is the main entry point if we want one clean JSON file for model training.

Example: build a `Kawasaki` dataset from all OLX pages while cheaply filtering the search stage first.

```bash
cd /home/admin/Desktop/projekty/moto-flip-finder
PYTHONPATH=src .venv/bin/python -m moto_flip_finder.build_ready_motorcycles_dataset \
  --url "https://www.olx.pl/motoryzacja/motocykle-skutery/q-kawasaki/" \
  --max-pages 50 \
  --max-records 500 \
  --search-delay-seconds 0.5 \
  --detail-delay-seconds 0.2 \
  --detail-max-workers 6 \
  --dataset-label kawasaki_all_pages \
  --keyword-filter kawasaki \
  --required-brand Kawasaki
```

Outputs:

- filtered search records in `data/raw/`
- validated detail records in `data/raw/`
- final ready dataset in `data/ready/`

Example final file:

- `data/ready/olx_kawasaki_all_pages_ready_<timestamp>.json`

A ready record is intentionally compact and model-friendly. It contains fields such as:

- `source`
- `url`
- `title`
- `price_pln`
- `negotiable`
- `location`
- `short_description`
- `full_description`
- `image_urls`
- `brand`
- `year`
- `technical_state`
- `origin_country`
- `seller_type`
- `engine_cc`
- `vehicle_type`
- `mileage_km`
- `validation_confidence`
- `validation_summary`
- `data_issues`

### 4. Train a price model on ready data

```bash
cd /home/admin/Desktop/projekty/moto-flip-finder
PYTHONPATH=src .venv/bin/python -m moto_flip_finder.train_ready_price_model \
  --ready-file data/ready/olx_kawasaki_all_pages_ready_20260331T172001Z.json \
  --required-brand Kawasaki \
  --backend auto \
  --min-family-records 3 \
  --model-output data/models/kawasaki_ready_price_model_v3.joblib \
  --report-output data/processed/kawasaki_ready_price_model_report_v3.json \
  --predictions-output data/processed/kawasaki_ready_price_predictions_v3.json \
  --search-iterations 8 \
  --cv-folds 5
```

Outputs:

- trained model in `data/models/`
- training report in `data/processed/`
- per-record predictions in `data/processed/`
- HTML opportunity report with links and photos in `data/processed/`

## ML Design

This repository is meant to show practical ML engineering rather than a notebook-only workflow.

Main modeling decisions:

- `scikit-learn` is the default for serious runs because the data is mostly tabular and tree models are robust on small and medium datasets.
- `auto` switches to a comparable-based median estimator when the dataset is too small for a sensible supervised model.
- `PyTorch` is kept as an explicit experimental backend for neural-network work on the same cleaned dataset.

Why this setup makes sense:

- it avoids overclaiming with a neural net on tiny data
- it keeps the pipeline usable even when a scrape produces only a modest sample
- it makes the modeling choice easy to explain in an interview

Validation and experiment structure:

- deterministic cleaning and filtering before training
- explicit holdout validation split
- cross-validation for the `scikit-learn` path
- saved JSON report with validation metrics
- saved predictions for the same dataset used for ranking and inspection
- fixed `random_seed` parameters in the training CLI

Key metrics exposed today:

- `MAE`
- `RMSE`
- median absolute error
- `MAPE`

Example with the optional `PyTorch` backend:

```bash
cd /home/admin/Desktop/projekty/moto-flip-finder
PYTHONPATH=src .venv/bin/python -m moto_flip_finder.train_ready_price_model \
  --ready-file data/ready/olx_kawasaki_all_pages_ready_20260331T172001Z.json \
  --required-brand Kawasaki \
  --backend torch \
  --min-family-records 3 \
  --report-limit 20 \
  --search-iterations 4 \
  --cv-folds 5
```

### 5. Run The Full Brand Pipeline In One Command

If we want one practical `v1` flow for a single brand, we can now run the whole path:

- OLX search scrape
- detail fetch and heuristic validation
- ready dataset build
- automatic choice of the most sensible price backend
- saved predictions for the ready dataset
- saved HTML report with listing links and photos

Example:

```bash
cd /home/admin/Desktop/projekty/moto-flip-finder
.venv/bin/moto-flip-finder brand-pipeline \
  --brand Kawasaki \
  --backend auto \
  --max-pages 50 \
  --max-records 500 \
  --search-delay-seconds 0.5 \
  --detail-delay-seconds 0.2 \
  --detail-max-workers 6 \
  --min-family-records 3 \
  --search-iterations 8 \
  --cv-folds 5
```

Default behavior of this command:

- inferred OLX search URL like `.../q-kawasaki/`
- keyword filter defaults to the brand name
- final ready dataset is restricted to the exact brand
- with enough data, the pipeline usually resolves to `scikit-learn`
- with smaller data, it can resolve to the comparable-based fallback
- saved model defaults to `data/models/kawasaki_ready_price_model_v1.joblib` when `scikit-learn` wins
- saved report defaults to `data/processed/kawasaki_ready_price_model_v1_report.json`
- saved predictions default to `data/processed/kawasaki_ready_price_model_v1_predictions.json`
- saved HTML report defaults to `data/processed/kawasaki_ready_price_model_v1_report.html`

If the resolved backend is not the default `scikit-learn` one, the output stem includes the real backend, for example:

- `kawasaki_ready_price_model_comparable_v1`
- `kawasaki_ready_price_model_torch_v1`

The HTML report includes:

- ranked listings by positive `price_delta_pln`
- clickable source links
- primary listing photo
- listed price vs predicted price
- short explanation based on the listing fields

The report is print-friendly, so you can open it in a browser and save it as PDF.

## CLI

Top-level release commands:

- `moto-flip-finder brand-pipeline`
- `moto-flip-finder build-ready-dataset`
- `moto-flip-finder train-ready-price`
- `moto-flip-finder train-legacy-price`
- `moto-flip-finder evaluate-damaged`
- `moto-flip-finder gsxr-analysis`

## Current ML Snapshot

Current `Kawasaki` baseline after family cleanup and outlier trimming:

- validation `MAE`: about `2829.57 PLN`
- validation `RMSE`: about `4538.68 PLN`
- validation `median absolute error`: about `1398.61 PLN`
- validation `MAPE`: about `13.58%`

Interpretation:

- the model is already useful as a baseline ranking and sanity-check layer
- it is not yet a production-grade final pricing model
- the next quality jump will come from better `model_family` segmentation and more data per family

## Why These Models

For interview purposes, the current model stack is intentional:

- `ExtraTrees` / `RandomForest` are strong baselines for mixed tabular features such as year, engine size, seller type, mileage, and normalized model family.
- the comparable-based fallback shows that the system can still produce reasoned outputs when data is too sparse for a supervised model.
- the optional `PyTorch` path demonstrates how the same training interface can support neural-network experiments without changing the scraping and feature pipeline.

This is closer to real ML engineering than forcing one model family everywhere.

## Legacy GSX-R Flow

The older GSX-R-specific pipeline is still present and tested:

- `src/moto_flip_finder/sources/olx/import_search.py`
- `src/moto_flip_finder/sources/olx/import_details.py`
- `src/moto_flip_finder/build_datasets.py`
- `src/moto_flip_finder/gsxr_ai_analysis.py`
- `src/moto_flip_finder/market_value.py`
- `src/moto_flip_finder/evaluate_damaged_listings.py`
- `src/moto_flip_finder/repair_estimator.py`
- `src/moto_flip_finder/deal_evaluator.py`

That flow is still the best place for:

- damage analysis
- repair estimation
- expected profit
- deal ranking
- HTML deal reports

## Repository Layout

Core files for the current OLX-to-ready-data pipeline:

- `src/moto_flip_finder/main.py`
- `src/moto_flip_finder/sources/olx/import_motorcycles_search.py`
- `src/moto_flip_finder/sources/olx/import_motorcycles_details.py`
- `src/moto_flip_finder/motorcycle_listing_validation.py`
- `src/moto_flip_finder/build_ready_motorcycles_dataset.py`
- `src/moto_flip_finder/price_model.py`
- `src/moto_flip_finder/torch_price_model.py`
- `src/moto_flip_finder/train_price_model.py`
- `src/moto_flip_finder/train_ready_price_model.py`
- `src/moto_flip_finder/run_brand_price_pipeline.py`

Important legacy-but-active files:

- `src/moto_flip_finder/description_analysis_provider.py`
- `src/moto_flip_finder/damage_analysis.py`
- `src/moto_flip_finder/market_value.py`
- `src/moto_flip_finder/evaluate_damaged_listings.py`

## Testing

Run the full test suite:

```bash
cd /home/admin/Desktop/projekty/moto-flip-finder
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/pytest -q -p no:cacheprovider
```

Current status:

- `93 passed`

## Data Hygiene

Generated artifacts are intentionally not meant for git:

- `data/raw/`
- `data/ready/`
- `data/models/`
- generated JSON files in `data/processed/`
- generated HTML reports

The repository should keep code, tests, and docs. Large run outputs stay local.

## What We Do Next

The next practical roadmap is:

1. let the user choose a target scope such as brand, family, year range, or price band
2. scrape OLX broadly for that scope
3. build a clean ready dataset
4. train or reuse a pricing model
5. compare specialist models against more general ones
6. generate ranked reports for manual review

Near-term technical priorities:

1. better `model_family` and `model_variant` extraction across more brands
2. per-family price models for brands like `Kawasaki`, `Honda`, `Yamaha`, `Suzuki`
3. stronger filtering of weird business cases before model training
4. a future generalized model for all motorcycles on OLX
5. stronger experiment tracking and model comparison for the `PyTorch` branch
