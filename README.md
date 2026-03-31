# moto-flip-finder

`moto-flip-finder` is a Python project for scraping motorcycle listings from OLX, enriching them with OpenAI, building ready-to-train datasets, and training baseline pricing models.

The project started around damaged Suzuki GSX-R listings, but the current main direction is broader:

1. scrape OLX search results for motorcycles
2. fetch listing details
3. use OpenAI to validate and enrich missing fields
4. reject bad listings such as quads, generic junk, or inconsistent records
5. save a clean `ready` dataset
6. train `scikit-learn` pricing models on the cleaned data

## Current State

What is working today:

- OLX search scraping with pagination and deduplication
- OLX detail scraping with structured field extraction
- OpenAI-backed listing validation and enrichment
- ready-dataset builder for brand-focused subsets such as `Kawasaki`
- `scikit-learn` price model training on `ready` datasets
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

The project currently depends on:

- `openai`
- `python-dotenv`
- `beautifulsoup4`
- `pandas`
- `scikit-learn`

## Environment

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Typical variables:

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5-nano
OPENAI_TIMEOUT_SECONDS=60
MOTO_FLIP_FINDER_ANALYSIS_DEBUG=0
```

Notes:

- if `OPENAI_API_KEY` is set, motorcycle validation and description analysis use OpenAI
- if OpenAI fails, the validation layer falls back to an empty validation result instead of crashing the run
- `gpt-5-nano` is a practical low-cost default for large OLX batches
- when `MOTO_FLIP_FINDER_ANALYSIS_DEBUG=1`, raw OpenAI JSON is printed to `stderr`

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

### 2. Fetch details and validate with OpenAI

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

The detail stage extracts hard fields from OLX and lets OpenAI resolve uncertain ones such as:

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
- OpenAI-enriched detail records in `data/raw/`
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

- `src/moto_flip_finder/sources/olx/import_motorcycles_search.py`
- `src/moto_flip_finder/sources/olx/import_motorcycles_details.py`
- `src/moto_flip_finder/motorcycle_listing_validation.py`
- `src/moto_flip_finder/build_ready_motorcycles_dataset.py`
- `src/moto_flip_finder/price_model.py`
- `src/moto_flip_finder/train_price_model.py`
- `src/moto_flip_finder/train_ready_price_model.py`

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

Current status before this commit:

- `73 passed`

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
3. validate and enrich listings with OpenAI
4. build a clean ready dataset
5. train or reuse a pricing model
6. use AI again for damage or risk analysis where needed
7. generate ranked reports for manual review

Near-term technical priorities:

1. better `model_family` and `model_variant` extraction across more brands
2. per-family price models for brands like `Kawasaki`, `Honda`, `Yamaha`, `Suzuki`
3. stronger filtering of weird business cases before model training
4. a future generalized model for all motorcycles on OLX
