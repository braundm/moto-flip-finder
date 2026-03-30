# moto-flip-finder

`moto-flip-finder` is a Python project for finding motorcycle flip opportunities from marketplace listings.

Today the project is centered on OLX and Suzuki GSX-R data, but the pipeline is already structured like a real product:

1. the user chooses what motorcycle to inspect
2. the scraper collects broad marketplace data
3. listing records are normalized and split into healthy vs suspicious
4. AI analyzes listing descriptions
5. market value and repair cost are estimated
6. the project generates ranked reports for manual review

## What Works Today

The repository currently supports:

- OLX search scraping for broad `gsxr` result sets
- OLX detail scraping with normalized structured fields
- healthy vs damaged candidate separation
- OpenAI-backed description analysis with heuristic fallback
- generation-aware comparable selection for valuation
- damaged listing evaluation with:
  - `healthy_market_value`
  - `repair_estimate`
  - `expected_profit`
  - `roi_percent`
  - `deal_score`
- AI listing analysis on real `*_details.json` files
- HTML report generation with top deals and downloaded listing images

## Current User Flow

The intended flow for the user is:

1. pick motorcycle parameters
2. scrape listings
3. enrich and normalize details
4. classify listings
5. run AI analysis
6. generate valuation and report

In the current codebase, the most important user-controlled parameters are:

- engine capacity, for example `600` or `125`
- year range
- how many records to analyze
- how many top offers to return at the end

## Project Layout

- `src/moto_flip_finder/sources/olx/import_search.py`
  OLX search scraper with pagination
- `src/moto_flip_finder/sources/olx/import_details.py`
  OLX detail scraper and field normalization
- `src/moto_flip_finder/build_datasets.py`
  dataset split into healthy and damaged records
- `src/moto_flip_finder/description_analysis_provider.py`
  OpenAI provider and response normalization
- `src/moto_flip_finder/damage_analysis.py`
  provider selection and heuristic fallback
- `src/moto_flip_finder/market_value.py`
  comparable selection and market value estimation
- `src/moto_flip_finder/evaluate_damaged_listings.py`
  profitability analysis and HTML report generation
- `src/moto_flip_finder/gsxr_ai_analysis.py`
  AI analysis runner for real OLX detail records
- `src/moto_flip_finder/repair_estimator.py`
  repair cost estimation
- `src/moto_flip_finder/deal_evaluator.py`
  listing-level evaluation logic used by sample flow

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv/bin/pip install -e .
```

## Environment

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Important variables:

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5.1
```

Notes:

- if `OPENAI_API_KEY` is set, description analysis uses OpenAI
- if the OpenAI request fails, the code falls back to heuristics
- if you want to override the model for your account, change `OPENAI_MODEL`

## End-to-End Pipeline

### 1. Search Import

Scrape broad OLX search results for `gsxr`:

```bash
cd /home/admin/Desktop/projekty/moto-flip-finder
PYTHONPATH=src .venv/bin/python -m moto_flip_finder.sources.olx.import_search \
  --url "https://www.olx.pl/motoryzacja/motocykle-skutery/q-gsxr/" \
  --max-pages 10 \
  --delay-seconds 1.0
```

Output:

- `data/raw/olx_gsxr_<timestamp>_raw.json`
- `data/raw/olx_gsxr_<timestamp>_records.json`

### 2. Detail Import

Fetch full details for saved records:

```bash
cd /home/admin/Desktop/projekty/moto-flip-finder
LATEST="$(ls -t data/raw/*_records.json | head -n 1)"
PYTHONPATH=src .venv/bin/python -m moto_flip_finder.sources.olx.import_details \
  --records-file "$LATEST" \
  --all-records \
  --delay-seconds 1.0
```

Output:

- `data/raw/olx_gsxr_<timestamp>_details.json`

### 3. Build Processed Datasets

Split detail records into all / healthy / damaged:

```bash
cd /home/admin/Desktop/projekty/moto-flip-finder
PYTHONPATH=src .venv/bin/python -m moto_flip_finder.build_datasets
```

Output:

- `data/processed/all_gsxr.json`
- `data/processed/healthy_comps.json`
- `data/processed/damaged_candidates.json`

### 4. Evaluate Damaged Listings

Run profitability analysis and generate reports:

```bash
cd /home/admin/Desktop/projekty/moto-flip-finder
MOTO_FLIP_FINDER_ANALYSIS_DEBUG=1 PYTHONPATH=src .venv/bin/python -m moto_flip_finder.evaluate_damaged_listings \
  --top-profit-limit 3
```

Output:

- `data/processed/damaged_evaluations.json`
- `data/processed/top_deals.json`
- `data/processed/top_profit_report.html`
- `data/processed/top_profit_assets/`

### 5. Run AI Analysis on Real Listings

Analyze current OLX detail records through the OpenAI provider using user parameters.

Example for `GSX-R 600`, years `2004-2008`, analyze `20` records, return top `10`:

```bash
cd /home/admin/Desktop/projekty/moto-flip-finder
OPENAI_MODEL=gpt-5.1 MOTO_FLIP_FINDER_ANALYSIS_DEBUG=1 PYTHONPATH=src .venv/bin/python -m moto_flip_finder.gsxr_ai_analysis \
  --engine-cc 600 \
  --year-from 2004 \
  --year-to 2008 \
  --max-records 20 \
  --top-n 10
```

Example for `GSX-R 125`, years `2017-2021`, return top `8`:

```bash
cd /home/admin/Desktop/projekty/moto-flip-finder
OPENAI_MODEL=gpt-5.1 MOTO_FLIP_FINDER_ANALYSIS_DEBUG=1 PYTHONPATH=src .venv/bin/python -m moto_flip_finder.gsxr_ai_analysis \
  --engine-cc 125 \
  --year-from 2017 \
  --year-to 2021 \
  --max-records 20 \
  --top-n 8
```

Output:

- `data/processed/gsxr_ai_analysis.json`
- `data/processed/gsxr_ai_top_candidates.json`

## Key Output Files

### `data/raw/*_records.json`

Search-stage listing records, used as input for detail scraping.

### `data/raw/*_details.json`

Normalized detail records with fields like:

- `title`
- `price_pln`
- `url`
- `location`
- `city`
- `region`
- `short_description`
- `full_description`
- `image_urls`
- `brand`
- `model_family`
- `engine_cc`
- `normalized_model`
- `year`
- `technical_state`
- `origin_country`
- `seller_type`
- `attributes`

### `data/processed/healthy_comps.json`

Healthy comparable motorcycles for market value estimation.

### `data/processed/damaged_candidates.json`

Suspicious or damaged listings that should be evaluated further.

### `data/processed/damaged_evaluations.json`

Each record contains:

- `original_listing`
- `healthy_market_value`
- `comparables_count`
- `valuation_status`
- `valuation_confidence`
- `generation_hint`
- `matched_same_generation`
- `selected_comparables_strategy`
- `damage_analysis`
- `repair_estimate`
- `expected_profit`
- `roi_percent`
- `deal_score`

### `data/processed/top_profit_report.html`

Human-readable HTML report with:

- first downloaded OLX image
- listing title
- purchase price
- estimated profit
- location
- link
- short explanation why the bike is worth checking

## Normalization And Valuation Notes

### Model normalization

Current normalization uses:

- `brand`
- `model_family`
- `engine_cc`

Examples:

- `Suzuki` + `gsxr` + `600` -> `gsxr_600`
- `Suzuki` + `gsxr` + `750` -> `gsxr_750`
- `Suzuki` + `gsxr` + `1000` -> `gsxr_1000`

### Generation hint

The valuation layer detects generation-like tokens from text, for example:

- `k1`, `k2`, `k3`, `k4`, `k5`, `k6`, `k7`, `k8`, `k9`
- `l0`, `l1`, `l2`, `l3`, `l4`, `l5`, `l6`, `l7`, `l8`, `l9`
- `m0`, `m1`, `m2`, and similar

Comparable selection priority is:

1. same `normalized_model`
2. same `generation_hint`, if available
3. same model within `year ±1`
4. same model within `year ±2`
5. same model fallback

### AI analysis behavior

The AI analysis runner builds input from:

- `title`
- `full_description`
- `attributes`

If the listing text is too sparse, the result is still saved but marked as low information value.

## Tests

Run the full suite:

```bash
cd /home/admin/Desktop/projekty/moto-flip-finder
PYTHONPATH=src .venv/bin/pytest -q
```

## Current Status Before Commit

At the time of the last cleanup pass:

- the full test suite passed
- `52 passed`
- the project generated:
  - processed datasets
  - valuation JSON
  - ranked JSON
  - AI analysis JSON
  - HTML report with local images

## Roadmap

The next practical steps for the project are:

1. generalize input parameters beyond GSX-R-only flow
2. let the user define the motorcycle family directly, not only `engine_cc`
3. improve OpenAI prompts and output normalization on noisy real-world descriptions
4. add a second report view by `deal_score`, not only `expected_profit`
5. optionally add ML-based ranking or anomaly scoring after the scraping stage
6. expand beyond OLX into more than one source

## Product Direction

The intended long-term user experience is:

1. the user enters search parameters:
   motorcycle family, displacement, year range, and result count
2. the system scrapes marketplaces
3. the system normalizes and filters listings
4. optional ML scoring narrows noisy results
5. AI analyzes condition, hidden risks, and starting probability
6. valuation compares the listing with healthy market comps
7. the system generates final ranked reports with images and links for manual review
