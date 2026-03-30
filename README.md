# moto-flip-finder

`moto-flip-finder` helps evaluate damaged motorcycle listings by parsing listing data, analyzing the description, estimating repair cost, and scoring flip potential.

## Current scope

The current MVP works on sample listing data and produces:

- parsed listing data
- normalized damage analysis
- repair cost estimate
- expected profit
- flip score

Description analysis uses:

- `OpenAI` when `OPENAI_API_KEY` is available
- heuristic analysis when no API key is configured
- heuristic fallback when an OpenAI request fails at runtime

## Project layout

- `src/moto_flip_finder/parser.py`: listing parsing
- `src/moto_flip_finder/damage_analysis.py`: provider selection and heuristic fallback
- `src/moto_flip_finder/description_analysis_provider.py`: OpenAI description analysis
- `src/moto_flip_finder/repair_estimator.py`: repair cost estimation
- `src/moto_flip_finder/deal_evaluator.py`: final deal evaluation
- `src/moto_flip_finder/main.py`: CLI-style local run on sample data

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv/bin/pip install -e .
```

## Environment

Copy `.env.example` to `.env` and fill in your OpenAI key:

```bash
cp .env.example .env
```

Expected variables:

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4.1-mini
```

## Run the program

Run from the project root:

```bash
PYTHONPATH=src .venv/bin/python -m moto_flip_finder.main
```

## Debug provider selection

To see which description-analysis provider was used:

```bash
MOTO_FLIP_FINDER_ANALYSIS_DEBUG=1 PYTHONPATH=src .venv/bin/python -m moto_flip_finder.main
```

You will see exactly one provider line on `stderr` for each analysis:

- `provider=openai`
- `provider=heuristic`
- `provider=heuristic-fallback`

When debug is enabled and OpenAI is used, the raw AI JSON is also printed to `stderr` as:

```text
[damage_analysis] raw_json={...}
```

## OpenAI output contract

The OpenAI description-analysis integration is constrained to the existing repair estimator model.

Allowed normalized damage names:

- `fairings`
- `lever`
- `mirror`
- `footpeg`
- `tank`
- `wheel`
- `exhaust`
- `forks`
- `frame`
- `swingarm`

Allowed severity values:

- `low`
- `medium`
- `high`
- `unknown`

The returned Python object remains the existing `DamageAnalysis` model.

## Tests

Run focused tests:

```bash
PYTHONPATH=src .venv/bin/pytest -q tests/test_damage_analysis.py tests/test_repair_estimator.py
```

The current tests cover:

- heuristic mode without API key
- OpenAI provider selection when API key is present
- fallback to heuristics on provider failure
- payload normalization
- invalid payload handling

## Status

This is still an MVP. The current repository is centered on local evaluation flow and OpenAI-backed description analysis, not yet on full scraping or photo analysis.
