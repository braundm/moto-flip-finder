# Roadmap

## Current Foundation

What already works:

- release-oriented `v1` CLI for the main project flows
- OLX motorcycle search scraping with pagination
- OLX detail scraping
- deterministic heuristic validation and normalization of uncertain listing fields
- ready-dataset generation for focused subsets such as `Kawasaki`
- one-command brand pipeline from scrape to saved model outputs
- HTML opportunity report with links and photos for brand price predictions
- baseline `scikit-learn` and optional `PyTorch` price models trained on ready data
- legacy GSX-R damage-analysis and profitability flow

## Near Term

1. improve `model_family` and `model_variant` extraction across more brands
2. add ready datasets for `Honda`, `Yamaha`, and `Suzuki`
3. train separate per-family price models, for example `Z`, `ZX`, `Ninja`, `Versys`
4. improve filtering of suspicious business cases before training
5. keep generated data local and commit only code, tests, and docs

## Mid Term

1. allow the user to choose a target scope such as brand, family, year range, or budget
2. run scrape -> heuristic validation -> ready dataset as one practical workflow
3. combine pricing, risk analysis, and report generation in one repeatable pipeline
4. compare generalized models against per-family specialist models

## Longer Term

1. expand from brand-specific datasets to a generalized all-motorcycles OLX dataset
2. add stronger normalization for model families and generations
3. blend ML pricing with stronger local damage and risk analysis
4. add a UI or dashboard for interactive filtering and review
