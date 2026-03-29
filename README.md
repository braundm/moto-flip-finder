# moto-flip-finder

AI-assisted system for identifying undervalued damaged motorcycles from classifieds by estimating repair cost, market value, and potential flip profit.

## Problem

Damaged motorcycle listings are difficult to evaluate quickly.  
This project aims to collect listings, analyze visible damage from photos and text, estimate repair cost, compare the asking price with healthy market value, and rank the best flip opportunities.

## Planned pipeline

1. Collect damaged motorcycle listings
2. Extract title, description, price, and images
3. Analyze listing text and photos
4. Estimate likely damage and repair cost
5. Compare with healthy market value
6. Score potential profitability
7. Rank the best opportunities

## Example output

Each listing should produce a structured result such as:

- listing price
- estimated repair cost
- estimated healthy market value
- expected profit
- risk level
- final flip score

## Tech stack

- Python
- Web scraping / data collection
- Vision LLM API for damage analysis
- Rule-based scoring and valuation
- JSON / CSV / database storage

## Project status

Early development.  
The repository is currently focused on architecture, data models, pipeline design, and MVP implementation.

## Roadmap

- [ ] Define listing data model
- [ ] Build single-listing parser
- [ ] Download and store listing images
- [ ] Add damage analysis module
- [ ] Add repair cost estimator
- [ ] Add healthy market price comparison
- [ ] Add profit scoring
- [ ] Add ranked results output
