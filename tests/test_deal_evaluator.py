from moto_flip_finder.deal_evaluator import evaluate_listing
from moto_flip_finder.parser import parse_listing


def test_evaluate_listing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    raw_listing = {
        "source": "olx",
        "listing_id": "001",
        "url": "https://example.com",
        "title": "Yamaha R6 po szlifie",
        "description": "Motocykl po szlifie. Do wymiany owiewki i klamka. Silnik odpala.",
        "price_pln": 18900,
        "brand": "Yamaha",
        "model": "R6",
        "year": 2008,
    }

    listing = parse_listing(raw_listing)
    evaluation = evaluate_listing(listing, healthy_market_value_pln=29000)

    assert evaluation.healthy_market_value_pln == 29000
    assert evaluation.repair_estimate.total_cost_pln > 0
    assert isinstance(evaluation.expected_profit_pln, int)
