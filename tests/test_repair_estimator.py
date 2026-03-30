from moto_flip_finder.damage_analysis import analyze_description
from moto_flip_finder.repair_estimator import estimate_repair_cost


def test_estimate_repair_cost(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    text = "Motocykl po szlifie. Do wymiany owiewki i klamka. Silnik odpala."

    analysis = analyze_description(text)
    estimate = estimate_repair_cost(analysis)

    assert estimate.parts_cost_pln == 1950
    assert estimate.labor_cost_pln == 480
    assert estimate.risk_buffer_pln == 3500
    assert estimate.total_cost_pln == 5930
