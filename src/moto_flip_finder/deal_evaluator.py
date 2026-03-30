from .damage_analysis import analyze_description
from .models import DealEvaluation, Listing
from .profit import calculate_expected_profit
from .repair_estimator import estimate_repair_cost


def evaluate_listing(
    listing: Listing,
    healthy_market_value_pln: int,
    other_costs_pln: int = 1000,
) -> DealEvaluation:
    damage_analysis = analyze_description(listing.description)
    repair_estimate = estimate_repair_cost(damage_analysis)

    expected_profit = calculate_expected_profit(
        purchase_price_pln=listing.price_pln,
        healthy_market_value_pln=healthy_market_value_pln,
        repair_cost_pln=repair_estimate.total_cost_pln,
        risk_buffer_pln=0,
        other_costs_pln=other_costs_pln,
    )

    score = 50

    if expected_profit > 0:
        score += 20
    if expected_profit > 3000:
        score += 15
    if damage_analysis.starts is True:
        score += 10
    if damage_analysis.severity == "high":
        score -= 20
    elif damage_analysis.severity == "medium":
        score -= 10

    if expected_profit < 0:
        score -= 20

    score = max(0, min(score, 100))
    is_pearl = expected_profit >= 3000 and damage_analysis.severity != "high"

    return DealEvaluation(
        listing=listing,
        damage_analysis=damage_analysis,
        repair_estimate=repair_estimate,
        healthy_market_value_pln=healthy_market_value_pln,
        expected_profit_pln=expected_profit,
        is_pearl=is_pearl,
        score=score,
    )