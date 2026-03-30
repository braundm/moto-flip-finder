from .deal_evaluator import evaluate_listing
from .parser import parse_listing
from .sample_data import SAMPLE_LISTINGS


HEALTHY_MARKET_VALUES = {
    "R6": 29000,
    "GSX-R 600": 24000,
    "CBR 600 RR": 27000,
}


def main():
    evaluations = []

    for raw_listing in SAMPLE_LISTINGS:
        listing = parse_listing(raw_listing)
        healthy_value = HEALTHY_MARKET_VALUES.get(listing.model, 0)
        evaluation = evaluate_listing(listing, healthy_value)
        evaluations.append(evaluation)

    evaluations.sort(key=lambda item: item.expected_profit_pln, reverse=True)

    for evaluation in evaluations:
        print("-" * 60)
        print(evaluation.listing.title)
        print("Price:", evaluation.listing.price_pln, "PLN")
        print("Healthy value:", evaluation.healthy_market_value_pln, "PLN")
        print("Repair total:", evaluation.repair_estimate.total_cost_pln, "PLN")
        print("Expected profit:", evaluation.expected_profit_pln, "PLN")
        print("Severity:", evaluation.damage_analysis.severity)
        print("Starts:", evaluation.damage_analysis.starts)
        print("Pearl:", evaluation.is_pearl)
        print("Score:", evaluation.score)
        
if __name__ == "__main__":
    main()