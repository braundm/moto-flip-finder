def calculate_expected_profit(
    purchase_price_pln: int,
    healthy_market_value_pln: int,
    repair_cost_pln: int,
    risk_buffer_pln: int,
    other_costs_pln: int,
) -> int:
    return (
        healthy_market_value_pln
        - purchase_price_pln
        - repair_cost_pln
        - risk_buffer_pln
        - other_costs_pln
    )

