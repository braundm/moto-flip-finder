from .models import DamageAnalysis, RepairEstimate


PARTS_COSTS = {
    "fairings": 1800,
    "lever": 150,
    "mirror": 200,
    "footpeg": 250,
    "tank": 1200,
    "wheel": 900,
    "exhaust": 800,
    "forks": 1800,
    "frame": 4000,
    "swingarm": 1500,
}

LABOR_COSTS = {
    "fairings": 400,
    "lever": 80,
    "mirror": 50,
    "footpeg": 100,
    "tank": 300,
    "wheel": 250,
    "exhaust": 150,
    "forks": 600,
    "frame": 1500,
    "swingarm": 700,
}


def estimate_repair_cost(analysis: DamageAnalysis) -> RepairEstimate:
    parts_cost = 0
    labor_cost = 0

    for item in analysis.suspected_damage:
        parts_cost += PARTS_COSTS.get(item, 0)
        labor_cost += LABOR_COSTS.get(item, 0)

    risk_buffer = 0

    for item in analysis.hidden_risks:
        if item == "frame":
            risk_buffer += 1500
        elif item == "forks":
            risk_buffer += 800
        elif item == "swingarm":
            risk_buffer += 700
        elif item == "wheel":
            risk_buffer += 500

    if analysis.starts is False:
        risk_buffer += 2000

    if analysis.severity == "high":
        risk_buffer += 1500
    elif analysis.severity == "medium":
        risk_buffer += 500

    total_cost = parts_cost + labor_cost + risk_buffer

    return RepairEstimate(
        parts_cost_pln=parts_cost,
        labor_cost_pln=labor_cost,
        risk_buffer_pln=risk_buffer,
        total_cost_pln=total_cost,
    )