from dataclasses import dataclass, field


@dataclass
class Listing:
    source: str
    listing_id: str
    url: str
    title: str
    description: str
    price_pln: int
    image_urls: list[str] = field(default_factory=list)
    brand: str | None = None
    model: str | None = None
    year: int | None = None


@dataclass
class DamageAnalysis:
    found_keywords: list[str] = field(default_factory=list)
    suspected_damage: list[str] = field(default_factory=list)
    hidden_risks: list[str] = field(default_factory=list)
    starts: bool | None = None
    severity: str = "unknown"


@dataclass
class RepairEstimate:
    parts_cost_pln: int
    labor_cost_pln: int
    risk_buffer_pln: int
    total_cost_pln: int
    
@dataclass
class DealEvaluation:
    listing: Listing
    damage_analysis: DamageAnalysis
    repair_estimate: RepairEstimate
    healthy_market_value_pln: int
    expected_profit_pln: int
    is_pearl: bool
    score: int