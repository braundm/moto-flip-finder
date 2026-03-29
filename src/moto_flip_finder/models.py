from dataclasses import dataclass


@dataclass
class Listing:
    source: str
    title: str
    price_pln: int
    url: str
    description: str
