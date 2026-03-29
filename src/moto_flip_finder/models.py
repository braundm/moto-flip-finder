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