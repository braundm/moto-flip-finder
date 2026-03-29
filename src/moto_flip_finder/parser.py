from .models import Listing


def parse_listing(data: dict) -> Listing:
    return Listing(
        source=data["source"],
        listing_id=data["listing_id"],
        url=data["url"],
        title=data["title"],
        description=data["description"],
        price_pln=int(data["price_pln"]),
        image_urls=data.get("image_urls", []),
        brand=data.get("brand"),
        model=data.get("model"),
        year=data.get("year"),
    )