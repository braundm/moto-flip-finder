
from moto_flip_finder.parser import parse_listing


def test_parse_listing():
    data = {
        "source": "olx",
        "listing_id": "1",
        "url": "https://example.com",
        "title": "Suzuki GSX-R uszkodzony",
        "description": "Do naprawy bok i klamka.",
        "price_pln": 12000,
    }

    listing = parse_listing(data)

    assert listing.source == "olx"
    assert listing.listing_id == "1"
    assert listing.price_pln == 12000
    assert listing.image_urls == []