from .parser import parse_listing

def main():
    raw_listing = {
        "source": "olx",
        "listing_id": "123456",
        "url": "https://www.olx.pl/d/oferta/test",
        "title": "Yamaha R6 po szlifie",
        "description": "Motocykl uszkodzony, do poprawek owiewki i klamka.",
        "price_pln": 18900,
        "image_urls": [
            "https://example.com/1.jpg",
            "https://example.com/2.jpg",
        ],
        "brand": "Yamaha",
        "model": "R6",
        "year": 2008,
    }

    listing = parse_listing(raw_listing)
    print(listing)


if __name__ == "__main__":
    main()