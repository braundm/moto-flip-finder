from moto_flip_finder.sources.olx.import_motorcycles_details import (
    build_detail_record,
    select_detail_targets,
)


class FakeValidationProvider:
    def validate(self, listing: dict) -> dict:
        assert listing["title"] == "Honda CBR 125 JC50 - 2012r"
        return {
            "vehicle_type": "motorcycle",
            "resolved_brand": "Honda",
            "resolved_price_pln": 8600,
            "resolved_engine_cc": 124,
            "resolved_year": 2013,
            "negotiable": True,
            "mileage_km": 38000,
            "is_sensible_listing": True,
            "reject_reason": None,
            "validation_confidence": "high",
            "validation_summary": "Listing looks consistent.",
            "data_issues": [],
        }


def test_select_detail_targets_deduplicates_and_respects_limit():
    records = [
        {"url": "https://www.olx.pl/d/oferta/1"},
        {"url": "https://www.olx.pl/d/oferta/1"},
        {"url": "https://www.olx.pl/d/oferta/2"},
    ]

    selected = select_detail_targets(records, max_records=1)

    assert selected == [{"url": "https://www.olx.pl/d/oferta/1"}]


def test_build_detail_record_extracts_generic_motorcycle_fields():
    seed_record = {
        "title": "Honda CBR 125 JC50 - 2012r",
        "price_pln": 8500,
        "url": "https://www.olx.pl/d/oferta/honda-cbr-125-CID5-ID1.html",
        "location": None,
        "image_urls": ["https://img.seed/1.jpg"],
        "short_description": "8 500 zł | Do negocjacji",
    }
    raw_payloads = {
        "html": """
        <html>
          <head>
            <meta property="og:description" content="Honda CBR 125 z 2012 roku." />
            <meta property="og:image" content="https://img.meta/1.jpg" />
          </head>
          <body>
            <div data-testid="ad-parameters-container">
              <p>Prywatne</p>
              <p>Marka: Honda</p>
              <p>Rok produkcji: 2012</p>
              <p>Poj. silnika: 125 cm³</p>
              <p>Stan techniczny: Nieuszkodzony</p>
              <p>Kraj pochodzenia: Polska</p>
            </div>
            <div data-testid="map-aside-section">
              <p>Lokalizacja</p>
              <p>Suwałki</p>
              <p>Podlaskie</p>
            </div>
          </body>
        </html>
        """,
        "json_ld": [
            {
                "@type": "Product",
                "name": "Honda CBR 125 JC50 - 2012r",
                "description": "Na sprzedaż Honda CBR 125 z 2012 roku.",
                "image": ["https://img.jsonld/1.jpg"],
                "offers": {"price": "8 500 PLN"},
                "areaServed": {"name": "Suwałki, Podlaskie"},
            }
        ],
        "next_data": {
            "props": {
                "pageProps": {
                    "ad": {
                        "description": "Na sprzedaż Honda CBR 125 z 2012 roku. Motocykl idealny do miasta.",
                        "images": [{"url": "https://img.next/1.jpg"}],
                    }
                }
            }
        },
    }

    detail = build_detail_record(
        seed_record,
        raw_payloads,
        validation_provider=FakeValidationProvider(),
    )

    assert detail["title"] == "Honda CBR 125 JC50 - 2012r"
    assert detail["price_pln"] == 8600
    assert detail["negotiable"] is True
    assert detail["brand"] == "Honda"
    assert detail["year"] == 2013
    assert detail["technical_state"] == "Nieuszkodzony"
    assert detail["origin_country"] == "Polska"
    assert detail["seller_type"] == "Prywatne"
    assert detail["engine_cc"] == 124
    assert detail["vehicle_type"] == "motorcycle"
    assert detail["mileage_km"] == 38000
    assert detail["location"] == "Suwałki, Podlaskie"
    assert detail["is_sensible_listing"] is True
    assert detail["validation_confidence"] == "high"
    assert detail["validation_summary"] == "Listing looks consistent."
    assert "https://img.jsonld/1.jpg" in detail["image_urls"]
    assert "https://img.next/1.jpg" in detail["image_urls"]
    assert detail["attributes"]["Marka"] == "Honda"


class FakeQuadRejectingProvider:
    def validate(self, listing: dict) -> dict:
        return {
            "vehicle_type": "quad",
            "resolved_brand": None,
            "resolved_price_pln": None,
            "resolved_engine_cc": None,
            "resolved_year": None,
            "negotiable": None,
            "mileage_km": None,
            "is_sensible_listing": False,
            "reject_reason": "quad listing",
            "validation_confidence": "high",
            "validation_summary": "This is a quad, not a motorcycle listing.",
            "data_issues": ["vehicle type is quad"],
        }


def test_build_detail_record_allows_model_to_reject_quad_listings():
    seed_record = {
        "title": "Quad Yamaha Grizzly 700",
        "price_pln": 41900,
        "url": "https://www.olx.pl/d/oferta/quad-yamaha-grizzly-700-CID5-ID1.html",
        "short_description": None,
    }
    raw_payloads = {
        "html": "<html></html>",
        "json_ld": [],
        "next_data": {},
    }

    detail = build_detail_record(
        seed_record,
        raw_payloads,
        validation_provider=FakeQuadRejectingProvider(),
    )

    assert detail["vehicle_type"] == "quad"
    assert detail["is_sensible_listing"] is False
    assert detail["reject_reason"] == "quad listing"
