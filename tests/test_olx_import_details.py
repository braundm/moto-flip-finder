from moto_flip_finder.sources.olx.import_details import (
    _extract_price_from_short_description,
    _is_listing_image_url,
    build_detail_record,
    select_detail_targets,
)


def test_select_detail_targets_defaults_to_mentions_600_only():
    records = [
        {"url": "https://www.olx.pl/d/oferta/1", "mentions_600": True},
        {"url": "https://www.olx.pl/d/oferta/2", "mentions_600": False},
        {"url": "https://www.olx.pl/d/oferta/1", "mentions_600": True},
    ]

    selected = select_detail_targets(records)

    assert selected == [{"url": "https://www.olx.pl/d/oferta/1", "mentions_600": True}]


def test_build_detail_record_extracts_full_description_images_and_attributes():
    seed_record = {
        "title": "Suzuki GSX-R 600",
        "price_pln": 22000,
        "url": "https://www.olx.pl/d/oferta/gsxr-600-test",
        "location": None,
        "image_urls": ["https://img.seed/1.jpg"],
        "short_description": "Krotki opis",
        "mentions_600": True,
    }
    raw_payloads = {
        "html": """
        <html>
          <head>
            <meta property="og:description" content="Pelny opis z meta" />
            <meta property="og:image" content="https://img.meta/1.jpg" />
          </head>
          <body>
            <div data-testid="ad-parameters-container">
              <p>Prywatne</p>
              <p>Marka: Suzuki</p>
              <p>Rok produkcji: 2008</p>
              <p>Poj. silnika: 600 cm3</p>
              <p>Stan techniczny: Uszkodzony</p>
              <p>Kraj pochodzenia: Niemcy</p>
            </div>
            <div data-testid="map-aside-section">
              <p>Lokalizacja</p>
              <p>Warszawa</p>
              <p>Mazowieckie</p>
            </div>
            <div data-cy="adParam_engine">Pojemnosc skokowa: 600 cm3</div>
          </body>
        </html>
        """,
        "json_ld": [
            {
                "@type": "Product",
                "name": "Suzuki GSX-R 600",
                "description": "Pelny opis ogloszenia GSX-R 600, zdrowy motocykl.",
                "image": ["https://img.jsonld/1.jpg", "https://img.jsonld/2.jpg"],
                "offers": {"price": "22 000 PLN"},
                "areaServed": {"name": "Warszawa, Mazowieckie"},
            }
        ],
        "next_data": {
            "props": {
                "pageProps": {
                    "ad": {
                        "description": "Pelny opis z next data",
                        "images": [{"url": "https://img.next/1.jpg"}],
                    }
                }
            }
        },
    }

    detail = build_detail_record(seed_record, raw_payloads)

    assert detail["full_description"] == "Pelny opis z next data"
    assert detail["mentions_600"] is True
    assert detail["looks_damaged"] is False
    assert detail["brand"] == "Suzuki"
    assert detail["model_family"] == "gsxr"
    assert detail["engine_cc"] == 600
    assert detail["normalized_model"] == "gsxr_600"
    assert detail["year"] == 2008
    assert detail["technical_state"] == "Uszkodzony"
    assert detail["origin_country"] == "Niemcy"
    assert detail["seller_type"] == "Prywatne"
    assert detail["city"] == "Warszawa"
    assert detail["region"] == "Mazowieckie"
    assert detail["location"] == "Warszawa, Mazowieckie"
    assert "https://img.jsonld/1.jpg" in detail["image_urls"]
    assert "https://img.next/1.jpg" in detail["image_urls"]
    assert detail["attributes"]["Rok produkcji"] == "2008"


def test_extract_price_from_short_description_parses_plain_price():
    assert _extract_price_from_short_description("56 999 zł") == 56999


def test_build_detail_record_prefers_short_description_price_without_model_digit_bleed():
    seed_record = {
        "title": "Suzuki gsxr 600 k5",
        "price_pln": 516200,
        "url": "https://www.olx.pl/d/oferta/gsxr-600-k5",
        "location": None,
        "image_urls": ["https://ireland.apollo.olxcdn.com/v1/files/real-image"],
        "short_description": "16 200 zł | do negocjacji",
    }
    raw_payloads = {
        "html": "<html></html>",
        "json_ld": [],
        "next_data": {},
    }

    detail = build_detail_record(seed_record, raw_payloads)

    assert detail["price_pln"] == 516200
    assert detail["normalized_model"] == "gsxr_600"


def test_is_listing_image_url_filters_ui_assets():
    assert _is_listing_image_url("https://ireland.apollo.olxcdn.com/v1/files/real-image") is True
    assert _is_listing_image_url("https://www.olx.pl/app/static/media/app_store.156ac6d41.svg") is False
    assert _is_listing_image_url("https://www.olx.pl/app/static/media/full-screen.5555ba1b6.svg") is False
    assert _is_listing_image_url("https://www.olx.pl/app/static/media/google_play.8cb1ced49.svg") is False
