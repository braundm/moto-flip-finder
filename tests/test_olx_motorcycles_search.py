from moto_flip_finder.sources.olx.import_motorcycles_search import (
    build_search_page_url,
    build_structured_records,
)


def test_build_structured_records_for_generic_motorcycles_keeps_neutral_fields():
    raw_payloads = {
        "json_ld": [
            {
                "@type": "ItemList",
                "itemListElement": [
                    {
                        "@type": "ListItem",
                        "url": "https://www.olx.pl/d/oferta/honda-cbr-125-CID5-ID1.html",
                        "name": "Honda CBR 125 JC50 - 2012r",
                        "description": "8 500 zł | Do negocjacji",
                        "image": ["https://img/1.jpg"],
                        "offers": {"price": "8 500 PLN"},
                        "address": {"addressLocality": "Suwałki"},
                    },
                    {
                        "@type": "ListItem",
                        "url": "https://www.olx.pl/d/oferta/yamaha-mt-07-CID5-ID2.html",
                        "name": "Yamaha MT-07 ABS",
                        "description": "Cena sztywna",
                        "image": ["https://img/2.jpg"],
                        "offers": {"price": "27 900 PLN"},
                        "address": {"addressLocality": "Warszawa"},
                    },
                ],
            }
        ],
        "next_data": None,
    }

    records = build_structured_records(
        raw_payloads,
        "https://www.olx.pl/motoryzacja/motocykle-skutery/",
    )

    assert len(records) == 2
    assert records[0] == {
        "title": "Honda CBR 125 JC50 - 2012r",
        "price_pln": 8500,
        "negotiable": None,
        "url": "https://www.olx.pl/d/oferta/honda-cbr-125-CID5-ID1.html",
        "location": "Suwałki",
        "image_urls": ["https://img/1.jpg"],
        "short_description": "8 500 zł | Do negocjacji",
    }
    assert records[1]["negotiable"] is None


def test_build_search_page_url_works_for_category_url():
    assert (
        build_search_page_url("https://www.olx.pl/motoryzacja/motocykle-skutery/", 2)
        == "https://www.olx.pl/motoryzacja/motocykle-skutery/?page=2"
    )
