from moto_flip_finder.sources.olx.import_search import (
    build_structured_records,
    build_search_page_url,
    _extract_price_from_text,
    detect_damaged_listing,
    detect_gsxr_600,
    infer_normalized_model,
    normalize_gsxr_text,
)


def test_normalize_gsxr_text_handles_common_variants():
    text = "Suzuki GSX-R i gsx r oraz GSXR"

    normalized = normalize_gsxr_text(text)

    assert normalized == "suzuki gsxr i gsxr oraz gsxr"


def test_detect_gsxr_600_works_from_title_or_description():
    assert detect_gsxr_600("Suzuki GSX-R", "Bardzo ladny 600") is True
    assert detect_gsxr_600("Suzuki GSXR 750", None) is False


def test_detect_damaged_listing_uses_simple_keywords():
    assert detect_damaged_listing("GSXR po szlifie", None) is True
    assert detect_damaged_listing("GSXR zadbany", "Odpala i jezdzi") is False


def test_infer_normalized_model_detects_gsxr():
    assert infer_normalized_model("Suzuki gsx r 600", None) == "gsxr"
    assert infer_normalized_model("Honda CBR 600", None) is None


def test_build_structured_records_keeps_non_damaged_and_non_600_offers():
    raw_payloads = {
        "json_ld": [
            {
                "@type": "ItemList",
                "itemListElement": [
                    {
                        "@type": "ListItem",
                        "url": "https://www.olx.pl/d/oferta/1",
                        "name": "Suzuki GSX-R 600",
                        "description": "Zdrowy motocykl, bezwypadkowy",
                        "image": ["https://img/1.jpg"],
                        "offers": {"price": "21 500 PLN"},
                        "address": {"addressLocality": "Warszawa"},
                    },
                    {
                        "@type": "ListItem",
                        "url": "https://www.olx.pl/d/oferta/2",
                        "name": "Suzuki GSXR 750 po szlifie",
                        "description": "Do naprawy, uszkodzony bok",
                        "image": ["https://img/2.jpg"],
                        "offers": {"price": "13 000 PLN"},
                        "address": {"addressLocality": "Krakow"},
                    },
                ],
            }
        ],
        "next_data": None,
    }

    records = build_structured_records(
        raw_payloads, "https://www.olx.pl/motoryzacja/motocykle/q-gsxr/"
    )

    assert len(records) == 2
    assert records[0]["normalized_model"] == "gsxr"
    assert records[0]["mentions_600"] is True
    assert records[0]["looks_damaged"] is False
    assert records[1]["mentions_600"] is False
    assert records[1]["looks_damaged"] is True


def test_build_search_page_url_adds_page_parameter():
    assert (
        build_search_page_url("https://www.olx.pl/motoryzacja/motocykle/q-gsxr/", 1)
        == "https://www.olx.pl/motoryzacja/motocykle/q-gsxr/"
    )
    assert (
        build_search_page_url("https://www.olx.pl/motoryzacja/motocykle/q-gsxr/", 3)
        == "https://www.olx.pl/motoryzacja/motocykle/q-gsxr/?page=3"
    )


def test_extract_price_from_text_does_not_merge_model_digits_with_price():
    text = "Suzuki gsxr 600 k5\n16 200 zł\n"

    price = _extract_price_from_text(text)

    assert price == 16200
