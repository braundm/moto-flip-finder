from moto_flip_finder.motorcycle_listing_validation import (
    empty_motorcycle_listing_validation,
    motorcycle_listing_validation_from_payload,
    validate_motorcycle_listing,
)


class FakeProvider:
    def validate(self, listing: dict) -> dict:
        return {
            "vehicle_type": "motorcycle",
            "resolved_brand": "Honda",
            "resolved_price_pln": 8700,
            "resolved_engine_cc": 125,
            "resolved_year": 2012,
            "negotiable": False,
            "mileage_km": 6400,
            "is_sensible_listing": True,
            "reject_reason": None,
            "validation_confidence": "medium",
            "validation_summary": "Looks like a normal listing.",
            "data_issues": [],
        }


class FailingProvider:
    def validate(self, listing: dict) -> dict:
        raise RuntimeError("boom")


def test_motorcycle_listing_validation_from_payload_normalizes_values():
    payload = motorcycle_listing_validation_from_payload(
        {
            "vehicle_type": "quad",
            "resolved_brand": "Yamaha",
            "resolved_price_pln": "41 900 zł",
            "resolved_engine_cc": "686 cm3",
            "resolved_year": "2019",
            "negotiable": "wrong",
            "mileage_km": "38 000 km",
            "is_sensible_listing": True,
            "reject_reason": "  ",
            "validation_confidence": "high",
            "validation_summary": " Looks consistent. ",
            "data_issues": ["price seems okay", "", 123],
        }
    )

    assert payload == {
        "vehicle_type": "quad",
        "resolved_brand": "Yamaha",
        "resolved_price_pln": 41900,
        "resolved_engine_cc": 686,
        "resolved_year": 2019,
        "negotiable": None,
        "mileage_km": 38000,
        "is_sensible_listing": True,
        "reject_reason": None,
        "validation_confidence": "high",
        "validation_summary": "Looks consistent.",
        "data_issues": ["price seems okay"],
    }


def test_validate_motorcycle_listing_returns_empty_payload_on_failure():
    result = validate_motorcycle_listing({"title": "Honda"}, provider=FailingProvider())

    assert result == empty_motorcycle_listing_validation()


def test_validate_motorcycle_listing_uses_provider_result():
    result = validate_motorcycle_listing({"title": "Honda"}, provider=FakeProvider())

    assert result["vehicle_type"] == "motorcycle"
    assert result["resolved_brand"] == "Honda"
    assert result["resolved_price_pln"] == 8700
    assert result["resolved_engine_cc"] == 125
    assert result["resolved_year"] == 2012
    assert result["negotiable"] is False
    assert result["mileage_km"] == 6400
    assert result["is_sensible_listing"] is True
