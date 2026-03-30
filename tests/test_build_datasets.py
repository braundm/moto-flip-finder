from moto_flip_finder.build_datasets import (
    build_processed_datasets,
    classify_record,
    is_damaged_record,
    is_healthy_record,
    summarize_records,
)


def test_build_processed_datasets_splits_records_for_pipeline():
    records = [
        {
            "title": "Suzuki GSX-R 600",
            "full_description": "Zadbany motocykl",
            "technical_state": "Nieuszkodzony",
            "looks_damaged": False,
            "normalized_model": "gsxr_600",
        },
        {
            "title": "Suzuki GSX-R 600 po szlifie",
            "full_description": "Do naprawy",
            "technical_state": "Uszkodzony",
            "looks_damaged": False,
            "normalized_model": "gsxr_600",
        },
        {
            "title": "Suzuki GSX-R 1000",
            "full_description": "Opis",
            "technical_state": "Uszkodzony",
            "looks_damaged": True,
            "normalized_model": "gsxr_1000",
        },
    ]

    datasets = build_processed_datasets(records)

    assert datasets["all_gsxr"][0]["classification_reason"] == (
        "healthy: technical_state=Nieuszkodzony"
    )
    assert datasets["healthy_comps"][0]["title"] == records[0]["title"]
    assert datasets["healthy_comps"][0]["dataset_class"] == "healthy"
    assert datasets["damaged_candidates"][0]["classification_reason"] == (
        "damaged: technical_state=Uszkodzony"
    )
    assert datasets["damaged_candidates"][1]["classification_reason"] == (
        "damaged: technical_state=Uszkodzony"
    )


def test_is_healthy_record_requires_non_damaged_and_technical_state():
    assert is_healthy_record(
        {"technical_state": "Nieuszkodzony", "looks_damaged": False}
    ) is True
    assert is_healthy_record(
        {
            "technical_state": "Uszkodzony",
            "looks_damaged": False,
        }
    ) is False


def test_is_damaged_record_uses_technical_state_or_signals():
    assert is_damaged_record(
        {"technical_state": "Uszkodzony", "looks_damaged": False}
    ) is True
    assert is_damaged_record(
        {"technical_state": None, "looks_damaged": True, "title": "GSX-R po dzwonie"}
    ) is True


def test_classify_record_returns_explicit_reason():
    dataset_class, reason = classify_record(
        {
            "technical_state": "Nieuszkodzony",
            "looks_damaged": False,
            "title": "Suzuki GSX-R 600",
            "full_description": "Zadbany motocykl",
        }
    )

    assert dataset_class == "healthy"
    assert reason == "healthy: technical_state=Nieuszkodzony"


def test_classify_record_prefers_damage_keyword_when_present():
    dataset_class, reason = classify_record(
        {
            "technical_state": "Nieuszkodzony",
            "looks_damaged": False,
            "title": "Suzuki GSX-R 600 po szlifie",
            "full_description": "",
        }
    )

    assert dataset_class == "damaged"
    assert reason == "damaged: keyword=po szlifie"


def test_classify_record_keeps_healthy_when_technical_state_is_clear_and_negated_phrases_exist():
    dataset_class, reason = classify_record(
        {
            "title": "Sprzedam Suzuki gsxr-600cc L4",
            "technical_state": "Nieuszkodzony",
            "looks_damaged": True,
            "full_description": (
                "Stan techniczny motocykla oceniam na bardzo dobry. "
                "Zero wkładu finansowego. 100% oryginał. "
                "Nic nie malowane i nie połamane. Bez żadnych uszkodzeń."
            ),
        }
    )

    assert dataset_class == "healthy"
    assert reason == "healthy: technical_state=Nieuszkodzony"


def test_summarize_records_counts_models():
    summary = summarize_records(
        [
            {"normalized_model": "gsxr_600", "technical_state": "Nieuszkodzony", "looks_damaged": False},
            {"normalized_model": "gsxr_600", "technical_state": "Uszkodzony", "looks_damaged": True},
            {"normalized_model": "gsxr_1000", "looks_damaged": False},
        ]
    )

    assert summary["all_records"] == 3
    assert summary["healthy_records"] == 1
    assert summary["damaged_records"] == 1
    assert summary["normalized_model_counts"] == {"gsxr_1000": 1, "gsxr_600": 2}
