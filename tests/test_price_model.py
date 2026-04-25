from moto_flip_finder.price_model import (
    detect_model_family,
    detect_model_hint,
    predict_price,
    score_records_with_price_model,
    train_price_model,
)
from moto_flip_finder.train_ready_price_model import (
    filter_training_records,
    run_ready_price_training,
)


def _healthy_record(index: int, *, model: str = "gsxr_600", engine_cc: int = 600) -> dict:
    year = 2005 + (index % 8)
    base_price = 12000 + (year - 2005) * 700
    if model == "gsxr_750":
        base_price += 3000
    if model == "gsxr_1000":
        base_price += 6000
    return {
        "url": f"https://example.com/{model}/{index}",
        "title": f"Suzuki {model} K{index % 9}",
        "full_description": "Zadbany motocykl z normalnym przebiegiem i serwisem.",
        "image_urls": ["https://example.com/image.jpg"] * (3 + index % 3),
        "brand": "Suzuki",
        "seller_type": "Prywatne" if index % 2 == 0 else "Firma",
        "origin_country": "Polska" if index % 3 == 0 else "Niemcy",
        "normalized_model": model,
        "engine_cc": engine_cc,
        "year": year,
        "price_pln": base_price,
    }


def _ready_kawasaki_record(
    index: int,
    *,
    family: str = "versys",
    engine_cc: int = 650,
    title: str | None = None,
    full_description: str | None = None,
    price_pln: int | None = None,
) -> dict:
    year = 2008 + (index % 10)
    base_price = price_pln if price_pln is not None else 15000 + (year - 2008) * 800
    if family == "ninja" and price_pln is None:
        base_price += 2500
    if family == "z" and price_pln is None:
        base_price += 7000
    return {
        "url": f"https://example.com/kawasaki/{family}/{index}",
        "title": title or f"Kawasaki {family.title()} {engine_cc}",
        "full_description": full_description or "Zadbany Kawasaki, serwisowany, gotowy do sezonu.",
        "image_urls": ["https://example.com/image.jpg"] * (2 + index % 4),
        "brand": "Kawasaki",
        "seller_type": "Prywatne",
        "origin_country": "Polska" if index % 2 == 0 else "Niemcy",
        "vehicle_type": "motorcycle",
        "technical_state": "Nieuszkodzony",
        "engine_cc": engine_cc,
        "year": year,
        "mileage_km": 15000 + index * 200,
        "negotiable": index % 3 == 0,
        "price_pln": base_price,
    }


def test_detect_model_hint_normalizes_kawasaki_variants():
    assert detect_model_hint("Kawasaki Ninja 1000sx 2020", None, "Kawasaki") == "z1000sx"
    assert detect_model_hint("Kawasaki ZX-4RR", None, "Kawasaki") == "zx4rr"
    assert detect_model_hint("Kawasaki Z 900 RS", None, "Kawasaki") == "z900rs"
    assert detect_model_hint("Kawasaki 1997 500", None, "Kawasaki") is None


def test_detect_model_family_groups_related_variants():
    assert detect_model_family("Kawasaki Z1000SX", None, "Kawasaki") == "sx"
    assert detect_model_family("Kawasaki Z900", None, "Kawasaki") == "z"
    assert detect_model_family("Kawasaki ZX-6R Ninja", None, "Kawasaki") == "zx"


def test_filter_training_records_removes_suspicious_and_rare_records():
    records = [_ready_kawasaki_record(index, family="versys", engine_cc=650) for index in range(6)]
    records.extend(
        _ready_kawasaki_record(index + 100, family="ninja", engine_cc=650)
        for index in range(6)
    )
    records.append(
        _ready_kawasaki_record(
            999,
            family="z",
            engine_cc=400,
            title="Kawasaki Z400/125 kat A1/B",
            full_description="Zarejestrowany jako 125, kat. B.",
            price_pln=25000,
        )
    )
    records.append(
        _ready_kawasaki_record(
            1000,
            family="vulcan",
            engine_cc=900,
            title="Kawasaki Vulcan 900 | Custom Bobber | projekt na zamówienie",
            full_description="Custom bobber project.",
            price_pln=28900,
        )
    )
    filtered = filter_training_records(records, required_brand="Kawasaki", min_family_records=3)
    titles = {record["title"] for record in filtered}
    assert "Kawasaki Z400/125 kat A1/B" not in titles
    assert "Kawasaki Vulcan 900 | Custom Bobber | projekt na zamówienie" not in titles
    assert len(filtered) == 12


def test_train_price_model_returns_working_estimator():
    records = [_healthy_record(index) for index in range(24)]
    records.extend(_healthy_record(index, model="gsxr_750", engine_cc=750) for index in range(24, 40))

    model_bundle = train_price_model(records, search_iterations=2, cv_folds=3)

    assert model_bundle["backend"] == "sklearn"
    assert model_bundle["best_candidate_name"] in {"extra_trees", "random_forest"}
    assert model_bundle["training_size"] > 0
    assert model_bundle["validation_metrics"]["mae_pln"] is not None
    assert len(model_bundle["feature_importances"]) > 0


def test_train_price_model_auto_falls_back_to_comparable_for_small_dataset():
    records = [_healthy_record(index) for index in range(8)]

    model_bundle = train_price_model(records)
    prediction = predict_price(
        {
            "title": "Suzuki gsxr_600 K5",
            "full_description": "Zadbany motocykl.",
            "normalized_model": "gsxr_600",
            "engine_cc": 600,
            "year": 2008,
            "brand": "Suzuki",
            "image_urls": ["https://example.com/1.jpg"],
        },
        model_bundle,
    )

    assert model_bundle["backend"] == "comparable"
    assert model_bundle["best_candidate_name"] == "same_model_trimmed_median"
    assert isinstance(prediction, int)
    assert prediction > 0


def test_train_price_model_handles_ready_style_records_without_normalized_model():
    records = [_ready_kawasaki_record(index, family="versys", engine_cc=650) for index in range(20)]
    records.extend(
        _ready_kawasaki_record(index, family="ninja", engine_cc=650)
        for index in range(20, 34)
    )
    records.extend(
        _ready_kawasaki_record(index, family="z", engine_cc=1000, title=f"Kawasaki Z1000 {index}")
        for index in range(34, 46)
    )

    model_bundle = train_price_model(records, search_iterations=2, cv_folds=3)

    prediction = predict_price(
        {
            "title": "Kawasaki Ninja 650",
            "full_description": "Serwisowany motocykl Kawasaki.",
            "brand": "Kawasaki",
            "vehicle_type": "motorcycle",
            "technical_state": "Nieuszkodzony",
            "engine_cc": 650,
            "year": 2015,
            "mileage_km": 22000,
            "image_urls": ["https://example.com/1.jpg"],
        },
        model_bundle,
    )

    assert model_bundle["training_size"] > 0
    assert isinstance(prediction, int)
    assert prediction > 0


def test_run_ready_price_training_returns_report_payload_and_predictions():
    ready_records = [_ready_kawasaki_record(index, family="versys", engine_cc=650) for index in range(20)]
    ready_records.extend(
        _ready_kawasaki_record(index, family="ninja", engine_cc=650)
        for index in range(20, 34)
    )
    ready_records.extend(
        _ready_kawasaki_record(index, family="z", engine_cc=1000, title=f"Kawasaki Z1000 {index}")
        for index in range(34, 46)
    )

    result = run_ready_price_training(
        ready_records,
        required_brand="Kawasaki",
        min_family_records=3,
        search_iterations=2,
        cv_folds=3,
    )

    assert len(result["training_records"]) > 0
    assert result["report_payload"]["required_brand"] == "Kawasaki"
    assert result["report_payload"]["input_record_count"] == len(ready_records)
    assert result["report_payload"]["training_record_count"] == len(result["training_records"])
    assert len(result["predictions"]) == len(ready_records)
    assert "listed_price_pln" in result["predictions"][0]
    assert "price_delta_pln" in result["predictions"][0]


def test_train_price_model_dispatches_to_torch_backend(monkeypatch):
    records = [_healthy_record(index) for index in range(24)]
    captured: dict[str, object] = {}

    def fake_train_torch_price_model(input_records, **kwargs):
        captured["records"] = input_records
        captured["kwargs"] = kwargs
        return {
            "backend": "torch",
            "best_candidate_name": "torch_mlp_medium",
            "training_size": 19,
            "validation_size": 5,
            "validation_metrics": {"mae_pln": 1234.56, "rmse_pln": 2345.67},
            "feature_importances": [],
            "estimator": {"model_type": "torch_mlp"},
        }

    monkeypatch.setattr(
        "moto_flip_finder.torch_price_model.train_torch_price_model",
        fake_train_torch_price_model,
    )

    model_bundle = train_price_model(
        records,
        backend="torch",
        search_iterations=4,
        cv_folds=3,
    )

    assert captured["records"] == records
    assert captured["kwargs"]["search_iterations"] == 4
    assert model_bundle["backend"] == "torch"
    assert model_bundle["best_candidate_name"] == "torch_mlp_medium"


def test_predict_price_dispatches_to_torch_backend(monkeypatch):
    captured: dict[str, object] = {}

    def fake_predict_torch_price(record, model_bundle):
        captured["record"] = record
        captured["model_bundle"] = model_bundle
        return 20500

    monkeypatch.setattr(
        "moto_flip_finder.torch_price_model.predict_torch_price",
        fake_predict_torch_price,
    )

    model_bundle = {
        "backend": "torch",
        "estimator": {"model_type": "torch_mlp"},
    }
    record = {
        "title": "Kawasaki Ninja 650",
        "brand": "Kawasaki",
        "engine_cc": 650,
        "year": 2015,
    }

    prediction = predict_price(record, model_bundle)

    assert prediction == 20500
    assert captured["record"] == record
    assert captured["model_bundle"] == model_bundle


def test_run_ready_price_training_passes_backend_to_train_price_model(monkeypatch):
    ready_records = [_ready_kawasaki_record(index) for index in range(20)]
    captured: dict[str, object] = {}

    def fake_train_price_model(records, **kwargs):
        captured["records"] = records
        captured["kwargs"] = kwargs
        return {
            "backend": kwargs["backend"],
            "best_candidate_name": "torch_mlp_small",
            "validation_metrics": {"mae_pln": 1111.11, "rmse_pln": 2222.22},
            "training_size": len(records),
            "validation_size": 4,
            "feature_importances": [],
            "estimator": {"model_type": "torch_mlp"},
        }

    monkeypatch.setattr(
        "moto_flip_finder.train_ready_price_model.train_price_model",
        fake_train_price_model,
    )
    monkeypatch.setattr(
        "moto_flip_finder.train_ready_price_model.build_enriched_ready_price_predictions",
        lambda ready_records, model_bundle: [
            {
                "original_listing": ready_records[0],
                "predicted_price_pln": 20000,
                "listed_price_pln": ready_records[0]["price_pln"],
                "price_delta_pln": 1000,
            }
        ],
    )

    result = run_ready_price_training(
        ready_records,
        backend="torch",
        required_brand="Kawasaki",
        min_family_records=1,
    )

    assert captured["kwargs"]["backend"] == "torch"
    assert result["model_bundle"]["backend"] == "torch"
    assert result["report_payload"]["required_brand"] == "Kawasaki"


def test_predict_price_returns_int_for_valid_record():
    records = [_healthy_record(index) for index in range(24)]
    model_bundle = train_price_model(records, search_iterations=2, cv_folds=3)

    prediction = predict_price(
        {
            "title": "Suzuki gsxr_600 K5",
            "full_description": "Zadbany motocykl.",
            "normalized_model": "gsxr_600",
            "engine_cc": 600,
            "year": 2008,
            "brand": "Suzuki",
            "seller_type": "Prywatne",
            "origin_country": "Polska",
            "image_urls": ["https://example.com/1.jpg"],
        },
        model_bundle,
    )

    assert isinstance(prediction, int)
    assert prediction > 0


def test_score_records_with_price_model_scores_each_input_record():
    records = [_healthy_record(index) for index in range(24)]
    model_bundle = train_price_model(records, search_iterations=2, cv_folds=3)

    scored = score_records_with_price_model(
        [
            {
                "title": "Suzuki GSX-R 600",
                "normalized_model": "gsxr_600",
                "engine_cc": 600,
                "year": 2007,
                "brand": "Suzuki",
                "image_urls": [],
            },
            {
                "title": "Suzuki GSX-R 750",
                "normalized_model": "gsxr_750",
                "engine_cc": 750,
                "year": 2009,
                "brand": "Suzuki",
                "image_urls": [],
            },
        ],
        model_bundle,
    )

    assert len(scored) == 2
    assert scored[0]["predicted_price_pln"] is not None
