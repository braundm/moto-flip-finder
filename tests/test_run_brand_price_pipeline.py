import json
from pathlib import Path

from moto_flip_finder.run_brand_price_pipeline import (
    build_default_brand_search_url,
    run_brand_price_pipeline,
)


def test_build_default_brand_search_url_slugifies_brand_name():
    assert (
        build_default_brand_search_url("Triumph Motorcycles")
        == "https://www.olx.pl/motoryzacja/motocykle-skutery/q-triumph-motorcycles/"
    )


def test_run_brand_price_pipeline_uses_brand_defaults_and_saves_outputs(
    tmp_path: Path,
    monkeypatch,
):
    ready_path = tmp_path / "ready.json"
    ready_records = [
        {
            "url": "https://example.com/a",
            "title": "Kawasaki Versys 650",
            "full_description": "Serwisowany motocykl Kawasaki.",
            "brand": "Kawasaki",
            "vehicle_type": "motorcycle",
            "technical_state": "Nieuszkodzony",
            "engine_cc": 650,
            "year": 2015,
            "price_pln": 21900,
            "image_urls": [],
        }
    ]
    ready_path.write_text(json.dumps({"records": ready_records}, ensure_ascii=False), encoding="utf-8")

    captured_calls: dict[str, object] = {}

    def fake_build_ready_motorcycles_dataset(**kwargs):
        captured_calls["build_ready"] = kwargs
        return tmp_path / "records.json", tmp_path / "details.json", ready_path

    def fake_run_ready_price_training(loaded_records, **kwargs):
        captured_calls["training"] = {
            "loaded_records": loaded_records,
            **kwargs,
        }
        return {
            "training_records": loaded_records,
            "model_bundle": {
                "backend": "sklearn",
                "best_candidate_name": "extra_trees",
                "validation_metrics": {
                    "mae_pln": 1234.56,
                    "rmse_pln": 2345.67,
                },
            },
            "report_payload": {
                "required_brand": "Kawasaki",
                "input_record_count": len(loaded_records),
                "training_record_count": len(loaded_records),
                "min_family_records": kwargs["min_family_records"],
            },
            "predictions": [
                {
                    "original_listing": loaded_records[0],
                    "predicted_price_pln": 23000,
                    "listed_price_pln": 21900,
                    "price_delta_pln": 1100,
                }
            ],
        }

    def fake_save_price_model(model_bundle, output_path):
        captured_calls["model_output"] = {
            "model_bundle": model_bundle,
            "output_path": output_path,
        }
        return output_path

    def fake_save_price_model_report(report_payload, output_path):
        captured_calls["report_output"] = {
            "report_payload": report_payload,
            "output_path": output_path,
        }
        return output_path

    def fake_save_ready_price_predictions(predictions, output_path):
        captured_calls["predictions_output"] = {
            "predictions": predictions,
            "output_path": output_path,
        }
        return output_path

    def fake_save_ready_price_report(predictions, output_path, **kwargs):
        captured_calls["html_report_output"] = {
            "predictions": predictions,
            "output_path": output_path,
            **kwargs,
        }
        return output_path

    monkeypatch.setattr(
        "moto_flip_finder.run_brand_price_pipeline.build_ready_motorcycles_dataset",
        fake_build_ready_motorcycles_dataset,
    )
    monkeypatch.setattr(
        "moto_flip_finder.run_brand_price_pipeline.run_ready_price_training",
        fake_run_ready_price_training,
    )
    monkeypatch.setattr(
        "moto_flip_finder.run_brand_price_pipeline.save_price_model",
        fake_save_price_model,
    )
    monkeypatch.setattr(
        "moto_flip_finder.run_brand_price_pipeline.save_price_model_report",
        fake_save_price_model_report,
    )
    monkeypatch.setattr(
        "moto_flip_finder.run_brand_price_pipeline.save_ready_price_predictions",
        fake_save_ready_price_predictions,
    )
    monkeypatch.setattr(
        "moto_flip_finder.run_brand_price_pipeline.save_ready_price_report",
        fake_save_ready_price_report,
    )

    result = run_brand_price_pipeline(
        brand="Kawasaki",
        max_pages=7,
        max_records=25,
        raw_output_dir=tmp_path / "raw",
        ready_output_dir=tmp_path / "ready",
    )

    assert captured_calls["build_ready"] == {
        "search_url": "https://www.olx.pl/motoryzacja/motocykle-skutery/q-kawasaki/",
        "max_pages": 7,
        "max_records": 25,
        "raw_output_dir": tmp_path / "raw",
        "ready_output_dir": tmp_path / "ready",
        "search_delay_seconds": 0.5,
        "detail_delay_seconds": 0.2,
        "detail_max_workers": 6,
        "dataset_label": "kawasaki_all_pages",
        "keyword_filter": "Kawasaki",
        "required_brand": "Kawasaki",
    }
    assert captured_calls["training"]["loaded_records"] == ready_records
    assert captured_calls["training"]["backend"] == "auto"
    assert captured_calls["training"]["required_brand"] == "Kawasaki"
    assert captured_calls["report_output"]["report_payload"]["ready_file"] == str(ready_path)
    assert (
        captured_calls["report_output"]["report_payload"]["search_url"]
        == "https://www.olx.pl/motoryzacja/motocykle-skutery/q-kawasaki/"
    )
    assert captured_calls["report_output"]["report_payload"]["dataset_label"] == "kawasaki_all_pages"
    assert captured_calls["report_output"]["report_payload"]["keyword_filter"] == "Kawasaki"
    assert captured_calls["html_report_output"]["brand"] == "Kawasaki"
    assert captured_calls["html_report_output"]["backend"] == "sklearn"
    assert captured_calls["html_report_output"]["dataset_label"] == "kawasaki_all_pages"
    assert result["backend"] == "sklearn"
    assert result["requested_backend"] == "auto"
    assert result["model_path"] == Path("data/models/kawasaki_ready_price_model_v1.joblib")
    assert result["report_path"] == Path("data/processed/kawasaki_ready_price_model_v1_report.json")
    assert result["predictions_path"] == Path("data/processed/kawasaki_ready_price_model_v1_predictions.json")
    assert result["html_report_path"] == Path("data/processed/kawasaki_ready_price_model_v1_report.html")
    assert result["best_candidate_name"] == "extra_trees"


def test_run_brand_price_pipeline_uses_torch_backend_specific_filenames(
    tmp_path: Path,
    monkeypatch,
):
    ready_path = tmp_path / "ready.json"
    ready_path.write_text(
        json.dumps({"records": [{"url": "https://example.com/a", "title": "Kawasaki", "price_pln": 1}]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "moto_flip_finder.run_brand_price_pipeline.build_ready_motorcycles_dataset",
        lambda **kwargs: (tmp_path / "records.json", tmp_path / "details.json", ready_path),
    )
    monkeypatch.setattr(
        "moto_flip_finder.run_brand_price_pipeline.run_ready_price_training",
        lambda loaded_records, **kwargs: {
            "training_records": loaded_records,
            "model_bundle": {
                "backend": kwargs["backend"],
                "best_candidate_name": "torch_mlp_small",
                "validation_metrics": {"mae_pln": 900.0, "rmse_pln": 1200.0},
            },
            "report_payload": {},
            "predictions": [],
        },
    )
    monkeypatch.setattr(
        "moto_flip_finder.run_brand_price_pipeline.save_price_model",
        lambda model_bundle, output_path: output_path,
    )
    monkeypatch.setattr(
        "moto_flip_finder.run_brand_price_pipeline.save_price_model_report",
        lambda report_payload, output_path: output_path,
    )
    monkeypatch.setattr(
        "moto_flip_finder.run_brand_price_pipeline.save_ready_price_predictions",
        lambda predictions, output_path: output_path,
    )
    monkeypatch.setattr(
        "moto_flip_finder.run_brand_price_pipeline.save_ready_price_report",
        lambda predictions, output_path, **kwargs: output_path,
    )

    result = run_brand_price_pipeline(
        brand="Kawasaki",
        backend="torch",
    )

    assert result["backend"] == "torch"
    assert result["requested_backend"] == "torch"
    assert result["model_path"] == Path("data/models/kawasaki_ready_price_model_torch_v1.joblib")
    assert result["report_path"] == Path("data/processed/kawasaki_ready_price_model_torch_v1_report.json")
    assert result["predictions_path"] == Path("data/processed/kawasaki_ready_price_model_torch_v1_predictions.json")
    assert result["html_report_path"] == Path("data/processed/kawasaki_ready_price_model_torch_v1_report.html")


def test_run_brand_price_pipeline_uses_comparable_filenames_for_comparable_backend(
    tmp_path: Path,
    monkeypatch,
):
    ready_path = tmp_path / "ready.json"
    ready_path.write_text(
        json.dumps({"records": [{"url": "https://example.com/a", "title": "Kawasaki", "price_pln": 1}]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "moto_flip_finder.run_brand_price_pipeline.build_ready_motorcycles_dataset",
        lambda **kwargs: (tmp_path / "records.json", tmp_path / "details.json", ready_path),
    )
    monkeypatch.setattr(
        "moto_flip_finder.run_brand_price_pipeline.run_ready_price_training",
        lambda loaded_records, **kwargs: {
            "training_records": loaded_records,
            "model_bundle": {
                "backend": "comparable",
                "best_candidate_name": "same_model_trimmed_median",
                "validation_metrics": {"mae_pln": 700.0, "rmse_pln": 1000.0},
            },
            "report_payload": {},
            "predictions": [],
        },
    )
    monkeypatch.setattr(
        "moto_flip_finder.run_brand_price_pipeline.save_price_model",
        lambda model_bundle, output_path: output_path,
    )
    monkeypatch.setattr(
        "moto_flip_finder.run_brand_price_pipeline.save_price_model_report",
        lambda report_payload, output_path: output_path,
    )
    monkeypatch.setattr(
        "moto_flip_finder.run_brand_price_pipeline.save_ready_price_predictions",
        lambda predictions, output_path: output_path,
    )
    monkeypatch.setattr(
        "moto_flip_finder.run_brand_price_pipeline.save_ready_price_report",
        lambda predictions, output_path, **kwargs: output_path,
    )

    result = run_brand_price_pipeline(
        brand="Kawasaki",
    )

    assert result["backend"] == "comparable"
    assert result["requested_backend"] == "auto"
    assert result["model_path"] == Path("data/models/kawasaki_ready_price_model_comparable_v1.joblib")
    assert result["report_path"] == Path("data/processed/kawasaki_ready_price_model_comparable_v1_report.json")
    assert result["predictions_path"] == Path("data/processed/kawasaki_ready_price_model_comparable_v1_predictions.json")
    assert result["html_report_path"] == Path("data/processed/kawasaki_ready_price_model_comparable_v1_report.html")
