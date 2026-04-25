import pytest

import moto_flip_finder.main as main_module
from moto_flip_finder.main import main


def test_main_prints_help_when_no_args(capsys):
    main([])

    captured = capsys.readouterr()

    assert "moto-flip-finder v1.0.0" in captured.out
    assert "brand-pipeline" in captured.out


def test_main_prints_version(capsys):
    main(["--version"])

    captured = capsys.readouterr()

    assert captured.out.strip() == "1.0.0"


def test_main_dispatches_brand_pipeline(monkeypatch):
    captured: dict[str, object] = {}

    def fake_brand_pipeline_main():
        captured["called"] = True

    monkeypatch.setattr(
        "moto_flip_finder.main.run_brand_price_pipeline_main",
        fake_brand_pipeline_main,
    )
    monkeypatch.setitem(main_module.COMMANDS, "brand-pipeline", fake_brand_pipeline_main)

    main(["brand-pipeline", "--brand", "Kawasaki"])

    assert captured["called"] is True


def test_main_rejects_unknown_command():
    with pytest.raises(SystemExit) as exc_info:
        main(["unknown-command"])

    assert "Unknown command" in str(exc_info.value)
