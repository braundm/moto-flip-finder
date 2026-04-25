from moto_flip_finder.damage_analysis import (
    HeuristicDescriptionAnalysisProvider,
    analyze_description,
)
from moto_flip_finder.models import DamageAnalysis
from moto_flip_finder.description_analysis_provider import (
    _extract_response_payload,
    damage_analysis_from_payload,
)


def test_analyze_description_uses_heuristics_by_default():
    analysis = analyze_description(
        "Motocykl uszkodzony, po szlifie. Do poprawek owiewki i klamka. Silnik odpala."
    )

    assert analysis.starts is True
    assert analysis.severity == "medium"
    assert "fairings" in analysis.suspected_damage
    assert "lever" in analysis.suspected_damage


def test_analyze_description_emits_debug_output_for_heuristic_mode(
    monkeypatch, capsys
):
    monkeypatch.setenv("MOTO_FLIP_FINDER_ANALYSIS_DEBUG", "1")

    analyze_description("Po szlifie. Owiewki i klamka. Silnik odpala.")

    captured = capsys.readouterr()
    assert captured.err.strip() == "[damage_analysis] provider=heuristic"


def test_analyze_description_falls_back_when_provider_fails():
    class BrokenProvider:
        def analyze(self, text: str) -> DamageAnalysis:
            raise RuntimeError("provider unavailable")

    analysis = analyze_description(
        "Po dzwonie, krzywy przod, lagi. Nie odpala.",
        provider=BrokenProvider(),
    )

    assert analysis.starts is False
    assert analysis.severity == "high"
    assert "frame" in analysis.hidden_risks


def test_analyze_description_emits_debug_output_for_fallback(
    monkeypatch, capsys
):
    class BrokenProvider:
        def analyze(self, text: str) -> DamageAnalysis:
            raise RuntimeError("provider unavailable")

    monkeypatch.setenv("MOTO_FLIP_FINDER_ANALYSIS_DEBUG", "1")

    analyze_description("Po dzwonie, lagi. Nie odpala.", provider=BrokenProvider())

    captured = capsys.readouterr()
    assert captured.err.strip() == "[damage_analysis] provider=heuristic-fallback"

def test_heuristic_provider_stays_compatible_with_damage_analysis_model():
    provider = HeuristicDescriptionAnalysisProvider()

    analysis = provider.analyze("Do wymiany lusterko i podnozek. Silnik odpala.")

    assert isinstance(analysis, DamageAnalysis)
    assert analysis.suspected_damage == ["footpeg", "mirror"]
    assert analysis.hidden_risks == []


def test_damage_analysis_from_payload_rejects_non_object_payload():
    try:
        damage_analysis_from_payload(["not", "an", "object"])
    except ValueError as exc:
        assert str(exc) == "Damage analysis payload must be a JSON object"
    else:
        raise AssertionError("Expected ValueError for non-object payload")


def test_damage_analysis_from_payload_normalizes_damage_names_and_severity():
    analysis = damage_analysis_from_payload(
        {
            "found_keywords": ["szlif", "owiewka"],
            "suspected_damage": ["fairing", "mirror", "unknown-part", "rims"],
            "hidden_risks": ["front fork", "chassis", "bad-item"],
            "starts": "maybe",
            "severity": "CRITICAL",
        }
    )

    assert analysis.found_keywords == ["owiewka", "szlif"]
    assert analysis.suspected_damage == ["fairings", "mirror", "wheel"]
    assert analysis.hidden_risks == ["forks", "frame"]
    assert analysis.starts is None
    assert analysis.severity == "unknown"


def test_extract_response_payload_rejects_invalid_json():
    class FakeResponse:
        output_text = "{not-json}"

    try:
        _extract_response_payload(FakeResponse())
    except ValueError as exc:
        assert str(exc) == "Analysis response was not valid JSON"
    else:
        raise AssertionError("Expected ValueError for invalid JSON")


def test_extract_response_payload_emits_raw_json_debug(monkeypatch, capsys):
    class FakeResponse:
        output_text = '{"severity":"low"}'

    monkeypatch.setenv("MOTO_FLIP_FINDER_ANALYSIS_DEBUG", "1")

    payload = _extract_response_payload(FakeResponse())

    captured = capsys.readouterr()
    assert payload == {"severity": "low"}
    assert captured.err.strip() == '[damage_analysis] raw_json={"severity":"low"}'
