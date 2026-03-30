from moto_flip_finder.damage_analysis import (
    HeuristicDescriptionAnalysisProvider,
    analyze_description,
)
from moto_flip_finder.models import DamageAnalysis


def test_analyze_description_uses_heuristics_without_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    analysis = analyze_description(
        "Motocykl uszkodzony, po szlifie. Do poprawek owiewki i klamka. Silnik odpala."
    )

    assert analysis.starts is True
    assert analysis.severity == "medium"
    assert "fairings" in analysis.suspected_damage
    assert "lever" in analysis.suspected_damage


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


def test_analyze_description_uses_openai_provider_when_key_is_present(monkeypatch):
    class FakeOpenAIProvider:
        def analyze(self, text: str) -> DamageAnalysis:
            return DamageAnalysis(
                found_keywords=["llm"],
                suspected_damage=["fairings"],
                hidden_risks=["frame"],
                starts=True,
                severity="medium",
            )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "moto_flip_finder.damage_analysis.OpenAIDescriptionAnalysisProvider",
        FakeOpenAIProvider,
    )

    analysis = analyze_description("Any listing text")

    assert analysis == DamageAnalysis(
        found_keywords=["llm"],
        suspected_damage=["fairings"],
        hidden_risks=["frame"],
        starts=True,
        severity="medium",
    )


def test_heuristic_provider_stays_compatible_with_damage_analysis_model():
    provider = HeuristicDescriptionAnalysisProvider()

    analysis = provider.analyze("Do wymiany lusterko i podnozek. Silnik odpala.")

    assert isinstance(analysis, DamageAnalysis)
    assert analysis.suspected_damage == ["footpeg", "mirror"]
    assert analysis.hidden_risks == []
