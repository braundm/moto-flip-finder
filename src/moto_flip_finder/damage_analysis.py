from __future__ import annotations

import os

from .description_analysis_provider import (
    DescriptionAnalysisProvider,
    OpenAIDescriptionAnalysisProvider,
)
from .models import DamageAnalysis


DAMAGE_KEYWORDS = {
    "owiewki": "fairings",
    "klamka": "lever",
    "lusterko": "mirror",
    "podnozek": "footpeg",
    "bak": "tank",
    "felga": "wheel",
    "wydech": "exhaust",
    "lagi": "forks",
    "rama": "frame",
    "wahacz": "swingarm",
}

HIDDEN_RISK_KEYWORDS = {
    "po szlifie": ["forks", "frame", "swingarm"],
    "po dzwonie": ["frame", "forks", "wheel"],
    "uderzony": ["frame", "forks"],
    "krzywy": ["frame", "forks", "wheel"],
}

STARTS_TRUE_KEYWORDS = [
    "odpala",
    "jezdzi",
    "silnik odpala",
]

STARTS_FALSE_KEYWORDS = [
    "nie odpala",
    "uszkodzony silnik",
    "zatarty silnik",
]

HIGH_SEVERITY_KEYWORDS = [
    "rama",
    "lagi",
    "wahacz",
    "po dzwonie",
    "krzywy",
]


class HeuristicDescriptionAnalysisProvider(DescriptionAnalysisProvider):
    def analyze(self, text: str) -> DamageAnalysis:
        lowered = text.lower()

        found_keywords = []
        suspected_damage = []
        hidden_risks = []

        for keyword, damage_name in DAMAGE_KEYWORDS.items():
            if keyword in lowered:
                found_keywords.append(keyword)
                suspected_damage.append(damage_name)

        for keyword, risks in HIDDEN_RISK_KEYWORDS.items():
            if keyword in lowered:
                found_keywords.append(keyword)
                hidden_risks.extend(risks)

        starts = None

        for keyword in STARTS_FALSE_KEYWORDS:
            if keyword in lowered:
                found_keywords.append(keyword)
                starts = False

        for keyword in STARTS_TRUE_KEYWORDS:
            if keyword in lowered and starts is None:
                found_keywords.append(keyword)
                starts = True

        severity = "low"

        for keyword in HIGH_SEVERITY_KEYWORDS:
            if keyword in lowered:
                severity = "high"
                break

        if severity != "high" and suspected_damage:
            severity = "medium"

        suspected_damage = sorted(set(suspected_damage))
        hidden_risks = sorted(set(hidden_risks))
        found_keywords = sorted(set(found_keywords))

        return DamageAnalysis(
            found_keywords=found_keywords,
            suspected_damage=suspected_damage,
            hidden_risks=hidden_risks,
            starts=starts,
            severity=severity,
        )


def get_description_analysis_provider() -> DescriptionAnalysisProvider:
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIDescriptionAnalysisProvider()

    return HeuristicDescriptionAnalysisProvider()


def analyze_description(
    text: str,
    provider: DescriptionAnalysisProvider | None = None,
) -> DamageAnalysis:
    selected_provider = provider or get_description_analysis_provider()

    try:
        return selected_provider.analyze(text)
    except Exception:
        if isinstance(selected_provider, HeuristicDescriptionAnalysisProvider):
            raise

        return HeuristicDescriptionAnalysisProvider().analyze(text)
