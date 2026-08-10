"""Resolve standard, weighted and query-adaptive RRF modality weights."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from backend.app.services.retrieval.query_plan import QueryPlan


FUSION_MODES = ("legacy", "standard_rrf", "weighted_rrf", "adaptive_rrf")
RETRIEVAL_MODALITIES = ("visual", "caption", "ocr", "objects", "asr")
DEFAULT_WEIGHTED_RRF_WEIGHTS: dict[str, float] = {
    "visual": 0.50,
    "caption": 0.15,
    "ocr": 0.10,
    "objects": 0.05,
    "asr": 0.15,
}


@dataclass(frozen=True)
class ModalityWeighting:
    fusion_mode: str
    detected_modalities: tuple[str, ...]
    weights: dict[str, float]
    adjustments: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "fusion_mode": self.fusion_mode,
            "detected_modalities": list(self.detected_modalities),
            "weights": dict(self.weights),
            "adjustments": list(self.adjustments),
        }


def resolve_modality_weights(
    plan: QueryPlan,
    *,
    fusion_mode: str,
    base_weights: Mapping[str, float] | None = None,
) -> ModalityWeighting:
    mode = str(fusion_mode).casefold()
    if mode not in FUSION_MODES:
        raise ValueError(f"Unsupported fusion mode: {fusion_mode}")
    if mode == "standard_rrf":
        weights = {name: 1.0 for name in RETRIEVAL_MODALITIES}
        return ModalityWeighting(mode, plan.modality_hints, weights, ())

    weights = {
        name: float((base_weights or DEFAULT_WEIGHTED_RRF_WEIGHTS).get(name, 0.0))
        for name in RETRIEVAL_MODALITIES
    }
    if any(value < 0 for value in weights.values()):
        raise ValueError("RRF modality weights must be non-negative")
    adjustments: list[str] = []
    if mode == "adaptive_rrf":
        hints = set(plan.modality_hints)
        boosts: dict[str, dict[str, float]] = {
            "visual": {"visual": 1.50, "caption": 1.10},
            "caption": {"caption": 1.80, "visual": 1.10},
            "ocr": {"ocr": 3.00, "visual": 0.85},
            "asr": {"asr": 3.00, "visual": 0.80},
            "objects": {"objects": 2.50, "visual": 1.15},
            # Action has no standalone index. Its evidence is distributed to
            # the visual, caption and object branches.
            "action": {"visual": 1.35, "caption": 2.00, "objects": 1.35},
        }
        for hint in plan.modality_hints:
            for modality, multiplier in boosts.get(hint, {}).items():
                weights[modality] *= multiplier
                adjustments.append(f"{hint}->{modality}x{multiplier:g}")
        if "asr" not in hints:
            weights["asr"] *= 0.50
            adjustments.append("unhinted_asr->asrx0.5")
        if "ocr" not in hints:
            weights["ocr"] *= 0.70
            adjustments.append("unhinted_ocr->ocrx0.7")

    total = sum(weights.values())
    if total <= 0:
        raise ValueError("At least one RRF modality weight must be positive")
    normalized = {name: round(value / total, 8) for name, value in weights.items()}
    return ModalityWeighting(
        mode,
        plan.modality_hints,
        normalized,
        tuple(adjustments),
    )
