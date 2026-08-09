"""Optional local VLM reranking with deterministic failover and cache."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

from backend.app.services.retrieval.advanced_rerank import AdvancedRankedFrame


VLM_MODES = ("off", "optional", "required")
DEFAULT_VLM_MODEL = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
PROMPT_REVISION = "retrieval-rerank-v1"


@dataclass(frozen=True)
class VLMRerankReport:
    mode: str
    status: str
    model_name: str
    prompt_revision: str
    candidate_count: int
    cache_hits: int
    fallback_reason: str = ""


def rerank_with_vlm(
    candidates: Sequence[AdvancedRankedFrame],
    *,
    query: str,
    mode: str,
    cache_root: Path,
    image_resolver: Callable[[AdvancedRankedFrame], Path],
    runner: Callable[[str, Path], Mapping[str, object]] | None = None,
    model_name: str = DEFAULT_VLM_MODEL,
    model_revision: str = "main",
    top_m: int = 20,
    timeout_seconds: float = 120.0,
) -> tuple[list[AdvancedRankedFrame], VLMRerankReport]:
    mode = str(mode).casefold()
    if mode not in VLM_MODES:
        raise ValueError(f"Unsupported VLM mode: {mode}")
    deterministic = list(candidates)
    if mode == "off" or not deterministic:
        return deterministic, VLMRerankReport(
            mode=mode,
            status="disabled",
            model_name=model_name,
            prompt_revision=PROMPT_REVISION,
            candidate_count=0,
            cache_hits=0,
        )
    selected = deterministic[: min(max(0, int(top_m)), len(deterministic))]
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_hits = 0
    rescored: list[AdvancedRankedFrame] = []
    started = time.perf_counter()
    try:
        scorer = runner or build_local_vlm_runner(model_name, model_revision)
        for candidate in selected:
            if time.perf_counter() - started > timeout_seconds:
                raise TimeoutError(f"VLM rerank exceeded {timeout_seconds}s")
            image_path = image_resolver(candidate)
            cache_key = _cache_key(
                query=query,
                image_path=image_path,
                model_name=model_name,
                model_revision=model_revision,
            )
            cache_path = cache_root / f"{cache_key}.json"
            if cache_path.exists():
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                cache_hits += 1
            else:
                payload = dict(scorer(query, image_path))
                _validate_payload(payload)
                temp_path = cache_path.with_suffix(".tmp")
                temp_path.write_text(
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    encoding="utf-8",
                )
                temp_path.replace(cache_path)
            _validate_payload(payload)
            vlm_score = float(payload["score"])
            breakdown = dict(candidate.breakdown)
            breakdown["vlm"] = vlm_score
            rescored.append(
                replace(
                    candidate,
                    score=round(0.80 * candidate.score + 0.20 * vlm_score, 8),
                    breakdown=breakdown,
                )
            )
        rescored.extend(deterministic[len(selected) :])
        rescored.sort(
            key=lambda item: (
                -item.score,
                float(item.record.get("timestamp", 0.0)),
                str(item.record.get("candidate_id") or ""),
            )
        )
        return rescored, VLMRerankReport(
            mode=mode,
            status="passed",
            model_name=model_name,
            prompt_revision=PROMPT_REVISION,
            candidate_count=len(selected),
            cache_hits=cache_hits,
        )
    except Exception as exc:
        if mode == "required":
            raise RuntimeError(f"Required VLM rerank failed: {exc}") from exc
        return deterministic, VLMRerankReport(
            mode=mode,
            status="fallback",
            model_name=model_name,
            prompt_revision=PROMPT_REVISION,
            candidate_count=len(selected),
            cache_hits=cache_hits,
            fallback_reason=f"{type(exc).__name__}: {exc}",
        )


def _cache_key(
    *,
    query: str,
    image_path: Path,
    model_name: str,
    model_revision: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(query.encode("utf-8"))
    digest.update(model_name.encode("utf-8"))
    digest.update(model_revision.encode("utf-8"))
    digest.update(PROMPT_REVISION.encode("utf-8"))
    with image_path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_payload(payload: Mapping[str, object]) -> None:
    if "score" not in payload:
        raise ValueError("VLM output is missing score")
    score = float(payload["score"])
    if not 0.0 <= score <= 1.0:
        raise ValueError("VLM score must be within [0, 1]")


def build_local_vlm_runner(
    model_name: str,
    model_revision: str,
) -> Callable[[str, Path], Mapping[str, object]]:
    # Imports and model allocation are intentionally lazy.  The caller can
    # release SigLIP/GPU state immediately before the first invocation.
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForVision2Seq, AutoProcessor
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("Local VLM dependencies are unavailable") from exc

    processor = AutoProcessor.from_pretrained(model_name, revision=model_revision)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        revision=model_revision,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.to(device)
    model.eval()

    def run(query: str, image_path: Path) -> Mapping[str, object]:
        prompt = (
            "Return strict JSON only: {\"score\": number between 0 and 1}. "
            f"Score how well this frame answers the retrieval query: {query}"
        )
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=48, do_sample=False)
        text = processor.batch_decode(generated, skip_special_tokens=True)[0]
        start = text.rfind("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("VLM returned malformed JSON")
        value = json.loads(text[start : end + 1])
        if not isinstance(value, dict):
            raise ValueError("VLM JSON must be an object")
        return value

    return run
