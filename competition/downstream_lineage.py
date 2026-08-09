"""Fail-closed lineage manifests for Phase 4 competition artifacts.

The downstream builders intentionally stay independent: neighbors, segments,
and the lexical text index can each be rerun.  A sidecar manifest binds every
output to the exact canonical keyframe publish and direct input files that
produced it, so retrieval cannot silently mix artifacts from different runs.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from competition.keyframe_phase3 import (
    atomic_write_json,
    sha256_file,
    sha256_json,
)


PHASE4_DOWNSTREAM_CONTRACT_VERSION = 1


def artifact_entry(path: Path) -> dict[str, object]:
    """Return a stable identity for one required on-disk artifact."""

    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Downstream artifact not found: {resolved}")
    return {
        "path": resolved.as_posix(),
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _artifact_map(paths: Mapping[str, Path]) -> dict[str, dict[str, object]]:
    return {
        name: artifact_entry(paths[name])
        for name in sorted(paths)
    }


def build_stage_manifest(
    *,
    stage: str,
    canonical_sources: Sequence[Mapping[str, object]],
    input_paths: Mapping[str, Path],
    output_paths: Mapping[str, Path],
    config: Mapping[str, object],
) -> dict[str, Any]:
    """Build a passed manifest from artifacts that already exist."""

    if not stage.strip():
        raise ValueError("stage must be non-empty")
    canonical = [dict(source) for source in canonical_sources]
    if not canonical:
        raise ValueError("canonical_sources must not be empty")
    return {
        "version": PHASE4_DOWNSTREAM_CONTRACT_VERSION,
        "stage": stage,
        "status": "passed",
        "canonical_sources_sha256": sha256_json(canonical),
        "canonical_sources": canonical,
        "inputs": _artifact_map(input_paths),
        "outputs": _artifact_map(output_paths),
        "config": dict(config),
    }


def write_stage_manifest(
    manifest_path: Path,
    *,
    stage: str,
    canonical_sources: Sequence[Mapping[str, object]],
    input_paths: Mapping[str, Path],
    output_paths: Mapping[str, Path],
    config: Mapping[str, object],
) -> dict[str, Any]:
    """Atomically publish a Phase 4 manifest after its outputs are complete."""

    manifest = build_stage_manifest(
        stage=stage,
        canonical_sources=canonical_sources,
        input_paths=input_paths,
        output_paths=output_paths,
        config=config,
    )
    atomic_write_json(manifest_path, manifest)
    return manifest


def validate_stage_manifest(
    manifest_path: Path,
    *,
    stage: str,
    canonical_sources: Sequence[Mapping[str, object]],
    input_paths: Mapping[str, Path],
    output_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Validate lineage and every direct input/output byte hash.

    The stored config remains audit metadata.  Validation compares the actual
    source and output artifacts because downstream consumers do not necessarily
    know the CLI arguments used by the upstream stage.
    """

    if not manifest_path.is_file():
        raise RuntimeError(
            f"{stage} lineage manifest is missing; rerun {stage}: {manifest_path}"
        )
    try:
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"{stage} lineage manifest is unreadable; rerun {stage}: {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict):
        raise RuntimeError(f"{stage} lineage manifest must be an object")
    if (
        manifest.get("version") != PHASE4_DOWNSTREAM_CONTRACT_VERSION
        or manifest.get("stage") != stage
        or manifest.get("status") != "passed"
    ):
        raise RuntimeError(f"{stage} lineage manifest is outdated; rerun {stage}")

    canonical = [dict(source) for source in canonical_sources]
    if (
        manifest.get("canonical_sources_sha256") != sha256_json(canonical)
        or manifest.get("canonical_sources") != canonical
    ):
        raise RuntimeError(
            f"{stage} artifacts are stale for current canonical keyframes; rerun {stage}"
        )
    try:
        current_inputs = _artifact_map(input_paths)
        current_outputs = _artifact_map(output_paths)
    except (FileNotFoundError, OSError) as exc:
        raise RuntimeError(f"{stage} artifacts are missing; rerun {stage}") from exc
    if manifest.get("inputs") != current_inputs:
        raise RuntimeError(f"{stage} inputs changed; rerun {stage}")
    if manifest.get("outputs") != current_outputs:
        raise RuntimeError(f"{stage} outputs changed; rerun {stage}")
    return manifest


__all__ = [
    "PHASE4_DOWNSTREAM_CONTRACT_VERSION",
    "artifact_entry",
    "build_stage_manifest",
    "validate_stage_manifest",
    "write_stage_manifest",
]
