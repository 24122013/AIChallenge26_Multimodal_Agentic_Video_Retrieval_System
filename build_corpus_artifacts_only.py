import argparse
import json
import logging
from pathlib import Path

import numpy as np
import backend.app.pipelines.offline_pipeline as p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--expected-videos", type=int, default=873)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bge-batch-size", type=int, default=16)
    args = ap.parse_args()

    root = Path(args.data_dir)
    expected = args.expected_videos

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    # ------------------------------------------------------------
    # Discover video IDs từ selected embeddings.
    # Không cần raw MP4.
    # ------------------------------------------------------------
    prefix = "siglip2_so400m_patch16_384_"

    video_ids = sorted({
        x.stem[len(prefix):]
        for x in (root / "embeddings").glob(
            "siglip2_so400m_patch16_384_L*_V*.npy"
        )
    })

    print(f"[PRECHECK] discovered videos = {len(video_ids)}")

    if len(video_ids) != expected:
        raise RuntimeError(
            f"Expected {expected} videos, found {len(video_ids)}"
        )

    config = p.OfflinePipelineConfig(
        output_dir=root,
        device=args.device,
        resume=True,
        force=False,
        build_corpus=True,
        bge_batch_size=args.bge_batch_size,
    )

    # ------------------------------------------------------------
    # Artifact-only validator.
    #
    # Bỏ:
    #   - raw MP4 SHA / identity validation
    #   - selected JPG validation
    #
    # Giữ:
    #   - selection report
    #   - canonical metadata
    #   - selected embeddings
    #   - caption / OCR / object alignment
    #   - dense corpus inputs
    #   - candidate ledger
    # ------------------------------------------------------------
    def artifact_only_validate_selected_bundle(
        *,
        video_path,
        config,
        paths,
        require_completion,
        source_signature=None,
    ):
        print(f"[VALIDATE] {paths.video_id}", flush=True)
        selection_report = p._read_json(paths.selection_report)

        if selection_report.get("status") != "passed":
            raise ValueError(
                f"{paths.video_id}: selection report not passed"
            )

        guarantees = selection_report.get("guarantees")
        if (
            not isinstance(guarantees, dict)
            or guarantees.get("constraints_satisfied") is not True
        ):
            raise ValueError(
                f"{paths.video_id}: selection guarantees failed"
            )

        records = p._read_jsonl(paths.selected_metadata)

        if not records:
            raise ValueError(
                f"{paths.video_id}: selected metadata empty"
            )

        if any(
            str(r.get("video_id") or "") != paths.video_id
            for r in records
        ):
            raise ValueError(
                f"{paths.video_id}: wrong video_id in canonical metadata"
            )

        candidate_ids = p._candidate_ids(records)
        frame_ids = p._frame_ids(records)

        # ---------- selected embeddings ----------
        embeddings = np.load(
            paths.selected_embeddings,
            allow_pickle=False,
        )

        embedding_records = p._read_jsonl(
            paths.selected_embedding_metadata
        )

        p.validate_embedding_artifacts(
            embeddings,
            embedding_records,
        )

        if p._candidate_ids(embedding_records) != candidate_ids:
            raise ValueError(
                f"{paths.video_id}: selected embedding candidate order mismatch"
            )

        if p._frame_ids(embedding_records) != frame_ids:
            raise ValueError(
                f"{paths.video_id}: selected embedding frame order mismatch"
            )

        # ---------- selected modalities ----------
        for label, path in (
            ("caption", paths.selected_captions),
            ("OCR", paths.selected_ocr),
            ("object", paths.selected_objects),
        ):
            modality_records = p._read_jsonl(path)

            p._validate_modality_records(
                label=f"selected {label}",
                records=modality_records,
                expected_candidates=records,
                video_id=paths.video_id,
                required_nonempty_field=(
                    "caption" if label == "caption" else None
                ),
            )

            if set(p._frame_ids(modality_records)) != set(frame_ids):
                raise ValueError(
                    f"{paths.video_id}: {label} frame IDs mismatch"
                )

        # ---------- dense corpus inputs ----------
        dense_inputs = (
            paths.dense_embeddings,
            paths.dense_embedding_metadata,
            paths.dense_captions,
            paths.dense_ocr,
            paths.dense_objects,
            paths.candidate_ledger,
        )

        for path in dense_inputs:
            if not path.is_file():
                raise FileNotFoundError(
                    f"{paths.video_id}: missing corpus input: {path}"
                )

        # Validate dense SigLIP arrays/metadata as well.
        p.validate_embedding_source(
            paths.dense_embeddings,
            paths.dense_embedding_metadata,
            paths.video_id,
        )

        dense_count = int(
            selection_report.get("candidate_count", -1)
        )

        if dense_count <= 0:
            # fallback từ dense embedding metadata
            dense_count = len(
                p._read_jsonl(paths.dense_embedding_metadata)
            )

        if dense_count <= 0:
            raise ValueError(
                f"{paths.video_id}: invalid dense candidate count"
            )

        # Completion report phải tồn tại vì corpus source contract
        # hash chính file này.
        if not paths.completion_report.is_file():
            raise FileNotFoundError(
                f"{paths.video_id}: missing completion report"
            )

        completion = p._read_json(paths.completion_report)

        if completion.get("status") != "passed":
            raise ValueError(
                f"{paths.video_id}: completion report not passed"
            )

        validation = {
            "status": "passed",
            "video_id": paths.video_id,
            "dense_candidate_count": dense_count,
            "selected_count": len(records),
            "artifact_only_validation": True,
        }

        return validation, tuple(records)

    # ------------------------------------------------------------
    # Build VideoArtifacts placeholders.
    # video_path không được đọc vì validator đã được thay.
    # ------------------------------------------------------------
    videos = []

    for i, video_id in enumerate(video_ids, 1):
        paths = p.PerVideoPaths.from_config(video_id, config)

        videos.append(
            p.VideoArtifacts(
                video_id=video_id,
                video_path=Path(
                    f"/artifact-only/{video_id}.mp4"
                ),
                paths=paths,
                selected_count=0,
                dense_candidate_count=0,
                skipped=True,
                validation={},
            )
        )

        if i % 100 == 0 or i == len(video_ids):
            print(
                f"[PRECHECK] prepared {i}/{len(video_ids)}"
            )

    # ------------------------------------------------------------
    # Chỉ monkeypatch handoff validator.
    # Corpus implementation còn lại giữ nguyên.
    # ------------------------------------------------------------
    original_validator = p._validate_selected_bundle
    p._validate_selected_bundle = artifact_only_validate_selected_bundle

    try:
        print()
        print("========================================")
        print(f"BUILDING CORPUS FROM {len(videos)} VIDEOS")
        print("RAW VIDEO / JPG VALIDATION: BYPASSED")
        print("CORPUS VALIDATION: ENABLED")
        print("========================================")
        print()

        report = p.build_corpus_indexes(
            tuple(videos),
            config,
        )

    finally:
        p._validate_selected_bundle = original_validator

    print()
    print("========== RESULT ==========")
    print("status             :", report.get("status"))
    print("video_count        :", report.get("video_count"))
    print(
        "selected_keyframes:",
        report.get("selected_keyframe_count"),
    )
    print(
        "dense_candidates  :",
        report.get("dense_candidate_count"),
    )

    segments = report.get("segments_events") or {}
    print(
        "segment video IDs :",
        len(segments.get("video_ids", [])),
    )

    bge = report.get("bge_m3") or {}
    print(
        "BGE vectors       :",
        bge.get("vector_count"),
    )

    if report.get("status") != "passed":
        raise RuntimeError("Corpus validation FAILED")

    if report.get("video_count") != expected:
        raise RuntimeError(
            f"Corpus contains {report.get('video_count')} "
            f"videos instead of {expected}"
        )

    if len(segments.get("video_ids", [])) != expected:
        raise RuntimeError(
            "segments/events does not contain "
            f"{expected} video IDs"
        )

    print()
    print(
        f"SUCCESS: corpus contains exactly {expected} videos"
    )


if __name__ == "__main__":
    main()
