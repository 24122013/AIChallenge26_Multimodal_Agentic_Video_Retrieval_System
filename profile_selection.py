"""Profile ONLY the multimodal selection stage for one already-processed video."""
import cProfile
import pstats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backend.app.pipelines.offline_pipeline import (
    OfflinePipelineConfig,
    PerVideoPaths,
    _extract_all_dense_features,
    _load_or_run_dense_candidate_generation,
    _load_or_run_dense_materialization,
    _load_or_run_shot_detection,
    _run_multimodal_selection,
    _file_signature,
)


def main() -> None:
    video_id = sys.argv[1] if len(sys.argv) > 1 else "L21_V014"
    video_path = Path(f"data/raw/video/{video_id}.mp4")
    if not video_path.is_file():
        raise SystemExit(f"Video not found: {video_path}")

    config = OfflinePipelineConfig(
        output_dir=Path("data"),
        device="cuda",
        resume=True,
        force=False,
    )
    paths = PerVideoPaths.from_config(video_id, config)
    source_signature = _file_signature(video_path)

    print("Loading shots (from checkpoint)...")
    shots = _load_or_run_shot_detection(
        video_path, config, paths, source_signature=source_signature
    )
    print("Loading candidates (from checkpoint)...")
    candidates = _load_or_run_dense_candidate_generation(video_path, shots, config, paths)
    print("Loading materialization (from checkpoint)...")
    materialized = _load_or_run_dense_materialization(
        video_path, shots, candidates, config, paths
    )
    print("Loading dense features: SigLIP2/OCR/objects/caption (from checkpoint)...")
    features = _extract_all_dense_features(video_path, materialized, config, paths)

    print(f"Profiling selection stage for {video_id} "
          f"(dense_pool={len(materialized.records)})...")
    profiler = cProfile.Profile()
    profiler.enable()
    result, contract = _run_multimodal_selection(shots, materialized, features, config, paths)
    profiler.disable()

    print(f"Selection done: {len(result.final_records)} keyframes selected.")

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(25)
    stats.dump_stats("/tmp/selection_profile.prof")
    print("\nFull profile saved to /tmp/selection_profile.prof")


if __name__ == "__main__":
    main()
