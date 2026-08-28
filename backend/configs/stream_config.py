import os
from dataclasses import dataclass

@dataclass(frozen=True)
class StreamConfig:
    # Uses the canonical root from your .env
    SOURCE_VIDEO_URL: str = os.getenv("RETRIEVAL_TRAKE_VIDEO_ROOT", "data/raw/video")
    
    # Standard output directory for canonical keyframes
    RETRIEVAL_KEYFRAME_ROOT: str = os.getenv("RETRIEVAL_KEYFRAME_ROOT", "data/keyframes")

    # Full dense candidate pool used by online dense refinement. Some returned
    # candidates are intentionally not copied into the canonical keyframe root.
    RETRIEVAL_DENSE_KEYFRAME_ROOT: str = os.getenv(
        "RETRIEVAL_DENSE_KEYFRAME_ROOT",
        "data/dense_keyframes",
    )
    B2_RCLONE_REMOTE: str = os.getenv(
    "B2_RCLONE_REMOTE",
    "b2remote:AIO-DataDominator-Storage",
    )   

    B2_KEYFRAME_PREFIX: str = os.getenv(
        "B2_KEYFRAME_PREFIX",
        "artifacts/keyframes",
    )

    B2_DENSE_KEYFRAME_PREFIX: str = os.getenv(
        "B2_DENSE_KEYFRAME_PREFIX",
        "artifacts/dense_keyframes",
    )
    
    B2_VIDEO_REMOTE: str = os.getenv(
    "B2_VIDEO_REMOTE",
    "b2remote:AIO-DataDominator-Storage",
    )

    B2_VIDEO_PREFIX: str = os.getenv(
        "B2_VIDEO_PREFIX",
        "raw/video/video",
    )

    NEIGHBOR_FRAME_PATH: str = os.getenv("NEIGHBOR_FRAME_PATH", 'data/metadata/neighbors_all.jsonl')
