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

    NEIGHBOR_FRAME_PATH: str = os.getenv("NEIGHBOR_FRAME_PATH", 'data/metadata/neighbors_all.jsonl')
