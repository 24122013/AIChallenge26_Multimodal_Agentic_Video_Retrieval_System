import os
from dataclasses import dataclass

@dataclass(frozen=True)
class StreamConfig:
    STREAM_FROM_LOCAL: bool = os.getenv("STREAM_FROM_LOCAL", "False").lower() in ("true", "1")

    # If local: SOURCE_VIDEO_URL in "data/raw/video/"
    SOURCE_VIDEO_URL: str = os.getenv("SOURCE_VIDEO_URL", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/raw/video')))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1")