import os
from dataclasses import dataclass

@dataclass(frozen=True)
class StreamConfig:
    # If local: SOURCE_VIDEO_URL in "data/raw/video/"
    SOURCE_VIDEO_URL: str = os.getenv("SOURCE_VIDEO_URL", "./data/raw/video")