from dataclasses import dataclass, asdict, field
from typing import Optional

@dataclass(frozen=True)
class ImagePayload:
    mode: str  # 'link', 'upload', 'generate'
    image_url: Optional[str] = None
    image_b64: Optional[str] = None
    image_prompt: Optional[str] = None

    def to_dict(self):
        return asdict(self)

@dataclass(frozen=True)
class TextSearchPayload:
    query: str
    mode: str = "visual"
    top_k: int = 20
    
    def to_dict(self):
        return asdict(self)