"""Streaming API for Video (Phase 1: Stream from local machine)."""
from __future__ import annotations

try:  # pragma: no cover - depends on optional API runtime.
    import logging
    import os
    import json
    from typing import Optional
    from functools import lru_cache
    from fastapi import APIRouter, Header, HTTPException
    from fastapi.responses import StreamingResponse
    from backend.configs.stream_config import StreamConfig
    from backend.app.services.video.stream_utils import get_video_path, get_byte_range, generate_full_video, generate_video_chunk

    # Base directory for video sources
    video_router = APIRouter(prefix="/api/video", tags=["video"])
    logger = logging.getLogger(__name__)
    STREAM_CONFIG = StreamConfig()

    @lru_cache(maxsize=1)
    def load_neighbors_metadata():
        """Loads the neighbors JSONL file into memory (cached)."""
        metadata_path = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                '../../data/metadata/neighbors_all.jsonl'
            )
        )
        neighbors_map = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        neighbors_map[record["frame_id"]] = record
        else:
            logger.warning(f"Neighbors metadata not found at {metadata_path}")
            
        return neighbors_map

    # GET /api/video/frame_neighbor/{frame_id}
    @video_router.get("/frame_neighbor/{frame_id}")
    def get_frame_neighbor(frame_id: str):
        """Returns the neighbor frames for a specific frame."""
        metadata = load_neighbors_metadata()
        if frame_id not in metadata:
            raise HTTPException(status_code=404, detail="Frame neighbors not found")
        return metadata[frame_id]

    # GET /api/video/stream/{video_name} — HTTP Range streaming
    @video_router.get("/stream/{video_name}")
    def stream_video(video_name: str, range: Optional[str] = Header(None)):
        """Streams local video files using HTTP Range Requests (Status 206)."""
        video_file = video_name + ".mp4"
        file_path = get_video_path(config=STREAM_CONFIG, video_file=video_file)
        file_size = os.path.getsize(file_path)

        # No Range Header Provided - Send the whole file
        if not range:
            headers = {
                'Content-Length': str(file_size),
                'Accept-Ranges': 'bytes'
            }
            return StreamingResponse(
                generate_full_video(), 
                status_code=200, 
                media_type='video/mp4', 
                headers=headers
            )
        else:
            # Parse the Range Header
            start_byte, end_byte, length = get_byte_range(range=range, file_size=file_size)

            # Return the 206 Partial Content response
            headers = {
                'Content-Length': str(length),
                'Accept-Ranges': 'bytes',
                'Content-Range': f'bytes {start_byte}-{end_byte}/{file_size}',
            }
            
            return StreamingResponse(
                generate_video_chunk(start_byte=start_byte, chunk_len=length, file_path=file_path),
                status_code=206, 
                media_type='video/mp4', 
                headers=headers
            )

except ImportError:  # pragma: no cover
    video_router = None