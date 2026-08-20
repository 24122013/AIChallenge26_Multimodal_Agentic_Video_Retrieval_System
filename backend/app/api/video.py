"""Streaming API for Video (Phase 1: Stream from local machine)."""
from __future__ import annotations

try:
    import logging
    import os
    import json
    from typing import Optional
    from functools import lru_cache
    from fastapi import APIRouter, Header, HTTPException
    from fastapi.responses import StreamingResponse, FileResponse
    from backend.configs.stream_config import StreamConfig
    from backend.app.services.video.stream_utils import get_video_path, get_byte_range, generate_full_video, generate_video_chunk

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

    @video_router.get("/frame_neighbor/{frame_id}")
    def get_frame_neighbor(frame_id: str):
        """Returns the neighbor frames mapped to internal API routes."""
        metadata = load_neighbors_metadata()
        record = metadata.get(frame_id)
        
        if not record:
            raise HTTPException(status_code=404, detail="Frame neighbors not found")
        
        return {
            "frame_id": record["frame_id"],
            "video_id": record["video_id"],
            "timestamp": record["timestamp"],
            "target_url": f"/video/frame/{record['frame_id']}",
            "neighbors_before": [
                {
                    "frame_id": nb["frame_id"],
                    "delta_seconds": nb["delta_seconds"],
                    "url": f"/video/frame/{nb['frame_id']}"
                } for nb in record.get("neighbors_before", [])
            ],
            "neighbors_after": [
                {
                    "frame_id": na["frame_id"],
                    "delta_seconds": na["delta_seconds"],
                    "url": f"/video/frame/{na['frame_id']}"
                } for na in record.get("neighbors_after", [])
            ]
        }

    @video_router.get("/frame/{frame_id}")
    def get_frame_image(frame_id: str):
        """Serves the keyframe JPEG directly from the local disk config."""
        # Sanitize to prevent directory traversal
        safe_frame_id = os.path.basename(frame_id)
        file_path = os.path.join(STREAM_CONFIG.RETRIEVAL_KEYFRAME_ROOT, f"{safe_frame_id}.jpg")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Frame image not found")
            
        return FileResponse(file_path, media_type="image/jpeg")

    @video_router.get("/stream/{video_name}")
    def stream_video(video_name: str, range: Optional[str] = Header(None)):
        """Streams local video files using HTTP Range Requests."""
        video_file = video_name + ".mp4"
        file_path = get_video_path(config=STREAM_CONFIG, video_file=video_file)
        file_size = os.path.getsize(file_path)

        if not range:
            headers = {
                'Content-Length': str(file_size),
                'Accept-Ranges': 'bytes'
            }
            return StreamingResponse(
                generate_full_video(file_path=file_path), 
                status_code=200,
                media_type='video/mp4', 
                headers=headers
            )
        else:
            start_byte, end_byte, length = get_byte_range(range=range, file_size=file_size)
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

except ImportError:
    video_router = None

if __name__ == "__main__":
    @lru_cache(maxsize=1)
    def load_neighbors_metadata():
        """Loads the neighbors JSONL file into memory (cached)."""
        metadata_path = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                '../../../data/metadata/neighbors_all.jsonl'
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
            print(f"Neighbors metadata not found at {metadata_path}")
            
        return neighbors_map

    metadata = load_neighbors_metadata()
    print(metadata)