"""Streaming API for Video (Phase 1: Stream from local machine)."""
from __future__ import annotations

try:
    import logging
    import os
    import json
    import subprocess
    from typing import Optional
    from functools import lru_cache
    from fastapi import APIRouter, Header, HTTPException
    from fastapi.responses import StreamingResponse, FileResponse, Response
    from backend.configs.stream_config import StreamConfig
    from backend.app.services.video.stream_utils import get_video_path, get_byte_range, generate_full_video, generate_video_chunk

    video_router = APIRouter(prefix="/api/video", tags=["video"])
    logger = logging.getLogger(__name__)
    STREAM_CONFIG = StreamConfig()

    @lru_cache(maxsize=1)
    def load_neighbors_metadata(metadata_path: str):
        """Loads the neighbors JSONL file into memory (cached)."""
        neighbors_map = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        neighbors_map[(record["video_id"], record["frame_id"])] = record
        else:
            logger.warning(f"Neighbors metadata not found at {metadata_path}")
            
        return neighbors_map

    @video_router.get("/frame_neighbor/{video_id}/{frame_id}")
    def get_frame_neighbor(video_id: str, frame_id: str):
        """Returns the neighbor frames mapped to internal API routes."""
        metadata = load_neighbors_metadata(STREAM_CONFIG.NEIGHBOR_FRAME_PATH)
        record = metadata.get((video_id, frame_id))
        
        if not record:
            raise HTTPException(status_code=404, detail="Frame neighbors not found")
        
        return {
            "frame_id": record["frame_id"],
            "video_id": record["video_id"],
            "timestamp": record["timestamp"],
            "target_url": (
                f"/api/video/frame/{record['video_id']}/{record['frame_id']}"
            ),
            "neighbors_before": [
                {
                    "frame_id": nb["frame_id"],
                    "delta_seconds": nb["delta_seconds"],
                    "url": (
                        f"/api/video/frame/{record['video_id']}/{nb['frame_id']}"
                    )
                } for nb in record.get("neighbors_before", [])
            ],
            "neighbors_after": [
                {
                    "frame_id": na["frame_id"],
                    "delta_seconds": na["delta_seconds"],
                    "url": (
                        f"/api/video/frame/{record['video_id']}/{na['frame_id']}"
                    )
                } for na in record.get("neighbors_after", [])
            ]
        }

    # @video_router.get("/frame/{video_id}/{frame_id}")
    # def get_frame_image(video_id: str, frame_id: str):
    #     """Serve a selected keyframe or a dense retrieval candidate JPEG."""
    #     # Sanitize to prevent directory traversal
    #     safe_frame_id = os.path.basename(frame_id)
    #     file_name = f"{safe_frame_id}.jpg"
    #     candidate_paths = (
    #         os.path.join(STREAM_CONFIG.RETRIEVAL_KEYFRAME_ROOT, video_id, file_name),
    #         os.path.join(
    #             STREAM_CONFIG.RETRIEVAL_DENSE_KEYFRAME_ROOT,
    #             video_id,
    #             file_name,
    #         ),
    #     )

    #     for file_path in candidate_paths:
    #         if os.path.isfile(file_path):
    #             return FileResponse(file_path, media_type="image/jpeg")

    #     raise HTTPException(status_code=404, detail="Frame image not found")
    @video_router.get("/frame/{video_id}/{frame_id}")
    def get_frame_image(video_id: str, frame_id: str):
        """
        Serve a selected keyframe or dense retrieval candidate JPEG.

        Supports frame_id formats:
        - 29884
        - 000029884
        - FRAME_L22_V001_000029884

        Priority:
        1. Local selected keyframes
        2. Local dense keyframes
        3. B2 selected keyframes
        4. B2 dense keyframes
        """

        safe_video_id = os.path.basename(video_id)
        safe_frame_id = os.path.basename(frame_id)

        # ---------------------------------------------------------
        # Normalize frame filename
        # ---------------------------------------------------------

        prefix = f"FRAME_{safe_video_id}_"

        if safe_frame_id.startswith(prefix):
            # Frontend already sent:
            # FRAME_L22_V001_000029884
            raw_number = safe_frame_id[len(prefix):]
        else:
            # Frontend/API sent:
            # 29884 or 000029884
            raw_number = safe_frame_id

        try:
            frame_number = int(raw_number)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid frame_id: {frame_id}",
            )

        file_name = f"FRAME_{safe_video_id}_{frame_number:09d}.jpg"

        # ---------------------------------------------------------
        # 1. Local storage
        # ---------------------------------------------------------

        candidate_paths = (
            os.path.join(
                STREAM_CONFIG.RETRIEVAL_KEYFRAME_ROOT,
                safe_video_id,
                file_name,
            ),
            os.path.join(
                STREAM_CONFIG.RETRIEVAL_DENSE_KEYFRAME_ROOT,
                safe_video_id,
                file_name,
            ),
        )

        for file_path in candidate_paths:
            if os.path.isfile(file_path):
                return FileResponse(
                    file_path,
                    media_type="image/jpeg",
                    headers={
                        "Cache-Control": "public, max-age=86400",
                    },
                )

        # ---------------------------------------------------------
        # 2. Backblaze B2 fallback
        # ---------------------------------------------------------

        remote = STREAM_CONFIG.B2_RCLONE_REMOTE.rstrip("/")

        b2_candidates = (
            (
                f"{remote}/"
                f"{STREAM_CONFIG.B2_KEYFRAME_PREFIX.strip('/')}/"
                f"{safe_video_id}/{file_name}"
            ),
            (
                f"{remote}/"
                f"{STREAM_CONFIG.B2_DENSE_KEYFRAME_PREFIX.strip('/')}/"
                f"{safe_video_id}/{file_name}"
            ),
        )

        for remote_path in b2_candidates:
            try:
                result = subprocess.run(
                    ["rclone", "cat", remote_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30,
                    check=False,
                )

            except subprocess.TimeoutExpired:
                logger.warning(
                    "Timeout fetching B2 frame: %s",
                    remote_path,
                )
                continue

            if result.returncode == 0 and result.stdout:
                return Response(
                    content=result.stdout,
                    media_type="image/jpeg",
                    headers={
                        "Cache-Control": "public, max-age=86400",
                    },
                )

        logger.warning(
            "Frame not found: video=%s frame=%s normalized=%s",
            safe_video_id,
            safe_frame_id,
            file_name,
        )

        raise HTTPException(
            status_code=404,
            detail="Frame image not found",
        )

    @video_router.get("/stream/{video_name}")
    def stream_video(
        video_name: str,
        range: Optional[str] = Header(None),
    ):
        """
        Stream video.

        Priority:
        1. Local MP4
        2. Backblaze B2 via rclone with HTTP Range support
        """

        safe_video_name = os.path.basename(video_name)
        video_file = f"{safe_video_name}.mp4"

        # =========================================================
        # 1. TRY LOCAL VIDEO FIRST
        # =========================================================

        local_path = os.path.join(
            STREAM_CONFIG.SOURCE_VIDEO_URL,
            video_file,
        )

        if os.path.isfile(local_path):
            file_size = os.path.getsize(local_path)

            if not range:
                headers = {
                    "Content-Length": str(file_size),
                    "Accept-Ranges": "bytes",
                }

                return StreamingResponse(
                    generate_full_video(file_path=local_path),
                    status_code=200,
                    media_type="video/mp4",
                    headers=headers,
                )

            start_byte, end_byte, length = get_byte_range(
                range=range,
                file_size=file_size,
            )

            headers = {
                "Content-Length": str(length),
                "Accept-Ranges": "bytes",
                "Content-Range": (
                    f"bytes {start_byte}-{end_byte}/{file_size}"
                ),
            }

            return StreamingResponse(
                generate_video_chunk(
                    start_byte=start_byte,
                    chunk_len=length,
                    file_path=local_path,
                ),
                status_code=206,
                media_type="video/mp4",
                headers=headers,
            )

        # =========================================================
        # 2. BACKBLAZE B2 FALLBACK
        # =========================================================

        remote = STREAM_CONFIG.B2_VIDEO_REMOTE.rstrip("/")
        prefix = STREAM_CONFIG.B2_VIDEO_PREFIX.strip("/")

        remote_path = (
            f"{remote}/{prefix}/{video_file}"
        )

        # ---------------------------------------------------------
        # Get remote file size
        # ---------------------------------------------------------

        try:
            stat_result = subprocess.run(
                [
                    "rclone",
                    "lsjson",
                    "--stat",
                    remote_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False,
                text=True,
            )

        except subprocess.TimeoutExpired:
            raise HTTPException(
                status_code=504,
                detail="Timeout while checking video on B2",
            )

        if stat_result.returncode != 0:
            logger.warning(
                "B2 video not found: %s | %s",
                remote_path,
                stat_result.stderr,
            )

            raise HTTPException(
                status_code=404,
                detail="Video not found",
            )

        try:
            stat_data = json.loads(stat_result.stdout)
            file_size = int(stat_data["Size"])
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            logger.exception(
                "Could not parse B2 video size: %s",
                remote_path,
            )

            raise HTTPException(
                status_code=500,
                detail="Could not determine video size",
            )

        # =========================================================
        # Helper: stream rclone stdout without loading video in RAM
        # =========================================================

        def rclone_stream(
            offset: int = 0,
            count: int | None = None,
        ):
            command = [
                "rclone",
                "cat",
                remote_path,
                "--offset",
                str(offset),
            ]

            if count is not None:
                command.extend([
                    "--count",
                    str(count),
                ])

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            try:
                assert process.stdout is not None

                while True:
                    chunk = process.stdout.read(1024 * 1024)

                    if not chunk:
                        break

                    yield chunk

            finally:
                if process.stdout:
                    process.stdout.close()

                if process.poll() is None:
                    process.terminate()

                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

        # =========================================================
        # Browser requested entire file
        # =========================================================

        if not range:
            headers = {
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes",
            }

            return StreamingResponse(
                rclone_stream(
                    offset=0,
                    count=file_size,
                ),
                status_code=200,
                media_type="video/mp4",
                headers=headers,
            )

        # =========================================================
        # Browser requested byte range
        # =========================================================

        try:
            start_byte, end_byte, length = get_byte_range(
                range=range,
                file_size=file_size,
            )

        except Exception as exc:
            logger.warning(
                "Invalid video range %s: %s",
                range,
                exc,
            )

            raise HTTPException(
                status_code=416,
                detail="Invalid Range header",
            )

        headers = {
            "Content-Length": str(length),
            "Accept-Ranges": "bytes",
            "Content-Range": (
                f"bytes {start_byte}-{end_byte}/{file_size}"
            ),
        }

        return StreamingResponse(
            rclone_stream(
                offset=start_byte,
                count=length,
            ),
            status_code=206,
            media_type="video/mp4",
            headers=headers,
        )

except ImportError:
    video_router = None
