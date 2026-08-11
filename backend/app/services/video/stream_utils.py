import os
import re
from typing import Optional
from fastapi import HTTPException
from backend.configs.stream_config import StreamConfig

def get_video_path(config: StreamConfig, video_file: str) -> Optional[str]:
    """Returns video path (First version: streaming videos from local)"""
    safe_name = os.path.basename(video_file)
    file_path = os.path.join(config.SOURCE_VIDEO_URL, safe_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Video resource not found")
    else:
        return file_path

def byte_range_sanity_check(start_byte: int, end_byte: int, file_size: int):
    if start_byte >= file_size or end_byte >= file_size or start_byte > end_byte:
        raise HTTPException(status_code=416, detail="Requested Range Not Satisfiable")

def get_byte_range(range: str, file_size: int) -> tuple[int]:
    byte_match = re.search(r'bytes=(\d+)-(\d*)', range)
    if not byte_match:
        raise HTTPException(status_code=416, detail="Invalid Byte Range Format")

    start_byte = int(byte_match.group(1))
    end_byte_str = byte_match.group(2)
    end_byte = int(end_byte_str) if end_byte_str else file_size - 1
    
    # Sanity checks for ranges
    byte_range_sanity_check(start_byte=start_byte, end_byte=end_byte, file_size=file_size)

    length = end_byte - start_byte + 1
    
    return start_byte, end_byte, length

def generate_full_video(file_path: str):
    """Generator to yield the whole video"""
    with open(file_path, 'rb') as f:
        while chunk := f.read(1024 * 1024):
            yield chunk

def generate_video_chunk(start_byte: int, chunk_len: int, file_path: str):
    """Generator to yield the specific video byte range"""

    with open(file_path, 'rb') as f:
        f.seek(start_byte)
        remaining = chunk_len
        while remaining > 0:
            read_size = min(remaining, 1024 * 1024)
            data = f.read(read_size)
            if not data:
                break
            remaining -= len(data)
            yield data
