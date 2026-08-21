"""Multimodal retrieval service orchestrating concurrent vector/DB searches."""
from __future__ import annotations

import asyncio
import time
from typing import List, Dict

from backend.app.services.retrieval.retrieval_manager import search_visual
from backend.app.models.search import SearchPayload
from backend.app.models.retrieval import RetrievalResult, VisualSearchResponse


def deduplicate_and_merge(nested_results: List[List[dict] | Exception], top_k: int) -> List[RetrievalResult]:
    """
    Flattens the arrays, merges modality_scores by segment_id, and calculates the final score.
    Safely ignores failed tasks and missing keys.
    """
    merged_map: Dict[str, dict] = {}
    
    for result_group in nested_results:
        # Skip if the async task returned an exception instead of a list
        if isinstance(result_group, Exception):
            continue
        if not hasattr(result_group, "results"):
            continue
        
        results = result_group.results
        for item in results:
            data = item.to_dict()
            seg_id = data.get("segment_id", None)

            # segment_id must not be None - as it's the unique key
            if seg_id is None or seg_id == "":
                raise Exception(f"segment_id must be in data! Got {seg_id}")
            
            # Use seg_id as the unique key for merging frames
            if seg_id not in merged_map:
                merged_map[seg_id] = data
            else:
                # Scene exists! Merge the new method scores safely
                merged_map[seg_id]["modality_scores"].update(data.get("modality_scores", {}))
                
    # Calculate Final Scores & Primary Source
    final_results = []
    for data in merged_map.values():
        # TODO: Add processing of data
        final_results.append(RetrievalResult(**data))
        
    # Sort by final score descending and truncate
    final_results.sort(key=lambda x: x.score, reverse=True)
    return final_results[:top_k]


async def execute_multimodal_search(payload: SearchPayload) -> VisualSearchResponse:
    """
    Dispatches concurrent search tasks across requested modalities and aggregates results.
    """
    tasks = []
    # Default to 20 if config or top_k is missing
    top_k = getattr(payload.config, "topK", 20) if payload.config else 20
    
    # Dispatch concurrent search tasks based on payload presence
    if getattr(payload, "clipQuery", None):
        tasks.append(search_visual(payload.clipQuery, top_k))
        
    if getattr(payload, "ocrQuery", None):
        # TODO: Implement
        pass
        
    if getattr(payload, "image", None) and (payload.image.image_url or payload.image.image_b64):
        # TODO: Implement
        pass
        
    if getattr(payload, "colorHex", None):
        # TODO: Implement
        pass

    if not tasks:
        return VisualSearchResponse(
            search_payload=payload,
            top_k=top_k,
            latency_ms=0,
            results=[]
        )
    
    t0 = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    final_results = deduplicate_and_merge(results, top_k)
    latency = round((time.time() - t0) * 1000)
    
    return VisualSearchResponse(
        search_payload=payload,
        top_k=top_k,
        latency_ms=latency,
        results=final_results
    )