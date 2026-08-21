import { useState, useCallback } from 'react';
import type { VideoScene, NeighborFrame } from '../types';
import { API_PROXY } from '../constants/proxy';

export interface NeighborResponse {
  frame_id: string;
  video_id: string;
  timestamp: number;
  target_url: string;
  neighbors_before: NeighborFrame[];
  neighbors_after: NeighborFrame[];
}

function useVideo(displayedResults: VideoScene[]) {
  const [activeVideo, setActiveVideo] = useState<VideoScene | null>(null);
  const [neighborData, setNeighborData] = useState<NeighborResponse | null>(null);
  const [lastSelectedScene, setLastSelectedScene] = useState<VideoScene | null>(null);

  const handleSelectResultVideo = useCallback(async (scene: VideoScene) => {
      setActiveVideo({ ...scene, video_url: `${API_PROXY}/video/stream/${scene.video_id}` });
      setLastSelectedScene(scene);
      setNeighborData(null);
      
      try {
        const res = await fetch(
          `${API_PROXY}/video/frame_neighbor/${encodeURIComponent(scene.video_id)}/${encodeURIComponent(scene.frame_id)}`,
        );
        if (res.ok) {
            const data = (await res.json()) as NeighborResponse;
            console.log("Neighbor API Response:", data);
            setNeighborData(data);
        }
      } catch (error) {
        console.error("Failed to fetch neighbor frames:", error);
      }
    }, []);

    const handleNextVideo = useCallback(() => {
      if (!activeVideo || displayedResults.length === 0) return;
      const idx = displayedResults.findIndex(r => r.frame_id === activeVideo.frame_id);
      if (idx !== -1 && idx < displayedResults.length - 1) {
        handleSelectResultVideo(displayedResults[idx + 1]);
      }
    }, [activeVideo, displayedResults, handleSelectResultVideo]);
  
    const handlePrevVideo = useCallback(() => {
      if (!activeVideo || displayedResults.length === 0) return;
      const idx = displayedResults.findIndex(r => r.frame_id === activeVideo.frame_id);
      if (idx > 0) {
        handleSelectResultVideo(displayedResults[idx - 1]);
      }
    }, [activeVideo, displayedResults, handleSelectResultVideo]);

  return { 
    activeVideo, 
    setActiveVideo, 
    neighborData, 
    setNeighborData, 
    lastSelectedScene, 
    handleSelectResultVideo, 
    handleNextVideo, 
    handlePrevVideo 
  };
}

export default useVideo;
