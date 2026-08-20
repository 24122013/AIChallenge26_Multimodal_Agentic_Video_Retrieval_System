import { useState, useCallback } from 'react';
import type { VideoScene } from '../types';

export interface NeighborDetail {
  frame_id: string;
  delta_seconds: number;
  url: string;
}

export interface NeighborResponse {
  frame_id: string;
  video_id: string;
  timestamp: number;
  target_url: string;
  neighbors_before: NeighborDetail[];
  neighbors_after: NeighborDetail[];
}

function useVideo(displayedResults: VideoScene[]) {
  const [activeVideo, setActiveVideo] = useState<VideoScene | null>(null);
  const [neighborData, setNeighborData] = useState<NeighborResponse | null>(null);
  const [lastSelectedScene, setLastSelectedScene] = useState<VideoScene | null>(null);

  const handleSelectResultVideo = useCallback(async (scene: VideoScene) => {
      setActiveVideo({ ...scene, video_url: `/api/video/stream/${scene.video_id}` });
      setLastSelectedScene(scene);
      setNeighborData(null);
      
      try {
          const res = await fetch(`/api/video/frame_neighbor/${scene.frame_id}`);
          if (res.ok) {
              const data = (await res.json()) as NeighborResponse;
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