import { useState, useCallback } from 'react';
import type { VideoScene } from '../types';

function useVideo(displayedResults: VideoScene[]) {
  const [activeVideo, setActiveVideo] = useState<VideoScene | null>(null);
  const [neighborFrames, setNeighborFrames] = useState<{ time: number; label: string; isActive?: boolean }[]>([]);
  const [lastSelectedScene, setLastSelectedScene] = useState<VideoScene | null>(null);

  const handleSelectResultVideo = useCallback((scene: VideoScene) => {
      setActiveVideo({ ...scene, video_url: `/api/video/stream/${scene.video_id}` });
      setLastSelectedScene(scene);
      setNeighborFrames([
        { time: Math.max(0, scene.timestamp - 2), label: '-2s Context' },
        { time: scene.timestamp, label: 'Target Frame', isActive: true },
        { time: scene.timestamp + 2, label: '+2s Context' },
      ]);
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

  return { activeVideo, setActiveVideo, neighborFrames, setNeighborFrames, lastSelectedScene, handleSelectResultVideo, handleNextVideo, handlePrevVideo };
}

export default useVideo;