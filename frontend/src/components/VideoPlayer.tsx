import React, { useRef, useEffect, useState } from 'react';
import type { VideoScene } from '../types';
import type { NeighborResponse } from '../hooks/useVideo';
import { Play, X, ChevronLeft, ChevronRight, Info } from 'lucide-react';
import { API_PROXY } from '../constants/proxy';

const TARGET_FPS = 30;
const FRAMES_TO_STEP = 5;
const STEP_TIME = FRAMES_TO_STEP / TARGET_FPS;

interface VideoPlayerProps {
    activeVideo: VideoScene | null;
    neighborData: NeighborResponse | null;
    onClose: () => void;
    onFinalSubmit: (sceneId: string) => void;
    onNext: () => void;
    onPrev: () => void;
    displayedResults: VideoScene[]; 
    onSelectResult?: (scene: VideoScene) => void;
}

const VideoPlayer: React.FC<VideoPlayerProps> = ({ 
    activeVideo,
    neighborData,
    onClose, 
    onFinalSubmit, 
    onNext, 
    onPrev,
    displayedResults,
    onSelectResult 
}) => {
    const videoPlayerRef = useRef<HTMLVideoElement>(null);
    const modalRef = useRef<HTMLDivElement>(null);

    const [submitStage, setSubmitStage] = useState<'idle' | 'fact_check'>('idle');
    const [prevSceneId, setPrevSceneId] = useState<string | undefined>(activeVideo?.frame_id);
    const [isVideoMode, setIsVideoMode] = useState<boolean>(false);
    const [playTimestamp, setPlayTimestamp] = useState<number>(0);

    if (activeVideo?.frame_id !== prevSceneId) {
        setPrevSceneId(activeVideo?.frame_id);
        setSubmitStage('idle');
        setIsVideoMode(false);
        setPlayTimestamp(activeVideo?.timestamp || 0);
    }

    useEffect(() => {
        if (isVideoMode && videoPlayerRef.current && activeVideo?.video_url) {
            videoPlayerRef.current.src = activeVideo.video_url;
            videoPlayerRef.current.currentTime = playTimestamp;
            videoPlayerRef.current.play().catch(() => console.warn("Auto-play blocked"));
        }
    }, [isVideoMode, activeVideo, playTimestamp]);

    useEffect(() => {
        if (!activeVideo) return;

        // Hotkeys
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'Escape') {
                e.preventDefault();
                if (isVideoMode) {
                    setIsVideoMode(false);
                } else {
                    onClose();
                }
            } else if (e.key === 'Enter') {
                e.preventDefault();
                if (submitStage === 'idle') {
                    setSubmitStage('fact_check');
                } else {
                    onFinalSubmit(activeVideo.frame_id);
                    setSubmitStage('idle');
                }
            } else if (e.key === 'Backspace') {
                if (submitStage === 'fact_check') {
                    e.preventDefault();
                    setSubmitStage('idle');
                }
            } else if (e.key === '[') {
                e.preventDefault();
                if (videoPlayerRef.current && isVideoMode) {
                    videoPlayerRef.current.currentTime = Math.max(0, videoPlayerRef.current.currentTime - STEP_TIME);
                }
            } else if (e.key === ']') {
                e.preventDefault();
                if (videoPlayerRef.current && isVideoMode) {
                    videoPlayerRef.current.currentTime = Math.min(videoPlayerRef.current.duration, videoPlayerRef.current.currentTime + STEP_TIME);
                }
            } else if (e.key === 'ArrowLeft') {
                e.preventDefault();
                onPrev();
                const currentIndex = displayedResults.findIndex(v => v.frame_id === activeVideo.frame_id);
                if (currentIndex > 0 && onSelectResult) {
                    onSelectResult(displayedResults[currentIndex - 1]);
                }
            } else if (e.key === 'ArrowRight') {
                e.preventDefault();
                onNext();
                const currentIndex = displayedResults.findIndex(v => v.frame_id === activeVideo.frame_id);
                if (currentIndex < displayedResults.length - 1 && onSelectResult) {
                    onSelectResult(displayedResults[currentIndex + 1]);
                }
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [activeVideo, submitStage, isVideoMode, onFinalSubmit, onClose, onNext, onPrev, displayedResults, onSelectResult]);

    const handleOverlayClick = (e: React.MouseEvent<HTMLDivElement>) => {
        if (modalRef.current && !modalRef.current.contains(e.target as Node)) onClose();
    };

    const handlePlayRequest = (timestamp: number) => {
        setPlayTimestamp(timestamp);
        setIsVideoMode(true);
    };

    if (!activeVideo) return null;

    const allNeighbors = neighborData 
        ? [
            ...neighborData.neighbors_before.map(frame => ({
                ...frame,
                url: `${API_PROXY}/video/frame/${neighborData.video_id}/${frame.frame_id}`,
                delta_seconds: frame.delta_seconds // Already calculated or passed from backend
            })), 
            { 
                frame_id: neighborData.frame_id, 
                delta_seconds: 0, 
                url: `${API_PROXY}/video/frame/${neighborData.video_id}/${neighborData.frame_id}`, 
                isTarget: true, 
                timestamp: neighborData.timestamp 
            }, 
            ...neighborData.neighbors_after.map(frame => ({
                ...frame,
                url: `${API_PROXY}/video/frame/${neighborData.video_id}/${frame.frame_id}`,
                delta_seconds: frame.delta_seconds
            }))
          ] 
        : [];

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-fade-in" onClick={handleOverlayClick}>
            <div ref={modalRef} className={`bg-white dark:bg-zinc-950 rounded-2xl border shadow-2xl w-full max-w-6xl max-h-[95vh] relative flex flex-col transition-all duration-200 overflow-hidden ${submitStage === 'fact_check' ? 'border-amber-500 ring-2 ring-amber-500/50' : 'border-zinc-200 dark:border-zinc-800'}`}>
                
                {/* Header */}
                <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900/50 shrink-0">
                    <div className="flex items-center gap-4">
                        <h2 className="text-lg font-bold tracking-tight text-zinc-900 dark:text-zinc-100 flex items-center gap-2">
                            <Info className="w-5 h-5 text-blue-500" />
                            Scene Review
                        </h2>
                        <div className="h-4 w-px bg-zinc-300 dark:bg-zinc-700" />
                        <span className="font-mono text-sm bg-zinc-200 dark:bg-zinc-800 px-2 py-0.5 rounded text-zinc-700 dark:text-zinc-300">
                            {activeVideo.video_id}
                        </span>
                        <span className="text-sm font-medium text-zinc-500">
                            @ {activeVideo.timestamp.toFixed(2)}s
                        </span>
                    </div>

                    <div className="flex gap-4 items-center">
                        {submitStage === 'fact_check' ? (
                            <div className="bg-amber-100 dark:bg-amber-900/30 text-amber-800 dark:text-amber-400 px-3 py-1 rounded-lg border border-amber-300 dark:border-amber-700/50 text-xs font-medium animate-pulse">
                                Confirm: <kbd className="bg-white dark:bg-amber-950 px-1.5 py-0.5 border dark:border-amber-800 rounded">Enter</kbd> | Cancel: <kbd className="bg-white dark:bg-amber-950 px-1.5 py-0.5 border dark:border-amber-800 rounded">Backspace</kbd>
                            </div>
                        ) : (
                            <div className="flex items-center gap-1 text-zinc-500 dark:text-zinc-400 text-xs font-medium bg-white dark:bg-zinc-900 px-2 py-1 rounded border border-zinc-200 dark:border-zinc-800 shadow-sm">
                                <kbd className="bg-zinc-100 dark:bg-zinc-800 px-1.5 py-0.5 rounded border border-zinc-200 dark:border-zinc-700">←</kbd>
                                <kbd className="bg-zinc-100 dark:bg-zinc-800 px-1.5 py-0.5 rounded border border-zinc-200 dark:border-zinc-700">→</kbd>
                                <span className="ml-1">Navigate</span>
                            </div>
                        )}
                        <div className="h-4 w-px bg-zinc-300 dark:bg-zinc-700" />
                        <button onClick={onClose} className="p-1.5 text-zinc-500 hover:text-zinc-900 dark:hover:text-white bg-zinc-100 hover:bg-zinc-200 dark:bg-zinc-800 dark:hover:bg-zinc-700 rounded-full transition-colors">
                            <X className="w-5 h-5" />
                        </button>
                    </div>
                </div>

                {/* Main Body */}
                <div className="flex flex-col lg:flex-row flex-1 min-h-0 bg-white dark:bg-zinc-950">
                    
                    {/* Left: Visuals */}
                    <div className="flex-1 flex flex-col min-w-0 border-r border-zinc-200 dark:border-zinc-800 bg-black/5 dark:bg-black/40">
                        <div className="relative flex-1 flex items-center justify-center p-4 min-h-[300px]">
                            {isVideoMode ? (
                                <video 
                                    ref={videoPlayerRef} 
                                    controls 
                                    autoPlay 
                                    className="w-full h-full object-contain bg-black rounded-lg shadow-inner" 
                                />
                            ) : (
                                <div className="relative w-full h-full group flex items-center justify-center">
                                    <img 
                                        src={`${API_PROXY}/video/frame/${activeVideo.video_id}/${activeVideo.frame_id}`} 
                                        alt={activeVideo.frame_id} 
                                        className="w-full h-full object-contain rounded-lg shadow-md bg-black" 
                                        loading="lazy" 
                                    />
                                    <div className="absolute inset-0 flex items-center justify-center bg-black/0 group-hover:bg-black/20 transition-all rounded-lg">
                                        <button 
                                            onClick={() => handlePlayRequest(activeVideo.timestamp)}
                                            className="bg-blue-600/90 hover:bg-blue-500 text-white p-5 rounded-full shadow-2xl transform transition-transform hover:scale-110 backdrop-blur-sm"
                                        >
                                            <Play className="w-10 h-10 ml-1" fill="currentColor" />
                                        </button>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Neighbor Frames Strip */}
                        {allNeighbors.length > 0 && (
                            <div className="h-32 bg-zinc-100 dark:bg-zinc-900 border-t border-zinc-200 dark:border-zinc-800 p-2 shrink-0 flex gap-2 overflow-x-auto overflow-y-hidden custom-scrollbar">
                                {allNeighbors.map((frame, idx) => {
                                    const isTarget = 'isTarget' in frame;
                                    return (
                                        <button 
                                            key={idx} 
                                            onClick={() => handlePlayRequest(frame.timestamp || (activeVideo.timestamp + frame.delta_seconds))}
                                            className={`relative h-full aspect-video shrink-0 rounded-md overflow-hidden border-2 transition-all group ${isTarget ? 'border-blue-500 shadow-md ring-2 ring-blue-500/30' : 'border-transparent hover:border-zinc-400 dark:hover:border-zinc-600'}`}
                                            title="Click to play from this timestamp"
                                        >
                                            <img src={`${API_PROXY}/video/frame/${neighborData?.video_id}/${frame.frame_id}`} alt={frame.frame_id} className="w-full h-full object-cover" loading="lazy" />
                                            
                                            <div className="absolute inset-0 bg-black/0 group-hover:bg-black/30 flex items-center justify-center transition-colors">
                                                <Play className="w-6 h-6 text-white opacity-0 group-hover:opacity-100 drop-shadow-md" fill="currentColor" />
                                            </div>

                                            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/90 to-transparent p-1.5">
                                                <div className="flex justify-between items-center text-[10px] font-mono text-white">
                                                    <span>{isTarget ? 'Target' : (frame.delta_seconds > 0 ? `+${frame.delta_seconds}s` : `${frame.delta_seconds}s`)}</span>
                                                </div>
                                            </div>
                                        </button>
                                    );
                                })}
                            </div>
                        )}
                    </div>

                    {/* Right: Detailed Report */}
                    <div className="w-full lg:w-80 xl:w-96 shrink-0 flex flex-col h-[40vh] lg:h-auto overflow-y-auto custom-scrollbar">
                        <div className="p-5 flex flex-col gap-6">
                            
                            <div>
                                <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-400 dark:text-zinc-500 mb-2">Retrieval Scores</h3>
                                <div className="grid grid-cols-2 gap-2">
                                    {Object.entries(activeVideo.modality_scores || {}).map(([method, score]) => (
                                        <div key={method} className="bg-zinc-50 dark:bg-zinc-900/50 border border-zinc-200 dark:border-zinc-800 rounded p-2 flex justify-between items-center">
                                            <span className="text-xs font-medium text-zinc-600 dark:text-zinc-400 capitalize">{method}</span>
                                            <span className="text-sm font-mono font-bold text-zinc-900 dark:text-zinc-100">{(score * 100).toFixed(1)}%</span>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {activeVideo.caption && (
                                <div>
                                    <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-400 dark:text-zinc-500 mb-2">Video Caption</h3>
                                    <div className="text-sm text-zinc-800 dark:text-zinc-200 bg-blue-50 dark:bg-blue-900/20 border border-blue-100 dark:border-blue-800/50 p-3 rounded-lg leading-relaxed">
                                        {activeVideo.caption}
                                    </div>
                                </div>
                            )}

                            {activeVideo.ocr_text && (
                                <div>
                                    <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-400 dark:text-zinc-500 mb-2">OCR Content</h3>
                                    <div className="text-sm font-mono text-zinc-700 dark:text-zinc-300 bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 p-3 rounded-lg max-h-32 overflow-y-auto custom-scrollbar whitespace-pre-wrap">
                                        {activeVideo.ocr_text}
                                    </div>
                                </div>
                            )}

                            {activeVideo.asr_text && (
                                <div>
                                    <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-400 dark:text-zinc-500 mb-2">Audio Transcript (ASR)</h3>
                                    <div className="text-sm text-zinc-700 dark:text-zinc-300 bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 p-3 rounded-lg max-h-32 overflow-y-auto custom-scrollbar">
                                        {activeVideo.asr_text}
                                    </div>
                                </div>
                            )}

                            {activeVideo.objects && activeVideo.objects.length > 0 && (
                                <div>
                                    <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-400 dark:text-zinc-500 mb-2">Detected Objects</h3>
                                    <div className="flex flex-wrap gap-1.5">
                                        {activeVideo.objects.map((obj, i) => (
                                            <span key={i} className="px-2 py-1 bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 text-xs rounded-md border border-zinc-200 dark:border-zinc-700">
                                                {obj}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Add QA / Trake specialized blocks if they exist in ExtendedScene typing natively */}
                            {('answer' in activeVideo) && (
                                <div>
                                    <h3 className="text-xs font-bold uppercase tracking-wider text-purple-500 dark:text-purple-400 mb-2">QA Answer Extract</h3>
                                    <div className="text-sm text-purple-900 dark:text-purple-100 bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800/50 p-3 rounded-lg">
                                        {activeVideo.answer}
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                {/* Footer Controls */}
                <div className="flex items-center justify-between p-4 border-t border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900/50 shrink-0">
                    <div className="flex gap-2">
                        <button onClick={onPrev} className="px-4 py-2 flex items-center gap-1 text-sm font-medium text-zinc-700 dark:text-zinc-300 bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-lg hover:bg-zinc-50 dark:hover:bg-zinc-700 transition-colors">
                            <ChevronLeft className="w-4 h-4" /> Prev Scene
                        </button>
                        <button onClick={onNext} className="px-4 py-2 flex items-center gap-1 text-sm font-medium text-zinc-700 dark:text-zinc-300 bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-lg hover:bg-zinc-50 dark:hover:bg-zinc-700 transition-colors">
                            Next Scene <ChevronRight className="w-4 h-4" />
                        </button>
                    </div>

                    <button 
                        onClick={() => onFinalSubmit(activeVideo.frame_id)} 
                        className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-bold rounded-lg shadow-sm transition-colors"
                    >
                        {submitStage === 'fact_check' ? 'Confirm Submission' : 'Submit Scene as Answer'}
                    </button>
                </div>

            </div>
        </div>
    );
};

export default VideoPlayer;
