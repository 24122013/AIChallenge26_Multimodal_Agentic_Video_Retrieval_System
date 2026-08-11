import React, { useRef, useEffect, useState } from 'react';
import type { VideoScene, ContextFrame } from '../../types';

const TARGET_FPS = 30;
const FRAMES_TO_STEP = 5;
const STEP_TIME = FRAMES_TO_STEP / TARGET_FPS;

interface VideoPlayerProps {
    activeVideo: VideoScene | null;
    neighborFrames: ContextFrame[];
    onClose: () => void;
    onFinalSubmit: (sceneId: string) => void;
    onNext: () => void;
    onPrev: () => void;
    // We pass the displayedResults so the player can manually trigger the click on the next/prev item
    displayedResults: VideoScene[]; 
    onSelectResult?: (scene: VideoScene) => void;
}

const VideoPlayer: React.FC<VideoPlayerProps> = ({ 
    activeVideo, 
    neighborFrames, 
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

    if (activeVideo?.frame_id !== prevSceneId) {
        setPrevSceneId(activeVideo?.frame_id);
        setSubmitStage('idle');
    }

    useEffect(() => {
        if (videoPlayerRef.current && activeVideo?.video_url) {
            videoPlayerRef.current.src = activeVideo.video_url;
            videoPlayerRef.current.currentTime = activeVideo.timestamp;
            videoPlayerRef.current.play().catch(() => console.warn("Auto-play blocked"));
        }
    }, [activeVideo]);

    useEffect(() => {
        if (!activeVideo) return;

        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'Escape') {
                e.preventDefault();
                onClose();
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
                if (videoPlayerRef.current) {
                    videoPlayerRef.current.currentTime = Math.max(0, videoPlayerRef.current.currentTime - STEP_TIME);
                }
            } else if (e.key === ']') {
                e.preventDefault();
                if (videoPlayerRef.current) {
                    videoPlayerRef.current.currentTime = Math.min(videoPlayerRef.current.duration, videoPlayerRef.current.currentTime + STEP_TIME);
                }
            } else if (e.key === 'ArrowLeft') {
                e.preventDefault();
                onPrev();
                
                // Manually mark the PREV video as clicked without a useEffect loop
                const currentIndex = displayedResults.findIndex(v => v.frame_id === activeVideo.frame_id);
                if (currentIndex > 0 && onSelectResult) {
                    onSelectResult(displayedResults[currentIndex - 1]);
                }
            } else if (e.key === 'ArrowRight') {
                e.preventDefault();
                onNext();
                
                // Manually mark the NEXT video as clicked without a useEffect loop
                const currentIndex = displayedResults.findIndex(v => v.frame_id === activeVideo.frame_id);
                if (currentIndex < displayedResults.length - 1 && onSelectResult) {
                    onSelectResult(displayedResults[currentIndex + 1]);
                }
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [activeVideo, submitStage, onFinalSubmit, onClose, onNext, onPrev, displayedResults, onSelectResult]);

    const handleOverlayClick = (e: React.MouseEvent<HTMLDivElement>) => {
        if (modalRef.current && !modalRef.current.contains(e.target as Node)) onClose();
    };

    if (!activeVideo) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4 animate-fade-in" onClick={handleOverlayClick}>
            <div ref={modalRef} className={`bg-white dark:bg-zinc-900 rounded-2xl border p-5 shadow-2xl w-full max-w-4xl relative flex flex-col transition-all duration-200 ${submitStage === 'fact_check' ? 'border-amber-500 ring-2 ring-amber-500/20' : 'border-zinc-200 dark:border-zinc-800'}`}>
                <button onClick={onClose} className="absolute top-4 right-4 z-10 p-2 text-zinc-400 hover:text-zinc-800 dark:hover:text-white bg-zinc-100 hover:bg-zinc-200 dark:bg-zinc-800 dark:hover:bg-zinc-700 rounded-full transition">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                </button>

                <div className="flex items-center justify-between mb-4 pr-12">
                    <h2 className="text-lg font-semibold tracking-tight truncate">Scene Review: <span className="text-zinc-500 font-mono text-sm">{activeVideo.video_id}</span></h2>
                    <div className="flex gap-2 items-center text-xs font-semibold">
                        {submitStage === 'fact_check' ? (
                            <div className="bg-amber-50 text-amber-800 px-3 py-1 rounded-lg border border-amber-200 animate-pulse">
                                <span>⚠️ Fact-check Mode: Press <kbd className="bg-white px-1 border rounded shadow-sm text-black">Enter</kbd> to confirm/unsubmit | <kbd className="bg-white px-1 border rounded shadow-sm text-black">Backspace</kbd> to cancel</span>
                            </div>
                        ) : (
                            <div className="text-zinc-500 px-3 py-1 bg-zinc-100 dark:bg-zinc-800/50 rounded-lg border border-zinc-200 dark:border-zinc-800">
                                Scenes: <kbd className="bg-white dark:bg-zinc-700 px-1 border rounded">←</kbd> <kbd className="bg-white dark:bg-zinc-700 px-1 border rounded">→</kbd> | Fwd/Rwd: <kbd className="bg-white dark:bg-zinc-700 px-1 border rounded">[</kbd> <kbd className="bg-white dark:bg-zinc-700 px-1 border rounded">]</kbd>
                            </div>
                        )}
                    </div>
                </div>

                <div className="aspect-video w-full bg-black rounded-lg overflow-hidden relative border border-zinc-800 shadow-inner">
                    <video ref={videoPlayerRef} controls autoPlay className="w-full h-full object-contain" />
                </div>

                {neighborFrames.length > 0 && (
                    <div className="mt-5 border-t border-zinc-200 dark:border-zinc-800 pt-4">
                        <span className="text-xs font-bold uppercase tracking-wider text-zinc-400">Context Segments</span>
                        <div className="flex gap-3 mt-3">
                            {neighborFrames.map((frame, index) => (
                                <button
                                    key={index}
                                    onClick={() => { if (videoPlayerRef.current) videoPlayerRef.current.currentTime = frame.time; }}
                                    className={`flex-1 p-2 text-sm border rounded-lg transition-all text-center ${frame.isActive ? 'bg-blue-50 border-blue-400 text-blue-600 font-medium' : 'bg-zinc-50 hover:bg-zinc-100 border-zinc-200'}`}
                                >
                                    <div>{frame.label}</div>
                                    <div className="opacity-70 text-xs mt-0.5">{frame.time.toFixed(1)}s</div>
                                </button>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default VideoPlayer;