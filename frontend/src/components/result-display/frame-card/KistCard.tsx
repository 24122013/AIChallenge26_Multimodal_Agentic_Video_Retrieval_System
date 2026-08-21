import React from 'react';
import type { VideoScene } from '../../../types';
import { METHOD_COLORS } from '../../../constants/mode-icons';
import { API_PROXY } from '../../../constants/proxy';

interface KistCardProps {
    scene: VideoScene;
    index: number;
    cardSize: 'sm' | 'md' | 'lg';
    isClicked: boolean;
    isSubmitted: boolean;
    isDraggingThis: boolean;
    sequenceTitle?: string;
    outputFormat: string;
    onSelectResult: (scene: VideoScene) => void;
    onFinalSubmit: (sceneId: string) => void;
    onDragStart: (e: React.DragEvent) => void;
    onDrop: (e: React.DragEvent) => void;
    onDragEnd: () => void;
    handleCopyOutput: (text: string) => void;
}

export default function KistCard({
    scene, cardSize, isClicked, isSubmitted, isDraggingThis, sequenceTitle, outputFormat,
    onSelectResult, onFinalSubmit, onDragStart, onDrop, onDragEnd, handleCopyOutput
}: KistCardProps) {
    const scoreList = Object.entries(scene.modality_scores || {}).filter(([m]) => m !== 'rerank');
    const rerankScore = scene.modality_scores?.['rerank'];

    let cardStyle = "bg-white dark:bg-zinc-900 border-zinc-200 dark:border-zinc-800";
    if (isSubmitted) cardStyle = "bg-emerald-50/60 dark:bg-emerald-950/20 border-emerald-400 shadow-inner";
    else if (isClicked) cardStyle = "bg-yellow-100/80 dark:bg-yellow-800/40 border-zinc-300 dark:border-zinc-700/80";

    return (
        <div 
            draggable
            onDragStart={onDragStart}
            onDragOver={(e) => e.preventDefault()}
            onDrop={onDrop}
            onDragEnd={onDragEnd}
            className={`border rounded-xl p-3 flex flex-col justify-between transition-all duration-200 cursor-grab active:cursor-grabbing ${cardStyle} ${isDraggingThis ? 'opacity-40 scale-[0.98]' : ''}`}
        >
            <div>
                {sequenceTitle && (
                    <div className="mb-2 -mt-1 flex items-center justify-between">
                        <span className="text-[10px] font-bold uppercase text-emerald-600 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-900/30 px-1.5 py-0.5 rounded border border-emerald-200 dark:border-emerald-800">{sequenceTitle}</span>
                    </div>
                )}
                <div className="relative aspect-video w-full bg-black/5 dark:bg-white/5 rounded-lg overflow-hidden mb-3 border border-zinc-200 dark:border-zinc-800">
                    <img src={`${API_PROXY}/video/frame/${scene.video_id}/${scene.frame_id}`} alt={scene.frame_id} className="object-cover w-full h-full" loading="lazy" />
                </div>
                
                <div className="flex justify-between items-start mb-2 gap-2">
                    <span className={`font-bold font-mono px-2 py-0.5 rounded bg-zinc-200/60 dark:bg-zinc-800/80 truncate ${cardSize === 'sm' ? 'text-[10px] max-w-[100px]' : 'text-xs max-w-[150px]'}`}>{scene.video_id}</span>
                    <p className={`${cardSize === 'sm' ? 'text-[10px]' : 'text-xs'} text-zinc-500 font-mono mt-0.5`}>@ {scene.timestamp.toFixed(1)}s</p>
                </div>
                
                {cardSize !== 'sm' && (
                    <div className="flex items-center gap-1 flex-wrap mb-2">
                        {scoreList.map(([method, score]) => (
                            <div key={method} className={`flex items-center gap-1 text-[9px] px-1.5 py-0.5 rounded border border-black/5 dark:border-white/5 ${METHOD_COLORS[method] || 'bg-zinc-200 text-zinc-800'}`}>
                                <span className="font-semibold uppercase tracking-tight">{method}</span>
                                <span className="font-mono font-bold opacity-80">{(score * 100).toFixed(0)}%</span>
                            </div>
                        ))}
                    </div>
                )}

                {cardSize === 'lg' && rerankScore !== undefined && (
                    <div className="mb-2">
                        <span className="text-[10px] font-mono bg-indigo-50 dark:bg-indigo-950/40 text-indigo-600 px-1.5 py-0.5 rounded border border-indigo-100">
                            Rerank Mod: <strong className="font-bold">+{Math.round(rerankScore * 100)}%</strong>
                        </span>
                    </div>
                )}

                {cardSize !== 'sm' && (
                    <div 
                        className="bg-white/80 dark:bg-zinc-950/40 p-2 rounded text-xs border border-zinc-100 dark:border-zinc-800 mb-2 cursor-pointer hover:bg-zinc-50 dark:hover:bg-zinc-900 transition-colors"
                        onClick={() => handleCopyOutput(outputFormat)}
                        title="Click to copy output format"
                    >
                        <span className="text-zinc-800 dark:text-zinc-200 font-mono truncate block text-[10px]">{outputFormat}</span>
                    </div>
                )}
            </div>

            <div className={`flex ${cardSize === 'sm' ? 'flex-col gap-1.5' : 'gap-2 mt-auto pt-2'}`}>
                <button onClick={() => onSelectResult(scene)} className={`flex-1 bg-zinc-900 dark:bg-zinc-200 hover:bg-zinc-800 text-white dark:text-zinc-900 font-medium transition-colors rounded ${cardSize === 'sm' ? 'text-[10px] py-1.5' : 'text-xs py-2'}`}>
                    Review
                </button>
                <button onClick={() => onFinalSubmit(scene.frame_id)} className={`font-semibold transition-colors border rounded ${cardSize === 'sm' ? 'text-[10px] py-1.5 px-2' : 'text-xs py-2 px-3'} ${isSubmitted ? 'bg-red-50 text-red-600 border-red-200 hover:bg-red-100 dark:bg-red-950/30 dark:text-red-400' : 'border-zinc-200 hover:bg-zinc-50 dark:border-zinc-700 dark:hover:bg-zinc-800'}`}>
                    {isSubmitted ? 'Unsubmit' : 'Submit'}
                </button>
            </div>
        </div>
    );
}