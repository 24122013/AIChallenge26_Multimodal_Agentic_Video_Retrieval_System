import React, { useState, useMemo } from 'react';
import type { VideoScene } from '../../types';
import { METHOD_COLORS } from '../../constants/mode-icons';
import { API_PROXY } from '../../constants/proxy';

type CardSize = 'sm' | 'md' | 'lg';
type GroupingMode = 'none' | 'video' | 'modality' | 'tens';

// Hook localized to KIST display
function useDraggableResults(initialResults: VideoScene[]) {
    const [dragState, setDragState] = useState({
        idsStr: '',
        orderedIds: [] as string[]
    });

    const currentIdsStr = initialResults.map(r => r.frame_id).join(',');

    if (currentIdsStr !== dragState.idsStr) {
        setDragState({
            idsStr: currentIdsStr,
            orderedIds: initialResults.map(r => r.frame_id)
        });
    }

    const [draggedId, setDraggedId] = useState<string | null>(null);

    const handleDrop = (e: React.DragEvent, targetId: string) => {
        e.preventDefault();
        const sourceId = e.dataTransfer.getData('text/plain');
        if (!sourceId || sourceId === targetId) return;

        setDragState(prev => {
            const newOrder = [...prev.orderedIds];
            const srcIdx = newOrder.indexOf(sourceId);
            const tgtIdx = newOrder.indexOf(targetId);
            if (srcIdx > -1 && tgtIdx > -1) {
                newOrder.splice(srcIdx, 1);
                newOrder.splice(tgtIdx, 0, sourceId);
            }
            return { ...prev, orderedIds: newOrder };
        });
        setDraggedId(null);
    };

    const sortedResults = useMemo(() => {
        return [...initialResults].sort((a, b) => {
            const idxA = dragState.orderedIds.indexOf(a.frame_id);
            const idxB = dragState.orderedIds.indexOf(b.frame_id);
            return (idxA > -1 ? idxA : 999) - (idxB > -1 ? idxB : 999);
        });
    }, [initialResults, dragState.orderedIds]);

    return { sortedResults, draggedId, setDraggedId, handleDrop };
}

interface KistDisplayProps {
    results: VideoScene[];
    cardSize: CardSize;
    groupBy: GroupingMode;
    onSelectResult: (scene: VideoScene) => void;
    onFinalSubmit: (sceneId: string) => void;
    clickedSceneIds: Set<string>;
    submittedSceneIds: Set<string>;
}

const getOutputString = (scene: VideoScene): string => {
    return `${scene.video_id}, ${scene.frame_index ?? scene.frame_id}`;
};

export default function KistDisplay({
    results,
    cardSize,
    groupBy,
    onSelectResult,
    onFinalSubmit,
    clickedSceneIds,
    submittedSceneIds
}: KistDisplayProps) {
    const { sortedResults: draggableResults, draggedId, setDraggedId, handleDrop } = useDraggableResults(results);

    const groupedResults = useMemo(() => {
        if (groupBy === 'none') return { 'All Results': draggableResults };
        
        const groups: Record<string, VideoScene[]> = {};
        draggableResults.forEach((scene, index) => {
            let key = 'Other';
            if (groupBy === 'video') {
                key = scene.video_id;
            } else if (groupBy === 'tens') {
                const start = Math.floor(index / 10) * 10 + 1;
                const end = Math.min((Math.floor(index / 10) + 1) * 10, draggableResults.length);
                key = `Batch ${Math.floor(index / 10) + 1} (${start} - ${end})`;
            } else if (groupBy === 'modality') {
                if (scene.modality_scores && Object.keys(scene.modality_scores).length > 0) {
                    key = Object.keys(scene.modality_scores).reduce((a, b) => scene.modality_scores[a] > scene.modality_scores[b] ? a : b);
                } else {
                    key = 'Unknown';
                }
            }

            if (!groups[key]) groups[key] = [];
            groups[key].push(scene);
        });
        return groups;
    }, [draggableResults, groupBy]);

    const handleCopyOutput = (text: string) => {
        navigator.clipboard.writeText(text).catch(err => console.error("Failed to copy:", err));
    };

    const getGridClass = () => {
        switch (cardSize) {
            case 'sm': return 'grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 2xl:grid-cols-8 gap-3';
            case 'lg': return 'grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6';
            case 'md':
            default: return 'grid-cols-1 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-4';
        }
    };

    const renderCard = (scene: VideoScene, index: number, sequenceTitle?: string) => {
        const scoreList = Object.entries(scene.modality_scores || {}).filter(([m]) => m !== 'rerank');
        const rerankScore = scene.modality_scores?.['rerank'];

        const isClicked = clickedSceneIds.has(scene.frame_id);
        const isSubmitted = submittedSceneIds.has(scene.frame_id);

        let cardStyle = "bg-white dark:bg-zinc-900 border-zinc-200 dark:border-zinc-800";
        if (isSubmitted) cardStyle = "bg-emerald-50/60 dark:bg-emerald-950/20 border-emerald-400 shadow-inner";
        else if (isClicked) cardStyle = "bg-yellow-100/80 dark:bg-yellow-800/40 border-zinc-300 dark:border-zinc-700/80";

        const isDraggingThis = draggedId === scene.frame_id;
        const outputFormat = getOutputString(scene);

        return (
            <div 
                key={`${scene.frame_id}-${index}`} 
                draggable
                onDragStart={(e) => {
                    e.dataTransfer.setData('text/plain', scene.frame_id);
                    setDraggedId(scene.frame_id);
                }}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => handleDrop(e, scene.frame_id)}
                onDragEnd={() => setDraggedId(null)}
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

                    {cardSize === 'lg' && scene.ocr_text && (
                        <div className="bg-white/80 dark:bg-zinc-950/40 p-2.5 rounded text-xs border border-zinc-100 dark:border-zinc-800 mb-3 h-14 overflow-hidden">
                            <span className="font-bold block text-[9px] uppercase text-zinc-400 mb-0.5">OCR Content</span>
                            <span className="text-zinc-700 dark:text-zinc-300 italic line-clamp-2 text-[11px]">&ldquo;{scene.ocr_text}&rdquo;</span>
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
    };

    return (
        <div className="flex flex-col gap-8 pb-12">
            {Object.entries(groupedResults).map(([groupName, scenes]) => {
                if (scenes.length === 0) return null; 

                return (
                    <div key={groupName} className="flex flex-col gap-3">
                        {groupBy !== 'none' && (
                            <div className="flex items-center gap-3">
                                <h4 className="text-sm font-bold text-zinc-800 dark:text-zinc-200 capitalize">{groupName}</h4>
                                <div className="h-px flex-1 bg-zinc-200 dark:bg-zinc-800" />
                                <span className="text-xs font-medium text-zinc-500">{scenes.length} items</span>
                            </div>
                        )}
                        <div className={`grid ${getGridClass()}`}>
                            {scenes.map((scene, index) => renderCard(scene, index))}
                        </div>
                    </div>
                );
            })}
        </div>
    );
}