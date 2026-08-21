import React from 'react';
import type { TemporalData, VideoScene } from '../../types';
import { ArrowRight, Clock, Activity } from 'lucide-react';
import { METHOD_COLORS } from '../../constants/mode-icons';
import { API_PROXY } from '../../constants/proxy';

interface TemporalDisplayProps {
    temporalData: TemporalData;
    onSelectResult: (scene: VideoScene) => void;
    onFinalSubmit: (sceneId: string) => void;
    clickedSceneIds: Set<string>;
    submittedSceneIds: Set<string>;
}

export default function TemporalDisplay({
    temporalData,
    onSelectResult,
    onFinalSubmit,
    clickedSceneIds,
    submittedSceneIds,
}: TemporalDisplayProps) {

    const handleCopyOutput = (text: string) => {
        navigator.clipboard.writeText(text).catch(err => console.error("Failed to copy:", err));
    };

    return (
        <div className="flex flex-col gap-8 max-w-7xl mx-auto w-full pb-12">
            <div className="flex items-center gap-4 border-b border-zinc-200 dark:border-zinc-800 pb-4">
                <h3 className="text-lg font-semibold text-zinc-800 dark:text-zinc-200 flex items-center gap-2">
                    <Clock className="w-5 h-5 text-emerald-500" /> 
                    Temporal Sequences
                </h3>
                <span className="text-xs font-mono bg-emerald-50 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400 px-2 py-1 rounded border border-emerald-200 dark:border-emerald-800/50">
                    {temporalData.temporal_matches.length} chains mapped
                </span>
            </div>

            <div className="flex flex-col gap-6">
                {temporalData.temporal_matches.map((match, mIdx) => {
                    const outputFormat = `${match.video_id}, ${match.events.map(e => e.frame_index ?? e.frame_id).join(', ')}`;
                    
                    return (
                        <div key={match.chain_id || mIdx} className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-2xl p-5 shadow-sm overflow-hidden relative">
                            {/* Card Header */}
                            <div className="flex flex-wrap items-center justify-between gap-4 mb-5 border-b border-zinc-100 dark:border-zinc-800/50 pb-4">
                                <div className="flex items-center gap-3">
                                    <span className="text-sm font-bold bg-zinc-100 dark:bg-zinc-800 px-2 py-1 rounded-md text-zinc-700 dark:text-zinc-300">
                                        Rank #{mIdx + 1}
                                    </span>
                                    <span className="font-mono text-sm font-semibold text-zinc-800 dark:text-zinc-200">
                                        {match.video_id}
                                    </span>
                                    <div className="h-4 w-px bg-zinc-300 dark:bg-zinc-700" />
                                    <div className="flex items-center gap-1.5 text-xs text-zinc-500 font-mono">
                                        <Clock className="w-3.5 h-3.5" />
                                        <span>{match.start_time.toFixed(1)}s - {match.end_time.toFixed(1)}s</span>
                                    </div>
                                </div>
                                <div className="flex items-center gap-3">
                                    <div 
                                        className="text-xs font-mono bg-zinc-50 dark:bg-zinc-800/50 border border-zinc-200 dark:border-zinc-700 px-2 py-1 rounded cursor-pointer hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors text-zinc-600 dark:text-zinc-400"
                                        onClick={() => handleCopyOutput(outputFormat)}
                                        title="Click to copy sequence output"
                                    >
                                        Copy: {outputFormat.length > 30 ? outputFormat.slice(0, 30) + '...' : outputFormat}
                                    </div>
                                    <span className="text-sm font-mono font-bold text-emerald-600 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-950/30 px-2 py-1 rounded border border-emerald-100 dark:border-emerald-900/50">
                                        Score: {(match.score * 100).toFixed(1)}%
                                    </span>
                                </div>
                            </div>

                            {/* Filmstrip of Events */}
                            <div className="flex items-stretch gap-2 overflow-x-auto pb-4 custom-scrollbar">
                                {match.events.map((ev, eIdx) => {
                                    const mockScene = {
                                        ...ev,
                                        score: ev.score || 0,
                                        timestamp: ev.timestamp || 0
                                    } as unknown as VideoScene;

                                    const isSubmitted = submittedSceneIds.has(mockScene.frame_id);
                                    const isClicked = clickedSceneIds.has(mockScene.frame_id);

                                    let cardStyle = "border-zinc-200 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-950/50";
                                    if (isSubmitted) cardStyle = "border-emerald-400 dark:border-emerald-500/50 bg-emerald-50/30 dark:bg-emerald-950/20";
                                    else if (isClicked) cardStyle = "border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-900";

                                    return (
                                        <React.Fragment key={`${match.chain_id}-ev-${eIdx}`}>
                                            <div className={`flex flex-col min-w-[220px] max-w-[280px] shrink-0 border rounded-xl overflow-hidden p-2.5 transition-colors ${cardStyle}`}>
                                                
                                                <div className="flex items-center justify-between mb-2 px-1">
                                                    <span className="text-[10px] font-bold uppercase text-zinc-500 flex items-center gap-1">
                                                        <Activity className="w-3 h-3" /> Event {eIdx + 1}
                                                    </span>
                                                    <span className="text-xs font-mono font-semibold text-zinc-600 dark:text-zinc-400">
                                                        @ {mockScene.timestamp.toFixed(1)}s
                                                    </span>
                                                </div>

                                                <div className="aspect-video w-full bg-black/5 dark:bg-white/5 rounded-lg overflow-hidden mb-3 border border-zinc-200 dark:border-zinc-800">
                                                    <img src={`${API_PROXY}/video/frame/${mockScene.video_id}/${mockScene.frame_id}`} alt={mockScene.frame_id} className="object-cover w-full h-full" loading="lazy" />
                                                </div>

                                                {/* Modality Scores */}
                                                <div className="flex items-center gap-1 flex-wrap mb-3 px-1 min-h-[24px]">
                                                    {Object.entries(ev.modality_scores || {}).filter(([m]) => m !== 'rrf').map(([method, score]) => (
                                                        <div key={method} className={`flex items-center gap-1 text-[9px] px-1.5 py-0.5 rounded border border-black/5 dark:border-white/5 ${METHOD_COLORS[method] || 'bg-zinc-200 text-zinc-800'}`}>
                                                            <span className="font-semibold uppercase tracking-tight">{method}</span>
                                                            <span className="font-mono font-bold opacity-80">{(score * 100).toFixed(0)}%</span>
                                                        </div>
                                                    ))}
                                                </div>
                                                
                                                <div className="flex gap-2 mt-auto">
                                                    <button 
                                                        onClick={() => onSelectResult(mockScene)} 
                                                        className="flex-1 bg-zinc-200 hover:bg-zinc-300 dark:bg-zinc-800 dark:hover:bg-zinc-700 text-zinc-800 dark:text-zinc-200 text-[10px] py-1.5 rounded transition-colors font-medium"
                                                    >
                                                        Review
                                                    </button>
                                                    <button 
                                                        onClick={() => onFinalSubmit(mockScene.frame_id)} 
                                                        className={`text-[10px] py-1.5 px-3 rounded font-semibold transition-colors border ${isSubmitted ? 'bg-red-50 text-red-600 border-red-200 dark:bg-red-950/30 dark:text-red-400' : 'border-zinc-200 hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-800'}`}
                                                    >
                                                        {isSubmitted ? 'Undo' : 'Accept'}
                                                    </button>
                                                </div>
                                            </div>
                                            
                                            {/* Connector Arrow */}
                                            {eIdx < match.events.length - 1 && (
                                                <div className="flex items-center justify-center shrink-0 px-1 text-zinc-300 dark:text-zinc-700">
                                                    <ArrowRight className="w-5 h-5" />
                                                </div>
                                            )}
                                        </React.Fragment>
                                    );
                                })}
                            </div>

                        </div>
                    );
                })}
            </div>
        </div>
    );
}