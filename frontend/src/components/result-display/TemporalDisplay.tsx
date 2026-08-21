import React from 'react';
import type { TemporalData, VideoScene } from '../../types';
import { ArrowRight, Clock } from 'lucide-react';
import TemporalCard from './frame-card/TemporalCard';

interface TemporalDisplayProps {
    temporalData: TemporalData;
    cardSize: 'sm' | 'md' | 'lg';
    onSelectResult: (scene: VideoScene) => void;
    onFinalSubmit?: (sceneId: string) => void; 
    clickedSceneIds: Set<string>;
    submittedSceneIds: Set<string>;
}

export default function TemporalDisplay({
    temporalData, cardSize, onSelectResult, clickedSceneIds, submittedSceneIds,
}: TemporalDisplayProps) {
    
    const handleCopyOutput = (text: string) => {
        navigator.clipboard.writeText(text).catch(err => console.error("Failed to copy:", err));
    };

    return (
        <div className="flex flex-col gap-8 mx-auto w-full pb-12">
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
                            
                            {/* Card Header Restored (Solves the unused variables issue) */}
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
                                    
                                    return (
                                        <React.Fragment key={`${match.chain_id}-ev-${eIdx}`}>
                                            <TemporalCard
                                                scene={mockScene}
                                                eventIndex={eIdx}
                                                cardSize={cardSize}
                                                isClicked={clickedSceneIds.has(mockScene.frame_id)}
                                                isSubmitted={submittedSceneIds.has(mockScene.frame_id)}
                                                onSelectResult={onSelectResult}
                                            />
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