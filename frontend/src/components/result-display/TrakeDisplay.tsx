import type { TrakeData, VideoScene, CandidateEvent } from '../../types';
import { API_PROXY } from '../../constants/proxy';

interface TrakeDisplayProps {
    trakeData: TrakeData;
    cardSize: 'sm' | 'md' | 'lg';
    onSelectResult: (scene: VideoScene) => void;
    onFinalSubmit: (sceneId: string) => void;
    clickedTrakeIds: Set<string>;
    submittedTrakeIds: Set<string>;
}

const getGridClass = (size: 'sm' | 'md' | 'lg') => {
    switch (size) {
        case 'sm': return 'grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3';
        case 'lg': return 'grid-cols-1 md:grid-cols-2 gap-6';
        case 'md':
        default: return 'grid-cols-1 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-4';
    }
};

export default function TrakeDisplay({
    trakeData,
    cardSize,
    onSelectResult,
    onFinalSubmit,
    clickedTrakeIds = new Set(),
    submittedTrakeIds = new Set(),
}: TrakeDisplayProps) {

    if (trakeData.hypotheses.length === 0) {
        const warnings = trakeData.warnings ?? trakeData.trace?.warnings ?? [];
        return (
            <div role="status" className="mx-auto w-full max-w-4xl rounded-xl border border-amber-200 bg-amber-50 p-6 text-amber-900 dark:border-amber-900/60 dark:bg-amber-950/30 dark:text-amber-200">
                <h3 className="font-semibold">No complete TRAKE sequence found</h3>
                <p className="mt-2 text-sm">
                    Status: {trakeData.status || 'insufficient_support'}. The system did not return a partial or low-support sequence.
                </p>
                {warnings.length > 0 && (
                    <ul className="mt-3 list-disc space-y-1 pl-5 text-xs">
                        {warnings.map(warning => <li key={warning}>{warning}</li>)}
                    </ul>
                )}
            </div>
        );
    }

    return (
        <div className="flex flex-col gap-8 max-w-7xl mx-auto w-full pb-12">
            <div className="flex items-center gap-4 border-b border-zinc-200 dark:border-zinc-800 pb-4">
                <h3 className="text-lg font-semibold text-zinc-800 dark:text-zinc-200">TRAKE Hypotheses</h3>
                <span className="text-xs font-mono bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400 px-2 py-1 rounded">
                    {trakeData.hypotheses.length} sequences mapped
                </span>
            </div>

            {trakeData.hypotheses.map((hypothesis, hIdx) => {
                const sequenceId = hypothesis.path_id || `${hypothesis.video_id}:${hypothesis.frame_ids.join('-')}`;
                const isSubmitted = submittedTrakeIds.has(sequenceId);
                
                return (
                    <div 
                        key={sequenceId || hIdx}
                        className={`bg-white dark:bg-zinc-900 border transition-all rounded-2xl p-5 shadow-sm ${
                            isSubmitted 
                                ? 'border-emerald-400 dark:border-emerald-500/50 ring-1 ring-emerald-400 bg-emerald-50/20 dark:bg-emerald-950/20' 
                                : 'border-zinc-200 dark:border-zinc-800'
                        }`}
                    >
                        {/* Sequence Level Header & Actions */}
                        <div className="flex flex-wrap items-center justify-between gap-4 mb-6 pb-4 border-b border-zinc-100 dark:border-zinc-800/80">
                            <div className="flex items-center gap-3">
                                <span className="text-sm font-bold bg-zinc-100 dark:bg-zinc-800 px-2 py-1 rounded-md text-zinc-700 dark:text-zinc-300">
                                    Rank #{hIdx + 1}
                                </span>
                                <span className="font-mono text-sm font-semibold text-zinc-800 dark:text-zinc-200">
                                    {hypothesis.video_id}
                                </span>
                                <div className="h-4 w-px bg-zinc-300 dark:bg-zinc-700 hidden sm:block" />
                                <span className="text-sm font-mono text-zinc-500">
                                    Confidence: {(hypothesis.score * 100).toFixed(1)}%
                                </span>
                            </div>
                            
                            {/* Sequence Level Submit Button */}
                            <button 
                                onClick={() => onFinalSubmit(sequenceId)}
                                className={`text-xs py-2 px-4 rounded-lg font-semibold transition-colors border ${
                                    isSubmitted 
                                        ? 'bg-red-50 text-red-600 border-red-200 hover:bg-red-100 dark:bg-red-950/30 dark:text-red-400' 
                                        : 'bg-zinc-900 text-white dark:bg-zinc-200 dark:text-zinc-900 hover:bg-zinc-800 dark:hover:bg-zinc-300 border-transparent'
                                }`}
                            >
                                {isSubmitted ? 'Undo Sequence' : 'Accept Sequence'}
                            </button>
                        </div>

                        {/* Individual Events within the Sequence */}
                        <div className={`grid ${getGridClass(cardSize)}`}>
                            {hypothesis.events?.map((ev: CandidateEvent, eIdx: number) => {
                                const sceneData = ev.result;
                                if (!sceneData) return null;
                                const frameIdx = hypothesis.frame_ids[eIdx] ?? sceneData.frame_index;

                                const mockScene: VideoScene = {
                                    ...sceneData,
                                    timestamp: sceneData.timestamp ?? 0,
                                    score: ev.normalized_score ?? 0,
                                    asr_text: '',
                                    neighbors: sceneData.neighbors.map(neighbor => ({
                                        frame_id: neighbor.frame_id,
                                        timestamp: neighbor.timestamp,
                                        delta_seconds: neighbor.timestamp - (sceneData.timestamp ?? 0),
                                    })),
                                };

                                const isClicked = clickedTrakeIds.has(mockScene.frame_id);

                                return (
                                    <div key={eIdx} className={`flex flex-col border rounded-xl overflow-hidden p-2.5 transition-colors ${
                                        isClicked 
                                            ? 'border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-900/80' 
                                            : 'border-zinc-200 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-950/50'
                                    }`}>
                                        <span className="text-[10px] font-bold uppercase text-zinc-500 mb-2 ml-1 flex items-center gap-1">
                                            Event {ev.event_index + 1}
                                        </span>
                                        <div className="aspect-video w-full bg-black/5 dark:bg-white/5 rounded-lg overflow-hidden mb-3 border border-zinc-200 dark:border-zinc-800">
                                            <img src={`${API_PROXY}/video/frame/${mockScene.video_id}/${mockScene.frame_id}`} alt={mockScene.frame_id} className="object-cover w-full h-full" loading="lazy" />
                                        </div>
                                        <div className="flex justify-between items-center gap-2 px-1 mb-3">
                                            <span className={`${cardSize === 'sm' ? 'text-[10px]' : 'text-xs'} font-mono font-medium text-zinc-600 dark:text-zinc-400`}>
                                                @ {mockScene.timestamp.toFixed(1)}s
                                            </span>
                                            <span className={`${cardSize === 'sm' ? 'text-[10px]' : 'text-xs'} font-mono font-semibold text-zinc-700 dark:text-zinc-300`}>
                                                frame_idx: {frameIdx}
                                            </span>
                                        </div>
                                        <div className="flex gap-2 mt-auto">
                                            <button 
                                                onClick={() => onSelectResult(mockScene)} 
                                                className={`w-full bg-zinc-200 hover:bg-zinc-300 dark:bg-zinc-800 dark:hover:bg-zinc-700 text-zinc-800 dark:text-zinc-200 rounded transition-colors font-medium ${cardSize === 'sm' ? 'text-[10px] py-1.5' : 'text-xs py-2'}`}
                                            >
                                                Review
                                            </button>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                );
            })}
        </div>
    );
}
