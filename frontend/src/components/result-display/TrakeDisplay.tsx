import type { TrakeData, VideoScene, CandidateEvent } from '../../types';
import { useSearchInteractions } from '../../hooks/useSearchInteractions';
import { API_PROXY } from '../../constants/proxy';

interface TrakeDisplayProps {
    trakeData: TrakeData;
    onSelectResult: (scene: VideoScene) => void;
    onFinalSubmit: (sceneId: string) => void;
}

export default function TrakeDisplay({
    trakeData,
    onSelectResult,
    onFinalSubmit,
}: TrakeDisplayProps) {
    const { submittedSceneIds } = useSearchInteractions();

    return (
        <div className="flex flex-col gap-8 max-w-5xl mx-auto w-full pb-12">
            <div className="flex items-center gap-4 border-b border-zinc-200 dark:border-zinc-800 pb-4">
                <h3 className="text-lg font-semibold text-zinc-800 dark:text-zinc-200">TRAKE Hypotheses</h3>
                <span className="text-xs font-mono bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400 px-2 py-1 rounded">
                    {trakeData.hypotheses.length} sequences mapped
                </span>
            </div>

            {trakeData.hypotheses.map((hypothesis, hIdx) => (
                <div key={hIdx} className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-2xl p-5 shadow-sm">
                    <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-3">
                            <span className="text-sm font-bold bg-zinc-100 dark:bg-zinc-800 px-2 py-1 rounded-md text-zinc-700 dark:text-zinc-300">
                                Rank #{hIdx + 1}
                            </span>
                            <span className="font-mono text-sm font-semibold text-zinc-800 dark:text-zinc-200">
                                {hypothesis.video_id}
                            </span>
                        </div>
                        <span className="text-sm font-mono text-zinc-500">
                            Confidence: {(hypothesis.score * 100).toFixed(1)}%
                        </span>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                        {hypothesis.events?.map((ev: CandidateEvent, eIdx: number) => {
                            const sceneData = ev.result;
                            if (!sceneData) return null;

                            const mockScene = {
                                video_id: sceneData.video_id,
                                frame_id: sceneData.frame_id,
                                timestamp: sceneData.timestamp || 0,
                                score: ev.normalized_score || 0,
                            } as unknown as VideoScene;

                            const isSubmitted = submittedSceneIds.has(mockScene.frame_id);

                            return (
                                <div key={eIdx} className="flex flex-col border border-zinc-100 dark:border-zinc-800 rounded-xl overflow-hidden bg-zinc-50/50 dark:bg-zinc-950/50 p-2">
                                    <span className="text-[10px] font-bold uppercase text-zinc-400 mb-2 ml-1">Event {ev.event_index + 1}</span>
                                    <div className="aspect-video w-full bg-black/5 dark:bg-white/5 rounded-lg overflow-hidden mb-3">
                                        <img src={`${API_PROXY}/video/frame/${mockScene.video_id}/${mockScene.frame_id}`} alt={mockScene.frame_id} className="object-cover w-full h-full" loading="lazy" />
                                    </div>
                                    <div className="flex justify-between items-center px-1 mb-3">
                                        <span className="text-xs font-mono text-zinc-500">@ {mockScene.timestamp.toFixed(1)}s</span>
                                    </div>
                                    <div className="flex gap-2">
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
                            );
                        })}
                    </div>
                </div>
            ))}
        </div>
    );
}