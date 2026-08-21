import type { QAData, VideoScene } from '../../types';
import { useSearchInteractions } from '../../hooks/useSearchInteractions';
import { API_PROXY } from '../../constants/proxy';

interface QaDisplayProps {
    qaData: QAData;
    onSelectResult: (scene: VideoScene) => void;
    onFinalSubmit: (sceneId: string) => void;
}

const getOutputString = (scene: VideoScene, answerText: string): string => {
    return `${scene.video_id}, ${scene.frame_index ?? scene.frame_id}, ${answerText}`;
};

export default function QaDisplay({
    qaData,
    onSelectResult,
    onFinalSubmit
}: QaDisplayProps) {
    const { clickedSceneIds, submittedSceneIds } = useSearchInteractions();

    const handleCopyOutput = (text: string) => {
        navigator.clipboard.writeText(text).catch(err => console.error("Failed to copy:", err));
    };

    return (
        <div className="flex flex-col max-w-6xl mx-auto w-full">
            <div className="py-12 px-4 md:px-8 mb-8">
                <span className="text-xs font-bold uppercase tracking-widest text-zinc-400 dark:text-zinc-500 mb-4 block">Synthesized Answer</span>
                <p className="text-3xl md:text-4xl font-semibold leading-tight text-zinc-900 dark:text-zinc-100 max-w-4xl tracking-tight">
                    {qaData.answer.answer_text}
                </p>
            </div>

            <div className="h-px w-full bg-zinc-200 dark:bg-zinc-800 mb-12" />

            <div className="px-4 md:px-8">
                <div className="flex items-center justify-between mb-6">
                    <h3 className="text-lg font-medium text-zinc-800 dark:text-zinc-200">Supporting Evidence</h3>
                    <span className="text-sm font-mono text-zinc-500">{qaData.evidence.length} sources</span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                    {qaData.evidence.map((evidence, index) => {
                        const mockScene = {
                            video_id: evidence.video_id,
                            frame_id: evidence.frame_id,
                            frame_index: evidence.frame_index,
                            timestamp: evidence.timestamp,
                            score: evidence.retrieval_score,
                            caption: evidence.caption,
                            ocr_text: evidence.ocr_text,
                            objects: evidence.objects,
                            modality_scores: {} 
                        } as unknown as VideoScene;

                        const isClicked = clickedSceneIds.has(evidence.frame_id);
                        const isSubmitted = submittedSceneIds.has(evidence.frame_id);
                        
                        let borderStyle = "border-zinc-200 dark:border-zinc-800";
                        if (isSubmitted) borderStyle = "border-emerald-400 dark:border-emerald-500 ring-1 ring-emerald-400 shadow-sm";
                        else if (isClicked) borderStyle = "border-zinc-300 dark:border-zinc-700 bg-zinc-50/50 dark:bg-zinc-900/30";

                        const outputFormat = getOutputString(mockScene, qaData.answer.answer_text);

                        return (
                            <div key={`${evidence.frame_id}-${index}`} className={`flex flex-col p-4 rounded-2xl border transition-all ${borderStyle}`}>
                                <div className="relative aspect-video w-full bg-zinc-100 dark:bg-zinc-900 rounded-lg overflow-hidden mb-4">
                                    <img src={`${API_PROXY}/video/frame/${evidence.video_id}/${evidence.frame_id}`} alt={evidence.frame_id} className="object-cover w-full h-full" loading="lazy" />
                                </div>
                                
                                <div className="flex justify-between items-center mb-3">
                                    <span className="font-mono text-xs font-bold text-zinc-700 dark:text-zinc-300 px-2 py-1 bg-zinc-100 dark:bg-zinc-800 rounded">
                                        {evidence.video_id}
                                    </span>
                                    <span className="text-xs font-mono text-zinc-500">@ {evidence.timestamp.toFixed(1)}s</span>
                                </div>

                                <div 
                                    className="mb-4 bg-zinc-50 dark:bg-zinc-900/50 p-2 rounded-lg cursor-pointer hover:bg-zinc-100 dark:hover:bg-zinc-900 transition-colors border border-transparent hover:border-zinc-200 dark:hover:border-zinc-800"
                                    onClick={() => handleCopyOutput(outputFormat)}
                                    title="Click to copy export format"
                                >
                                    <span className="text-[10px] text-zinc-500 font-mono truncate block">{outputFormat}</span>
                                </div>

                                <div className="flex gap-2 mt-auto">
                                    <button 
                                        onClick={() => onSelectResult(mockScene)} 
                                        className="flex-1 bg-zinc-900 dark:bg-zinc-200 hover:bg-zinc-800 text-white dark:text-zinc-900 text-xs py-2 rounded-lg font-medium transition-colors"
                                    >
                                        Review
                                    </button>
                                    <button 
                                        onClick={() => onFinalSubmit(evidence.frame_id)} 
                                        className={`text-xs py-2 px-4 rounded-lg font-semibold transition-colors border ${isSubmitted ? 'bg-red-50 text-red-600 border-red-200 hover:bg-red-100 dark:bg-red-950/30 dark:text-red-400' : 'border-zinc-200 hover:bg-zinc-50 dark:border-zinc-700 dark:hover:bg-zinc-800'}`}
                                    >
                                        {isSubmitted ? 'Undo' : 'Accept'}
                                    </button>
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>
        </div>
    );
}