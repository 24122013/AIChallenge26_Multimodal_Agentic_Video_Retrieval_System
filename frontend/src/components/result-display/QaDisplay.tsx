import type { QAData, VideoScene } from '../../types';
import KistDisplay from './KistDisplay';

type GroupingMode = 'none' | 'video' | 'modality' | 'tens';

interface QaDisplayProps {
    qaData: QAData;
    results: VideoScene[];
    cardSize: 'sm' | 'md' | 'lg';
    groupBy: GroupingMode;
    onSelectResult: (scene: VideoScene) => void;
    onFinalSubmit: (sceneId: string) => void;
    clickedSceneIds: Set<string>;
    submittedSceneIds: Set<string>;
}

export default function QaDisplay({
    qaData,
    results,
    cardSize,
    groupBy,
    onSelectResult,
    onFinalSubmit,
    clickedSceneIds,
    submittedSceneIds,
}: QaDisplayProps) {
    const answer = qaData.answer;
    const hasGeneratedAnswer = answer.status === 'answered' && Boolean(answer.answer?.trim());
    const answerIssue = answer.status === 'error' || answer.status === 'insufficient_evidence';

    return (
        <div className="flex flex-col gap-5 pb-12">
            {hasGeneratedAnswer && (
                <div className="rounded-xl border border-blue-200 bg-blue-50 px-4 py-3 dark:border-blue-900/60 dark:bg-blue-950/30">
                    <span className="mb-1 block text-[10px] font-bold uppercase tracking-wider text-blue-600 dark:text-blue-400">
                        Optional grounded answer
                    </span>
                    <p className="text-sm text-zinc-800 dark:text-zinc-200">{answer.answer}</p>
                </div>
            )}

            {answerIssue && (
                <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800 dark:border-amber-900/60 dark:bg-amber-950/30 dark:text-amber-300">
                    Không thể sinh câu trả lời tự động. Các frame liên quan bên dưới vẫn có thể được kiểm tra và chọn thủ công.
                </div>
            )}

            <div className="flex items-center justify-between border-b border-zinc-200 pb-3 dark:border-zinc-800">
                <h3 className="text-sm font-semibold text-zinc-800 dark:text-zinc-200">Related frames</h3>
                <span className="rounded-md border border-zinc-200 bg-zinc-100 px-2.5 py-1 font-mono text-xs text-zinc-500 dark:border-zinc-700 dark:bg-zinc-800/50">
                    {results.length} frames
                </span>
            </div>

            {results.length > 0 ? (
                <KistDisplay
                    results={results}
                    cardSize={cardSize}
                    groupBy={groupBy}
                    onSelectResult={onSelectResult}
                    onFinalSubmit={onFinalSubmit}
                    clickedSceneIds={clickedSceneIds}
                    submittedSceneIds={submittedSceneIds}
                />
            ) : (
                <div className="flex h-64 items-center justify-center rounded-xl border-2 border-dashed border-zinc-300 text-sm text-zinc-500 dark:border-zinc-700/50">
                    No related frames match the current filters.
                </div>
            )}
        </div>
    );
}
