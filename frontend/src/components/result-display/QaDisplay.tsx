import type { QAData, VideoScene } from '../../types';
import { useSearchInteractions } from '../../hooks/useSearchInteractions';
import QaCard from './frame-card/QaCard';

interface QaDisplayProps {
    qaData: QAData;
    cardSize: 'sm' | 'md' | 'lg';
    onSelectResult: (scene: VideoScene) => void;
    onFinalSubmit?: (sceneId: string) => void; 
}

const getOutputString = (scene: VideoScene, answerText: string): string => {
    return `${scene.video_id}, ${scene.frame_index ?? scene.frame_id}, ${answerText}`;
};

const getGridClass = (size: 'sm' | 'md' | 'lg') => {
    switch (size) {
        case 'sm': return 'grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 2xl:grid-cols-8 gap-x-4 gap-y-8';
        case 'lg': return 'grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-x-8 gap-y-10';
        case 'md':
        default: return 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-x-6 gap-y-10';
    }
};

export default function QaDisplay({ qaData, cardSize, onSelectResult }: QaDisplayProps) {
    const { clickedSceneIds } = useSearchInteractions();

    const handleCopyOutput = (text: string) => {
        navigator.clipboard.writeText(text).catch(err => console.error("Failed to copy:", err));
    };

    // Safely extract data with fallbacks to prevent undefined crashes
    const answerText = qaData?.answer?.answer_text ?? "Generating answer...";
    const evidenceList = qaData?.evidence ?? [];

    return (
        <div className="flex flex-col mx-auto w-full">
            
            {/* Centered, clean typographic layout for the answer */}
            <div className="py-12 px-4 md:px-8 mb-4 max-w-5xl mx-auto text-center">
                <span className="text-xs font-bold uppercase tracking-widest text-zinc-400 dark:text-zinc-500 mb-6 block">
                    Synthesized Answer
                </span>
                <p className="text-3xl md:text-4xl font-medium leading-relaxed text-zinc-900 dark:text-zinc-100 tracking-tight">
                    {answerText}
                </p>
            </div>

            <div className="px-4 md:px-8 pb-12">
                <div className="flex items-center justify-between mb-8 border-b border-zinc-200 dark:border-zinc-800 pb-4">
                    <h3 className="text-lg font-medium text-zinc-800 dark:text-zinc-200">
                        Supporting Evidence
                    </h3>
                    <span className="text-sm font-mono text-zinc-500 bg-zinc-100 dark:bg-zinc-800/50 px-2.5 py-1 rounded-md border border-zinc-200/50 dark:border-zinc-700/50">
                        {evidenceList.length} sources
                    </span>
                </div>

                <div className={`grid ${getGridClass(cardSize)}`}>
                    {evidenceList.map((evidence, index) => {
                        const mockScene = { ...evidence } as unknown as VideoScene;
                        return (
                            <QaCard 
                                key={`${evidence.frame_id}-${index}`}
                                scene={mockScene}
                                cardSize={cardSize}
                                isClicked={clickedSceneIds.has(evidence.frame_id)}
                                outputFormat={getOutputString(mockScene, answerText)}
                                onSelectResult={onSelectResult}
                                handleCopyOutput={handleCopyOutput}
                            />
                        );
                    })}
                </div>
            </div>
        </div>
    );
}