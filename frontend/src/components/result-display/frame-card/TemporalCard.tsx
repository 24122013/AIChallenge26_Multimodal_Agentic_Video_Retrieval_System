import type { VideoScene } from '../../../types';
import { Activity } from 'lucide-react';
import { METHOD_COLORS } from '../../../constants/mode-icons';
import { API_PROXY } from '../../../constants/proxy';

interface TemporalCardProps {
    scene: VideoScene;
    eventIndex: number;
    cardSize: 'sm' | 'md' | 'lg';
    isClicked: boolean;
    isSubmitted: boolean;
    onSelectResult: (scene: VideoScene) => void;
}

export default function TemporalCard({
    scene, eventIndex, cardSize, isClicked, isSubmitted, onSelectResult
}: TemporalCardProps) {
    
    let cardStyle = "border-zinc-200 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-950/50";
    if (isSubmitted) cardStyle = "border-emerald-400 dark:border-emerald-500/50 bg-emerald-50/30 dark:bg-emerald-950/20";
    else if (isClicked) cardStyle = "border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-900";

    let sizeClasses = "min-w-[220px] max-w-[280px]";
    if (cardSize === 'sm') sizeClasses = "min-w-[160px] max-w-[200px]";
    if (cardSize === 'lg') sizeClasses = "min-w-[300px] max-w-[380px]";

    return (
        <div className={`flex flex-col shrink-0 border rounded-xl overflow-hidden p-2.5 transition-colors ${cardStyle} ${sizeClasses}`}>
            <div className="flex items-center justify-between mb-2 px-1">
                <span className="text-[10px] font-bold uppercase text-zinc-500 flex items-center gap-1">
                    <Activity className="w-3 h-3" /> Event {eventIndex + 1}
                </span>
                <span className="text-xs font-mono font-semibold text-zinc-600 dark:text-zinc-400">
                    @ {scene.timestamp.toFixed(1)}s
                </span>
            </div>

            <div className="aspect-video w-full bg-black/5 dark:bg-white/5 rounded-lg overflow-hidden mb-3 border border-zinc-200 dark:border-zinc-800">
                <img src={`${API_PROXY}/video/frame/${scene.video_id}/${scene.frame_id}`} alt={scene.frame_id} className="object-cover w-full h-full" loading="lazy" />
            </div>

            {cardSize !== 'sm' && (
                <div className="flex items-center gap-1 flex-wrap mb-3 px-1 min-h-[24px]">
                    {Object.entries(scene.modality_scores || {}).filter(([m]) => m !== 'rrf').map(([method, score]) => (
                        <div key={method} className={`flex items-center gap-1 text-[9px] px-1.5 py-0.5 rounded border border-black/5 dark:border-white/5 ${METHOD_COLORS[method] || 'bg-zinc-200 text-zinc-800'}`}>
                            <span className="font-semibold uppercase tracking-tight">{method}</span>
                            <span className="font-mono font-bold opacity-80">{(score as number * 100).toFixed(0)}%</span>
                        </div>
                    ))}
                </div>
            )}
            
            <div className="flex mt-auto">
                {/* Temporal card specifically omits Submit button per instructions */}
                <button 
                    onClick={() => onSelectResult(scene)} 
                    className={`w-full bg-zinc-200 hover:bg-zinc-300 dark:bg-zinc-800 dark:hover:bg-zinc-700 text-zinc-800 dark:text-zinc-200 rounded transition-colors font-medium ${cardSize === 'sm' ? 'text-[10px] py-1.5' : 'text-xs py-2'}`}
                >
                    Review
                </button>
            </div>
        </div>
    );
}