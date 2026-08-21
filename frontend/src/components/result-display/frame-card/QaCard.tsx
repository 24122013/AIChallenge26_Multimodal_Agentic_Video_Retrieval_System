import type { VideoScene } from '../../../types';
import { API_PROXY } from '../../../constants/proxy';
import { Copy, Eye } from 'lucide-react';

interface QaCardProps {
    scene: VideoScene;
    cardSize: 'sm' | 'md' | 'lg';
    isClicked: boolean;
    outputFormat: string;
    onSelectResult: (scene: VideoScene) => void;
    handleCopyOutput: (text: string) => void;
}

export default function QaCard({
    scene, cardSize, isClicked, outputFormat, onSelectResult, handleCopyOutput
}: QaCardProps) {
    
    const wrapperStyle = isClicked 
        ? "ring-1 ring-zinc-300 dark:ring-zinc-700 bg-zinc-50/50 dark:bg-zinc-900/20" 
        : "hover:bg-zinc-50 dark:hover:bg-zinc-900/20 border border-transparent hover:border-zinc-100 dark:hover:border-zinc-800";

    return (
        <div className={`flex flex-col p-2.5 -m-2.5 rounded-2xl transition-all ${wrapperStyle}`}>
            
            {/* Image Container with Hover Overlay */}
            <div 
                className="relative aspect-video w-full bg-zinc-100 dark:bg-zinc-900 rounded-xl overflow-hidden mb-3 cursor-pointer group shadow-sm border border-black/5 dark:border-white/5"
                onClick={() => onSelectResult(scene)}
            >
                <img 
                    src={`${API_PROXY}/video/frame/${scene.video_id}/${scene.frame_id}`} 
                    alt={scene.frame_id} 
                    className="object-cover w-full h-full transition-transform duration-500 group-hover:scale-[1.02]" 
                    loading="lazy" 
                />
                
                <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors flex items-center justify-center">
                    <div className="opacity-0 group-hover:opacity-100 bg-white/95 dark:bg-zinc-900/95 text-zinc-900 dark:text-white px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all transform translate-y-2 group-hover:translate-y-0 shadow-sm border border-zinc-200 dark:border-zinc-700">
                        <Eye className="w-3.5 h-3.5" /> Review
                    </div>
                </div>
            </div>
            
            {/* Minimal Details Row */}
            <div className="flex justify-between items-start px-1 gap-3">
                <span className={`font-medium text-zinc-800 dark:text-zinc-200 truncate ${cardSize === 'sm' ? 'text-[11px] max-w-[120px]' : 'text-sm max-w-[200px]'}`}>
                    {scene.video_id}
                </span>
                
                <button 
                    onClick={() => handleCopyOutput(outputFormat)}
                    className="text-zinc-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors shrink-0 p-1 -m-1"
                    title="Copy export format"
                >
                    <Copy className="w-3.5 h-3.5" />
                </button>
            </div>
        </div>
    );
}