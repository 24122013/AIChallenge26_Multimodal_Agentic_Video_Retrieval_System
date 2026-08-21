import { Settings2, ChevronRight } from "lucide-react";
import { useState } from "react";
import { cn } from "../../libs/utils";

interface ConfigPanelProps {
    topK: number;
    setTopK: (val: number) => void;
    handleNumberSync: (val: string, setter: (val: number) => void, min: number, max: number) => void;
}

export default function ConfigPanel({ topK, setTopK, handleNumberSync }: ConfigPanelProps) {
    const [showConfig, setShowConfig] = useState(false);

    return (
        <div className="flex-1 flex flex-col border border-black/10 dark:border-white/10 rounded-xl bg-white dark:bg-white/5 overflow-hidden min-h-0">
            <button 
                onClick={() => setShowConfig(!showConfig)} 
                className="shrink-0 flex items-center justify-between w-full p-3 text-sm font-semibold dark:text-white bg-black/5 dark:bg-black/40 hover:bg-black/10 dark:hover:bg-black/60 transition-colors"
            >
                <div className="flex items-center gap-2"><Settings2 className="w-4 h-4" /> Tuning Parameters</div>
                <ChevronRight className={cn("w-4 h-4 transition-transform", showConfig && "rotate-90")} />
            </button>

            {showConfig && (
                <div className="flex-1 p-4 flex flex-col overflow-y-auto overflow-x-hidden gap-5 text-sm">
                    <div className="flex flex-col gap-2">
                        <div className="flex items-center justify-between">
                            <label className="text-xs text-gray-500 dark:text-gray-400 font-medium">Top K Results</label>
                            <input
                                type="number"
                                value={topK}
                                onChange={(e) => handleNumberSync(e.target.value, setTopK, 1, 200)}
                                className="w-16 p-1 text-right text-xs bg-black/5 dark:bg-black/40 rounded border-none focus:ring-1 focus:ring-blue-500 dark:text-white hide-arrows"
                            />
                        </div>
                        <input
                            type="range"
                            min={1}
                            max={200}
                            value={topK}
                            onChange={(e) => setTopK(parseInt(e.target.value))}
                            className="w-full accent-blue-600"
                        />
                    </div>
                </div>
            )}
        </div>
    );
}