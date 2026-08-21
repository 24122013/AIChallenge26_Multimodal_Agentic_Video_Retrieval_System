import React from 'react';
import { Plus, X } from "lucide-react";
import Textarea from '../ui/textarea';
import type { NavBoxType } from './types';

interface QueryInputExpandProps {
    expandedQueries: string[];
    setExpandedQueries: (queries: string[]) => void;
    expandedRefs: React.RefObject<(HTMLTextAreaElement | null)[]>;
    handleEnterNavigation: (e: React.KeyboardEvent, box: NavBoxType, index?: number) => void;
}

export default function QueryInputExpand({
    expandedQueries,
    setExpandedQueries,
    expandedRefs,
    handleEnterNavigation
}: QueryInputExpandProps) {
    return (
        <div className="flex flex-col gap-2 mt-2 pt-2 border-t border-black/5 dark:border-white/5">
            {expandedQueries.map((eq, index) => (
                <div key={index} className="flex items-start gap-2 bg-white/50 dark:bg-black/20 rounded-lg border border-black/10 dark:border-white/10 p-2 focus-within:border-blue-500 transition-colors">
                    <span className="text-xs font-semibold text-zinc-400 mt-1">{index + 1}.</span>
                    <Textarea
                        ref={el => { expandedRefs.current[index] = el; }}
                        value={eq}
                        onChange={e => {
                            const newQueries = [...expandedQueries];
                            newQueries[index] = e.target.value;
                            setExpandedQueries(newQueries);
                        }}
                        onKeyDown={e => handleEnterNavigation(e, 'extended', index)}
                        placeholder="Expanded context..."
                        className="w-full bg-transparent border-none text-sm resize-none focus-visible:ring-0 dark:text-white min-h-[30px] p-0"
                    />
                    <button 
                        onClick={() => {
                            const newQueries = [...expandedQueries];
                            newQueries.splice(index, 1);
                            setExpandedQueries(newQueries);
                        }}
                        className="text-zinc-400 hover:text-red-500 transition-colors p-1"
                    >
                        <X className="w-3 h-3" />
                    </button>
                </div>
            ))}
            
            <button
                onClick={() => {
                    setExpandedQueries([...expandedQueries, ""]);
                    setTimeout(() => expandedRefs.current[expandedQueries.length]?.focus(), 50);
                }}
                className="flex items-center gap-1 text-xs text-blue-600 dark:text-blue-400 font-medium py-1.5 px-2 hover:bg-blue-50 dark:hover:bg-blue-950/30 rounded-lg w-max transition-colors"
            >
                <Plus className="w-3 h-3" /> Add Extended Query
            </button>
        </div>
    );
}