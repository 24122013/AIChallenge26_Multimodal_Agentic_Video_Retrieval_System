import React from 'react';
import { Search, HelpCircle, Activity, Clock, ChevronDown } from 'lucide-react';
import Textarea from '../ui/textarea';
import { cn } from '../../libs/utils';
import { type KistMode, KIST_MODES } from '../../constants/mode-icons';
import type { NavBoxType } from './types';

interface QueryInputPrimaryProps {
    selectedModel: string;
    kistQueries: Record<KistMode, string>;
    setKistQueries: React.Dispatch<React.SetStateAction<Record<KistMode, string>>>;
    kistMode: KistMode;
    setKistMode: (mode: KistMode) => void;
    qaQuery: string;
    setQaQuery: (q: string) => void;
    trakeQuery: string;
    setTrakeQuery: (q: string) => void;
    temporalQuery: string;
    setTemporalQuery: (q: string) => void;
    isModeSelectorOpen: boolean;
    setIsModeSelectorOpen: (open: boolean) => void;
    highlightedModeIndex: number;
    setHighlightedModeIndex: (idx: number) => void;
    kistRef: React.RefObject<HTMLTextAreaElement | null>;
    qaRef: React.RefObject<HTMLTextAreaElement | null>;
    trakeRef: React.RefObject<HTMLTextAreaElement | null>;
    temporalRef: React.RefObject<HTMLTextAreaElement | null>;
    handleEnterNavigation: (e: React.KeyboardEvent, box: NavBoxType, index?: number) => void;
    children?: React.ReactNode;
}

export default function QueryInputPrimary({
    selectedModel,
    kistQueries, setKistQueries, kistMode, setKistMode,
    qaQuery, setQaQuery, trakeQuery, setTrakeQuery, temporalQuery, setTemporalQuery,
    isModeSelectorOpen, setIsModeSelectorOpen, highlightedModeIndex, setHighlightedModeIndex,
    kistRef, qaRef, trakeRef, temporalRef,
    handleEnterNavigation,
    children
}: QueryInputPrimaryProps) {
    const kistPlaceholder = kistMode === 'temporal'
        ? 'Khoảnh khắc đầu tiên người dẫn xuất hiện trên xích lô...'
        : `Describe context or object for ${kistMode}...`;
    
    return (
        <div className="bg-white dark:bg-white/5 rounded-xl border border-black/10 dark:border-white/10 p-2 focus-within:border-blue-500 transition-colors">
            
            {selectedModel === "KIST" && (
                <>
                    <div className="flex items-center justify-between mb-1 px-2 relative">
                        <span className="text-xs font-semibold text-black/50 dark:text-white/50 flex items-center gap-1">
                            <Search className="w-3 h-3" /> Query Input
                        </span>
                        
                        <button
                            onClick={() => {
                                setIsModeSelectorOpen(!isModeSelectorOpen);
                                setHighlightedModeIndex(KIST_MODES.indexOf(kistMode));
                            }}
                            className="flex items-center justify-between min-w-[90px] gap-1 bg-black/5 dark:bg-black/40 text-xs text-gray-700 dark:text-gray-300 rounded px-2 py-1 outline-none focus:ring-1 focus:ring-blue-500 cursor-pointer border-none"
                        >
                            <span className="capitalize">{kistMode}</span>
                            <ChevronDown className="w-3 h-3 opacity-50" />
                        </button>

                        {isModeSelectorOpen && (
                            <>
                                <div className="fixed inset-0 z-40" onClick={() => setIsModeSelectorOpen(false)} />
                                <div className="absolute top-full right-0 mt-1 w-32 bg-white dark:bg-[#1a1a1a] border border-black/10 dark:border-white/10 rounded-md shadow-xl z-50 py-1 flex flex-col">
                                    {KIST_MODES.map((mode, idx) => (
                                        <div
                                            key={mode}
                                            onClick={() => {
                                                setKistMode(mode as KistMode);
                                                setIsModeSelectorOpen(false);
                                            }}
                                            className={cn(
                                                "px-3 py-1.5 text-xs cursor-pointer capitalize hover:bg-black/5 dark:hover:bg-white/10 text-gray-700 dark:text-gray-300",
                                                highlightedModeIndex === idx && "bg-blue-50 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 font-medium"
                                            )}
                                        >
                                            {mode}
                                        </div>
                                    ))}
                                    <div className="px-3 py-1 mt-1 text-[10px] text-gray-400 dark:text-gray-500 border-t border-black/10 dark:border-white/10 leading-tight">
                                        Use <kbd className="font-sans">↑</kbd> <kbd className="font-sans">↓</kbd> to navigate<br/> <kbd className="font-sans">Enter</kbd> to select
                                    </div>
                                </div>
                            </>
                        )}
                    </div>
                    <Textarea
                        ref={kistRef}
                        value={kistQueries[kistMode]}
                        onChange={e => setKistQueries(prev => ({ ...prev, [kistMode]: e.target.value }))}
                        onKeyDown={e => handleEnterNavigation(e, 'main_query')}
                        placeholder={kistPlaceholder}
                        className="w-full bg-transparent border-none text-sm resize-none focus-visible:ring-0 dark:text-white min-h-[50px]"
                    />
                </>
            )}

            {selectedModel === "QA" && (
                <>
                    <span className="text-xs font-semibold text-black/50 dark:text-white/50 flex items-center gap-1 mb-1 px-2">
                        <HelpCircle className="w-3 h-3" /> Question
                    </span>
                    <Textarea
                        ref={qaRef}
                        value={qaQuery}
                        onChange={e => setQaQuery(e.target.value)}
                        onKeyDown={e => handleEnterNavigation(e, 'main_query')}
                        placeholder="Enter your question..."
                        className="w-full bg-transparent border-none text-sm resize-none focus-visible:ring-0 dark:text-white min-h-[70px]"
                    />
                </>
            )}

            {selectedModel === "TRAKE" && (
                <>
                    <span className="text-xs font-semibold text-black/50 dark:text-white/50 flex items-center gap-1 mb-1 px-2">
                        <Activity className="w-3 h-3" /> TRAKE Event Sequence
                    </span>
                    <Textarea
                        ref={trakeRef}
                        value={trakeQuery}
                        onChange={e => setTrakeQuery(e.target.value)}
                        onKeyDown={e => handleEnterNavigation(e, 'main_query')}
                        placeholder={'Mô tả video/context (không bắt buộc)\nE1: Sự kiện thứ nhất...\nE2: Sự kiện thứ hai...\nE3: Sự kiện thứ ba...'}
                        aria-label="TRAKE context and ordered E1, E2 event sequence"
                        className="w-full bg-transparent border-none text-sm resize-y focus-visible:ring-0 dark:text-white min-h-[140px]"
                    />
                    <p className="px-2 pb-1 text-[10px] leading-relaxed text-zinc-500 dark:text-zinc-400">
                        Mỗi sự kiện bắt đầu bằng E1:, E2:, ...; Enter để xuống dòng, Ctrl/⌘ + Enter để tìm kiếm.
                    </p>
                </>
            )}

            {selectedModel === "TEMPORAL" && (
                <>
                    <span className="text-xs font-semibold text-black/50 dark:text-white/50 flex items-center gap-1 mb-1 px-2">
                        <Clock className="w-3 h-3" /> Temporal Evidence Query
                    </span>
                    <Textarea
                        ref={temporalRef}
                        value={temporalQuery}
                        onChange={e => setTemporalQuery(e.target.value)}
                        onKeyDown={e => handleEnterNavigation(e, 'main_query')}
                        placeholder="Describe an ordered evidence chain for QA..."
                        className="w-full bg-transparent border-none text-sm resize-none focus-visible:ring-0 dark:text-white min-h-[70px]"
                    />
                </>
            )}

            {children}
        </div>
    );
}
