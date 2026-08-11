"use client";

import {
    ArrowRight, Settings2, Search, HelpCircle,
    ChevronRight, ChevronLeft, Link as LinkIcon, Upload, Wand2, Palette, ChevronDown
} from "lucide-react";
import { useCallback, useState, useRef, useEffect } from "react";
import { type ImageInputMode, type ColorScheme, COLOR_OPTIONS } from '../../constants/image-input';
import type { TextSearchPayload, SearchPayload, ImagePayload } from '../../types';
import Textarea from '../ui/textarea';
import { cn } from "../../libs/utils";
import ModelDropdownMenu from "../ModelsDropdownMenu";
import { type RetrievalMethod, type KistMode, KIST_MODES } from '../../constants/mode-icons';

interface SearchSideBarProps {
    onSearch: (payload: SearchPayload) => void;
    isExpanded: boolean;
    setIsExpanded: (expanded: boolean) => void;
}

interface ExtendedTextSearchPayload extends Omit<TextSearchPayload, 'mode'> {
    mode: RetrievalMethod;
}

export default function SearchSideBar({ onSearch, isExpanded, setIsExpanded }: SearchSideBarProps) {
    const [showConfig, setShowConfig] = useState(false);
    
    // Model Selection
    const [selectedModel, setSelectedModel] = useState<string>("KIST");

    // KIST State
    const [kistQueries, setKistQueries] = useState<Record<KistMode, string>>({
        visual: "", hybrid: "", caption: "", ocr: "", asr: "", object: "", temporal: ""
    });
    const [kistMode, setKistMode] = useState<KistMode>("visual");
    
    // Hotkey State - Mode Selector (Ctrl + Q)
    const [isModeSelectorOpen, setIsModeSelectorOpen] = useState(false);
    const [highlightedModeIndex, setHighlightedModeIndex] = useState(0);

    // QA State
    const [qaQuery, setQaQuery] = useState("");

    // Image Add-on State
    const [imageMode, setImageMode] = useState<ImageInputMode>('upload');
    const [imageLink, setImageLink] = useState("");
    const [imagePrompt, setImagePrompt] = useState("");
    const [selectedColor, setSelectedColor] = useState<ColorScheme>('none');

    // Config Matrix State
    const [ragTopN, setRagTopN] = useState<number>(100);
    const [rerankTopM, setRerankTopM] = useState<number>(25);
    const [tradSearchParam, setTradSearchParam] = useState<number>(50);
    const [enableExpansion, setEnableExpansion] = useState(false);

    // DOM Refs
    const fileInputRef = useRef<HTMLInputElement>(null);
    const uploadZoneRef = useRef<HTMLLabelElement>(null);
    const kistRef = useRef<HTMLTextAreaElement>(null);
    const qaRef = useRef<HTMLTextAreaElement>(null);
    const imageGenRef = useRef<HTMLTextAreaElement>(null);
    const imageLinkRef = useRef<HTMLInputElement>(null);
    const colorRefs = useRef<(HTMLButtonElement | null)[]>([]);

    const handleNumberSync = (val: string, setter: (val: number) => void, min: number, max: number) => {
        const num = parseInt(val);
        if (isNaN(num)) return;
        if (num >= min && num <= max) setter(num);
    };

    const handleSearch = useCallback(async () => {
        const imagePayload: ImagePayload = { mode: imageMode };
        if (imageMode === 'link' && imageLink.trim()) {
            imagePayload.image_url = imageLink.trim();
        } else if (imageMode === 'generate' && imagePrompt.trim()) {
            imagePayload.image_prompt = imagePrompt.trim();
        } else if (imageMode === 'upload' && fileInputRef.current?.files?.[0]) {
            const file = fileInputRef.current.files[0];
            const b64 = await new Promise<string>((resolve) => {
                const reader = new FileReader();
                reader.onload = () => resolve((reader.result as string).split(',')[1]);
                reader.readAsDataURL(file);
            });
            imagePayload.image_b64 = b64;
        }

        const text_queries: ExtendedTextSearchPayload[] = [];

        if (selectedModel === "KIST") {
            (Object.keys(kistQueries) as KistMode[]).forEach((mode) => {
                if (kistQueries[mode].trim()) {
                    text_queries.push({
                        query: kistQueries[mode].trim(),
                        mode: mode,
                        top_k: ragTopN
                    });
                }
            });
        } else if (selectedModel === "QA" && qaQuery.trim()) {
            text_queries.push({ 
                query: qaQuery.trim(), 
                mode: 'qa', 
                top_k: ragTopN 
            });
        }

        const selectedColorObj = COLOR_OPTIONS.find(c => c.id === selectedColor);
        const colorHex = selectedColorObj && selectedColorObj.colorHex !== 'transparent' ? selectedColorObj.colorHex : undefined;

        const payload = {
            text_queries,
            image: imagePayload,
            colorHex,
            config: {
                ragTopN,
                rerankTopM,
                tradSearchParam,
                enableExpansion,
                model: selectedModel,
                topK: ragTopN,
            },
        };

        onSearch(payload as unknown as SearchPayload);

        setKistQueries({
            visual: "", hybrid: "", caption: "", ocr: "", asr: "", object: "", temporal: ""
        });
        setQaQuery("");
        setImageLink("");
        setImagePrompt("");
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
    }, [
        selectedModel, kistQueries, qaQuery,
        imageMode, imageLink, imagePrompt, 
        ragTopN, rerankTopM, tradSearchParam, enableExpansion, 
        selectedColor, onSearch
    ]);

    const handleSearchRef = useRef(handleSearch);
    useEffect(() => { handleSearchRef.current = handleSearch; }, [handleSearch]);

    useEffect(() => {
        const handleGlobalKeyDown = (e: KeyboardEvent) => {
            if (isModeSelectorOpen) {
                if (e.key === 'ArrowUp') {
                    e.preventDefault();
                    setHighlightedModeIndex(prev => (prev - 1 + KIST_MODES.length) % KIST_MODES.length);
                    return;
                } else if (e.key === 'ArrowDown') {
                    e.preventDefault();
                    setHighlightedModeIndex(prev => (prev + 1) % KIST_MODES.length);
                    return;
                } else if (e.key === 'Enter') {
                    e.preventDefault();
                    setKistMode(KIST_MODES[highlightedModeIndex]);
                    setIsModeSelectorOpen(false);
                    return;
                } else if (e.key === 'Escape') {
                    e.preventDefault();
                    setIsModeSelectorOpen(false);
                    return;
                }
            }

            if (e.ctrlKey && e.code === 'KeyQ') {
                e.preventDefault();
                if (selectedModel === "KIST") {
                    setIsModeSelectorOpen(prev => {
                        if (!prev) setHighlightedModeIndex(KIST_MODES.indexOf(kistMode));
                        return !prev;
                    });
                }
                return;
            }

            if (e.ctrlKey || e.metaKey) {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    handleSearchRef.current();
                } else if (e.key.toLowerCase() === 'i') {
                    e.preventDefault();
                    setImageMode('upload');
                    setTimeout(() => uploadZoneRef.current?.focus(), 0);
                } else if (e.key.toLowerCase() === 'l') {
                    e.preventDefault();
                    setImageMode('link');
                    setTimeout(() => imageLinkRef.current?.focus(), 0);
                }
            }
        };

        window.addEventListener('keydown', handleGlobalKeyDown);
        return () => window.removeEventListener('keydown', handleGlobalKeyDown);
    }, [isModeSelectorOpen, highlightedModeIndex, kistMode, selectedModel]);

    const handleEnterNavigation = (
        e: React.KeyboardEvent,
        currentBox: 'kist' | 'qa' | 'imageLink' | 'imageGen' | 'uploadZone'
    ) => {
        if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey && !e.metaKey) {
            e.preventDefault();

            if (currentBox === 'kist' && !isModeSelectorOpen) {
                setImageMode('upload');
                setTimeout(() => uploadZoneRef.current?.focus(), 0);
            } else if (currentBox === 'qa') {
                handleSearchRef.current(); 
            } else if (currentBox === 'uploadZone') {
                setImageMode('link');
                setTimeout(() => imageLinkRef.current?.focus(), 0);
            } else if (currentBox === 'imageLink') {
                setImageMode('generate');
                setTimeout(() => imageGenRef.current?.focus(), 0);
            } else if (['imageGen', 'imageLink', 'uploadZone'].includes(currentBox)) {
                const selectedIdx = Math.max(0, COLOR_OPTIONS.findIndex(c => c.id === selectedColor));
                colorRefs.current[selectedIdx]?.focus();
            }
        } else if (e.key === 'Enter' && e.shiftKey) {
            e.preventDefault();
            if (currentBox === 'uploadZone') {
                setTimeout(() => kistRef.current?.focus(), 0);
            } else if (currentBox === 'imageLink') {
                setImageMode('upload');
                setTimeout(() => uploadZoneRef.current?.focus(), 0);
            } else if (currentBox === 'imageGen') {
                setImageMode('link');
                setTimeout(() => imageLinkRef.current?.focus(), 0);
            } else if (['uploadZone', 'imageLink', 'imageGen'].includes(currentBox)) {
                kistRef.current?.focus();
            }
        } else if (currentBox === 'uploadZone' && e.ctrlKey && e.key === 'Enter') {
            fileInputRef.current?.click();
        }
    };

    const handleColorKeyDown = (e: React.KeyboardEvent, index: number) => {
        if (e.key === 'Enter' && e.shiftKey) {
            e.preventDefault();
            setImageMode('generate');
            setTimeout(() => imageGenRef.current?.focus(), 0);
        } else if (e.key === 'ArrowRight') {
            e.preventDefault();
            const next = (index + 1) % COLOR_OPTIONS.length;
            setSelectedColor(COLOR_OPTIONS[next].id);
            colorRefs.current[next]?.focus();
        } else if (e.key === 'ArrowLeft') {
            e.preventDefault();
            const prev = (index - 1 + COLOR_OPTIONS.length) % COLOR_OPTIONS.length;
            setSelectedColor(COLOR_OPTIONS[prev].id);
            colorRefs.current[prev]?.focus();
        }
    };

    if (!isExpanded) {
        return (
            <div className="fixed left-4 top-1/2 -translate-y-1/2">
                <button onClick={() => setIsExpanded(true)} className="p-3 bg-blue-600 text-white rounded-full shadow-lg">
                    <Search className="w-5 h-5" />
                </button>
            </div>
        );
    }

    return (
        <aside className="w-72 min-w-[17rem] h-full flex-shrink-0 border-r border-zinc-200 dark:border-zinc-800 bg-slate-50 dark:bg-zinc-900/40 overflow-y flex flex-col">
            <div className="w-full h-screen bg-black/5 dark:bg-[#121212] border-r border-black/10 dark:border-white/10 flex flex-col relative">
                <button onClick={() => setIsExpanded(false)} className="absolute -right-3 top-6 bg-white dark:bg-gray-800 border border-black/10 p-1 rounded-full shadow-sm z-10">
                    <ChevronLeft className="w-4 h-4 text-black/70 dark:text-white/70" />
                </button>

                <div className="flex-1 flex flex-col overflow-hidden px-2 pt-5 pb-3">
                    <div className="flex items-center justify-between shrink-0 mb-4 px-2">
                        <h2 className="text-lg font-bold dark:text-white">Retrieval Engine</h2>
                        <ModelDropdownMenu selectedModel={selectedModel} onSelect={setSelectedModel} />
                    </div>

                    {/* --- KIST MODE UI --- */}
                    {selectedModel === "KIST" && (
                        <div className="flex flex-col gap-3 shrink-0 mb-5">
                            <div className="bg-white dark:bg-white/5 rounded-xl border border-black/10 dark:border-white/10 p-2 focus-within:border-blue-500 transition-colors">
                                
                                {/* KIST Mode Selector */}
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
                                            {/* Overlay đóng menu khi click ra ngoài */}
                                            <div className="fixed inset-0 z-40" onClick={() => setIsModeSelectorOpen(false)} />
                                            <div className="absolute top-full right-0 mt-1 w-32 bg-white dark:bg-[#1a1a1a] border border-black/10 dark:border-white/10 rounded-md shadow-xl z-50 py-1 flex flex-col">
                                                {KIST_MODES.map((mode, idx) => (
                                                    <div
                                                        key={mode}
                                                        onClick={() => {
                                                            setKistMode(mode);
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
                                    onKeyDown={e => handleEnterNavigation(e, 'kist')}
                                    placeholder={`Describe context or object for ${kistMode}...`}
                                    className="w-full bg-transparent border-none text-sm resize-none focus-visible:ring-0 dark:text-white min-h-[50px]"
                                />
                            </div>

                            {/* Image Add-on (KIST) */}
                            <div className="bg-white dark:bg-white/5 rounded-xl border border-black/10 dark:border-white/10 p-3 flex flex-col gap-3 shrink-0">
                                <div className="flex items-center gap-1 p-1 bg-black/5 dark:bg-black/40 rounded-lg">
                                    {[
                                        { id: 'upload', icon: Upload, label: 'Upload' },
                                        { id: 'link', icon: LinkIcon, label: 'Link' },
                                        { id: 'generate', icon: Wand2, label: 'Generate' }
                                    ].map(mode => (
                                        <button
                                            key={mode.id}
                                            onClick={() => setImageMode(mode.id as ImageInputMode)}
                                            className={cn(
                                                "flex-1 flex items-center justify-center gap-1.5 text-xs py-1.5 rounded-md transition-all",
                                                imageMode === mode.id ? "bg-white dark:bg-gray-700 shadow-sm text-blue-600 dark:text-blue-400 font-medium" : "text-gray-500 hover:text-gray-900 dark:hover:text-gray-300"
                                            )}
                                        >
                                            <mode.icon className="w-3 h-3" /> {mode.label}
                                        </button>
                                    ))}
                                </div>

                                <div className="mt-1">
                                    {imageMode === 'upload' && (
                                        <label
                                            ref={uploadZoneRef}
                                            tabIndex={0}
                                            onKeyDown={e => handleEnterNavigation(e, 'uploadZone')}
                                            className="flex w-full border border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-4 text-center cursor-pointer hover:bg-gray-50 dark:hover:bg-white/5 focus:outline-none focus:ring-1 focus:ring-blue-500"
                                        >
                                            <input ref={fileInputRef} type="file" accept="image/*" className="hidden" />
                                            <span className="text-sm text-gray-500 w-full">Drag &amp; drop or click to upload</span>
                                        </label>
                                    )}
                                    {imageMode === 'link' && (
                                        <input
                                            ref={imageLinkRef}
                                            type="text"
                                            value={imageLink}
                                            onChange={e => setImageLink(e.target.value)}
                                            onKeyDown={e => handleEnterNavigation(e, 'imageLink')}
                                            placeholder="Paste image URL here..."
                                            className="w-full text-sm p-2 rounded-lg bg-black/5 dark:bg-black/40 border-none focus:ring-1 focus:ring-blue-500 dark:text-white"
                                        />
                                    )}
                                    {imageMode === 'generate' && (
                                        <Textarea
                                            ref={imageGenRef}
                                            value={imagePrompt}
                                            onChange={e => setImagePrompt(e.target.value)}
                                            onKeyDown={e => handleEnterNavigation(e, 'imageGen')}
                                            placeholder="Prompt for image generation..."
                                            className="w-full bg-black/5 dark:bg-black/40 border-none text-sm rounded-lg p-2 focus-visible:ring-1 focus-visible:ring-blue-500 dark:text-white"
                                        />
                                    )}
                                </div>

                                <div className="flex items-center justify-between border-t border-black/10 dark:border-white/10 pt-3">
                                    <span className="text-xs text-gray-500 flex items-center gap-1"><Palette className="w-3 h-3" />Color</span>
                                    <div className="flex gap-1.5">
                                        {COLOR_OPTIONS.map((c, index) => (
                                            <button
                                                key={c.id}
                                                ref={el => { colorRefs.current[index] = el; }}
                                                onClick={() => setSelectedColor(c.id)}
                                                onKeyDown={e => handleColorKeyDown(e, index)}
                                                title={c.id}
                                                className={cn(
                                                    "w-5 h-5 rounded-full transition-transform focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-[#121212]",
                                                    c.colorHex !== 'transparent' && `bg-[${c.colorHex}]`,
                                                    c.border,
                                                    selectedColor === c.id ? "scale-125 ring-2 ring-blue-500 ring-offset-1 dark:ring-offset-gray-900" : "hover:scale-110"
                                                )}
                                                style={c.colorHex !== 'transparent' ? { backgroundColor: c.colorHex } : {}}
                                            />
                                        ))}
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* --- QA MODE UI --- */}
                    {selectedModel === "QA" && (
                        <div className="flex flex-col gap-3 shrink-0 mb-5">
                            <div className="bg-white dark:bg-white/5 rounded-xl border border-black/10 dark:border-white/10 p-2 focus-within:border-blue-500 transition-colors">
                                <span className="text-xs font-semibold text-black/50 dark:text-white/50 flex items-center gap-1 mb-1 px-2">
                                    <HelpCircle className="w-3 h-3" /> Question
                                </span>
                                <Textarea
                                    ref={qaRef}
                                    value={qaQuery}
                                    onChange={e => setQaQuery(e.target.value)}
                                    onKeyDown={e => handleEnterNavigation(e, 'qa')}
                                    placeholder="Enter your question..."
                                    className="w-full bg-transparent border-none text-sm resize-none focus-visible:ring-0 dark:text-white min-h-[70px]"
                                />
                            </div>
                        </div>
                    )}

                    <div className="flex-1 flex flex-col border border-black/10 dark:border-white/10 rounded-xl bg-white dark:bg-white/5 overflow-hidden min-h-0">
                        <button onClick={() => setShowConfig(!showConfig)} className="shrink-0 flex items-center justify-between w-full p-3 text-sm font-semibold dark:text-white bg-black/5 dark:bg-black/40 hover:bg-black/10 dark:hover:bg-black/60 transition-colors">
                            <div className="flex items-center gap-2"><Settings2 className="w-4 h-4" /> Tuning Parameters</div>
                            <ChevronRight className={cn("w-4 h-4 transition-transform", showConfig && "rotate-90")} />
                        </button>

                        {showConfig && (
                            <div className="flex-1 p-4 flex flex-col overflow-y-auto overflow-x-hidden gap-5 text-sm">
                                <div className="flex items-center justify-between">
                                    <span className="font-medium dark:text-white/80">Query Expansion</span>
                                    <input
                                        type="checkbox"
                                        checked={enableExpansion}
                                        onChange={e => setEnableExpansion(e.target.checked)}
                                        className="w-4 h-4 accent-blue-600"
                                    />
                                </div>
                                {[
                                    { label: "Top N (RAG)", value: ragTopN, setter: setRagTopN, min: 10, max: 1000 },
                                    { label: "Top M (Rerank)", value: rerankTopM, setter: setRerankTopM, min: 5, max: 200 },
                                    { label: "Trad Search Weight", value: tradSearchParam, setter: setTradSearchParam, min: 0, max: 100 }
                                ].map((param, idx) => (
                                    <div key={idx} className="flex flex-col gap-2">
                                        <div className="flex items-center justify-between">
                                            <label className="text-xs text-gray-500 dark:text-gray-400 font-medium">{param.label}</label>
                                            <input
                                                type="number"
                                                value={param.value}
                                                onChange={(e) => handleNumberSync(e.target.value, param.setter, param.min, param.max)}
                                                className="w-16 p-1 text-right text-xs bg-black/5 dark:bg-black/40 rounded border-none focus:ring-1 focus:ring-blue-500 dark:text-white hide-arrows"
                                            />
                                        </div>
                                        <input
                                            type="range"
                                            min={param.min}
                                            max={param.max}
                                            value={param.value}
                                            onChange={(e) => param.setter(parseInt(e.target.value))}
                                            className="w-full accent-blue-600"
                                        />
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>

                <div className="shrink-0 p-4 bg-white/80 dark:bg-[#121212]/80 backdrop-blur-md border-t border-black/10 dark:border-white/10">
                    <button
                        onClick={handleSearch}
                        className="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl shadow-md font-medium transition-all flex items-center justify-center gap-2"
                    >
                        Execute Search <ArrowRight className="w-4 h-4" />
                    </button>
                </div>
            </div>
        </aside>
    );
}