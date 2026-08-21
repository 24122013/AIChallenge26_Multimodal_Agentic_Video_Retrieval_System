"use client";

import { ArrowRight, Search, ChevronLeft } from "lucide-react";
import React, { useCallback, useState, useRef, useEffect } from "react";
import { type ImageInputMode, type ColorScheme, COLOR_OPTIONS } from '../../constants/image-input';
import type { TextSearchPayload, SearchPayload, ImagePayload } from '../../types';
import ModelDropdownMenu from "../ModelsDropdownMenu";
import { type KistMode, KIST_MODES } from '../../constants/mode-icons';

import ConfigPanel from './ConfigPanel';
import ImageAddon from './ImageAddon';
import QueryInputExpand from "./QueryInputExpand";
import QueryInputPrimary from "./QueryInputPrimary";
import type { NavBoxType } from './types';

interface SearchSideBarProps {
    onSearch: (payload: SearchPayload) => void;
    isExpanded: boolean;
    setIsExpanded: (expanded: boolean) => void;
}

export default function SearchSideBar({ onSearch, isExpanded, setIsExpanded }: SearchSideBarProps) {
    const [selectedModel, setSelectedModel] = useState<string>("KIST");

    const [kistQueries, setKistQueries] = useState<Record<KistMode, string>>({
        visual: "", hybrid: "", caption: "", ocr: "", asr: "", object: ""
    });
    const [kistMode, setKistMode] = useState<KistMode>("visual");
    const [qaQuery, setQaQuery] = useState("");
    const [trakeQuery, setTrakeQuery] = useState("");
    const [temporalQuery, setTemporalQuery] = useState("");
    const [expandedQueries, setExpandedQueries] = useState<string[]>([]);
    
    const [isModeSelectorOpen, setIsModeSelectorOpen] = useState(false);
    const [highlightedModeIndex, setHighlightedModeIndex] = useState(0);

    const [imageMode, setImageMode] = useState<ImageInputMode>('upload');
    const [imageLink, setImageLink] = useState("");
    const [imagePrompt, setImagePrompt] = useState("");
    const [selectedColor, setSelectedColor] = useState<ColorScheme>('none');
    const [topK, setTopK] = useState<number>(100);

    const fileInputRef = useRef<HTMLInputElement>(null);
    const uploadZoneRef = useRef<HTMLLabelElement>(null);
    const kistRef = useRef<HTMLTextAreaElement>(null);
    const qaRef = useRef<HTMLTextAreaElement>(null);
    const trakeRef = useRef<HTMLTextAreaElement>(null);
    const temporalRef = useRef<HTMLTextAreaElement>(null);
    const expandedRefs = useRef<(HTMLTextAreaElement | null)[]>([]);
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
        if (selectedModel === "KIST") {
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
        }

        const validExpanded = expandedQueries.filter(q => q.trim().length > 0);
        const text_queries: TextSearchPayload[] = [];

        if (selectedModel === "KIST") {
            (Object.keys(kistQueries) as KistMode[]).forEach((mode) => {
                if (kistQueries[mode].trim()) {
                    text_queries.push({
                        query: kistQueries[mode].trim(),
                        mode: mode,
                        top_k: topK,
                        expanded_queries: validExpanded.length > 0 ? validExpanded : undefined
                    });
                }
            });
        } else if (selectedModel === "QA" && qaQuery.trim()) {
            text_queries.push({ 
                query: qaQuery.trim(), 
                mode: 'qa', 
                top_k: topK,
                expanded_queries: validExpanded.length > 0 ? validExpanded : undefined
            });
        } else if (selectedModel === "TRAKE" && trakeQuery.trim()) {
            text_queries.push({
                query: trakeQuery.trim(),
                mode: 'trake',
                top_k: topK,
                expanded_queries: validExpanded.length > 0 ? validExpanded : undefined
            });
        } else if (selectedModel === "TEMPORAL" && temporalQuery.trim()) {
            text_queries.push({
                query: temporalQuery.trim(),
                mode: 'temporal',
                top_k: topK
            });
        }

        const selectedColorObj = COLOR_OPTIONS.find(c => c.id === selectedColor);
        const colorHex = selectedColorObj && selectedColorObj.colorHex !== 'transparent' ? selectedColorObj.colorHex : undefined;

        const payload = {
            text_queries,
            image: selectedModel === "KIST" ? imagePayload : undefined,
            colorHex: selectedModel === "KIST" ? colorHex : undefined,
            config: {
                topK,
                model: selectedModel,
            },
        };

        onSearch(payload as unknown as SearchPayload);

        setKistQueries({ visual: "", hybrid: "", caption: "", ocr: "", asr: "", object: "" });
        setQaQuery("");
        setTrakeQuery("");
        setTemporalQuery("");
        setExpandedQueries([]);
        setImageLink("");
        setImagePrompt("");
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
    }, [
        selectedModel, kistQueries, qaQuery, trakeQuery, temporalQuery, expandedQueries,
        imageMode, imageLink, imagePrompt, 
        topK, selectedColor, onSearch
    ]);

    const handleSearchRef = useRef(handleSearch);
    useEffect(() => { handleSearchRef.current = handleSearch; }, [handleSearch]);

    // Derived reference to access current length reliably inside keydown
    const expandedLenRef = useRef(expandedQueries.length);
    useEffect(() => { expandedLenRef.current = expandedQueries.length; }, [expandedQueries]);

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

            if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'e') {
                e.preventDefault();
                if (selectedModel !== "TEMPORAL") {
                    setExpandedQueries(prev => [...prev, ""]);
                    setTimeout(() => {
                        expandedRefs.current[expandedLenRef.current]?.focus();
                    }, 50);
                }
                return;
            }

            if (e.ctrlKey || e.metaKey) {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    handleSearchRef.current();
                } else if (selectedModel === "KIST") {
                    if (e.key.toLowerCase() === 'i') {
                        e.preventDefault();
                        setImageMode('upload');
                        setTimeout(() => uploadZoneRef.current?.focus(), 0);
                    } else if (e.key.toLowerCase() === 'l') {
                        e.preventDefault();
                        setImageMode('link');
                        setTimeout(() => imageLinkRef.current?.focus(), 0);
                    }
                }
            }
        };

        window.addEventListener('keydown', handleGlobalKeyDown);
        return () => window.removeEventListener('keydown', handleGlobalKeyDown);
    }, [isModeSelectorOpen, highlightedModeIndex, kistMode, selectedModel]);

    const focusMainQuery = () => {
        if (selectedModel === "KIST") kistRef.current?.focus();
        else if (selectedModel === "QA") qaRef.current?.focus();
        else if (selectedModel === "TRAKE") trakeRef.current?.focus();
        else if (selectedModel === "TEMPORAL") temporalRef.current?.focus();
    };

    const goToNextSection = () => {
        if (selectedModel === "KIST") {
            setImageMode('upload');
            setTimeout(() => uploadZoneRef.current?.focus(), 0);
        } else {
            handleSearchRef.current(); 
        }
    };

    const handleEnterNavigation = (
        e: React.KeyboardEvent,
        currentBox: NavBoxType,
        extendedIndex?: number
    ) => {
        if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey && !e.metaKey) {
            e.preventDefault();

            if (currentBox === 'main_query' && !isModeSelectorOpen) {
                if (expandedQueries.length > 0) {
                    expandedRefs.current[0]?.focus();
                } else {
                    goToNextSection();
                }
            } else if (currentBox === 'extended' && typeof extendedIndex === 'number') {
                if (extendedIndex < expandedQueries.length - 1) {
                    expandedRefs.current[extendedIndex + 1]?.focus();
                } else {
                    goToNextSection();
                }
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
            if (currentBox === 'extended' && typeof extendedIndex === 'number') {
                if (extendedIndex > 0) {
                    expandedRefs.current[extendedIndex - 1]?.focus();
                } else {
                    focusMainQuery();
                }
            } else if (currentBox === 'uploadZone') {
                if (expandedQueries.length > 0) expandedRefs.current[expandedQueries.length - 1]?.focus();
                else focusMainQuery();
            } else if (currentBox === 'imageLink') {
                setImageMode('upload');
                setTimeout(() => uploadZoneRef.current?.focus(), 0);
            } else if (currentBox === 'imageGen') {
                setImageMode('link');
                setTimeout(() => imageLinkRef.current?.focus(), 0);
            } else if (['uploadZone', 'imageLink', 'imageGen'].includes(currentBox)) {
                if (expandedQueries.length > 0) expandedRefs.current[expandedQueries.length - 1]?.focus();
                else focusMainQuery();
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
            <div className="fixed z-[50] left-4 top-1/2 -translate-y-1/2">
                <button onClick={() => setIsExpanded(true)} className="p-3 bg-blue-600 text-white rounded-full shadow-lg">
                    <Search className="w-5 h-5" />
                </button>
            </div>
        );
    }

    return (
        <aside className="w-72 min-w-[17rem] h-full flex-shrink-0 z-[30] border-r border-zinc-200 dark:border-zinc-800 bg-slate-50 dark:bg-zinc-900/40 overflow-y-auto flex flex-col">
            <div className="w-full h-full bg-black/5 dark:bg-[#121212] border-r border-black/10 dark:border-white/10 flex flex-col relative">
                <button onClick={() => setIsExpanded(false)} className="absolute z-100 -right-3 top-6 bg-white dark:bg-gray-800 border border-black/10 p-1 rounded-full shadow-sm">
                    <ChevronLeft className="w-4 h-4 text-black/70 dark:text-white/70" />
                </button>

                <div className="flex-1 flex flex-col px-2 pt-5 pb-3">
                    <div className="flex items-center justify-between shrink-0 mb-4 px-2">
                        <h2 className="text-lg font-bold dark:text-white">Retrieval Engine</h2>
                        <ModelDropdownMenu 
                            selectedModel={selectedModel} 
                            onSelect={(model) => {
                                setSelectedModel(model);
                                setExpandedQueries([]);
                            }} 
                        />
                    </div>

                    <div className="flex flex-col gap-3 shrink-0 mb-5">
                        <QueryInputPrimary
                            selectedModel={selectedModel}
                            kistQueries={kistQueries} setKistQueries={setKistQueries}
                            kistMode={kistMode} setKistMode={setKistMode}
                            qaQuery={qaQuery} setQaQuery={setQaQuery}
                            trakeQuery={trakeQuery} setTrakeQuery={setTrakeQuery}
                            temporalQuery={temporalQuery} setTemporalQuery={setTemporalQuery}
                            isModeSelectorOpen={isModeSelectorOpen} setIsModeSelectorOpen={setIsModeSelectorOpen}
                            highlightedModeIndex={highlightedModeIndex} setHighlightedModeIndex={setHighlightedModeIndex}
                            kistRef={kistRef} qaRef={qaRef} trakeRef={trakeRef} temporalRef={temporalRef}
                            handleEnterNavigation={handleEnterNavigation}
                        >
                            {selectedModel !== "TEMPORAL" && (
                                <QueryInputExpand 
                                    expandedQueries={expandedQueries}
                                    setExpandedQueries={setExpandedQueries}
                                    expandedRefs={expandedRefs}
                                    handleEnterNavigation={handleEnterNavigation}
                                />
                            )}
                        </QueryInputPrimary>

                        {selectedModel === "KIST" && (
                            <ImageAddon 
                                imageMode={imageMode} setImageMode={setImageMode}
                                imageLink={imageLink} setImageLink={setImageLink}
                                imagePrompt={imagePrompt} setImagePrompt={setImagePrompt}
                                selectedColor={selectedColor} setSelectedColor={setSelectedColor}
                                uploadZoneRef={uploadZoneRef} fileInputRef={fileInputRef}
                                imageLinkRef={imageLinkRef} imageGenRef={imageGenRef}
                                colorRefs={colorRefs}
                                handleEnterNavigation={handleEnterNavigation}
                                handleColorKeyDown={handleColorKeyDown}
                            />
                        )}
                    </div>

                    <ConfigPanel 
                        topK={topK} 
                        setTopK={setTopK} 
                        handleNumberSync={handleNumberSync} 
                    />
                </div>

                <div className="shrink-0 p-4 bg-white/80 dark:bg-[#121212]/80 backdrop-blur-md border-t border-black/10 dark:border-white/10 sticky bottom-0">
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