import React from 'react';
import { Upload, Link as LinkIcon, Wand2, Palette } from "lucide-react";
import { type ImageInputMode, type ColorScheme, COLOR_OPTIONS } from '../../constants/image-input';
import { cn } from "../../libs/utils";
import Textarea from '../ui/textarea';
import type { NavBoxType } from './types';

interface ImageAddonProps {
    imageMode: ImageInputMode;
    setImageMode: (mode: ImageInputMode) => void;
    imageLink: string;
    setImageLink: (link: string) => void;
    imagePrompt: string;
    setImagePrompt: (prompt: string) => void;
    selectedColor: ColorScheme;
    setSelectedColor: (color: ColorScheme) => void;
    uploadZoneRef: React.RefObject<HTMLLabelElement | null>;
    fileInputRef: React.RefObject<HTMLInputElement | null>;
    imageLinkRef: React.RefObject<HTMLInputElement | null>;
    imageGenRef: React.RefObject<HTMLTextAreaElement | null>;
    colorRefs: React.RefObject<(HTMLButtonElement | null)[]>;
    handleEnterNavigation: (e: React.KeyboardEvent, currentBox: NavBoxType, index?: number) => void;
    handleColorKeyDown: (e: React.KeyboardEvent, index: number) => void;
}

export default function ImageAddon({
    imageMode, setImageMode,
    imageLink, setImageLink,
    imagePrompt, setImagePrompt,
    selectedColor, setSelectedColor,
    uploadZoneRef, fileInputRef, imageLinkRef, imageGenRef, colorRefs,
    handleEnterNavigation, handleColorKeyDown
}: ImageAddonProps) {
    return (
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
                        <span className="text-sm text-gray-500 w-full">Drag &amp; drop or click</span>
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
                        className="w-full text-sm p-2 rounded-lg bg-black/5 dark:bg-black/40 border-none focus:ring-1 focus:ring-blue-500 dark:text-white outline-none"
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
    );
}