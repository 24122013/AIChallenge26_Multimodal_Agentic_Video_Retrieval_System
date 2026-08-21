import React from 'react';

export const MODES = ["KIST", "QA", "TRAKE", "TEMPORAL"];

export const MODE_LABELS: Record<string, string> = {
    KIST: "KIST",
    QA: "QA",
    TRAKE: "TRAKE",
    TEMPORAL: "Temporal Evidence",
};

export const MODE_ICONS: Record<string, React.ReactNode> = {
    "KIST": <span className="text-sm">🅺</span>,
    "QA": <span className="text-sm">🆀</span>,
    "TRAKE": <span className="text-sm">🆃</span>,
    "TEMPORAL": <span className="text-sm">⏱️</span>,
};

// These labels are UI retrieval profiles; visual and temporal map to canonical
// backend KIS routes while the diagnostic visual route remains API-only.
export const KIST_MODES = ["visual", "hybrid", "caption", "ocr", "object", "temporal"] as const;
export type KistMode = typeof KIST_MODES[number];

export const RETRIEVAL_METHODS: string[] = [
    "visual", "kis_visual", "hybrid", "caption", "ocr", "object", "kis_temporal",
    "qa", "trake", "temporal",
];
export type RetrievalMethod = "visual" | "kis_visual" | "hybrid" | "caption" | "ocr" | "object" | "kis_temporal" | "qa" | "trake" | "temporal";

export const METHOD_COLORS: Record<string, string> = {
    visual: 'bg-blue-100 text-blue-700 dark:bg-blue-950/40 dark:text-blue-400',
    kis_visual: 'bg-blue-100 text-blue-700 dark:bg-blue-950/40 dark:text-blue-400',
    hybrid: 'bg-fuchsia-100 text-fuchsia-700 dark:bg-fuchsia-950/40 dark:text-fuchsia-400',
    caption: 'bg-pink-100 text-pink-700 dark:bg-pink-950/40 dark:text-pink-400',
    ocr: 'bg-violet-100 text-violet-700 dark:bg-violet-950/40 dark:text-violet-400',
    object: 'bg-orange-100 text-orange-700 dark:bg-orange-950/40 dark:text-orange-400',
    kis_temporal: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-400',
    qa: 'bg-rose-100 text-rose-700 dark:bg-rose-950/40 dark:text-rose-400',
    trake: 'bg-teal-100 text-teal-700 dark:bg-teal-950/40 dark:text-teal-400',
    temporal: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-400',
};
