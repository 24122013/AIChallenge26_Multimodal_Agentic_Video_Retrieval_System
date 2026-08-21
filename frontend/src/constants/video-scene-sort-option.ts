import { type SortKey } from "../types"; 

const METHOD_LABELS: Record<string, string> = {
    'visual': 'Visual match',
    'caption': 'Caption relevance',
    'ocr': 'OCR relevance',
    'asr': 'ASR relevance',
    'objects': 'Object match',
    // 'hybrid': 'Hybrid match',
    // 'qa': 'QA score',
    // 'temporal': 'Temporal score',
    // 'rerank': 'Re-rank score',
};

export const SORT_OPTIONS: { key: SortKey; label: string }[] = [
    { key: 'arrival', label: 'Arrival order' },
    ...Object.entries(METHOD_LABELS).map(([key, label]) => ({
        key: key as SortKey,
        label
    }))
];