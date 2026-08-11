// ── Data types for result grid ─────────────────────────────────────────────
export type SearchSource = 'clip' | 'ocr' | 'image' | 'color' | 'rerank' | (string & {});

// Add this to match your Python NeighborFrame
export interface NeighborFrame {
    frame_id: string;
    timestamp: number;
    keyframe_path?: string;
}
  
export interface VideoScene {
    video_id: string;
    frame_id: string;
    timestamp: number;
    score: number;
    
    segment_id: string;
    shot_id: string;
    
    faiss_index: number | null;
    frame_index: number | null;
    
    keyframe_path: string;
    thumbnail_path: string;
    
    timestamp_source: string;
    timestamp_confidence: number;
    
    caption: string;
    ocr_text: string;
    asr_text: string;
    objects: string[];
    
    modality_scores: Record<string, number>;
    neighbors: NeighborFrame[];

    // Note: Backend doesn't send this. Make it optional or construct it in useSearch
    video_url?: string;
    source?: SearchSource;
}

export type SortKey = 'arrival' | 'clip_score' | 'ocr_score' | 'image_score' | 'color_score' | 'rerank_score' | (string & {});

export interface ContextFrame {
    time: number;
    label: string;
    isActive?: boolean;
}

export interface TelemetryDetails {
    query_string?: string;
    results_returned?: number;
    frame_id?: string;
    video_name?: string;
    timestamp?: number;
    chosen_frame_id?: string;
    matching_query?: string;
}

// ── Search configuration mirrors ChatBox state ──────────────────────────────
export interface SearchConfig {
    ragTopN: number;
    rerankTopM: number;
    tradSearchParam: number;
    enableExpansion: boolean;
    model: string;
    topK: number;
}

// ── Image add-on payload ─────────────────────────────────────────────────────
export type ImageInputMode = 'upload' | 'link' | 'generate';

export interface ImagePayload {
    mode: ImageInputMode;
    image_url?: string | null;
    image_b64?: string | null;
    image_prompt?: string | null;
}

// ── Text Search Payload (Matches New Backend Dataclass) ──────────────────────
export interface TextSearchPayload {
    query: string;
    mode: string; // "visual", "hybrid", "caption", "ocr", "asr", "objects", "qa", "temporal"
    top_k: number;
}

// ── Full payload sent from ChatBox → App ─────────────────────────────────────
export interface SearchPayload {
    text_queries: TextSearchPayload[]; // Replaces clipQuery/ocrQuery
    image?: ImagePayload;
    colorHex?: string;
    config: SearchConfig;
}

export interface SearchLog {
    id: string;
    query: string;
    payload: SearchPayload; // TODO: Include config modification for Payload (e.g., topK)
    resultsCount: number;
    latency: number;
    submissionsCount: number;
    methods: string[];
    results: Array<{ id: string; name: string; score: number }>;
}