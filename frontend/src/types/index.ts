export type SearchSource = 'clip' | 'ocr' | 'image' | 'color' | 'rerank' | (string & {});

export type ActiveTask = "KIST" | "QA" | "TRAKE" | "TEMPORAL";

export interface SearchResultNeighbor {
    video_id: string;
    frame_id: string;
    timestamp: number;
    shot_id: string;
    segment_id: string;
    faiss_index: number;
    frame_index: number;
    keyframe_path: string;
    thumbnail_path: string;
}

export interface NeighborFrame {
    frame_id: string;
    delta_seconds: number;
    timestamp?: number; // optional; needed for the timestamp visualization
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

    video_url?: string;
    source?: SearchSource;
    answer?: string;
    context_sources?: unknown[];
}

export type SortKey = 'arrival' | 'clip_score' | 'ocr_score' | 'image_score' | 'color_score' | 'rerank_score' | (string & {});

export interface SearchConfig {
    model: string;
    topK: number;
}

export type ImageInputMode = 'upload' | 'link' | 'generate';

export interface ImagePayload {
    mode: ImageInputMode;
    image_url?: string | null;
    image_b64?: string | null;
    image_prompt?: string | null;
}

export interface TextSearchPayload {
    query: string;
    mode: "visual" | "kis_visual" | "hybrid" | "caption" | "ocr" | "object" | "kis_temporal" | "qa" | "trake" | "temporal";
    top_k: number;
    expanded_queries?: string[];
}

export interface SearchPayload {
    text_queries: TextSearchPayload[];
    image?: ImagePayload;
    colorHex?: string;
    config: SearchConfig;
}

export interface SearchLog {
    id: string;
    taskId: string;
    taskType: string;
    searchMode: string;
    query: string;
    payload: SearchPayload;
    resultsCount: number;
    latency: number;
    solveTime: number;
    submissionsCount: number;
    clickedCandidates: Array<{ id: string; timestamp: number }>;
    submittedCandidates: Array<{ id: string; timestamp: number }>;
    correctness: boolean | null;
    methods: string[];
    results: Array<{ id: string; name: string; score: number }>;
}

// --- NEW API RESPONSE TYPES ---

export interface ApiResponse<T> {
    success: boolean;
    data: T;
    message: string | null;
}

export interface VisualSearchData {
    query: string;
    top_k: number;
    latency_ms: number;
    task?: "visual";
    results: VisualSearchResult[];
}

export interface KistData {
    schema_version: string;
    query: string;
    task: "kis" | "avs";
    top_k: number;
    latency_ms: number;
    candidates: Candidate[];
    requested_task?: "kis" | "kis_visual" | "kis_temporal" | "avs";
    query_plan?: QueryPlan;
    routing_trace?: Record<string, unknown>;
}

export interface VisualSearchResult {
    video_id: string;
    frame_id: string;
    timestamp: number;
    score: number;
    segment_id: string;
    shot_id: string;
    faiss_index: number | null;
    frame_index: number;
    keyframe_path: string;
    thumbnail_path: string;
    timestamp_source: string;
    timestamp_confidence: number;
    caption: string;
    ocr_text: string;
    objects: string[];
    modality_scores: Record<string, number>;
    neighbors: SearchResultNeighbor[];
}

export interface TrakeData {
    schema_version: string;
    query: string;
    task: "trake";
    top_k: number;
    latency_ms: number;
    
    // Output TRAKE:
    hypotheses: TrakeHypothesis[];
    status: "ok" | "insufficient_support" | "timeout" | string;
    warnings: string[];
    event_plan?: Record<string, unknown>;
    trace?: {
        status?: string;
        warnings?: string[];
        latency?: Record<string, number>;
        preflight?: Record<string, unknown>;
    };
}

export interface TrakeHypothesis {
    rank: number;
    video_id: string;
    frame_ids: number[];
    score: number;
    score_breakdown: Record<string, unknown>;
    path_id: string;
    events: CandidateEvent[];
    lineage: Array<Record<string, unknown>>;
    warnings: string[];
}

export interface QAData {
    schema_version: string;
    query: string;
    task: "qa";
    top_k: number;
    latency_ms: number;
    
    // Output QA:
    answer: QAAnswer;
    evidence: QAEvidence[];
}

export interface TemporalData {
    schema_version: string;
    query: string;
    task: "temporal"; 
    top_k: number;
    latency_ms: number;
    
    // Output Temporal:
    evidence: QAEvidence[];
    candidates: Candidate[];
    temporal_matches: TemporalMatch[]; 
    routing_trace: RoutingTrace;
}

export interface QueryPlan {
    original_query: string;
    normalized_query: string;
    retrieval_query: string;
    requested_profile: string;
    profile: string;
    profile_source: string;
    temporal_relation: string;
    temporal_events: string[];
    temporal_event_ids: string[];
    temporal_cues: string[];
    modality_hints: string[];
    modality_scope: Array<"visual" | "caption" | "ocr" | "objects">;
    quoted_phrases: string[];
    expansions: string[];
    expansion_plan: ExpansionPlan;
    modality_queries: Record<string, string>;
    reasons: string[];
    task_mode: string;
    answer_type: string;
    retrieval_statement: string;
    known_constraints: Record<string, string[]>;
    constraint_roles: Record<string, Record<string, string>>;
    needs_temporal: boolean;
    answer_event_index: number | null;
    confidence: number;
}

export interface ExpansionPlan {
    original: string;
    variants: QueryVariant[];
    decomposition: QueryDecomposition;
    protected_literals: ProtectedLiterals;
    status: string;
    fallback_reason: string;
    provider_name: string;
    model_name: string;
    model_revision: string;
    prompt_revision: string;
    cache_hit: boolean;
    provider_paraphrases: string[];
    decomposition_rejections: string[];
}

export interface QueryVariant {
    text: string;
    type: string;
    weight: number;
    accepted: boolean;
    rejection_reason: string;
}

export interface QueryDecomposition {
    objects: string[];
    attributes: string[];
    actions: string[];
    relations: string[];
    ocr_literals: string[];
    scene_terms: string[];
}

export interface ProtectedLiterals {
    quoted: string[];
    ocr_literals: string[];
    numbers: string[];
    counts: string[];
    colors: string[];
    codes: string[];
    proper_names: string[];
    negations: string[];
    relations: string[];
}

export interface Candidate {
    video_id: string;
    keyframe_id: string;
    frame_id: string;
    timestamp: number;
    shot_id: string;
    segment_id: string;
    visual_score: number | null;
    caption_score: number | null;
    ocr_score: number | null;
    object_score: number | null;
    bge_score: number | null;
    fusion_score: number;
    rerank_score: number;
    score: number;
    neighbors: SearchResultNeighbor[];
    segment_context: unknown | null;
    frame_index: number;
    faiss_index: number | null;
    keyframe_path: string;
    thumbnail_path: string;
    caption: string;
    ocr_text: string;
    objects: string[];
    modality_scores: Record<string, number>;
    temporal: TemporalCandidateInfo | Record<string, never>; // empty {} and populated objects
    context_sources: unknown[];
    cses_selection?: CsesSelection | null;
    score_breakdown?: Record<string, number>;
    score_contributions?: Record<string, number>;

    timestamp_source?: string;
    timestamp_confidence?: number;

    events?: CandidateEvent[];
    rank?: number;
}

export interface FrameResult {
    video_id: string;
    frame_id: string;
    timestamp: number;
    score: number;
    segment_id: string;
    shot_id: string;
    faiss_index: number;
    frame_index: number;
    keyframe_path: string;
    thumbnail_path: string;
    timestamp_source: string;
    timestamp_confidence: number;
    caption: string;
    ocr_text: string;
    objects: string[];
    modality_scores: Record<string, number>;
    neighbors: SearchResultNeighbor[];
}

export interface CandidateEvent {
    event_index: number;
    normalized_score: number;
    rank: number;
    retrieval_query: string;
    warnings: string[];
    result: FrameResult;
}

export interface TemporalCandidateInfo {
    temporal_event_index: number;
    temporal_match_rank: number;
    temporal_match_mode: string;
    temporal_chain_id: string;
    temporal_event_query: string;
    temporal_event_role: string;
    temporal_chain_score: number;
}

export interface SearchContext {
    enabled: boolean;
    neighbors_enabled: boolean;
    segments_enabled: boolean;
    index: unknown | null;
}

export interface RoutingTrace {
    queries: TraceQuery[];
    modality_queries: TraceModalityQuery[];
    modality_hints: string[];
    weights: Record<string, number>;
    hint_boost: number;
    rrf_k: number;
    per_modality_limit: number;
    fusion_pool_size: number;
    rerank_pool_size: number;
    fallback_used: boolean;
    fallback_reasons: string[];
    reranker: string;
    constraint_rerank: ConstraintRerank;
    temporal_route: TemporalRoute;
    temporal_handoff: boolean;
    feature_flags: Record<string, boolean>;
}

export interface TraceQuery {
    query: string;
    source: string;
    event_index?: number; 
}

export interface TraceModalityQuery {
    query: string;
    modality: string;
    candidate_count: number;
}

export interface ConstraintRerank {
    per_event?: PerEventConstraint[]; 
    applied: boolean;
    status: string;
    weight: number;
    min_signal: number;
    max_signal: number;
    candidate_count?: number; 
    context_constraints?: Record<string, string[]>; 
}

export interface PerEventConstraint {
    applied: boolean;
    status: string;
    weight: number;
    min_signal: number;
}

export interface TemporalRoute {
    executed: boolean;
    event_queries: string[];
    event_count: number;
    external_expansions_ignored?: boolean;
    match_count: number;
    match_mode: string;
    warnings: string[];
    answer_eligible: boolean;
    reason: string | null;
    answer_event_index?: number;
    chain_id?: string;
    chain_score?: number;
    canonical_segment_context?: CanonicalSegmentContext;
}

export interface CanonicalSegmentContext {
    enabled: boolean;
    mapped_candidate_count: number;
}

export interface QAEvidence {
    evidence_id: string;
    video_id: string;
    frame_id: string;
    frame_index: number;
    shot_id: string;
    timestamp: number;
    image_path: string;
    caption: string;
    ocr_text: string;
    objects: string[];
    source_modalities: string[];
    retrieval_score: number;
    base_retrieval_score: number;
    constraint_score: number;
    matched_constraints: string[];
    temporal_event_index: number | null;
    temporal_match_rank: number | null;
    temporal_match_mode: string;
    temporal_chain_id: string;
    temporal_event_query: string;
    temporal_event_role: string;
    temporal_chain_score: number | null;
    warnings: string[];
}

export interface TemporalMatch {
    video_id: string;
    score: number;
    start_time: number;
    end_time: number;
    chain_id: string;
    match_mode: string;
    warnings: string[];
    events: TemporalMatchEvent[];
}

export interface TemporalMatchEvent {
    video_id: string;
    frame_id: string;
    timestamp: number;
    score: number;
    segment_id: string;
    shot_id: string;
    faiss_index: number | null;
    frame_index: number;
    keyframe_path: string;
    thumbnail_path: string;
    timestamp_source: string;
    timestamp_confidence: number;
    caption: string;
    ocr_text: string;
    objects: string[];
    modality_scores: Record<string, number>;
    neighbors: SearchResultNeighbor[];
}

export interface QAAnswer {
    status: "answered" | "insufficient_evidence" | "disabled" | "error";
    answer: string | null;
    answer_type: string;
    confidence: number;
    evidence_ids: string[];
    reason?: string | null;
}

export interface CsesSelection {
    row: number;
    selection_rank: number;
    selection_gain: number;
    relevance: number;
    visual_coverage_gain: number;
    temporal_coverage_gain: number;
    preserved_event_ids: string[];
    temporal_bin: number;
}

export type SearchData = VisualSearchData | KistData | QAData | TrakeData | TemporalData;

export type TelemetryDetails = Record<string, unknown>;
