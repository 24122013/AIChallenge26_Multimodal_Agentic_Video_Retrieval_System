// doc: TRAKE -> Ranked `hypotheses`
export interface TrakeData {
    schema_version: string;
    query: string;
    task: string; // "trake"
    
    // Output TRAKE:
    hypotheses: Candidate[]; 

    /* 
      --- CÁC FIELD BỎ QUA TỪ BACKEND ĐỂ NHẸ PAYLOAD ---
      - event_plan
      - candidates
      - trace
      - latency_ms, requested_task, top_k
    */
}

// QA -> `evidence`, `answer`
export interface QAData {
    schema_version: string;
    query: string;
    task: string; // "qa"
    
    // Output QA:
    answer: QAAnswer;
    evidence: QAEvidence[];

    /* 
      --- SKIPPED FIELDS FROM BACKEND ---
      - query_plan
      - candidates
      - context
      - routing_trace
      - answer_report
      - answer_eligible, preflight_block_reason, temporal_matches, experiment_id, latency_ms
    */
}

export interface Candidate {
    rank: number;
    video_id: string;
    frame_ids: number[];
    score: number;
    score_breakdown: ScoreBreakdown;
    path_id: string;
    events: CandidateEvent[];
    lineage: Lineage[];
    warnings: string[];
}

export interface ScoreBreakdown {
    base_score: number;
    context_score: number;
    coverage: number;
    duplicate_location_penalty: number;
    event_scores: number[];
    gap_penalty: number;
    gap_penalty_method: string;
    gap_units: string;
    mean_event_score: number;
    refinement: ScoreRefinement;
    video_score: number;
  }
  
export interface ScoreRefinement {
    applied: boolean;
    coarse_score: number;
    confidences: number[];
    final_score: number;
    strategies: string[];
}
  
export interface CandidateEvent {
    event_index: number;
    normalized_score: number;
    rank: number;
    retrieval_query: string;
    warnings: string[];
    result: FrameResult;
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
    objects: unknown[];
    modality_scores: {
      visual: number;
    };
    neighbors: unknown[];
}

export interface Lineage {
    event_index: number;
    video_id: string;
    original_frame_index: number;
    internal_frame_id: string;
    source: string;
}

export interface QAAnswer {
    status: string;
    answer: string | null;
    answer_type: string;
    confidence: number;
    evidence_ids: string[];
    reason: string;
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
    matched_constraints: unknown[];
    temporal_event_index: number | null;
    temporal_match_rank: number | null;
    temporal_match_mode: string;
    temporal_chain_id: string;
    temporal_event_query: string;
    temporal_event_role: string;
    temporal_chain_score: number | null;
    warnings: string[];
}