import { useState, useCallback, useRef } from 'react';
import type {
    ApiResponse,
    Candidate,
    FrameResult,
    QAEvidence,
    SearchData,
    SearchPayload,
    SortKey,
    VideoScene,
    VisualSearchResult,
} from '../types';
import { API_PROXY } from '../constants/proxy';

interface UseSearchReturn {
    sortedResults: VideoScene[];
    isSearching: boolean;
    searchError: string | null;
    latency: number;
    sortBy: SortKey;
    setSortBy: (key: SortKey) => void;
    executeSearch: (payload: SearchPayload) => Promise<void>;
    apiResponseData: ApiResponse<SearchData> | null;
}

function getScore(scene: VideoScene, key: SortKey): number {
    if (key === 'arrival') return 0;
    if (key === 'score') return scene.score;
    return scene.modality_scores?.[key] ?? -1;
}

function candidateToScene(candidate: Candidate): VideoScene {
    return {
        video_id: candidate.video_id,
        frame_id: candidate.frame_id || candidate.keyframe_id,
        timestamp: candidate.timestamp,
        score: candidate.score,
        segment_id: candidate.segment_id,
        shot_id: candidate.shot_id,
        faiss_index: candidate.faiss_index,
        frame_index: candidate.frame_index,
        keyframe_path: candidate.keyframe_path,
        thumbnail_path: candidate.thumbnail_path,
        timestamp_source: candidate.timestamp_source ?? 'retrieval_metadata',
        timestamp_confidence: candidate.timestamp_confidence ?? 1,
        caption: candidate.caption,
        ocr_text: candidate.ocr_text,
        asr_text: '',
        objects: candidate.objects,
        modality_scores: candidate.modality_scores,
        neighbors: candidate.neighbors.map(neighbor => ({
            frame_id: neighbor.frame_id,
            timestamp: neighbor.timestamp,
            delta_seconds: neighbor.timestamp - candidate.timestamp,
        })),
        context_sources: candidate.context_sources,
    };
}

function frameToScene(frame: FrameResult | VisualSearchResult, score = frame.score): VideoScene {
    return {
        ...frame,
        score,
        asr_text: '',
        neighbors: frame.neighbors.map(neighbor => ({
            frame_id: neighbor.frame_id,
            timestamp: neighbor.timestamp,
            delta_seconds: neighbor.timestamp - frame.timestamp,
        })),
    };
}

function evidenceToScene(evidence: QAEvidence): VideoScene {
    return {
        video_id: evidence.video_id,
        frame_id: evidence.frame_id,
        timestamp: evidence.timestamp,
        score: evidence.retrieval_score,
        segment_id: '',
        shot_id: evidence.shot_id,
        faiss_index: null,
        frame_index: evidence.frame_index,
        keyframe_path: evidence.image_path,
        thumbnail_path: evidence.image_path,
        timestamp_source: 'qa_evidence',
        timestamp_confidence: 1,
        caption: evidence.caption,
        ocr_text: evidence.ocr_text,
        asr_text: '',
        objects: evidence.objects,
        modality_scores: {
            retrieval_score: evidence.retrieval_score,
            base_retrieval_score: evidence.base_retrieval_score,
            constraint_score: evidence.constraint_score,
        },
        neighbors: [],
    };
}

function extractScenes(data: SearchData): VideoScene[] {
    if ('results' in data) return data.results.map(result => frameToScene(result));
    if (data.task === 'kis' || data.task === 'avs') {
        return data.candidates.map(candidateToScene);
    }
    if (data.task === 'qa') return data.evidence.map(evidenceToScene);
    if (data.task === 'temporal') return data.candidates.map(candidateToScene);
    if (data.task === 'trake') {
        return data.hypotheses.flatMap(hypothesis =>
            hypothesis.events.map(event => frameToScene(event.result, event.normalized_score)),
        );
    }
    return [];
}

export function useSearch(): UseSearchReturn {
    const [results, setResults] = useState<VideoScene[]>([]);
    const [isSearching, setIsSearching] = useState(false);
    const [searchError, setSearchError] = useState<string | null>(null);
    const [latency, setLatency] = useState(0);
    const [sortBy, setSortBy] = useState<SortKey>('arrival');
    const [apiResponseData, setApiResponseData] = useState<ApiResponse<SearchData> | null>(null);
    const runIdRef = useRef(0);

    const clearResults = useCallback(() => {
        setResults([]);
        setLatency(0);
        setApiResponseData(null);
        setSearchError(null);
    }, []);

    const executeSearch = useCallback(async (payload: SearchPayload) => {
        const runId = ++runIdRef.current;
        clearResults();
        setIsSearching(true);

        const validQueries = payload.text_queries?.filter(item => item.query.trim()) ?? [];
        if (validQueries.length === 0) {
            setSearchError('Please enter a search query.');
            setIsSearching(false);
            return;
        }

        try {
            const responses = await Promise.all(validQueries.map(async queryParam => {
                const response = await fetch(`${API_PROXY}/search`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: queryParam.query,
                        mode: queryParam.mode,
                        top_k: payload.config.topK || queryParam.top_k || 20,
                        expanded_queries: queryParam.expanded_queries ?? [],
                    }),
                });
                const body = await response.json().catch(() => null) as
                    | (Partial<ApiResponse<SearchData>> & { detail?: string })
                    | null;
                if (!response.ok) {
                    throw new Error(body?.message || body?.detail || `Search failed (HTTP ${response.status}).`);
                }
                return body as ApiResponse<SearchData>;
            }));

            if (runIdRef.current !== runId) return;
            const aggregatedResults: VideoScene[] = [];
            const seenIds = new Set<string>();
            let totalLatency = 0;

            responses.forEach(response => {
                totalLatency += response.data.latency_ms ?? 0;
                extractScenes(response.data).forEach(scene => {
                    const uniqueId = `${scene.video_id}:${scene.frame_id}`;
                    if (!seenIds.has(uniqueId)) {
                        seenIds.add(uniqueId);
                        aggregatedResults.push(scene);
                    }
                });
            });

            setApiResponseData(responses[0] ?? null);
            setResults(aggregatedResults);
            setLatency(totalLatency);
        } catch (error) {
            if (runIdRef.current === runId) {
                setSearchError(error instanceof Error ? error.message : 'Search failed unexpectedly.');
            }
        } finally {
            if (runIdRef.current === runId) setIsSearching(false);
        }
    }, [clearResults]);

    const sortedResults = sortBy === 'arrival'
        ? results
        : [...results].sort((a, b) => getScore(b, sortBy) - getScore(a, sortBy));

    return {
        sortedResults,
        isSearching,
        searchError,
        latency,
        sortBy,
        setSortBy,
        executeSearch,
        apiResponseData,
    };
}
