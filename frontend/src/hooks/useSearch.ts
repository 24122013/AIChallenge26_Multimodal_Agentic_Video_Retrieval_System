import { useState, useCallback, useRef } from 'react';
import type {
    VideoScene, SearchPayload, SortKey,
    ApiResponse, VisualSearchData, QAData, TrakeData, TemporalData
} from '../types';
import { API_PROXY } from '../constants/proxy';

interface UseSearchReturn {
    sortedResults: VideoScene[];
    isSearching: boolean;
    latency: number;
    sortBy: SortKey;
    setSortBy: (key: SortKey) => void;
    executeSearch: (payload: SearchPayload) => Promise<void>;
    apiResponseData: ApiResponse<VisualSearchData> | ApiResponse<QAData> | ApiResponse<TrakeData> | ApiResponse<TemporalData> | null;
}

// Stable sort key accessor
function getScore(scene: VideoScene, key: SortKey): number {
    if (key === 'arrival') return 0;
    if (key === 'score') return scene.score;
    // Updated to match backend's dict name
    return scene.modality_scores?.[key] ?? -1;
}

// Hook
export function useSearch(): UseSearchReturn {
    const [results, setResults] = useState<VideoScene[]>([]);
    const [isSearching, setIsSearching] = useState<boolean>(false);
    const [latency, setLatency] = useState<number>(0);
    const [sortBy, setSortBy] = useState<SortKey>('arrival');
    const [apiResponseData, setApiResponseData] = useState<ApiResponse<VisualSearchData> | ApiResponse<QAData> | ApiResponse<TrakeData> | ApiResponse<TemporalData> | null>(null);

    // Track a run-ID so stale async tasks from a cancelled search don't pollute results
    const runIdRef = useRef(0);

    const clearResults = useCallback(() => {
        setResults([]);
        setLatency(0);
    }, []);

    // Main entry point: fire unified searches
    const executeSearch = useCallback(async (payload: SearchPayload) => {
        const runId = ++runIdRef.current;
        clearResults();
        setIsSearching(true);
    
        console.log("🚀 [Frontend] Strictly Typed SearchPayload:", payload);
    
        // Filter out empty queries, but DO NOT skip 'qa' mode
        const validQueries = payload.text_queries?.filter(q => q.query.trim() !== "") || [];
    
        if (validQueries.length === 0) {
            console.warn("⚠️ No valid text queries provided. Aborting fetch.");
            setIsSearching(false);
            return;
        }
    
        try {
            const searchPromises = validQueries.map(async (queryParam) => {
                const requestBody = {
                    query: queryParam.query,
                    mode: queryParam.mode,
                    top_k: payload.config.topK || queryParam.top_k || 20
                };
    
                console.log(`📡 [Network] Sending request to /api/search for mode [${queryParam.mode}]:`, requestBody);
    
                const res = await fetch(`${API_PROXY}/search`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestBody),
                });
    
                if (!res.ok) {
                    const errorData = await res.json().catch(async () => ({ detail: await res.text() }));
                    const errorMsg = errorData.detail || `HTTP ${res.status}`;
                    console.error(`❌ [Backend Crash] Mode [${queryParam.mode}] Details:`, errorMsg);
                    return null;
                }
    
                // Return the full JSON response instead of setting state mid-loop
                return await res.json();
            });
    
            const responses = await Promise.all(searchPromises);
    
            if (runIdRef.current !== runId) return;
    
            let totalLatency = 0;
            const aggregatedResults: VideoScene[] = [];
            const seenIds = new Set<string>();
            let primaryResponseData = null;
    
            responses.forEach((jsonResponse) => {
                if (!jsonResponse) return;
    
                // Capture the first valid response payload for UI rendering (like QA data)
                if (!primaryResponseData) {
                    primaryResponseData = jsonResponse;
                }
    
                const searchData = jsonResponse.data;
                totalLatency += searchData?.latency_ms ?? 0;
                
                const scenes: VideoScene[] = searchData?.results ?? [];
                scenes.forEach(scene => {
                    const uniqueId = scene.frame_id ? String(scene.frame_id) : null;
                    
                    if (uniqueId) {
                        if (!seenIds.has(uniqueId)) {
                            seenIds.add(uniqueId);
                            aggregatedResults.push(scene);
                        }
                    } else {
                        aggregatedResults.push(scene);
                    }
                });
            });
    
            // Set state once after all promises resolve
            if (primaryResponseData) {
                setApiResponseData(primaryResponseData);
            }
            setResults(aggregatedResults);
            setLatency(totalLatency);
    
        } catch (err) {
            console.error('💥 [useSearch] execution error:', err);
        } finally {
            if (runIdRef.current === runId) {
                setIsSearching(false);
            }
        }
    }, [clearResults]);

    // Sorted view (derived, never mutates `results` order)
    const sortedResults: VideoScene[] = sortBy === 'arrival'
        ? results
        : [...results].sort((a, b) => getScore(b, sortBy) - getScore(a, sortBy));

    return {
        sortedResults,
        isSearching,
        latency,
        sortBy,
        setSortBy,
        executeSearch,
        apiResponseData,
    };
}