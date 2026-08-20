import { useState, useCallback, useRef } from 'react';
import type { VideoScene, SearchPayload, SortKey } from '../types';

// Relative paths — Vite proxy forwards /search/* → http://localhost:8000
const API_BASE = '/api';

interface UseSearchReturn {
    sortedResults: VideoScene[];
    isSearching: boolean;
    latency: number;
    sortBy: SortKey;
    setSortBy: (key: SortKey) => void;
    executeSearch: (payload: SearchPayload) => Promise<void>;
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

        // 1. Extract all queries, explicitly skipping "qa" mode
        const kistQueries = payload.text_queries?.filter(q => q.mode !== 'qa') || [];

        if (kistQueries.length === 0) {
            console.warn("⚠️ No KIST text queries provided. Aborting fetch.");
            setIsSearching(false);
            return;
        }

        try {
            // 2. Map each KIST query to a separate fetch promise
            const searchPromises = kistQueries.map(async (queryParam) => {
                if (queryParam.query != "") {
                    const requestBody = {
                        query: queryParam.query,
                        mode: queryParam.mode,
                        top_k: payload.config.topK || queryParam.top_k || 20
                    };
    
                    console.log(`📡 [Network] Sending request to /api/search for mode [${queryParam.mode}]:`, requestBody);
    
                    const res = await fetch(`${API_BASE}/search`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(requestBody),
                    });
    
                    if (!res.ok) {
                        const errorData = await res.json().catch(async () => ({ detail: await res.text() }));
                        const errorMsg = errorData.detail || `HTTP ${res.status}`;
                        console.error(`❌ [Backend Crash] Mode [${queryParam.mode}] Details:`, errorMsg);
                        // Return null for this specific query so others can still succeed
                        return null;
                    }
    
                    const json = await res.json();
                    console.log(`✅ [Backend Response] Mode [${queryParam.mode}]:`, json);
                    return json.data;
                } else {
                    console.log(`None in the query`);
                    return {};
                }
            });

            // 3. Execute all API calls concurrently
            const responses = await Promise.all(searchPromises);

            // Abort state update if a new search was triggered while we were waiting
            if (runIdRef.current !== runId) return;

            // 4. Aggregate results and latencies
            let totalLatency = 0;
            const aggregatedResults: VideoScene[] = [];
            const seenIds = new Set<string>();

            responses.forEach((searchData) => {
                if (!searchData) return;

                totalLatency += searchData.latency_ms ?? 0;
                
                const scenes: VideoScene[] = searchData.results ?? [];
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
    };
}