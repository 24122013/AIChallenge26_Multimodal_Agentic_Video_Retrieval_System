import { useCallback, useState } from 'react';
import type { SearchLog, TelemetryDetails } from '../types';
import { API_PROXY } from '../constants/proxy';

export type SendTelemetryFn = (
    eventType: 'query' | 'click_result' | 'submit_result',
    details: TelemetryDetails,
    latencyMs?: number
) => Promise<void>;

export const useTelemetry = () => {
    // Telemetry Session State Log Dashboard
    const [searchLogs, setSearchLogs] = useState<SearchLog[]>([]);
    const [isLogModalOpen, setIsLogModalOpen] = useState<boolean>(false);

    const sendTelemetry: SendTelemetryFn = useCallback(async (
        eventType: 'query' | 'click_result' | 'submit_result',
        details: TelemetryDetails,
        latencyMs: number = 0
    ) => {
        try {
            await fetch(`${API_PROXY}/telemetry/log`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('access_token') || ''}`
                },
                body: JSON.stringify({
                    event_type: eventType,
                    timestamp: Date.now() / 1000,
                    latency_ms: latencyMs,
                    details: details
                })
            });
            console.log(`[Telemetry] ${eventType} logged. Latency: ${latencyMs}ms`);
        } catch (err) {
            console.error("Failed to ship user telemetry:", err);
        }
    }, []);

    return { sendTelemetry, searchLogs, setSearchLogs, isLogModalOpen, setIsLogModalOpen };
};