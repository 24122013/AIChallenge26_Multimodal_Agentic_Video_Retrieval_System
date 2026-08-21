import React from 'react';
import type { SearchLog } from '../types';

interface TelemetryLogModalProps {
  isOpen: boolean;
  onClose: () => void;
  searchLogs: SearchLog[];
  currentSearchId: string | null;
  onRetraceLog: (log: SearchLog) => void;
}

const TelemetryLogModal: React.FC<TelemetryLogModalProps> = ({
    isOpen,
    onClose,
    searchLogs,
    currentSearchId,
    onRetraceLog,
  }) => {
    if (!isOpen) return null;
  
    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4" onClick={() => onClose()}>
          <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl p-5 shadow-2xl w-full max-w-3xl max-h-[80vh] overflow-y-auto flex flex-col" onClick={e => e.stopPropagation()}>
            <div className="flex justify-between items-center mb-4 border-b pb-2 dark:border-zinc-800">
              <h3 className="text-md font-bold uppercase tracking-wider text-zinc-400">System Telemetry Log Sessions</h3>
              <button className="text-xs font-bold text-zinc-500 hover:text-black dark:hover:text-white" onClick={() => onClose()}>Close (Esc)</button>
            </div>

            {searchLogs.length === 0 ? (
              <p className="text-zinc-400 text-sm italic text-center py-8">No interaction records tracked for active session.</p>
            ) : (
              <div className="space-y-3">
                {searchLogs.map((log) => {
                  const isActive = log.id === currentSearchId;

                  return (
                    <details key={log.id} className={`group border rounded-lg overflow-hidden ${isActive ? 'bg-blue-50 dark:bg-blue-900/20 border-blue-300 dark:border-blue-800 ring-1 ring-blue-500 shadow-sm' : 'bg-zinc-50 dark:bg-zinc-950 border-zinc-200 dark:border-zinc-800'}`}>
                      <summary className={`flex items-center justify-between p-3 cursor-pointer select-none transition-colors ${isActive ? 'hover:bg-blue-100/60 dark:hover:bg-blue-900/40' : 'hover:bg-zinc-100 dark:hover:bg-zinc-900'}`}>
                        <div className="flex items-center gap-3 truncate max-w-[70%]">
                          <span className="text-sm font-semibold truncate dark:text-zinc-200">&ldquo;{log.query}&rdquo;</span>

                          {isActive && (
                            <span className="text-[9px] font-bold px-1.5 py-0.5 bg-blue-600 text-white rounded uppercase tracking-wider animate-pulse shadow-sm">
                              Active
                            </span>
                          )}

                          <div className="flex gap-1">
                            {log.methods.map(m => (
                              <span key={m} className="text-[9px] font-bold px-1.5 py-0.5 bg-zinc-200 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-400 rounded-full">{m}</span>
                            ))}
                          </div>
                        </div>
                        <div className="flex items-center gap-4 shrink-0 font-mono text-xs text-zinc-500">
                          <span>Latency: <strong className="text-zinc-700 dark:text-zinc-300">{log.latency}ms</strong></span>
                          <span>Subs: <strong className="text-emerald-600 font-bold">{log.submissionsCount}</strong></span>
                          <span className="group-open:rotate-180 transition-transform">▼</span>
                        </div>
                      </summary>

                      <div className="p-3 bg-white dark:bg-zinc-900 border-t border-zinc-200 dark:border-zinc-800 text-xs text-zinc-600 dark:text-zinc-400 space-y-1">
                        <div className="flex justify-between items-start">
                          <div>
                            <p><strong>Session ID:</strong> {log.id}</p>
                            <p><strong>Returned Datasets:</strong> {log.resultsCount} active nodes found</p>
                          </div>
                          <button
                            onClick={() => onRetraceLog(log)}
                            disabled={isActive}
                            className={`font-semibold py-1.5 px-3 rounded text-[11px] transition-colors ${isActive ? 'bg-zinc-200 text-zinc-400 dark:bg-zinc-800 dark:text-zinc-600 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-500 text-white'}`}
                          >
                            {isActive ? 'Current Search' : 'Retrace Search'}
                          </button>
                        </div>

                        {log.results.length > 0 && (
                          <div className="mt-2 pt-2 border-t dark:border-zinc-800">
                            <p className="font-semibold text-[10px] uppercase text-zinc-400 mb-1">Indexed Target Array Sample:</p>
                            <div className="max-h-24 overflow-y-auto font-mono text-[11px] space-y-0.5">
                              {log.results.slice(0, 5).map((r, idx) => (
                                <div key={idx} className="truncate">{r.name} - Score: {(r.score * 100).toFixed(1)}%</div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </details>
                  );
                })}
              </div>
            )}
          </div>
        </div>
    );
}

export default TelemetryLogModal;