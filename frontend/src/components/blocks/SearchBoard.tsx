import React, { useState, useMemo } from 'react';
import type { VideoScene, SortKey, SearchPayload } from '../../types';
import { SORT_OPTIONS } from '../../constants/video-scene-sort-option';
import SearchSidebar from './SearchSideBar';

const METHOD_COLORS: Record<string, string> = {
  visual: 'bg-blue-100 text-blue-700 dark:bg-blue-950/40 dark:text-blue-400',
  hybrid: 'bg-fuchsia-100 text-fuchsia-700 dark:bg-fuchsia-950/40 dark:text-fuchsia-400',
  caption: 'bg-pink-100 text-pink-700 dark:bg-pink-950/40 dark:text-pink-400',
  ocr: 'bg-violet-100 text-violet-700 dark:bg-violet-950/40 dark:text-violet-400',
  asr: 'bg-cyan-100 text-cyan-700 dark:bg-cyan-950/40 dark:text-cyan-400',
  object: 'bg-orange-100 text-orange-700 dark:bg-orange-950/40 dark:text-orange-400',
  qa: 'bg-rose-100 text-rose-700 dark:bg-rose-950/40 dark:text-rose-400',
  temporal: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-400',
  score: 'bg-indigo-100 text-indigo-700 dark:bg-indigo-950/40 dark:text-indigo-400',
};

// ─── SIMPLE CUSTOM HOOK FOR DRAG & DROP ───
function useDraggableResults(initialResults: VideoScene[]) {
  const [orderedIds, setOrderedIds] = useState<string[]>([]);
  const [draggedId, setDraggedId] = useState<string | null>(null);

  // Safely sync order ONLY when the incoming results actually change
  const currentIdsStr = initialResults.map(r => r.frame_id).join(',');
  React.useEffect(() => {
    setOrderedIds(initialResults.map(r => r.frame_id));
  }, [currentIdsStr]);

  const handleDrop = (e: React.DragEvent, targetId: string) => {
    e.preventDefault();
    const sourceId = e.dataTransfer.getData('text/plain');
    if (!sourceId || sourceId === targetId) return;

    setOrderedIds(prev => {
      const newOrder = [...prev];
      const srcIdx = newOrder.indexOf(sourceId);
      const tgtIdx = newOrder.indexOf(targetId);
      if (srcIdx > -1 && tgtIdx > -1) {
        newOrder.splice(srcIdx, 1);
        newOrder.splice(tgtIdx, 0, sourceId);
      }
      return newOrder;
    });
    setDraggedId(null);
  };

  const sortedResults = useMemo(() => {
    return [...initialResults].sort((a, b) => {
      const idxA = orderedIds.indexOf(a.frame_id);
      const idxB = orderedIds.indexOf(b.frame_id);
      return (idxA > -1 ? idxA : 999) - (idxB > -1 ? idxB : 999);
    });
  }, [initialResults, orderedIds]);

  return { sortedResults, draggedId, setDraggedId, handleDrop };
}

interface SearchBoardProps {
  sortedResults: VideoScene[];
  displayedResults: VideoScene[];
  isSearching: boolean;
  latency: number;
  isFilterDropdownOpen: boolean;
  openFilterDropdown: () => void;
  closeFilterDropdown: () => void;
  sortBy: SortKey;
  setSortBy: (val: SortKey) => void;
  activeSources: Set<string>;
  toggleSourceFilter: (source: string) => void;
  resetSources: () => void;
  availableSources: string[];
  clickedSceneIds: Set<string>;
  submittedSceneIds: Set<string>;
  searchLogsLength: number;
  onExecuteSearch: (payload: SearchPayload) => void;
  onOpenLogs: () => void;
  onSelectResult: (scene: VideoScene) => void;
  onFinalSubmit: (sceneId: string) => void;
}

const SearchBoard: React.FC<SearchBoardProps> = ({
  sortedResults,
  displayedResults,
  isSearching,
  latency,
  isFilterDropdownOpen,
  openFilterDropdown,
  closeFilterDropdown,
  sortBy,
  setSortBy,
  activeSources,
  toggleSourceFilter,
  resetSources,
  availableSources,
  clickedSceneIds,
  submittedSceneIds,
  searchLogsLength,
  onExecuteSearch,
  onOpenLogs,
  onSelectResult,
  onFinalSubmit
}) => {
  const isAnyLoading = Object.values(isSearching).some(Boolean);
  const [isSidebarExpanded, setIsSidebarExpanded] = useState(true);
  
  // Filtering states
  const [interactionFilter, setInteractionFilter] = useState<'all' | 'clicked' | 'unclicked'>('all');
  const [hideSubmitted, setHideSubmitted] = useState<boolean>(false);

  // 1. Apply Filters
  const filteredResults = useMemo(() => displayedResults.filter(scene => {
    if (hideSubmitted && submittedSceneIds.has(scene.frame_id)) return false;
    const isClicked = clickedSceneIds.has(scene.frame_id);
    if (interactionFilter === 'clicked') return isClicked;
    if (interactionFilter === 'unclicked') return !isClicked;
    return true;
  }), [displayedResults, hideSubmitted, submittedSceneIds, clickedSceneIds, interactionFilter]);

  // 2. Pass filtered results to our clean drag & drop hook
  const { sortedResults: draggableResults, draggedId, setDraggedId, handleDrop } = useDraggableResults(filteredResults);

  return (
    <div className="flex w-full h-screen overflow-hidden">
        <SearchSidebar 
            onSearch={onExecuteSearch} 
            isExpanded={isSidebarExpanded}
            setIsExpanded={setIsSidebarExpanded}
        />

        <main className="flex-1 h-full overflow-y-auto p-6 flex flex-col">
          <div className="flex-1 min-h-[300px]">
            <div className="flex flex-wrap items-center justify-between gap-3 mb-6">
              <div className="flex items-center gap-3">
                <h3 className="text-base font-semibold tracking-wide text-zinc-800 dark:text-zinc-200 uppercase">Search Results</h3>
                <button
                  onClick={() => onOpenLogs()}
                  className="text-xs bg-zinc-800 text-zinc-100 hover:bg-zinc-700 dark:bg-zinc-200 dark:text-zinc-900 dark:hover:bg-zinc-100 font-medium px-2.5 py-1 rounded-lg shadow transition-all"
                >
                  View Process Logs ({searchLogsLength})
                </button>
              </div>

              <div className="flex items-center gap-3">
                <label className="flex items-center gap-2 cursor-pointer text-sm text-zinc-700 dark:text-zinc-300 bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 px-3 py-1.5 rounded-lg shadow-sm">
                  <input 
                    type="checkbox" 
                    className="rounded border-zinc-300 text-blue-600 focus:ring-blue-500"
                    checked={hideSubmitted}
                    onChange={(e) => setHideSubmitted(e.target.checked)}
                  />
                  <span>Hide Submitted</span>
                </label>

                <select
                  value={interactionFilter}
                  onChange={e => setInteractionFilter(e.target.value as any)}
                  className="text-sm px-3 py-1.5 rounded-lg bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 text-zinc-700 dark:text-zinc-300 focus:outline-none focus:ring-2 focus:ring-blue-500 shadow-sm"
                >
                  <option value="all">All Status</option>
                  <option value="unclicked">Unclicked</option>
                  <option value="clicked">Clicked</option>
                </select>

                <div className="relative">
                  <button
                    onClick={() => openFilterDropdown()}
                    className="text-sm px-3 py-1.5 rounded-lg bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 text-zinc-700 dark:text-zinc-300 focus:outline-none focus:ring-2 focus:ring-blue-500 shadow-sm flex items-center gap-2"
                  >
                    <span>Groups ({activeSources.size})</span>
                    <span className="text-[10px]">▼</span>
                  </button>

                  {isFilterDropdownOpen && (
                    <>
                      <div className="fixed inset-0 z-10" onClick={closeFilterDropdown} />
                      <div className="absolute right-0 mt-2 w-48 bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-lg shadow-xl z-20 overflow-hidden">
                        <div className="p-2 border-b border-zinc-200 dark:border-zinc-700 flex justify-between items-center bg-zinc-50 dark:bg-zinc-900/50">
                          <span className="text-xs font-semibold text-zinc-600 dark:text-zinc-300">Toggle Sources</span>
                          <button onClick={() => resetSources()} className="text-[10px] text-blue-600 hover:text-blue-500 font-medium">Reset</button>
                        </div>
                        <div className="p-2 flex flex-col gap-1">
                          {availableSources.map(source => (
                            <label key={source} className="flex items-center gap-2 px-2 py-1.5 hover:bg-zinc-100 dark:hover:bg-zinc-700/50 rounded cursor-pointer transition-colors">
                              <input
                                type="checkbox"
                                className="rounded border-zinc-300 text-blue-600 focus:ring-blue-500"
                                checked={activeSources.has(source)}
                                onChange={() => toggleSourceFilter(source)}
                              />
                              <span className="text-sm text-zinc-700 dark:text-zinc-300 capitalize">{source}</span>
                            </label>
                          ))}
                        </div>
                      </div>
                    </>
                  )}
                </div>

                <select
                  value={sortBy}
                  onChange={e => setSortBy(e.target.value as SortKey)}
                  className="text-sm px-3 py-1.5 rounded-lg bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 text-zinc-700 dark:text-zinc-300 focus:outline-none focus:ring-2 focus:ring-blue-500 shadow-sm"
                >
                  {SORT_OPTIONS.map(o => (
                    <option key={o.key} value={o.key}>{o.label}</option>
                  ))}
                </select>

                {!isAnyLoading && latency > 0 && (
                  <span className="text-xs font-mono bg-emerald-50 text-emerald-700 dark:bg-emerald-950/30 dark:text-emerald-400 px-2 py-1.5 rounded border border-emerald-100 dark:border-emerald-900/50">
                    {latency} ms
                  </span>
                )}
              </div>
            </div>

            {/* States */}
            {sortedResults.length === 0 && !isAnyLoading && (
              <div className="h-64 border-2 border-dashed border-zinc-300 dark:border-zinc-800 rounded-xl flex items-center justify-center text-zinc-400 text-sm">No active indices queried.</div>
            )}

            {draggableResults.length === 0 && sortedResults.length > 0 && !isAnyLoading && (
              <div className="h-64 border-2 border-dashed border-zinc-300 dark:border-zinc-700/50 rounded-xl flex items-center justify-center text-zinc-500 text-sm bg-zinc-50 dark:bg-zinc-950/20">
                No results match your filters.
              </div>
            )}

            {/* Results Grid */}
            {draggableResults.length > 0 && (
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
                {draggableResults.map((scene, index) => {
                  const scoreList = Object.entries(scene.modality_scores || {}).filter(([m]) => m !== 'rerank');
                  const rerankScore = scene.modality_scores?.['rerank'];

                  const isClicked = clickedSceneIds.has(scene.frame_id);
                  const isSubmitted = submittedSceneIds.has(scene.frame_id);

                  let cardStyle = "bg-white dark:bg-zinc-900 border-zinc-200 dark:border-zinc-800";
                  if (isSubmitted) cardStyle = "bg-emerald-50/60 dark:bg-emerald-950/20 border-emerald-400 shadow-inner";
                  else if (isClicked) cardStyle = "bg-yellow-100/80 dark:bg-yellow-800/40 border-zinc-300 dark:border-zinc-700/80";

                  const isDraggingThis = draggedId === scene.frame_id;

                  return (
                    <div 
                      key={`${scene.frame_id}-${index}`} 
                      draggable
                      onDragStart={(e) => {
                        e.dataTransfer.setData('text/plain', scene.frame_id);
                        setDraggedId(scene.frame_id);
                      }}
                      onDragOver={(e) => e.preventDefault()}
                      onDrop={(e) => handleDrop(e, scene.frame_id)}
                      onDragEnd={() => setDraggedId(null)}
                      className={`border rounded-xl p-4 flex flex-col justify-between transition-all duration-200 cursor-grab active:cursor-grabbing ${cardStyle} ${isDraggingThis ? 'opacity-40 scale-[0.98]' : ''}`}
                    >
                      <div>
                        <div className="flex justify-between items-start mb-2 gap-2">
                          <span className="text-xs font-bold font-mono px-2 py-1 rounded bg-zinc-200/60 dark:bg-zinc-800/80 truncate max-w-[150px]">{scene.video_id}</span>
                          <p className="text-xs text-zinc-500 font-mono mt-1">@ {scene.timestamp.toFixed(1)} s</p>
                        </div>
                        
                        <div className="flex items-center gap-1.5 flex-wrap mb-3">
                          {scoreList.map(([method, score]) => (
                            <div key={method} className={`flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded border border-black/5 dark:border-white/5 ${METHOD_COLORS[method] || 'bg-zinc-200 text-zinc-800'}`}>
                              <span className="font-semibold uppercase tracking-tight">{method}</span>
                              <span className="font-mono font-bold opacity-80">{(score * 100).toFixed(0)}%</span>
                            </div>
                          ))}
                        </div>

                        {rerankScore !== undefined && (
                          <div className="mb-3">
                            <span className="text-[10px] font-mono bg-indigo-50 dark:bg-indigo-950/40 text-indigo-600 px-1.5 py-0.5 rounded border border-indigo-100">
                              Rerank Mod: <strong className="font-bold">+{Math.round(rerankScore * 100)}%</strong>
                            </span>
                          </div>
                        )}

                        <div className="bg-white/80 dark:bg-zinc-950/40 p-2.5 rounded text-xs border border-zinc-100 dark:border-zinc-800 mb-4 h-16 overflow-hidden">
                          <span className="font-bold block text-[10px] uppercase text-zinc-400 mb-1">OCR Content</span>
                          <span className="text-zinc-700 dark:text-zinc-300 italic line-clamp-2">&ldquo;{scene.ocr_text || 'None'}&rdquo;</span>
                        </div>
                      </div>

                      <div className="flex gap-2">
                        <button onClick={() => onSelectResult(scene)} className="flex-1 bg-zinc-900 dark:bg-zinc-200 hover:bg-zinc-800 text-white dark:text-zinc-900 text-xs py-2 rounded-lg font-medium transition-colors">
                          {isClicked ? 'Review Again' : 'Review Scene'}
                        </button>
                        <button onClick={() => onFinalSubmit(scene.frame_id)} className={`text-xs py-2 px-4 rounded-lg font-semibold transition-colors border ${isSubmitted ? 'bg-red-50 text-red-600 border-red-200 hover:bg-red-100 dark:bg-red-950/30 dark:text-red-400' : 'border-zinc-200 hover:bg-zinc-50 dark:border-zinc-700 dark:hover:bg-zinc-800'}`}>
                          {isSubmitted ? 'Unsubmit' : 'Submit'}
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </main>
      </div>
  );
};

export default SearchBoard;