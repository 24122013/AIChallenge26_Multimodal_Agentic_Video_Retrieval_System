import React, { useEffect, useState, useMemo } from 'react';
import type { VideoScene, SortKey, SearchPayload } from '../../types';
import { SORT_OPTIONS } from '../../constants/video-scene-sort-option';
import SearchSidebar from './SearchSideBar';
import { Grid, Rows, LayoutGrid } from 'lucide-react';
import { API_PROXY } from '../../constants/proxy';

const METHOD_COLORS: Record<string, string> = {
  visual: 'bg-blue-100 text-blue-700 dark:bg-blue-950/40 dark:text-blue-400',
  hybrid: 'bg-fuchsia-100 text-fuchsia-700 dark:bg-fuchsia-950/40 dark:text-fuchsia-400',
  caption: 'bg-pink-100 text-pink-700 dark:bg-pink-950/40 dark:text-pink-400',
  ocr: 'bg-violet-100 text-violet-700 dark:bg-violet-950/40 dark:text-violet-400',
  asr: 'bg-cyan-100 text-cyan-700 dark:bg-cyan-950/40 dark:text-cyan-400',
  object: 'bg-orange-100 text-orange-700 dark:bg-orange-950/40 dark:text-orange-400',
  qa: 'bg-rose-100 text-rose-700 dark:bg-rose-950/40 dark:text-rose-400',
  temporal: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-400',
  trake: 'bg-teal-100 text-teal-700 dark:bg-teal-950/40 dark:text-teal-400',
  score: 'bg-indigo-100 text-indigo-700 dark:bg-indigo-950/40 dark:text-indigo-400',
};

type ExtendedScene = VideoScene & {
    answer?: string;
    events?: Array<{ result?: { frame_index: number } }>;
};

type CardSize = 'sm' | 'md' | 'lg';
type GroupingMode = 'none' | 'video' | 'modality' | 'tens';

const getOutputString = (scene: ExtendedScene): string => {
    if (scene.events && Array.isArray(scene.events)) {
        const frameIndices = scene.events.map(e => e.result?.frame_index).join(', ');
        return `${scene.video_id}, ${frameIndices}`;
    }
    if (scene.answer) {
        return `${scene.video_id}, ${scene.frame_index ?? 0}, ${scene.answer}`;
    }
    return `${scene.video_id}, ${scene.frame_index ?? scene.frame_id}`;
};

function useDraggableResults(initialResults: VideoScene[]) {
  const [orderedIds, setOrderedIds] = useState<string[]>([]);
  const [draggedId, setDraggedId] = useState<string | null>(null);

  const currentIdsStr = initialResults.map(r => r.frame_id).join(',');
  useEffect(() => {
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
  
  const [interactionFilter, setInteractionFilter] = useState<'all' | 'clicked' | 'unclicked'>('all');
  const [hideSubmitted, setHideSubmitted] = useState<boolean>(false);
  const [cardSize, setCardSize] = useState<CardSize>('md');
  const [groupBy, setGroupBy] = useState<GroupingMode>('none');

  const filteredResults = useMemo(() => displayedResults.filter(scene => {
    if (hideSubmitted && submittedSceneIds.has(scene.frame_id)) return false;
    const isClicked = clickedSceneIds.has(scene.frame_id);
    if (interactionFilter === 'clicked') return isClicked;
    if (interactionFilter === 'unclicked') return !isClicked;
    return true;
  }), [displayedResults, hideSubmitted, submittedSceneIds, clickedSceneIds, interactionFilter]);

  const { sortedResults: draggableResults, draggedId, setDraggedId, handleDrop } = useDraggableResults(filteredResults);

  const groupedResults = useMemo(() => {
    if (groupBy === 'none') return { 'All Results': draggableResults };
    
    const groups: Record<string, VideoScene[]> = {};
    draggableResults.forEach((scene, index) => {
        let key = 'Other';
        if (groupBy === 'video') {
            key = scene.video_id;
        } else if (groupBy === 'tens') {
            const start = Math.floor(index / 10) * 10 + 1;
            const end = Math.min((Math.floor(index / 10) + 1) * 10, draggableResults.length);
            key = `Batch ${Math.floor(index / 10) + 1} (${start} - ${end})`;
        } else if (groupBy === 'modality') {
            if (scene.modality_scores && Object.keys(scene.modality_scores).length > 0) {
                key = Object.keys(scene.modality_scores).reduce((a, b) => scene.modality_scores[a] > scene.modality_scores[b] ? a : b);
            } else {
                key = 'Unknown';
            }
        }

        if (!groups[key]) groups[key] = [];
        groups[key].push(scene);
    });
    return groups;
  }, [draggableResults, groupBy]);

  const handleCopyOutput = (text: string) => {
      navigator.clipboard.writeText(text).catch(err => console.error("Failed to copy:", err));
  };

  const getGridClass = () => {
      switch (cardSize) {
          case 'sm': return 'grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 2xl:grid-cols-8 gap-3';
          case 'lg': return 'grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6';
          case 'md':
          default: return 'grid-cols-1 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-4';
      }
  };

  return (
    <div className="flex w-full h-screen overflow-hidden">
        <SearchSidebar 
            onSearch={onExecuteSearch} 
            isExpanded={isSidebarExpanded}
            setIsExpanded={setIsSidebarExpanded}
        />

        <main className="flex-1 h-full overflow-y-auto p-6 flex flex-col bg-zinc-50 dark:bg-[#09090b]">
          <div className="flex-1 min-h-[300px]">
            
            {/* Top Control Bar */}
            <div className="flex flex-wrap items-center justify-between gap-4 mb-6 bg-white dark:bg-zinc-900 p-4 rounded-xl shadow-sm border border-zinc-200 dark:border-zinc-800">
              <div className="flex items-center gap-3">
                <h3 className="text-sm font-bold tracking-wide text-zinc-800 dark:text-zinc-200 uppercase">Search Results</h3>
                <button
                  onClick={() => onOpenLogs()}
                  className="text-xs bg-zinc-800 text-zinc-100 hover:bg-zinc-700 dark:bg-zinc-200 dark:text-zinc-900 dark:hover:bg-zinc-100 font-medium px-2.5 py-1.5 rounded shadow transition-all"
                >
                  View Process Logs ({searchLogsLength})
                </button>
              </div>

              <div className="flex flex-wrap items-center gap-3">
                {/* View Options Group */}
                <div className="flex items-center bg-zinc-100 dark:bg-zinc-800/80 p-1 rounded-lg border border-zinc-200 dark:border-zinc-700">
                    <button onClick={() => setCardSize('sm')} className={`p-1.5 rounded ${cardSize === 'sm' ? 'bg-white dark:bg-zinc-600 shadow-sm text-blue-600 dark:text-blue-400' : 'text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300'}`} title="Compact View">
                        <Grid className="w-4 h-4" />
                    </button>
                    <button onClick={() => setCardSize('md')} className={`p-1.5 rounded ${cardSize === 'md' ? 'bg-white dark:bg-zinc-600 shadow-sm text-blue-600 dark:text-blue-400' : 'text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300'}`} title="Default View">
                        <LayoutGrid className="w-4 h-4" />
                    </button>
                    <button onClick={() => setCardSize('lg')} className={`p-1.5 rounded ${cardSize === 'lg' ? 'bg-white dark:bg-zinc-600 shadow-sm text-blue-600 dark:text-blue-400' : 'text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300'}`} title="Large View">
                        <Rows className="w-4 h-4" />
                    </button>
                </div>

                <div className="h-6 w-px bg-zinc-300 dark:bg-zinc-700 mx-1" />

                <select
                  value={groupBy}
                  onChange={e => setGroupBy(e.target.value as GroupingMode)}
                  className="text-xs px-2.5 py-1.5 rounded-lg bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 text-zinc-700 dark:text-zinc-300 focus:outline-none focus:ring-1 focus:ring-blue-500 shadow-sm"
                >
                  <option value="none">No Grouping</option>
                  <option value="video">Group by Video</option>
                  <option value="modality">Group by Modality</option>
                  <option value="tens">Chunk by 10s</option>
                </select>

                <div className="h-6 w-px bg-zinc-300 dark:bg-zinc-700 mx-1" />

                <label className="flex items-center gap-1.5 cursor-pointer text-xs font-medium text-zinc-700 dark:text-zinc-300 bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 px-2.5 py-1.5 rounded-lg shadow-sm">
                  <input 
                    type="checkbox" 
                    className="rounded border-zinc-300 text-blue-600 focus:ring-blue-500"
                    checked={hideSubmitted}
                    onChange={(e) => setHideSubmitted(e.target.checked)}
                  />
                  <span>Hide Submits</span>
                </label>

                <select
                  value={interactionFilter}
                  onChange={e => setInteractionFilter(e.target.value as "all" | "unclicked" | "clicked")}
                  className="text-xs px-2.5 py-1.5 rounded-lg bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 text-zinc-700 dark:text-zinc-300 focus:outline-none focus:ring-1 focus:ring-blue-500 shadow-sm"
                >
                  <option value="all">All Status</option>
                  <option value="unclicked">Unclicked</option>
                  <option value="clicked">Clicked</option>
                </select>

                <div className="relative">
                  <button
                    onClick={() => openFilterDropdown()}
                    className="text-xs px-2.5 py-1.5 rounded-lg bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 text-zinc-700 dark:text-zinc-300 focus:outline-none focus:ring-1 focus:ring-blue-500 shadow-sm flex items-center gap-1.5"
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
                  className="text-xs px-2.5 py-1.5 rounded-lg bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 text-zinc-700 dark:text-zinc-300 focus:outline-none focus:ring-1 focus:ring-blue-500 shadow-sm"
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

            {/* Empty States */}
            {sortedResults.length === 0 && !isAnyLoading && (
              <div className="h-64 border-2 border-dashed border-zinc-300 dark:border-zinc-800 rounded-xl flex items-center justify-center text-zinc-400 text-sm">No active indices queried.</div>
            )}

            {draggableResults.length === 0 && sortedResults.length > 0 && !isAnyLoading && (
              <div className="h-64 border-2 border-dashed border-zinc-300 dark:border-zinc-700/50 rounded-xl flex items-center justify-center text-zinc-500 text-sm bg-zinc-50 dark:bg-zinc-950/20">
                No results match your filters.
              </div>
            )}

            {/* Results Grid - Grouped */}
            {draggableResults.length > 0 && (
              <div className="flex flex-col gap-8 pb-12">
                {Object.entries(groupedResults).map(([groupName, scenes]) => (
                    <div key={groupName} className="flex flex-col gap-3">
                        
                        {groupBy !== 'none' && (
                            <div className="flex items-center gap-3">
                                <h4 className="text-sm font-bold text-zinc-800 dark:text-zinc-200 capitalize">{groupName}</h4>
                                <div className="h-px flex-1 bg-zinc-200 dark:bg-zinc-800" />
                                <span className="text-xs font-medium text-zinc-500">{scenes.length} items</span>
                            </div>
                        )}

                        <div className={`grid ${getGridClass()}`}>
                            {scenes.map((scene, index) => {
                                const scoreList = Object.entries(scene.modality_scores || {}).filter(([m]) => m !== 'rerank');
                                const rerankScore = scene.modality_scores?.['rerank'];

                                const isClicked = clickedSceneIds.has(scene.frame_id);
                                const isSubmitted = submittedSceneIds.has(scene.frame_id);

                                let cardStyle = "bg-white dark:bg-zinc-900 border-zinc-200 dark:border-zinc-800";
                                if (isSubmitted) cardStyle = "bg-emerald-50/60 dark:bg-emerald-950/20 border-emerald-400 shadow-inner";
                                else if (isClicked) cardStyle = "bg-yellow-100/80 dark:bg-yellow-800/40 border-zinc-300 dark:border-zinc-700/80";

                                const isDraggingThis = draggedId === scene.frame_id;
                                const outputFormat = getOutputString(scene as ExtendedScene);

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
                                    className={`border rounded-xl p-3 flex flex-col justify-between transition-all duration-200 cursor-grab active:cursor-grabbing ${cardStyle} ${isDraggingThis ? 'opacity-40 scale-[0.98]' : ''}`}
                                >
                                    <div>
                                    <div className="relative aspect-video w-full bg-black/5 dark:bg-white/5 rounded-lg overflow-hidden mb-3 border border-zinc-200 dark:border-zinc-800">
                                        <img src={`${API_PROXY}/video/frame/${scene.video_id}/${scene.frame_id}`} alt={scene.frame_id} className="object-cover w-full h-full" loading="lazy" />
                                    </div>
                                    
                                    <div className="flex justify-between items-start mb-2 gap-2">
                                        <span className={`font-bold font-mono px-2 py-0.5 rounded bg-zinc-200/60 dark:bg-zinc-800/80 truncate ${cardSize === 'sm' ? 'text-[10px] max-w-[100px]' : 'text-xs max-w-[150px]'}`}>{scene.video_id}</span>
                                        <p className={`${cardSize === 'sm' ? 'text-[10px]' : 'text-xs'} text-zinc-500 font-mono mt-0.5`}>@ {scene.timestamp.toFixed(1)}s</p>
                                    </div>
                                    
                                    {cardSize !== 'sm' && (
                                        <div className="flex items-center gap-1 flex-wrap mb-2">
                                        {scoreList.map(([method, score]) => (
                                            <div key={method} className={`flex items-center gap-1 text-[9px] px-1.5 py-0.5 rounded border border-black/5 dark:border-white/5 ${METHOD_COLORS[method] || 'bg-zinc-200 text-zinc-800'}`}>
                                            <span className="font-semibold uppercase tracking-tight">{method}</span>
                                            <span className="font-mono font-bold opacity-80">{(score * 100).toFixed(0)}%</span>
                                            </div>
                                        ))}
                                        </div>
                                    )}

                                    {cardSize === 'lg' && rerankScore !== undefined && (
                                        <div className="mb-2">
                                        <span className="text-[10px] font-mono bg-indigo-50 dark:bg-indigo-950/40 text-indigo-600 px-1.5 py-0.5 rounded border border-indigo-100">
                                            Rerank Mod: <strong className="font-bold">+{Math.round(rerankScore * 100)}%</strong>
                                        </span>
                                        </div>
                                    )}

                                    {cardSize !== 'sm' && (
                                        <div 
                                            className="bg-white/80 dark:bg-zinc-950/40 p-2 rounded text-xs border border-zinc-100 dark:border-zinc-800 mb-2 cursor-pointer hover:bg-zinc-50 dark:hover:bg-zinc-900 transition-colors"
                                            onClick={() => handleCopyOutput(outputFormat)}
                                            title="Click to copy output format"
                                        >
                                            <span className="text-zinc-800 dark:text-zinc-200 font-mono truncate block text-[10px]">{outputFormat}</span>
                                        </div>
                                    )}

                                    {cardSize === 'lg' && scene.ocr_text && (
                                        <div className="bg-white/80 dark:bg-zinc-950/40 p-2.5 rounded text-xs border border-zinc-100 dark:border-zinc-800 mb-3 h-14 overflow-hidden">
                                        <span className="font-bold block text-[9px] uppercase text-zinc-400 mb-0.5">OCR Content</span>
                                        <span className="text-zinc-700 dark:text-zinc-300 italic line-clamp-2 text-[11px]">&ldquo;{scene.ocr_text}&rdquo;</span>
                                        </div>
                                    )}
                                    </div>

                                    <div className={`flex ${cardSize === 'sm' ? 'flex-col gap-1.5' : 'gap-2 mt-auto pt-2'}`}>
                                    <button onClick={() => onSelectResult(scene)} className={`flex-1 bg-zinc-900 dark:bg-zinc-200 hover:bg-zinc-800 text-white dark:text-zinc-900 font-medium transition-colors rounded ${cardSize === 'sm' ? 'text-[10px] py-1.5' : 'text-xs py-2'}`}>
                                        Review
                                    </button>
                                    <button onClick={() => onFinalSubmit(scene.frame_id)} className={`font-semibold transition-colors border rounded ${cardSize === 'sm' ? 'text-[10px] py-1.5 px-2' : 'text-xs py-2 px-3'} ${isSubmitted ? 'bg-red-50 text-red-600 border-red-200 hover:bg-red-100 dark:bg-red-950/30 dark:text-red-400' : 'border-zinc-200 hover:bg-zinc-50 dark:border-zinc-700 dark:hover:bg-zinc-800'}`}>
                                        {isSubmitted ? 'Unsubmit' : 'Submit'}
                                    </button>
                                    </div>
                                </div>
                                );
                            })}
                        </div>
                    </div>
                ))}
              </div>
            )}
          </div>
        </main>
      </div>
  );
};

export default SearchBoard;