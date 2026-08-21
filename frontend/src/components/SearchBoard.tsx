import React, { useState, useMemo } from 'react';
import type { VideoScene, SortKey, SearchPayload, ActiveTask, ApiResponse, SearchData, QAData, TrakeData, TemporalData } from '../types';
import { SORT_OPTIONS } from '../constants/video-scene-sort-option';
import SearchSidebar from './searchbar/SearchSideBar';
import { Grid, Rows, LayoutGrid } from 'lucide-react';

import KistDisplay from './result-display/KistDisplay';
import QaDisplay from './result-display/QaDisplay';
import TrakeDisplay from './result-display/TrakeDisplay';
import TemporalDisplay from './result-display/TemporalDisplay';

type CardSize = 'sm' | 'md' | 'lg';
type GroupingMode = 'none' | 'video' | 'modality' | 'tens';

interface SearchBoardProps {
  activeTask: ActiveTask;
  apiResponseData?: ApiResponse<SearchData> | null;
  sortedResults: VideoScene[]; 
  displayedResults: VideoScene[];
  isSearching: boolean;
  searchError: string | null;
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
  searchLogsLength: number;
  onExecuteSearch: (payload: SearchPayload) => void;
  onOpenLogs: () => void;
  onSelectResult: (scene: VideoScene) => void;
  onFinalSubmit: (sceneId: string) => void;
  clickedSceneIds: Set<string>;
  submittedSceneIds: Set<string>;
  clickedTrakeIds: Set<string>;
  submittedTrakeIds: Set<string>;
}

const SearchBoard: React.FC<SearchBoardProps> = ({
  activeTask,
  apiResponseData,
  sortedResults, 
  displayedResults,
  isSearching,
  searchError,
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
  searchLogsLength,
  onExecuteSearch,
  onOpenLogs,
  onSelectResult,
  onFinalSubmit,
  clickedSceneIds = new Set(),
  submittedSceneIds = new Set(),
  clickedTrakeIds = new Set(),
  submittedTrakeIds = new Set()
}) => {
  const isAnyLoading = isSearching;
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

  const renderDisplay = () => {
    if (isAnyLoading) return null;

    const dataPayload = apiResponseData?.data;
    const resolvedTask = (dataPayload?.task || activeTask)?.toLowerCase();

    if (resolvedTask === 'qa' && dataPayload) {
        return (
            <QaDisplay 
                qaData={dataPayload as QAData}
                results={filteredResults}
                cardSize={cardSize}
                groupBy={groupBy}
                onSelectResult={onSelectResult}
                onFinalSubmit={onFinalSubmit}
                clickedSceneIds={clickedSceneIds}
                submittedSceneIds={submittedSceneIds}
            />
        );
    }

    if (resolvedTask === 'trake' && dataPayload) {
        return (
            <TrakeDisplay 
                trakeData={dataPayload as TrakeData}
                cardSize={cardSize}
                onSelectResult={onSelectResult}
                onFinalSubmit={onFinalSubmit}
                clickedTrakeIds={clickedTrakeIds}
                submittedTrakeIds={submittedTrakeIds}
            />
        );
    }

    if (resolvedTask === 'temporal' && (dataPayload as TemporalData)?.temporal_matches) {
        return (
            <TemporalDisplay
                temporalData={dataPayload as TemporalData}
                cardSize={cardSize}
                onSelectResult={onSelectResult}
                clickedSceneIds={clickedSceneIds}
                submittedSceneIds={submittedSceneIds}
            />
        );
    }

    const hasAnyResults = filteredResults.length > 0;

    if (!hasAnyResults && sortedResults.length > 0) {
        return (
            <div className="h-64 border-2 border-dashed border-zinc-300 dark:border-zinc-700/50 rounded-xl flex items-center justify-center text-zinc-500 text-sm bg-zinc-50 dark:bg-zinc-950/20">
                No results match your filters.
            </div>
        );
    }

    if (hasAnyResults) {
        return (
            <KistDisplay 
                results={filteredResults}
                cardSize={cardSize}
                groupBy={groupBy}
                onSelectResult={onSelectResult}
                onFinalSubmit={onFinalSubmit}
                clickedSceneIds={clickedSceneIds}
                submittedSceneIds={submittedSceneIds}
            />
        );
    }

    return (
        <div className="h-64 border-2 border-dashed border-zinc-300 dark:border-zinc-800 rounded-xl flex items-center justify-center text-zinc-400 text-sm">
            No active indices queried.
        </div>
    );
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
            <div className="flex flex-wrap items-center justify-between gap-4 mb-6 bg-white dark:bg-zinc-900 p-4 rounded-xl shadow-sm border border-zinc-200 dark:border-zinc-800">
              <div className="flex items-center gap-3">
                <h3 className="text-sm font-bold tracking-wide text-zinc-800 dark:text-zinc-200 uppercase">
                    {activeTask} Results
                </h3>
                <button
                  onClick={() => onOpenLogs()}
                  className="text-xs bg-zinc-800 text-zinc-100 hover:bg-zinc-700 dark:bg-zinc-200 dark:text-zinc-900 dark:hover:bg-zinc-100 font-medium px-2.5 py-1.5 rounded shadow transition-all"
                >
                  View Process Logs ({searchLogsLength})
                </button>
              </div>

              <div className="flex flex-wrap items-center gap-3">
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

                {(activeTask === 'KIST' || activeTask === 'QA' || activeTask === 'TEMPORAL') && (
                    <>
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
                    </>
                )}

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
                  onChange={e => setInteractionFilter(e.target.value as 'all' | 'clicked' | 'unclicked')}
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

            {searchError && (
              <div role="alert" className="mb-5 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900/60 dark:bg-red-950/30 dark:text-red-300">
                {searchError}
              </div>
            )}
            {renderDisplay()}
          </div>
        </main>
      </div>
  );
};

export default SearchBoard;
