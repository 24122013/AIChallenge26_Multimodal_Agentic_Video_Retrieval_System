import { useRef, useCallback, useMemo } from 'react';
import type { VideoScene, SearchPayload, SearchLog, ActiveTask } from './types';
import { useTelemetry } from './hooks/useTelemetry';
import { useSearch } from './hooks/useSearch';
import VideoPlayer from './components/VideoPlayer';
import TelemetryLogModal from './components/TelemetryLogModal';
import SearchBoard from './components/SearchBoard';
import useVideo from './hooks/useVideo';
import { useSearchInteractions } from './hooks/useSearchInteractions';
import { useResultFilters } from './hooks/useResultFilter';
import { RETRIEVAL_METHODS } from './constants/mode-icons';

export default function App() {
  const { sendTelemetry, searchLogs, setSearchLogs, isLogModalOpen, setIsLogModalOpen } = useTelemetry();
  const { sortedResults, isSearching, searchError, latency, sortBy, setSortBy, executeSearch, apiResponseData, elapsedSeconds, searchStage, cancelSearch } = useSearch();
  
  const {
    currentQuery, setCurrentQuery, currentSearchId, setCurrentSearchId,
    clickedSceneIds, setClickedSceneIds,
    submittedSceneIds, setSubmittedSceneIds,
    clickedTrakeIds, setClickedTrakeIds,
    submittedTrakeIds, setSubmittedTrakeIds
  } = useSearchInteractions();
  
  const {
    activeSources, setActiveSources, isFilterDropdownOpen, setIsFilterDropdownOpen,
    displayedResults
  } = useResultFilters(RETRIEVAL_METHODS, sortedResults);
  
  const {
    activeVideo, setActiveVideo, neighborData, setNeighborData,
    handleSelectResultVideo, handleNextVideo, handlePrevVideo
  } = useVideo(displayedResults);

  const searchStartTimesRef = useRef<Record<string, number>>({});

  const currentLog = searchLogs.find(l => l.id === currentSearchId);
  const activeTask = useMemo(() => {
    return (currentLog?.taskType?.toUpperCase() || 'KIST') as ActiveTask;
  }, [currentLog?.taskType]);

  const handleSelectResult = useCallback((scene: VideoScene) => {
    const clickTime = Date.now();
    
    if (activeTask === 'TRAKE') {
        setClickedTrakeIds(prev => new Set([...prev, scene.frame_id]));
    } else {
        setClickedSceneIds(prev => new Set([...prev, scene.frame_id]));
    }
    
    handleSelectResultVideo(scene);
    sendTelemetry('click_result', { frame_id: scene.frame_id }, 0);

    if (currentSearchId) {
      setSearchLogs(prev => prev.map(log => 
        log.id === currentSearchId 
          ? { 
              ...log, 
              clickedCandidates: [...(log.clickedCandidates || []), { id: scene.frame_id, timestamp: clickTime }] 
            }
          : log
      ));
    }
  }, [activeTask, setClickedSceneIds, setClickedTrakeIds, handleSelectResultVideo, sendTelemetry, currentSearchId, setSearchLogs]);

  const handleExecuteSearch = useCallback(async (payload: SearchPayload, retraceLogId?: string) => {
    const searchId = retraceLogId || crypto.randomUUID();
    setCurrentSearchId(searchId);
    
    const startTime = Date.now();
    if (!retraceLogId) {
      searchStartTimesRef.current[searchId] = startTime;
    }

    const primaryQuery = payload.text_queries.find(q => q.mode !== 'qa' && q.mode !== 'trake');
    const queryStr = primaryQuery?.query || payload.text_queries[0]?.query || 'multimodal criteria context';
    const taskType = payload.config.model === "QA" ? "qa"
      : payload.config.model === "TRAKE" ? "trake"
      : payload.config.model === "TEMPORAL" ? "temporal"
      : "kist";
    const searchMode = primaryQuery?.mode || payload.text_queries[0]?.mode || 'visual';

    setCurrentQuery(queryStr);
    
    if (!retraceLogId) {
      setSearchLogs(prev => [
        {
          id: searchId,
          taskId: `task-${crypto.randomUUID().slice(0, 8)}`,
          taskType: taskType,
          searchMode: searchMode,
          query: queryStr,
          payload,
          resultsCount: 0,
          latency: 0,
          solveTime: 0,
          submissionsCount: 0,
          clickedCandidates: [],
          submittedCandidates: [],
          correctness: null,
          methods: [searchMode],
          results: []
        },
        ...prev
      ]);
    }
    
    await executeSearch(payload);
  }, [executeSearch, setCurrentSearchId, setCurrentQuery, setSearchLogs]);

  const handleRetraceLog = useCallback((log: SearchLog) => {
    setIsLogModalOpen(false);
    handleExecuteSearch(log.payload, log.id);
  }, [handleExecuteSearch, setIsLogModalOpen]);

  const handleFinalSubmit = useCallback((id: string) => {
    const isTrake = activeTask === 'TRAKE';
    const isUnsubmitting = isTrake ? submittedTrakeIds.has(id) : submittedSceneIds.has(id);
    const submitTime = Date.now();

    if (isTrake) {
        setSubmittedTrakeIds(prev => {
            const newSet = new Set(prev);
            if (isUnsubmitting) {
                newSet.delete(id);
            } else {
                newSet.add(id);
            }
            return newSet;
        });
    } else {
        setSubmittedSceneIds(prev => {
            const newSet = new Set(prev);
            if (isUnsubmitting) {
                newSet.delete(id);
            } else {
                newSet.add(id);
            }
            return newSet;
        });
    }

    if (currentSearchId) {
      setSearchLogs(prev => prev.map(log => {
        if (log.id === currentSearchId) {
          const newSubmits = isUnsubmitting 
            ? log.submittedCandidates.filter(s => s.id !== id)
            : [...(log.submittedCandidates || []), { id: id, timestamp: submitTime }];
            
          let newSolveTime = log.solveTime;
          if (!isUnsubmitting && log.solveTime === 0 && searchStartTimesRef.current[currentSearchId]) {
            newSolveTime = submitTime - searchStartTimesRef.current[currentSearchId];
          }

          return { 
            ...log, 
            submissionsCount: Math.max(0, log.submissionsCount + (isUnsubmitting ? -1 : 1)),
            submittedCandidates: newSubmits,
            solveTime: newSolveTime
          };
        }
        return log;
      }));
    }
    
    const telemetryEvent = isUnsubmitting ? 'click_result' : 'submit_result';
    sendTelemetry(telemetryEvent, { chosen_id: id, matching_query: currentQuery });
  }, [activeTask, currentSearchId, currentQuery, submittedSceneIds, setSubmittedSceneIds, submittedTrakeIds, setSubmittedTrakeIds, setSearchLogs, sendTelemetry]);

  return (
    <div className="flex flex-col min-h-screen w-full bg-slate-100 dark:bg-[#09090b] font-sans text-slate-900 dark:text-slate-100 relative">
      <VideoPlayer
        activeVideo={activeVideo}
        neighborData={neighborData}
        onClose={() => { setActiveVideo(null); setNeighborData(null); }}
        onFinalSubmit={handleFinalSubmit}
        onNext={handleNextVideo}
        onPrev={handlePrevVideo}
        displayedResults={displayedResults}
        onSelectResult={handleSelectResult}
      />
      
      <TelemetryLogModal
        isOpen={isLogModalOpen}
        onClose={() => { setIsLogModalOpen(false); }}
        searchLogs={searchLogs}
        currentSearchId={currentSearchId}
        onRetraceLog={handleRetraceLog}
      />
      
      <SearchBoard
        activeTask={activeTask}
        apiResponseData={apiResponseData}
        sortedResults={sortedResults}
        displayedResults={displayedResults}
        isSearching={isSearching}
        searchError={searchError}
        latency={latency}
        elapsedSeconds={elapsedSeconds}
        searchStage={searchStage}
        onCancelSearch={cancelSearch}
        isFilterDropdownOpen={isFilterDropdownOpen}
        openFilterDropdown={() => setIsFilterDropdownOpen(true)}
        closeFilterDropdown={() => setIsFilterDropdownOpen(false)}
        sortBy={sortBy}
        setSortBy={setSortBy}
        activeSources={activeSources}
        toggleSourceFilter={(source: string) => {
          setActiveSources(prev => {
            const newSet = new Set(prev);
            if (newSet.has(source)) {
              newSet.delete(source);
            } else {
              newSet.add(source);
            }
            return newSet;
          });
        }}
        resetSources={() => setActiveSources(new Set(RETRIEVAL_METHODS))}
        availableSources={RETRIEVAL_METHODS}
        searchLogsLength={searchLogs.length}
        onExecuteSearch={handleExecuteSearch}
        onOpenLogs={() => setIsLogModalOpen(true)}
        onSelectResult={handleSelectResult}
        onFinalSubmit={handleFinalSubmit}
        clickedSceneIds={clickedSceneIds}
        submittedSceneIds={submittedSceneIds}
        clickedTrakeIds={clickedTrakeIds}
        submittedTrakeIds={submittedTrakeIds}
      />
    </div>
  );
}
