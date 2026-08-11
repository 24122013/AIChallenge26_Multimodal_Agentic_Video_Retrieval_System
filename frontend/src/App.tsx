import { useEffect, useRef, useCallback } from 'react';
import type { VideoScene, SearchPayload, SearchLog } from './types';
import { useTelemetry } from './hooks/useTelemetry';
import { useSearch } from './hooks/useSearch';
import VideoPlayer from './components/blocks/VideoPlayer';
import TelemetryLogModal from './components/blocks/TelemetryLogModal';
import SearchBoard from './components/blocks/SearchBoard';
import useVideo from './hooks/useVideo';
import { useSearchInteractions } from './hooks/useSearchInteractions';
import { useResultFilters } from './hooks/useResultFilter';
import { RETRIEVAL_METHODS } from './constants/mode-icons';

export default function App() {
  const { sendTelemetry, searchLogs, setSearchLogs, isLogModalOpen, setIsLogModalOpen } = useTelemetry();
  const { sortedResults, isSearching, latency, sortBy, setSortBy, executeSearch } = useSearch();
  const {
    currentQuery, setCurrentQuery, currentSearchId, setCurrentSearchId,
    clickedSceneIds, setClickedSceneIds, submittedSceneIds, setSubmittedSceneIds, lastSyncRef
  } = useSearchInteractions();
  const {
    activeSources, setActiveSources, isFilterDropdownOpen, setIsFilterDropdownOpen,
    displayedResults
  } = useResultFilters(RETRIEVAL_METHODS, sortedResults);
  const {
    activeVideo, setActiveVideo, neighborFrames, setNeighborFrames,
    lastSelectedScene, handleSelectResultVideo, handleNextVideo, handlePrevVideo
  } = useVideo(displayedResults);

  const handleSelectResult = useCallback((scene: VideoScene) => {
    setClickedSceneIds(prev => new Set([...prev, scene.frame_id]));
    handleSelectResultVideo(scene);
    sendTelemetry('click_result', { frame_id: scene.frame_id }, 0);
  }, [setClickedSceneIds, handleSelectResultVideo, sendTelemetry]);

  // Context Ref mapping for global Ctrl+P execution
  const contextRef = useRef({ activeVideo, lastSelectedScene, displayedResults });
  useEffect(() => {
    contextRef.current = { activeVideo, lastSelectedScene, displayedResults };
  }, [activeVideo, lastSelectedScene, displayedResults]);

  // Global Key Handlers
  useEffect(() => {
    const handleGlobalKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setIsLogModalOpen(false);
        setIsFilterDropdownOpen(false);
      }

      // Ctrl + P to open the player pop up
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'p') {
        e.preventDefault();
        const { activeVideo: currActive, lastSelectedScene: lastScene, displayedResults: results } = contextRef.current;

        if (!currActive) {
          if (lastScene) {
            handleSelectResult(lastScene);
          } else if (results.length > 0) {
            handleSelectResult(results[0]);
          }
        }
      }
    };
    window.addEventListener('keydown', handleGlobalKeyDown);
    return () => window.removeEventListener('keydown', handleGlobalKeyDown);
  }, [setIsLogModalOpen, setIsFilterDropdownOpen, handleSelectResult]);

  const handleExecuteSearch = useCallback(async (payload: SearchPayload, retraceLogId?: string) => {
    const searchId = retraceLogId || crypto.randomUUID();
    setCurrentSearchId(searchId);

    const clipQuery = payload.text_queries.find(q => q.mode === 'visual')?.query;
    const ocrQuery = payload.text_queries.find(q => q.mode === 'ocr')?.query;

    const queryStr = clipQuery || ocrQuery || 'multimodal criteria context';
    setCurrentQuery(queryStr);

    const activeMethods: string[] = [];
    if (clipQuery) activeMethods.push('visual');
    if (ocrQuery) activeMethods.push('ocr');
    if (payload.image?.mode && payload.image.mode !== 'upload') activeMethods.push('image');
    if (payload.colorHex && payload.colorHex !== 'transparent') activeMethods.push('color');

    await executeSearch(payload);

    if (!retraceLogId) {
      setSearchLogs(prev => [
        {
          id: searchId,
          query: queryStr,
          payload,
          resultsCount: sortedResults.length || 0,
          latency: latency,
          submissionsCount: 0,
          methods: activeMethods,
          results: []
        },
        ...prev
      ]);
    }
  }, [
    executeSearch, 
    latency, 
    sortedResults.length, 
    setCurrentSearchId, 
    setCurrentQuery, 
    setSearchLogs
  ]);

  const handleRetraceLog = useCallback((log: SearchLog) => {
    setIsLogModalOpen(false);
    handleExecuteSearch(log.payload, log.id);
  }, [handleExecuteSearch, setIsLogModalOpen]);

  useEffect(() => {
    if (!currentSearchId || sortedResults.length === 0) return;

    if (
      lastSyncRef.current.id !== currentSearchId || 
      lastSyncRef.current.count !== sortedResults.length
    ) {
      lastSyncRef.current = { id: currentSearchId, count: sortedResults.length };

      setSearchLogs(prev => {
        const currentLog = prev.find(l => l.id === currentSearchId);
        if (currentLog && currentLog.resultsCount === sortedResults.length) return prev;

        return prev.map(log =>
          log.id === currentSearchId
            ? {
              ...log,
                resultsCount: sortedResults.length,
                results: sortedResults.map(r => ({ 
                id: r.frame_id, 
                name: r.video_id, 
                score: r.score ?? 0 
              }))
            }
            : log
        );
      });
    }
  }, [lastSyncRef, sortedResults, currentSearchId, setSearchLogs]);

  const handleFinalSubmit = useCallback((frameId: string) => {
    // Determine if we are submitting or unsubmitting based on current state
    const isUnsubmitting = submittedSceneIds.has(frameId);

    setSubmittedSceneIds(prev => {
      const newSet = new Set(prev);
      if (isUnsubmitting) {
        newSet.delete(frameId);
      } else {
        newSet.add(frameId);
      }
      return newSet;
    });

    if (currentSearchId) {
      setSearchLogs(prev => prev.map(log =>
        log.id === currentSearchId 
          ? { ...log, submissionsCount: Math.max(0, log.submissionsCount + (isUnsubmitting ? -1 : 1)) } 
          : log
      ));
    }
    
    // 3. Send appropriate telemetry event
    const telemetryEvent = isUnsubmitting ? 'click_result' : 'submit_result';
    sendTelemetry(telemetryEvent, { chosen_frame_id: frameId, matching_query: currentQuery });

  }, [currentSearchId, currentQuery, submittedSceneIds, setSubmittedSceneIds, setSearchLogs, sendTelemetry]);

  return (
    <div className="flex flex-col min-h-screen w-full bg-slate-100 dark:bg-[#09090b] font-sans text-slate-900 dark:text-slate-100 relative">

      <VideoPlayer
        activeVideo={activeVideo}
        neighborFrames={neighborFrames}
        onClose={() => { setActiveVideo(null); setNeighborFrames([]); }}
        onFinalSubmit={handleFinalSubmit}
        onNext={handleNextVideo}
        onPrev={handlePrevVideo}
        displayedResults={displayedResults}
        onSelectResult={handleSelectResult}
      />

      <TelemetryLogModal
        isOpen={isLogModalOpen}
        onClose={() => { setIsLogModalOpen(false) }}
        searchLogs={searchLogs}
        currentSearchId={currentSearchId}
        onRetraceLog={handleRetraceLog}
      />

      <SearchBoard
        sortedResults={sortedResults}
        displayedResults={displayedResults}
        isSearching={isSearching}
        latency={latency}
        isFilterDropdownOpen={isFilterDropdownOpen}
        openFilterDropdown={() => setIsFilterDropdownOpen(true)}
        closeFilterDropdown={() => setIsFilterDropdownOpen(false)}
        sortBy={sortBy}
        setSortBy={setSortBy}
        activeSources={activeSources}
        toggleSourceFilter={(source: string) => {
          setActiveSources(prev => {
            const newSet = new Set(prev);
            if (newSet.has(source)) newSet.delete(source);
            else newSet.add(source);
            return newSet;
          });
        }}
        resetSources={() => setActiveSources(new Set(RETRIEVAL_METHODS))}
        availableSources={RETRIEVAL_METHODS}
        clickedSceneIds={clickedSceneIds}
        submittedSceneIds={submittedSceneIds}
        searchLogsLength={searchLogs.length}
        onExecuteSearch={handleExecuteSearch}
        onOpenLogs={() => setIsLogModalOpen(true)}
        onSelectResult={(scene: VideoScene) => handleSelectResult(scene)}
        onFinalSubmit={handleFinalSubmit}
      />
    </div>
  );
}