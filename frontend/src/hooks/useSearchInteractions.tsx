import { useState, useRef } from 'react';

export function useSearchInteractions() {
  const [currentQuery, setCurrentQuery] = useState<string>('');
  const [currentSearchId, setCurrentSearchId] = useState<string | null>(null);
  const [clickedSceneIds, setClickedSceneIds] = useState<Set<string>>(new Set());
  const [submittedSceneIds, setSubmittedSceneIds] = useState<Set<string>>(new Set());
  const lastSyncRef = useRef({ id: '', count: 0 });

  return {
    currentQuery, setCurrentQuery,
    currentSearchId, setCurrentSearchId,
    clickedSceneIds, setClickedSceneIds,
    submittedSceneIds, setSubmittedSceneIds,
    lastSyncRef
  };
}