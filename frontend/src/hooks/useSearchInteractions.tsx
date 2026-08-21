import { useState, useRef } from 'react';

export function useSearchInteractions() {
  // Global Search State
  const [currentQuery, setCurrentQuery] = useState<string>('');
  const [currentSearchId, setCurrentSearchId] = useState<string | null>(null);
  const lastSyncRef = useRef({ id: '', count: 0 });

  // KIST Task State
  const [clickedSceneIds, setClickedSceneIds] = useState<Set<string>>(new Set());
  const [submittedSceneIds, setSubmittedSceneIds] = useState<Set<string>>(new Set());

  // TRAKE Task State
  const [clickedTrakeIds, setClickedTrakeIds] = useState<Set<string>>(new Set());
  const [submittedTrakeIds, setSubmittedTrakeIds] = useState<Set<string>>(new Set());

  return {
    currentQuery, setCurrentQuery,
    currentSearchId, setCurrentSearchId,
    lastSyncRef,
    
    // KIST
    clickedSceneIds, setClickedSceneIds,
    submittedSceneIds, setSubmittedSceneIds,

    // TRAKE
    clickedTrakeIds, setClickedTrakeIds,
    submittedTrakeIds, setSubmittedTrakeIds
  };
}