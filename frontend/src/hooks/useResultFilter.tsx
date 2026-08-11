import { useState, useMemo } from 'react';
import type { VideoScene } from '../types'; // Adjust import as needed

export function useResultFilters(availableSources: string[], sortedResults: VideoScene[]) {
  const [activeSources, setActiveSources] = useState<Set<string>>(new Set(availableSources));
  const [isFilterDropdownOpen, setIsFilterDropdownOpen] = useState(false);

  // Moved derivation logic inside the hook
  const displayedResults = useMemo(() => {
    return sortedResults.filter(scene => {
      if (!scene.source) return true;
      return activeSources.has(scene.source);
    });
  }, [sortedResults, activeSources]);

  return {
    activeSources, setActiveSources,
    isFilterDropdownOpen, setIsFilterDropdownOpen,
    displayedResults
  };
}