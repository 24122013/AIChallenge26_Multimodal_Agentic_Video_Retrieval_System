import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const hook = readFileSync(new URL('../src/hooks/useSearch.ts', import.meta.url), 'utf8');
const board = readFileSync(new URL('../src/components/SearchBoard.tsx', import.meta.url), 'utf8');
const display = readFileSync(new URL('../src/components/result-display/TrakeDisplay.tsx', import.meta.url), 'utf8');
const sidebar = readFileSync(new URL('../src/components/searchbar/SearchSideBar.tsx', import.meta.url), 'utf8');

test('TRAKE search owns cancellation and a configurable timeout', () => {
    assert.match(hook, /new AbortController\(\)/);
    assert.match(hook, /VITE_TRAKE_TIMEOUT_MS/);
    assert.match(hook, /signal: controller\.signal/);
    assert.match(hook, /cancelSearch/);
});

test('SearchBoard keeps a visible progress panel with elapsed time and cancel', () => {
    assert.doesNotMatch(board, /if \(isAnyLoading\) return null/);
    assert.match(board, /TRAKE sequence search in progress/);
    assert.match(board, /elapsedSeconds/);
    assert.match(board, /onCancelSearch/);
});

test('TrakeDisplay explains empty and insufficient-support responses', () => {
    assert.match(display, /No complete TRAKE sequence found/);
    assert.match(display, /insufficient_support/);
    assert.match(display, /did not return a partial or low-support sequence/);
});

test('TRAKE displays the final original frame index for every event image', () => {
    assert.match(display, /hypothesis\.frame_ids\[eIdx\]/);
    assert.match(display, /frame_idx: \{frameIdx\}/);
});

test('TRAKE requests are fixed at 100 results in both search UI layers', () => {
    assert.match(hook, /queryParam\.mode\.toLowerCase\(\) === 'trake'[\s\S]*?\? 100/);
    assert.match(sidebar, /mode: 'trake',[\s\S]*?top_k: 100/);
    assert.match(sidebar, /fixedTopK=\{selectedModel === "TRAKE" \? 100 : undefined\}/);
});

test('SearchBoard exports the current KIS, QA, or TRAKE response without rerunning search', () => {
  assert.match(board, /Export to CSV/);
  assert.match(board, /\/search\/export-current/);
  assert.match(board, /query_id: queryId\.trim\(\)/);
  assert.match(board, /task: exportTask/);
  assert.match(board, /data: exportData/);
  assert.match(board, /top_k: 100/);
  assert.match(board, /candidates: sortedResults/);
  assert.match(board, /window\.prompt/);
});

test('QA allows a manual answer override when automatic answer mode is off', () => {
  assert.match(board, /QA answer for CSV/);
  assert.match(board, /manualQaAnswer/);
  assert.match(board, /manual_answer:/);
  assert.match(board, /Automatic answer mode is off/);
  assert.match(board, /maxLength=\{100\}/);
});
