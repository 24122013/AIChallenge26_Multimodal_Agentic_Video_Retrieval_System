import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const sidebar = readFileSync(
    new URL('../src/components/searchbar/SearchSideBar.tsx', import.meta.url),
    'utf8',
);
const types = readFileSync(
    new URL('../src/types/index.ts', import.meta.url),
    'utf8',
);
const board = readFileSync(
    new URL('../src/components/SearchBoard.tsx', import.meta.url),
    'utf8',
);

test('KIST Visual sends the canonical kis_visual backend mode', () => {
    assert.match(
        sidebar,
        /mode:\s*mode\s*===\s*['"]visual['"]\s*\?\s*['"]kis_visual['"]/s,
    );
    assert.match(types, /mode:\s*[^;]*"kis_visual"/s);
});

test('kis_visual response remains an ordinary KIS result display', () => {
    assert.match(board, /resolvedTask\s*===\s*['"]qa['"]/);
    assert.match(board, /resolvedTask\s*===\s*['"]trake['"]/);
    assert.match(board, /resolvedTask\s*===\s*['"]temporal['"]/);
    assert.match(board, /<KistDisplay/);
    assert.doesNotMatch(board, /resolvedTask\s*===\s*['"]kis_visual['"]/);
});
