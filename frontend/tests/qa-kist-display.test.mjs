import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const qaDisplay = readFileSync(
    new URL('../src/components/result-display/QaDisplay.tsx', import.meta.url),
    'utf8',
);
const searchBoard = readFileSync(
    new URL('../src/components/SearchBoard.tsx', import.meta.url),
    'utf8',
);

test('QA renders related frames with the same card grid as KIST', () => {
    assert.match(qaDisplay, /import KistDisplay from ['"]\.\/KistDisplay['"]/);
    assert.match(qaDisplay, /<KistDisplay/);
    assert.match(qaDisplay, /onFinalSubmit=\{onFinalSubmit\}/);
    assert.match(qaDisplay, /submittedSceneIds=\{submittedSceneIds\}/);
});

test('QA applies the shared result filters and grouping controls', () => {
    assert.match(searchBoard, /<QaDisplay[\s\S]*results=\{filteredResults\}/);
    assert.match(searchBoard, /activeTask === 'QA'/);
});
