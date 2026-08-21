import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const sidebar = readFileSync(
    new URL('../src/components/searchbar/SearchSideBar.tsx', import.meta.url),
    'utf8',
);
const modes = readFileSync(
    new URL('../src/constants/mode-icons.tsx', import.meta.url),
    'utf8',
);

test('KIST temporal sends the public kis_temporal backend mode', () => {
    assert.match(modes, /KIST_MODES\s*=\s*\[[^\]]*"temporal"/s);
    assert.match(
        sidebar,
        /:\s*mode\s*===\s*['"]temporal['"]\s*\?\s*['"]kis_temporal['"]\s*:\s*mode/s,
    );
});
