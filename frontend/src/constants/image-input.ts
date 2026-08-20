export type ImageInputMode = 'upload' | 'link' | 'generate';
export type ColorScheme = 'none' | 'black' | 'white' | 'red' | 'blue' | 'green' | 'yellow';

export const COLOR_OPTIONS: { id: ColorScheme; colorHex: string; border?: string }[] = [
    { id: 'none', colorHex: 'transparent', border: 'border-2 border-dashed border-gray-400' },
    { id: 'black', colorHex: '#000000' },
    { id: 'white', colorHex: '#ffffff', border: 'border border-gray-300' },
    { id: 'red', colorHex: '#ef4444' },
    { id: 'blue', colorHex: '#3b82f6' },
    { id: 'green', colorHex: '#22c55e' },
    { id: 'yellow', colorHex: '#eab308' },
];