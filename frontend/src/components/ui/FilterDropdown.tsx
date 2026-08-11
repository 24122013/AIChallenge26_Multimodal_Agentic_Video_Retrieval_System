import { useState, useRef, useEffect } from 'react';

interface FilterOption {
    id: string;
    label: string;
    colorClass?: string;
}

interface FilterDropdownProps {
    label: string;
    options: FilterOption[];
    activeIds: Set<string>;
    onChange: (newActiveIds: Set<string>) => void;
}

export function FilterDropdown({ label, options, activeIds, onChange }: FilterDropdownProps) {
    const [isOpen, setIsOpen] = useState(false);
    const dropdownRef = useRef<HTMLDivElement>(null);

    // Close dropdown on outside click
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const toggleOption = (id: string) => {
        const next = new Set(activeIds);
        if (next.has(id)) {
            next.delete(id);
        } else {
            next.add(id);
        }
        onChange(next);
    };

    const toggleAll = () => {
        if (activeIds.size === options.length) {
            onChange(new Set());
        } else {
            onChange(new Set(options.map(o => o.id)));
        }
    };

    return (
        <div className="relative" ref={dropdownRef}>
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="text-sm px-3 py-1.5 rounded-lg bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-50 dark:hover:bg-zinc-700/50 focus:outline-none focus:ring-2 focus:ring-blue-500 shadow-sm flex items-center gap-2 transition-colors"
            >
                <span>{label}</span>
                <span className="bg-zinc-100 dark:bg-zinc-900 px-1.5 py-0.5 rounded text-xs font-bold">
                    {activeIds.size}/{options.length}
                </span>
                <svg className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
            </button>

            {isOpen && (
                <div className="absolute top-full mt-2 left-0 w-48 bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-xl shadow-xl z-20 py-2 animate-fade-in">
                    <div className="px-3 pb-2 mb-2 border-b border-zinc-100 dark:border-zinc-700/50 flex justify-between items-center">
                        <span className="text-xs font-bold uppercase text-zinc-400">Sources</span>
                        <button onClick={toggleAll} className="text-[10px] text-blue-600 dark:text-blue-400 hover:underline">
                            {activeIds.size === options.length ? 'Clear All' : 'Select All'}
                        </button>
                    </div>
                    <div className="flex flex-col max-h-60 overflow-y-auto">
                        {options.map(option => (
                            <label key={option.id} className="flex items-center gap-3 px-3 py-1.5 hover:bg-zinc-50 dark:hover:bg-zinc-700/50 cursor-pointer">
                                <input
                                    type="checkbox"
                                    checked={activeIds.has(option.id)}
                                    onChange={() => toggleOption(option.id)}
                                    className="rounded border-zinc-300 text-blue-600 focus:ring-blue-500 bg-white dark:bg-zinc-900 dark:border-zinc-600"
                                />
                                <span className={`text-sm ${option.colorClass || 'text-zinc-700 dark:text-zinc-300'}`}>
                                    {option.label}
                                </span>
                            </label>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}