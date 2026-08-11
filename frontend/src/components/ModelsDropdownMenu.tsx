import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from "./ui/dropdown-menu";
import { cn } from '../libs/utils';
import { MODES, MODE_ICONS } from "../constants/mode-icons";
import { Bot, Check, ChevronDown } from "lucide-react";
import Button from "./ui/button/Button";
import { AnimatePresence, motion } from "framer-motion";

interface ModelDropdownMenuProps {
    selectedModel: string;
    onSelect: (model: string) => void;
}

export default function ModelDropdownMenu(
    { selectedModel, onSelect }: ModelDropdownMenuProps
) {
    return (
        <DropdownMenu>
            <DropdownMenuTrigger asChild>
            <Button
                variant="ghost"
                className="flex items-center gap-1 h-8 pl-1 pr-2 text-xs rounded-md dark:text-white hover:bg-black/10 dark:hover:bg-white/10 focus-visible:ring-1 focus-visible:ring-offset-0 focus-visible:ring-blue-500"
            >
                <AnimatePresence mode="wait">
                    <motion.div
                        key={selectedModel}
                        initial={{
                            opacity: 0,
                            y: -5,
                        }}
                        animate={{
                            opacity: 1,
                            y: 0,
                        }}
                        exit={{
                            opacity: 0,
                            y: 5,
                        }}
                        transition={{
                            duration: 0.15,
                        }}
                        className="flex items-center gap-1"
                    >
                        {
                            MODE_ICONS[
                                selectedModel
                            ]
                        }
                        {selectedModel}
                        <ChevronDown className="w-3 h-3 opacity-50" />
                    </motion.div>
                </AnimatePresence>
            </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent
                className={cn(
                    "min-w-[10rem]",
                    "border-black/10 dark:border-white/10",
                    "bg-gradient-to-b from-white via-white to-neutral-100 dark:from-neutral-950 dark:via-neutral-900 dark:to-neutral-800"
                )}
            >
                {MODES.map((model) => (
                    <DropdownMenuItem
                        key={model}
                        onSelect={() =>
                            onSelect(model)
                        }
                        className="flex items-center justify-between gap-2"
                    >
                        <div className="flex items-center gap-2">
                            {MODE_ICONS[model] || (
                                <Bot className="w-4 h-4 opacity-50" />
                            )}
                            <span>{model}</span>
                        </div>
                        {selectedModel ===
                            model && (
                            <Check className="w-4 h-4 text-blue-500" />
                        )}
                    </DropdownMenuItem>
                ))}
            </DropdownMenuContent>
        </DropdownMenu>
    );
};