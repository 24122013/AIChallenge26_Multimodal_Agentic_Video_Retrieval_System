import { ChevronDown } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { MODE_ICONS } from "../constants/mode-icons";
import Button from './ui/button/Button';

export default function DropdownButton({ selectedModel }: { selectedModel: string }) {
    return (
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
    );
}