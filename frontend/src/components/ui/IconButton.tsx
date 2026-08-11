import * as React from "react";
import { cn } from "../../libs/utils";

export interface IconButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    icon: React.ReactNode;
}

const IconButton = React.forwardRef<HTMLButtonElement, IconButtonProps>(
    ({ className, icon, ...props }, ref) => {
        return (
            <button
                ref={ref}
                // Inject the raw color directly to the DOM node
                style={{ backgroundColor: "oklch(48.8% 0.243 264.376)" }}
                className={cn(
                    "flex h-10 w-10 shrink-0 items-center justify-center rounded-md text-white",
                    // Use standard CSS filters for hover/active states
                    "transition-all duration-200 ease-in-out",
                    "hover:brightness-110 active:scale-95 active:brightness-90",
                    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2",
                    "disabled:pointer-events-none disabled:opacity-50",
                    "hover:bg-black/10 dark:hover:bg-white/10 focus-visible:ring-1 focus-visible:ring-offset-0 focus-visible:ring-blue-500",
                    className
                )}
                {...props}
            >
                {icon}
            </button>
        );
    }
);

IconButton.displayName = "IconButton";

export default IconButton;