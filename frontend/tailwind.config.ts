import type { Config } from "tailwindcss";

const config: Config = {
    content: [
        "./app/**/*.{js,ts,jsx,tsx,mdx}",
        "./components/**/*.{js,ts,jsx,tsx,mdx}", // Your UI and Blocks live here
    ],
    theme: {
        // shadcn will inject its specific color and radius variables here (when `npx shadcn-ui@latest init`)
        extend: {
            
        },
    },
    plugins: [],
};
export default config;