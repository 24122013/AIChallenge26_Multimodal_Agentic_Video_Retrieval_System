/// <reference types="vite/client" />

// Prolly needed for Tailwind/React projects:
declare module "*.gif" {
    const value: string;
    export default value;
}
declare module "*.png" {
    const value: string;
    export default value;
};
declare module "*.jpg";
declare module "*.jpeg";
declare module "*.svg?react" {
    import React from 'react';
    const value: React.FC<React.SVGProps<SVGSVGElement>>;
    export default value;
};
declare module "*.mp4";