// vite.config.ts
import { defineConfig } from 'vite';
import { API_PROXY } from './src/constants/proxy';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  server: {
    proxy: {
      [API_PROXY]: {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      }
    },
  },
});