import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Vite config for local development.
// Backend will run on http://localhost:8000 (FastAPI default).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
