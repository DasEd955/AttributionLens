import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Proxy /api/* to the Flask backend so fetch('/api/dashboard/stats') works
// in dev without CORS issues.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
