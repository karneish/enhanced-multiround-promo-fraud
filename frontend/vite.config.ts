import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 3000,
    proxy: {
      '/api/main': {
        target: 'http://127.0.0.1:5051',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/main/, '/api'),
      },
      '/api/adl': {
        target: 'http://127.0.0.1:5052',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/adl/, '/api'),
      },
      '/api/gen': {
        target: 'http://127.0.0.1:5053',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/gen/, '/api'),
      },
      '/api/ens': {
        target: 'http://127.0.0.1:5054',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/ens/, '/api'),
      },
    },
  },
});
