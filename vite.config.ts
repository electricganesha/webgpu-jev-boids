import { defineConfig } from 'vite';

// https://vite.dev/config/
export default defineConfig({
  server: {
    port: 5173,
    open: false,
    proxy: {
      '/typesafe-proxy': {
        target: 'https://api.typesafe.ai',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/typesafe-proxy/, ''),
      },
    },
  },
});
