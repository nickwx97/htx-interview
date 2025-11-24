import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./tests/setupTests.js'],
    include: ['tests/**/*.test.{js,jsx,ts,tsx}'],
    // Provide a minimal import.meta.env shim for tests so components can read VITE_API_BASE
    define: {
      'import.meta.env': JSON.stringify({ VITE_API_BASE: 'http://localhost:8000' })
    }
  },
})
