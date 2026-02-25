import { defineConfig } from 'wxt'

export default defineConfig({
  srcDir: 'src',
  modules: ['@wxt-dev/module-react'],
  manifest: {
    name: 'EDMS AI Assistant',
    description: 'AI-powered assistant for EDMS document management',
    version: '2.0.0',
    permissions: ['activeTab', 'storage', 'tabs'],
    host_permissions: [
      'http://localhost/*',
      'http://127.0.0.1/*',
      'http://localhost:3000/*',
      'http://localhost:3001/*',
      'http://localhost:8080/*',
      'http://localhost:8000/*',
      'https://next.edo.iba/*',
    ],
  },
})
