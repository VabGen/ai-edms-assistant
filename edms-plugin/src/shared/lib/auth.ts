/**
 * Извлекает JWT токен из localStorage / sessionStorage.
 * Перебирает известные ключи, а затем ищет eyJ… в остальных.
 */
export function getAuthToken(): string | null {
  try {
    const direct =
      localStorage.getItem('token') ||
      localStorage.getItem('access_token') ||
      sessionStorage.getItem('token')

    if (direct) return direct.replace(/^Bearer\s+/i, '').trim()

    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i)
      if (!key) continue
      if (!key.includes('auth') && !key.includes('user') && !key.includes('oidc')) continue

      const raw = localStorage.getItem(key)
      if (!raw?.includes('eyJ')) continue

      if (raw.startsWith('{')) {
        try {
          const parsed = JSON.parse(raw)
          const t = parsed.access_token ?? parsed.token ?? ''
          if (t) return t.replace(/^Bearer\s+/i, '').trim()
        } catch { /* ignore */ }
      }

      return raw.replace(/^Bearer\s+/i, '').trim()
    }
  } catch (e) {
    console.error('[EDMS] getAuthToken:', e)
  }
  return null
}
