const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

/** Извлекает UUID документа из текущего URL */
export function extractDocIdFromUrl(): string | null {
  try {
    return window.location.pathname.split('/').find(p => UUID_RE.test(p)) ?? null
  } catch {
    return null
  }
}
