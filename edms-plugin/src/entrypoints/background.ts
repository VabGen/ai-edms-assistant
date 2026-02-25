export default defineBackground({
  type: 'module',
  main() {
    const API = 'http://localhost:8000'
    const controllers = new Map<string, AbortController>()

    chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
      const reqId: string = msg.payload?.requestId ?? 'default'

      switch (msg.type) {
        // ── Abort ──────────────────────────────────────────────────────────
        case 'abortRequest': {
          controllers.get(reqId)?.abort()
          controllers.delete(reqId)
          return false
        }

        // ── Chat ───────────────────────────────────────────────────────────
        case 'sendChatMessage':
          doFetch(`${API}/chat`, msg.payload, reqId, sendResponse)
          return true

        // ── Summarize attachment ───────────────────────────────────────────
        case 'summarizeDocument':
          doFetch(`${API}/actions/summarize`, msg.payload, reqId, sendResponse)
          return true

        // ── Upload file ────────────────────────────────────────────────────
        case 'uploadFile':
          doUpload(msg.payload, sendResponse)
          return true

        // ── History ────────────────────────────────────────────────────────
        case 'getChatHistory':
          doGetHistory(msg.payload.thread_id, sendResponse)
          return true

        // ── New chat ───────────────────────────────────────────────────────
        case 'createNewChat':
          doFetch(`${API}/chat/new`, { user_token: msg.payload.user_token }, reqId, sendResponse)
          return true

        // ── Autofill appeal ────────────────────────────────────────────────
        case 'autofillAppeal':
          doFetch(`${API}/appeal/autofill`, {
            message:        msg.payload.message ?? 'Заполни обращение',
            user_token:     msg.payload.user_token,
            context_ui_id:  msg.payload.context_ui_id,
            file_path:      msg.payload.file_path ?? null,
          }, reqId, sendResponse)
          return true

        default:
          return false
      }
    })

    // ── Helpers ─────────────────────────────────────────────────────────────
    async function doFetch(
      url: string,
      payload: unknown,
      reqId: string,
      respond: (r: { success: boolean; data?: unknown; error?: string }) => void,
    ) {
      const ctrl = new AbortController()
      controllers.set(reqId, ctrl)
      try {
        const res  = await fetch(url, {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify(payload),
          signal:  ctrl.signal,
        })
        const data = await res.json()
        if (!res.ok) throw new Error(data.detail ?? `Ошибка сервера: ${res.status}`)
        respond({ success: true, data })
      } catch (e: any) {
        if (e.name === 'AbortError') respond({ success: false, error: 'Request aborted' })
        else                         respond({ success: false, error: e.message })
      } finally {
        controllers.delete(reqId)
      }
    }

    async function doUpload(
      payload: { fileData: string; fileName: string; user_token: string },
      respond: (r: { success: boolean; data?: unknown; error?: string }) => void,
    ) {
      try {
        const blob = await (await fetch(payload.fileData)).blob()
        const form = new FormData()
        form.append('file',       blob,              payload.fileName)
        form.append('user_token', payload.user_token)
        const res  = await fetch(`${API}/upload-file`, { method: 'POST', body: form })
        const data = await res.json()
        if (!res.ok) throw new Error(data.detail ?? 'Ошибка загрузки файла')
        respond({ success: true, data })
      } catch (e: any) {
        respond({ success: false, error: e.message })
      }
    }

    async function doGetHistory(
      threadId: string,
      respond: (r: { success: boolean; data?: unknown; error?: string }) => void,
    ) {
      try {
        const res  = await fetch(`${API}/chat/history/${threadId}`)
        const data = await res.json()
        if (!res.ok) throw new Error(data.detail ?? 'Ошибка истории')
        respond({ success: true, data })
      } catch (e: any) {
        respond({ success: false, error: e.message })
      }
    }
  },
})
