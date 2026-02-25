import { getAuthToken }        from '../../shared/lib/auth'
import { extractDocIdFromUrl } from '../../shared/lib/url'
import { toast }               from '../../shared/lib/toast'

export default defineContentScript({
  matches: [
    'http://localhost:3000/*',
    'http://localhost:3001/*',
    'http://localhost:8080/*',
    'https://next.edo.iba/*',
    'http://127.0.0.1:*/*',
  ],
  runAt: 'document_idle',

  main() {
    console.log('[EDMS] ✅ Attachments content script started')

    const observer = new MutationObserver(() => injectButtons())
    observer.observe(document.body, { childList: true, subtree: true })
    injectButtons()
  },
})

// ─── Selectors ────────────────────────────────────────────────────────────────
const ATTACHMENT_SEL = 'div.alert.alert-secondary:has(span.lead)'
const INJECTED_ATTR  = 'data-edms-injected'
const UUID_RE        = /[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/i

// ─── Inject ───────────────────────────────────────────────────────────────────
function injectButtons() {
  document.querySelectorAll<HTMLElement>(ATTACHMENT_SEL).forEach(el => {
    if (el.getAttribute(INJECTED_ATTR)) return
    el.setAttribute(INJECTED_ATTR, '1')

    const btn = createActionButton(el)
    const lead = el.querySelector('span.lead')
    lead?.insertAdjacentElement('afterend', btn)
  })
}

// ─── Button factory ───────────────────────────────────────────────────────────
function createActionButton(row: HTMLElement): HTMLElement {
  const wrapper = document.createElement('span')
  wrapper.style.cssText = 'position:relative;display:inline-flex;margin-left:10px;vertical-align:middle;'

  let hovering  = false
  let dropdown: HTMLElement | null = null

  const btn = document.createElement('button')
  btn.type  = 'button'
  btn.title = 'AI анализ'
  btn.style.cssText = [
    'all:unset',
    'width:26px', 'height:26px',
    'display:flex', 'align-items:center', 'justify-content:center',
    'border-radius:8px',
    'border:1px solid #f1f5f9',
    'background:#fff',
    'cursor:pointer',
    'transition:all .2s',
    'color:#94a3b8',
  ].join(';')

  btn.innerHTML = sparklesSVG()

  btn.addEventListener('mouseenter', () => {
    hovering = true
    btn.style.color = '#4f46e5'
    btn.style.borderColor = '#e0e7ff'
    btn.style.background = '#f5f7ff'
    showDropdown()
  })

  btn.addEventListener('mouseleave', () => {
    hovering = false
    btn.style.color = '#94a3b8'
    btn.style.borderColor = '#f1f5f9'
    btn.style.background = '#fff'
    setTimeout(() => { if (!hovering) hideDropdown() }, 200)
  })

  wrapper.appendChild(btn)

  function showDropdown() {
    hideDropdown()
    dropdown = document.createElement('div')
    dropdown.style.cssText = [
      'position:absolute', 'top:50%', 'left:32px',
      'transform:translateY(-50%)',
      'background:#fff',
      'border:1px solid #e2e8f0',
      'border-radius:12px',
      'box-shadow:0 10px 30px rgba(0,0,0,.12)',
      'width:145px', 'padding:5px',
      'z-index:99999',
      'animation:edms-fade-in .15s ease-out',
    ].join(';')

    const header = document.createElement('div')
    header.textContent = 'Анализ документа'
    header.style.cssText = 'padding:5px 8px;font-size:10px;font-weight:700;color:#94a3b8;text-transform:uppercase;border-bottom:1px solid #f1f5f9;margin-bottom:3px;'
    dropdown.appendChild(header)

    const actions = [
      { id: 'extractive',  label: 'Факты',    svg: chartSVG() },
      { id: 'abstractive', label: 'Пересказ', svg: fileTextSVG() },
      { id: 'thesis',      label: 'Тезисы',   svg: targetSVG() },
    ]

    actions.forEach(({ id, label, svg }) => {
      const item = document.createElement('div')
      item.style.cssText = 'display:flex;align-items:center;gap:8px;padding:7px 8px;cursor:pointer;border-radius:7px;transition:background .12s;'
      item.innerHTML = `<span style="color:#6366f1;display:flex;">${svg}</span><span style="font-size:12px;font-weight:500;color:#334155;">${label}</span>`

      item.addEventListener('mouseenter', () => item.style.background = '#f8faff')
      item.addEventListener('mouseleave', () => item.style.background = 'transparent')
      item.addEventListener('click', e => {
        e.stopPropagation()
        handleAction(id, row, getFileName(row), btn)
        hideDropdown()
      })
      dropdown!.appendChild(item)
    })

    dropdown.addEventListener('mouseenter', () => { hovering = true })
    dropdown.addEventListener('mouseleave', () => {
      hovering = false
      setTimeout(() => { if (!hovering) hideDropdown() }, 200)
    })

    wrapper.appendChild(dropdown)

    if (!document.getElementById('edms-anim')) {
      const style = document.createElement('style')
      style.id = 'edms-anim'
      style.textContent = '@keyframes edms-fade-in{from{opacity:0;transform:translateY(-50%) scale(.95)}to{opacity:1;transform:translateY(-50%) scale(1)}}'
      document.head.appendChild(style)
    }
  }

  function hideDropdown() {
    dropdown?.remove()
    dropdown = null
  }

  return wrapper
}

// ─── File helpers ──────────────────────────────────────────────────────────────
function getFileName(row: HTMLElement): string {
  return row.querySelector('span.lead')?.textContent?.trim() ?? 'Документ'
}

function getFileId(row: HTMLElement): string {
  const links = row.querySelectorAll('a')
  for (const a of links) {
    for (const attr of ['href', 'onclick', 'data-id', 'id', 'data-file-id']) {
      const m = a.getAttribute(attr)?.match(UUID_RE)
      if (m) return m[0]
    }
  }
  const pid = row.closest('[id]')?.id ?? row.closest('[data-id]')?.getAttribute('data-id') ?? ''
  const pm  = pid.match(UUID_RE)
  if (pm) return pm[0]
  const hm = row.innerHTML.match(UUID_RE)
  return hm ? hm[0] : ''
}

// ─── Action handler ───────────────────────────────────────────────────────────
function handleAction(summaryType: string, row: HTMLElement, fileName: string, btn: HTMLElement) {
  const token = getAuthToken()
  if (!token) {
    // Toast instead of alert()
    toast.error('Войдите в систему и попробуйте снова.', 'Авторизация не найдена')
    return
  }

  const fileId = getFileId(row)
  const docId  = extractDocIdFromUrl()

  btn.innerHTML = spinnerSVG()
  btn.style.color = '#6366f1'

  chrome.runtime.sendMessage({
    type: 'summarizeDocument',
    payload: {
      message:       fileName,
      user_token:    token,
      context_ui_id: docId,
      file_path:     fileId,
      human_choice:  summaryType,
    },
  }, res => {
    btn.innerHTML = sparklesSVG()
    btn.style.color = '#94a3b8'

    if (res?.success) {
      // Success toast
      toast.success(`Анализ файла "${fileName}" готов`, 'Добавлено вложение')

      window.postMessage({
        type: 'REFRESH_CHAT_HISTORY',
        messages: [
          { type: 'human', content: `Анализ файла: ${fileName}` },
          { type: 'ai',    content: res.data?.response ?? 'Анализ завершён.' },
        ],
      }, '*')
    } else {
      // Error toast instead of alert()
      const errMsg = res?.error ?? 'Неизвестная ошибка'
      toast.error(humanizeActionError(errMsg), 'Ошибка анализа')
    }
  })
}

// ─── Error humanizer ──────────────────────────────────────────────────────────
function humanizeActionError(raw: string): string {
  const lower = raw.toLowerCase()
  if (lower.includes('failed to fetch') || lower.includes('network'))
    return 'Нет соединения с сервером'
  if (lower.includes('timeout'))
    return 'Сервер не ответил вовремя'
  if (lower.includes('401') || lower.includes('unauthorized'))
    return 'Сессия истекла — обновите страницу'
  if (lower.includes('403'))
    return 'Нет доступа к этому файлу'
  if (lower.includes('500'))
    return 'Внутренняя ошибка сервера'
  return raw
}

// ─── SVGs ─────────────────────────────────────────────────────────────────────
function sparklesSVG() {
  return `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>`
}
function spinnerSVG() {
  return `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="animation:spin 1s linear infinite"><path d="M21 12a9 9 0 1 1-6.219-8.56"/><style>@keyframes spin{to{transform:rotate(360deg)}}</style></svg>`
}
function chartSVG() {
  return `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>`
}
function fileTextSVG() {
  return `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>`
}
function targetSVG() {
  return `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>`
}