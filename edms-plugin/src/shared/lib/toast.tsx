/**
 * Toast notification system for EDMS Chrome Extension
 * Renders outside React/Shadow DOM — directly in document.body
 * Supports: success | error | info
 */

export type ToastType = 'success' | 'error' | 'info'

export interface ToastOptions {
  message: string
  type?: ToastType
  duration?: number // ms, default 4000
  title?: string
}

// ─── Inject styles once ──────────────────────────────────────────────────────
function ensureStyles() {
  if (document.getElementById('edms-toast-styles')) return
  const style = document.createElement('style')
  style.id = 'edms-toast-styles'
  style.textContent = `
    #edms-toast-container {
      position: fixed;
      top: 16px;
      right: 16px;
      z-index: 2147483647;
      display: flex;
      flex-direction: column;
      gap: 10px;
      pointer-events: none;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    .edms-toast {
      pointer-events: auto;
      display: flex;
      align-items: flex-start;
      gap: 10px;
      min-width: 280px;
      max-width: 380px;
      padding: 12px 14px;
      border-radius: 14px;
      box-shadow:
        0 4px 24px rgba(0,0,0,0.14),
        0 1px 4px rgba(0,0,0,0.08),
        inset 0 1px 0 rgba(255,255,255,0.25);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid transparent;
      cursor: pointer;
      animation: edms-toast-in 0.35s cubic-bezier(0.22, 1, 0.36, 1) forwards;
      will-change: transform, opacity;
    }

    .edms-toast.edms-toast-out {
      animation: edms-toast-out 0.28s cubic-bezier(0.4, 0, 1, 1) forwards;
    }

    /* Success */
    .edms-toast--success {
      background: rgba(22, 163, 74, 0.92);
      border-color: rgba(134, 239, 172, 0.3);
      color: #fff;
    }
    .edms-toast--success .edms-toast-icon {
      background: rgba(255,255,255,0.2);
      color: #bbf7d0;
    }

    /* Error */
    .edms-toast--error {
      background: rgba(220, 38, 38, 0.92);
      border-color: rgba(252, 165, 165, 0.3);
      color: #fff;
    }
    .edms-toast--error .edms-toast-icon {
      background: rgba(255,255,255,0.2);
      color: #fecaca;
    }

    /* Info */
    .edms-toast--info {
      background: rgba(79, 70, 229, 0.92);
      border-color: rgba(165, 180, 252, 0.3);
      color: #fff;
    }
    .edms-toast--info .edms-toast-icon {
      background: rgba(255,255,255,0.2);
      color: #c7d2fe;
    }

    .edms-toast-icon {
      width: 30px;
      height: 30px;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
      font-size: 15px;
      line-height: 1;
    }

    .edms-toast-body {
      flex: 1;
      min-width: 0;
    }

    .edms-toast-title {
      font-size: 13px;
      font-weight: 700;
      line-height: 1.3;
      letter-spacing: -0.01em;
    }

    .edms-toast-msg {
      font-size: 12px;
      opacity: 0.88;
      margin-top: 2px;
      line-height: 1.4;
      word-break: break-word;
    }

    .edms-toast-close {
      flex-shrink: 0;
      width: 20px;
      height: 20px;
      display: flex;
      align-items: center;
      justify-content: center;
      opacity: 0.6;
      font-size: 16px;
      line-height: 1;
      transition: opacity 0.15s;
      cursor: pointer;
      background: none;
      border: none;
      color: inherit;
      padding: 0;
    }
    .edms-toast-close:hover { opacity: 1; }

    .edms-toast-progress {
      position: absolute;
      bottom: 0;
      left: 0;
      height: 3px;
      border-radius: 0 0 14px 14px;
      background: rgba(255,255,255,0.4);
      animation: edms-toast-progress linear forwards;
    }

    @keyframes edms-toast-in {
      from {
        opacity: 0;
        transform: translateX(60px) scale(0.94);
      }
      to {
        opacity: 1;
        transform: translateX(0) scale(1);
      }
    }

    @keyframes edms-toast-out {
      from {
        opacity: 1;
        transform: translateX(0) scale(1);
        max-height: 120px;
        margin-bottom: 0;
      }
      to {
        opacity: 0;
        transform: translateX(60px) scale(0.94);
        max-height: 0;
        margin-bottom: -10px;
      }
    }

    @keyframes edms-toast-progress {
      from { width: 100%; }
      to   { width: 0%; }
    }
  `
  document.head.appendChild(style)
}

// ─── Container ────────────────────────────────────────────────────────────────
function getContainer(): HTMLElement {
  let c = document.getElementById('edms-toast-container')
  if (!c) {
    c = document.createElement('div')
    c.id = 'edms-toast-container'
    document.body.appendChild(c)
  }
  return c
}

// ─── Icons ────────────────────────────────────────────────────────────────────
const ICONS: Record<ToastType, string> = {
  success: '✓',
  error:   '✕',
  info:    'i',
}

const TITLES: Record<ToastType, string> = {
  success: 'Успешно',
  error:   'Ошибка',
  info:    'Уведомление',
}

// ─── Show ─────────────────────────────────────────────────────────────────────
export function showToast(opts: ToastOptions): void {
  ensureStyles()
  const container = getContainer()

  const {
    message,
    type     = 'info',
    duration = 4000,
    title    = TITLES[type],
  } = opts

  // Toast element
  const toast = document.createElement('div')
  toast.className = `edms-toast edms-toast--${type}`
  toast.style.position = 'relative'
  toast.setAttribute('role', 'alert')
  toast.setAttribute('aria-live', 'assertive')

  toast.innerHTML = `
    <div class="edms-toast-icon">${ICONS[type]}</div>
    <div class="edms-toast-body">
      <div class="edms-toast-title">${escapeHtml(title)}</div>
      <div class="edms-toast-msg">${escapeHtml(message)}</div>
    </div>
    <button class="edms-toast-close" aria-label="Закрыть">✕</button>
    <div class="edms-toast-progress" style="animation-duration:${duration}ms"></div>
  `

  const dismiss = () => {
    toast.classList.add('edms-toast-out')
    toast.addEventListener('animationend', () => toast.remove(), { once: true })
  }

  toast.addEventListener('click', dismiss)
  toast.querySelector('.edms-toast-close')?.addEventListener('click', e => {
    e.stopPropagation()
    dismiss()
  })

  container.appendChild(toast)

  if (duration > 0) {
    setTimeout(dismiss, duration)
  }
}

// ─── Convenience aliases ──────────────────────────────────────────────────────
export const toast = {
  success: (message: string, title?: string) =>
    showToast({ message, type: 'success', title }),
  error: (message: string, title?: string) =>
    showToast({ message, type: 'error', title }),
  info: (message: string, title?: string) =>
    showToast({ message, type: 'info', title }),
}

// ─── Helper ───────────────────────────────────────────────────────────────────
function escapeHtml(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}