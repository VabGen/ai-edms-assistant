import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import dayjs from 'dayjs'

interface Props {
  content: string
  role: 'user' | 'assistant'
  isError?: boolean
}

// Detect if a message is an error message
export function isErrorMessage(content: string): boolean {
  return content.startsWith('__error__:')
}

// Extract clean error text from error message
export function extractErrorText(content: string): string {
  return content.replace(/^__error__:\s*/, '').replace(/^(Error:\s*|Ошибка:\s*)/i, '')
}

export function ChatMessage({ content, role, isError }: Props) {
  const isUser  = role === 'user'
  const isErr   = isError ?? isErrorMessage(content)
  const display = isErr ? extractErrorText(content) : content

  // ── Error bubble ────────────────────────────────────────────────────────────
  if (isErr) {
    return (
      <div
        className="flex w-full justify-start"
        style={{ animation: 'slideInRight .25s ease-out' }}
      >
        <div
          style={{
            maxWidth:     '88%',
            padding:      '12px 14px',
            borderRadius: '18px 18px 18px 4px',
            background:   'rgba(254,226,226,0.85)',
            border:       '1px solid rgba(252,165,165,0.5)',
            backdropFilter: 'blur(10px)',
            boxShadow:    '0 2px 12px rgba(220,38,38,0.1)',
          }}
        >
          {/* Header row */}
          <div style={{
            display:    'flex',
            alignItems: 'center',
            gap:        8,
            marginBottom: 6,
          }}>
            <span style={{
              width:        22,
              height:       22,
              borderRadius: 6,
              background:   'rgba(220,38,38,0.12)',
              display:      'flex',
              alignItems:   'center',
              justifyContent: 'center',
              fontSize:     13,
              flexShrink:   0,
            }}>
              ⚠️
            </span>
            <span style={{
              fontSize:    12,
              fontWeight:  700,
              color:       '#b91c1c',
              letterSpacing: '-0.01em',
            }}>
              Не удалось выполнить запрос
            </span>
          </div>

          {/* Error details */}
          <p style={{
            fontSize:   12,
            color:      '#991b1b',
            lineHeight: 1.5,
            margin:     0,
            opacity:    0.9,
          }}>
            {humanizeError(display)}
          </p>

          {/* Hint */}
          <p style={{
            fontSize:   11,
            color:      '#b91c1c',
            opacity:    0.6,
            marginTop:  6,
            marginBottom: 0,
          }}>
            Попробуйте повторить запрос или обновите страницу.
          </p>

          {/* Timestamp */}
          <div style={{
            fontSize:  10,
            opacity:   0.45,
            marginTop: 6,
            color:     '#7f1d1d',
          }}>
            {dayjs().format('HH:mm')}
          </div>
        </div>
      </div>
    )
  }

  // ── Normal bubble ───────────────────────────────────────────────────────────
  return (
    <div
      className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'}`}
      style={{ animation: 'slideInRight .25s ease-out' }}
    >
      <div className={[
        'max-w-[88%] px-4 py-3 text-sm leading-relaxed backdrop-blur-md shadow-sm border transition-all',
        isUser
          ? 'bg-indigo-600/80 text-white border-white/20 rounded-[20px] rounded-tr-[4px]'
          : 'bg-white/50 text-slate-800 border-white/40 rounded-[20px] rounded-tl-[4px]',
      ].join(' ')}>

        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            code({ inline, className, children, ...props }: any) {
              if (!inline) {
                return (
                  <pre className="overflow-x-auto p-3 my-2 rounded-xl bg-slate-900/80 font-mono text-xs text-indigo-100 border border-white/10">
                    <code className={className} {...props}>{children}</code>
                  </pre>
                )
              }
              return (
                <code
                  className={`font-mono text-[12px] px-1 py-0.5 rounded ${isUser ? 'bg-white/20' : 'bg-indigo-100/60 text-indigo-700'}`}
                  {...props}
                >
                  {children}
                </code>
              )
            },
            p: ({ children }) => (
              <p className="mb-2 last:mb-0 whitespace-pre-wrap break-words">{children}</p>
            ),
            ul: ({ children }) => <ul className="list-disc pl-5 mb-2 space-y-0.5">{children}</ul>,
            ol: ({ children }) => <ol className="list-decimal pl-5 mb-2 space-y-0.5">{children}</ol>,
            li: ({ children }) => <li className="marker:text-current">{children}</li>,
            a: ({ children, href }) => (
              <a
                href={href}
                className={`underline underline-offset-2 font-semibold ${isUser ? 'text-white' : 'text-indigo-600'}`}
                target="_blank"
                rel="noopener noreferrer"
              >
                {children}
              </a>
            ),
            blockquote: ({ children }) => (
              <blockquote
                className={`border-l-4 pl-3 italic my-2 ${isUser ? 'border-white/40' : 'border-indigo-200 text-slate-600'}`}
              >
                {children}
              </blockquote>
            ),
            table: ({ children }) => (
              <div className="overflow-x-auto my-2 rounded-xl border border-white/30 bg-white/20">
                <table className="w-full text-xs text-left border-collapse">{children}</table>
              </div>
            ),
            th: ({ children }) => (
              <th className={`p-2 border-b border-white/30 font-bold ${isUser ? 'bg-white/10' : 'bg-white/30'}`}>
                {children}
              </th>
            ),
            td: ({ children }) => <td className="p-2 border-b border-white/10">{children}</td>,
          }}
        >
          {display}
        </ReactMarkdown>

        <div
          className={`text-[10px] mt-1.5 opacity-50 flex items-center gap-1 ${isUser ? 'justify-end' : 'justify-start'}`}
        >
          <span>{dayjs().format('HH:mm')}</span>
        </div>
      </div>
    </div>
  )
}

// ─── Humanize technical error messages ────────────────────────────────────────
function humanizeError(raw: string): string {
  const lower = raw.toLowerCase()

  if (lower.includes('failed to fetch') || lower.includes('networkerror') || lower.includes('network'))
    return 'Нет соединения с сервером. Проверьте подключение к интернету и доступность сервиса.'

  if (lower.includes('timeout') || lower.includes('timed out'))
    return 'Сервер не ответил вовремя. Возможно, он перегружен — попробуйте чуть позже.'

  if (lower.includes('401') || lower.includes('unauthorized') || lower.includes('unauthenticated'))
    return 'Ошибка авторизации. Возможно, сессия истекла — обновите страницу или войдите снова.'

  if (lower.includes('403') || lower.includes('forbidden'))
    return 'Доступ запрещён. У вас нет прав на выполнение этого действия.'

  if (lower.includes('404') || lower.includes('not found'))
    return 'Запрашиваемый ресурс не найден. Возможно, он был удалён или перемещён.'

  if (lower.includes('500') || lower.includes('internal server'))
    return 'Внутренняя ошибка сервера. Мы уже работаем над её устранением.'

  if (lower.includes('503') || lower.includes('service unavailable'))
    return 'Сервис временно недоступен. Попробуйте повторить запрос через несколько минут.'

  if (lower.includes('aborted') || lower.includes('abort'))
    return 'Запрос был отменён.'

  // Return cleaned raw message if no pattern matched
  return raw || 'Произошла неизвестная ошибка.'
}