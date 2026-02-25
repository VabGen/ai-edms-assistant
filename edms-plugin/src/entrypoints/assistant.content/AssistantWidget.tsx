import { useState, useRef, useEffect, type FormEvent } from 'react'
import {
  Paperclip, X, Mic, Send, MessageSquare,
  Square, StopCircle, FileText, Search, List, History,
} from 'lucide-react'
import dayjs from 'dayjs'
import 'dayjs/locale/ru'

import { ChatMessage }        from '../../shared/ui/ChatMessage'
import { LiquidGlassFilter }  from '../../shared/ui/LiquidGlassFilter'
import { getAuthToken }       from '../../shared/lib/auth'
import { extractDocIdFromUrl } from '../../shared/lib/url'
import { sendMsg }            from '../../shared/lib/messaging'
import { toast }              from '../../shared/lib/toast'

dayjs.locale('ru')

// ─── Types ────────────────────────────────────────────────────────────────────
interface Message {
  role:        'user' | 'assistant'
  content:     string
  action_type?: string
  isError?:    boolean
}

interface Thread {
  id:      string
  preview: string
  date:    string
}

// ─── Keyframes injected into Shadow DOM ──────────────────────────────────────
const ANIM_STYLES = `
  @keyframes edms-ripple {
    0%   { transform: scale(0.8); opacity: 0.6; }
    100% { transform: scale(2.4); opacity: 0; }
  }
  @keyframes edms-fade-in {
    from { opacity: 0; transform: scale(0.93) translateY(8px); }
    to   { opacity: 1; transform: scale(1) translateY(0); }
  }
  @keyframes edms-wave {
    0%, 60%, 100% { transform: translateY(0);    opacity: 0.4; }
    30%            { transform: translateY(-6px); opacity: 1;   }
  }
  @keyframes edms-soundbar {
    0%, 100% { transform: scaleY(0.4); }
    50%      { transform: scaleY(1.2); }
  }
`

// ─── Sound Wave ───────────────────────────────────────────────────────────────
function SoundWave() {
  return (
    <>
      <style>{ANIM_STYLES}</style>
      <div style={{ display:'flex', alignItems:'flex-end', justifyContent:'center', gap:3, height:12, marginBottom:8 }}>
        {[0,1,2,3,4].map(i => (
          <div key={i} style={{
            width: 3, height: 10, borderRadius: 2,
            background: 'rgba(99,102,241,0.7)',
            transformOrigin: 'bottom',
            animation: `edms-soundbar 0.6s ease-in-out ${i * 80}ms infinite`,
          }} />
        ))}
      </div>
    </>
  )
}

// ─── Typing Dots ──────────────────────────────────────────────────────────────
function TypingDots() {
  return (
    <>
      <style>{ANIM_STYLES}</style>
      <div style={{
        display:'flex', alignItems:'center', gap:6,
        padding:'10px 14px',
        background:'rgba(255,255,255,0.35)',
        borderRadius:18,
        border:'1px solid rgba(255,255,255,0.4)',
        width:'fit-content',
      }}>
        {[0, 160, 320].map(d => (
          <div key={d} style={{
            width:7, height:7, borderRadius:'50%',
            background:'#818cf8',
            animation:`edms-wave 1.4s ease-in-out ${d}ms infinite`,
          }} />
        ))}
      </div>
    </>
  )
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Wrap text in error marker so ChatMessage renders the error bubble */
function makeErrorMessage(err: unknown): string {
  const raw = err instanceof Error ? err.message : String(err)
  return `__error__:${raw}`
}

/** Parse human-readable label from error for toast */
function getToastErrorText(err: unknown): string {
  const raw = (err instanceof Error ? err.message : String(err)).toLowerCase()
  if (raw.includes('failed to fetch') || raw.includes('networkerror'))
    return 'Нет соединения с сервером'
  if (raw.includes('timeout'))
    return 'Сервер не ответил вовремя'
  if (raw.includes('401') || raw.includes('unauthorized'))
    return 'Ошибка авторизации — обновите страницу'
  if (raw.includes('403'))
    return 'Доступ запрещён'
  return err instanceof Error ? err.message : String(err)
}

// ─── Main Widget ──────────────────────────────────────────────────────────────
export function AssistantWidget() {
  const [isEnabled,      setIsEnabled]      = useState(true)
  const [isOpen,         setIsOpen]         = useState(false)
  const [isSidebarOpen,  setIsSidebarOpen]  = useState(false)
  const [messages,       setMessages]       = useState<Message[]>([])
  const [threads,        setThreads]        = useState<Thread[]>([])
  const [threadId,       setThreadId]       = useState<string | null>(null)
  const [input,          setInput]          = useState('')
  const [loading,        setLoading]        = useState(false)
  const [attachedFile,   setAttachedFile]   = useState<{ path: string; name: string } | null>(null)
  const [isListening,    setIsListening]    = useState(false)
  const [recognition,    setRecognition]    = useState<any>(null)

  const bottomRef    = useRef<HTMLDivElement>(null)
  const fileRef      = useRef<HTMLInputElement>(null)
  const requestIdRef = useRef<string | null>(null)
  const serverPathRef = useRef<string | null>(null)

  // ── Bootstrap ────────────────────────────────────────────────────────────
  useEffect(() => {
    chrome.storage.local.get(['assistantEnabled'], r => {
      if (r.assistantEnabled !== undefined) setIsEnabled(r.assistantEnabled)
    })

    const onStorage = (
      changes: Record<string, chrome.storage.StorageChange>,
      area: string,
    ) => {
      if (area === 'local' && 'assistantEnabled' in changes) {
        const val = changes.assistantEnabled.newValue as boolean
        setIsEnabled(val)
        if (!val) setIsOpen(false)
      }
    }
    chrome.storage.onChanged.addListener(onStorage)

    const onWindowMsg = (e: MessageEvent) => {
      if (e.data?.type !== 'REFRESH_CHAT_HISTORY') return
      const { messages: raw, thread_id } = e.data as {
        messages: { type: string; content: string }[]
        thread_id?: string
      }
      const mapped = raw.map(m => ({
        role:    (m.type === 'human' ? 'user' : 'assistant') as Message['role'],
        content: m.content,
      }))
      if (thread_id) { setThreadId(thread_id); setMessages(mapped) }
      else setMessages(prev => [...prev, ...mapped])
      setIsOpen(true)
      setIsSidebarOpen(false)
      setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: 'smooth' }), 120)
    }
    window.addEventListener('message', onWindowMsg)

    const SR = (window as any).SpeechRecognition ?? (window as any).webkitSpeechRecognition
    if (SR) {
      const inst = new SR()
      inst.lang = 'ru-RU'
      inst.continuous = true
      inst.interimResults = true
      inst.onresult = (ev: any) => {
        let text = ''
        for (let i = ev.resultIndex; i < ev.results.length; i++)
          if (ev.results[i].isFinal) text += ev.results[i][0].transcript
        if (text) setInput(prev => (prev ? prev + ' ' : '') + text)
      }
      inst.onend  = () => setIsListening(false)
      inst.onerror = () => setIsListening(false)
      setRecognition(inst)
    }

    return () => {
      chrome.storage.onChanged.removeListener(onStorage)
      window.removeEventListener('message', onWindowMsg)
    }
  }, [])

  // ── Auto-scroll ───────────────────────────────────────────────────────────
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  // ── New chat ──────────────────────────────────────────────────────────────
  const newChat = async () => {
    if (messages.length && threadId) {
      const preview = (messages.find(m => m.role === 'user')?.content ?? 'Диалог').slice(0, 40)
      setThreads(prev =>
        prev.find(t => t.id === threadId)
          ? prev
          : [{ id: threadId, preview, date: dayjs().format('HH:mm') }, ...prev],
      )
    }
    setLoading(true)
    try {
      const res = await sendMsg<{ thread_id: string }>('createNewChat', {
        user_token: getAuthToken() ?? 'no_token',
      })
      setThreadId(res.thread_id)
      setMessages([])
      setIsSidebarOpen(false)
    } catch (err) {
      // Non-critical: just show toast, don't break chat
      toast.error(getToastErrorText(err), 'Не удалось создать диалог')
      console.error('[EDMS] newChat:', err)
    } finally {
      setLoading(false)
    }
  }

  // ── Load thread history ───────────────────────────────────────────────────
  const loadHistory = async (id: string) => {
    setLoading(true)
    setIsSidebarOpen(false)
    try {
      const res = await sendMsg<{ messages: { type: string; content: string }[] }>(
        'getChatHistory', { thread_id: id },
      )
      setMessages(res.messages.map(m => ({
        role:    (m.type === 'human' ? 'user' : 'assistant') as Message['role'],
        content: m.content,
      })))
      setThreadId(id)
    } catch (err) {
      toast.error(getToastErrorText(err), 'Не удалось загрузить историю')
      console.error('[EDMS] loadHistory:', err)
    } finally {
      setLoading(false)
    }
  }

  // ── Send message ──────────────────────────────────────────────────────────
  const send = async (e?: FormEvent, humanChoice?: string) => {
    if (e) e.preventDefault()
    if (!input.trim() && !attachedFile && !humanChoice) return
    if (loading) return

    if (isListening) { recognition?.stop(); setIsListening(false) }

    const token  = getAuthToken() ?? 'no_token'
    const docId  = extractDocIdFromUrl()
    const reqId  = Math.random().toString(36).slice(7)
    const tid    = threadId ?? `${token.slice(0, 8)}_${docId}`
    requestIdRef.current = reqId
    if (!threadId) setThreadId(tid)

    const userText = humanChoice
      ? ({ abstractive: 'Пересказ', extractive: 'Факты', thesis: 'Тезисы' } as Record<string,string>)[humanChoice] ?? humanChoice
      : attachedFile ? `${input} (Файл: ${attachedFile.name})` : input

    setMessages(prev => [...prev, { role: 'user', content: userText }])
    setLoading(true)
    const text = input
    setInput('')

    try {
      // Upload file if attached
      if (attachedFile && !humanChoice) {
        const up = await sendMsg<{ file_path: string }>('uploadFile', {
          fileData:   attachedFile.path,
          fileName:   attachedFile.name,
          user_token: token,
        })
        serverPathRef.current = up.file_path
      }

      const res = await sendMsg<any>('sendChatMessage', {
        message:       text,
        user_token:    token,
        requestId:     reqId,
        thread_id:     tid,
        context_ui_id: docId,
        file_path:     serverPathRef.current,
        human_choice:  humanChoice,
      })

      serverPathRef.current = null

      const content = res.response
        ?? res.content
        ?? res.message
        ?? (Array.isArray(res.messages) ? res.messages.at(-1)?.content : undefined)
        ?? 'Анализ завершён.'

      setMessages(prev => [...prev, {
        role: 'assistant',
        content,
        action_type: res.action_type,
      }])
    } catch (err: unknown) {
      const errStr = String(err)
      // Don't show error bubble for intentional abort
      if (!errStr.includes('aborted')) {
        setMessages(prev => [...prev, {
          role:    'assistant',
          content: makeErrorMessage(err),
          isError: true,
        }])
      }
      console.error('[EDMS] send:', err)
    } finally {
      setLoading(false)
      requestIdRef.current = null
      setAttachedFile(null)
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  const abort = () => {
    if (!requestIdRef.current) return
    chrome.runtime.sendMessage({ type: 'abortRequest', payload: { requestId: requestIdRef.current } })
    setLoading(false)
    requestIdRef.current = null
    setMessages(prev => [...prev, { role: 'assistant', content: '_Запрос отменён._' }])
  }

  const toggleMic = () => {
    if (!recognition) return
    if (isListening) recognition.stop()
    else { setInput(''); recognition.start(); setIsListening(true) }
  }

  // ── Action buttons (summarize choice) ─────────────────────────────────────
  const ActionButtons = ({ msg }: { msg: Message }) => {
    if (msg.action_type !== 'summarize_selection') return null
    return (
      <div className="mt-2 flex flex-wrap gap-2">
        {[
          { id: 'abstractive', label: 'Пересказ', icon: <FileText size={13} /> },
          { id: 'extractive',  label: 'Факты',    icon: <Search   size={13} /> },
          { id: 'thesis',      label: 'Тезисы',   icon: <List     size={13} /> },
        ].map(btn => (
          <button
            key={btn.id}
            type="button"
            onClick={e => { e.stopPropagation(); send(undefined, btn.id) }}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-xl border border-indigo-200 bg-white/60 text-indigo-700 hover:bg-indigo-600 hover:text-white hover:border-indigo-600 transition-all backdrop-blur-sm"
          >
            {btn.icon}{btn.label}
          </button>
        ))}
      </div>
    )
  }

  if (!isEnabled) return null

  // ─── Render ──────────────────────────────────────────────────────────────
  return (
    <div className="fixed bottom-5 right-5 z-[2147483647] flex flex-col items-end pointer-events-none select-none font-sans">
      <LiquidGlassFilter />

      {/* ── FAB ── */}
      {!isOpen && (
        <>
          <style>{ANIM_STYLES}</style>
          <button
            type="button"
            onClick={() => setIsOpen(true)}
            className="pointer-events-auto relative w-16 h-16 rounded-full flex items-center justify-center glass hover:scale-110 active:scale-95 transition-transform group"
          >
            <span style={{
              position:'absolute', inset:0,
              borderRadius:'50%',
              background:'rgba(99,102,241,0.18)',
              animation:'edms-ripple 3s cubic-bezier(0.4,0,0.2,1) infinite',
              pointerEvents:'none',
            }} />
            <MessageSquare
              size={28}
              className="text-indigo-600/90 group-hover:rotate-12 transition-transform z-10"
            />
          </button>
        </>
      )}

      {/* ── Chat window ── */}
      {isOpen && (
        <>
          <style>{ANIM_STYLES}</style>
          <div
            className="pointer-events-auto flex flex-col w-[480px] h-[720px] rounded-[28px] overflow-hidden glass"
            style={{ animation: 'edms-fade-in .3s cubic-bezier(.22,1,.36,1) forwards' }}
          >
            {/* Header */}
            <header className="flex items-center justify-between px-4 py-3 border-b border-white/20 shrink-0 bg-white/10">
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setIsSidebarOpen(v => !v)}
                  className={`p-2 rounded-xl transition-all flex flex-col justify-center items-center gap-[5px] w-9 h-9 ${isSidebarOpen ? 'bg-white/30 text-indigo-700' : 'text-slate-500 hover:bg-white/20'}`}
                  aria-label="История диалогов"
                >
                  <span className={`h-px w-4 bg-current rounded transition-all duration-300 origin-center ${isSidebarOpen ? 'rotate-45 translate-y-[6px]' : ''}`} />
                  <span className={`h-px w-4 bg-current rounded transition-all duration-200 ${isSidebarOpen ? 'opacity-0 scale-x-0' : ''}`} />
                  <span className={`h-px w-4 bg-current rounded transition-all duration-300 origin-center ${isSidebarOpen ? '-rotate-45 -translate-y-[6px]' : ''}`} />
                </button>
                <h3 className="font-bold text-slate-800/80 text-sm tracking-tight">EDMS Assistant</h3>
              </div>
              <button
                type="button"
                onClick={() => setIsOpen(false)}
                className="p-2 rounded-xl text-slate-400 hover:text-red-500 hover:bg-red-50/30 transition-colors"
                aria-label="Закрыть"
              >
                <X size={18} />
              </button>
            </header>

            {/* Body */}
            <div className="flex-1 flex overflow-hidden">
              {/* Sidebar */}
              <aside className={`shrink-0 flex flex-col bg-white/10 backdrop-blur-md border-r border-white/20 transition-all duration-300 overflow-hidden ${isSidebarOpen ? 'w-60' : 'w-0'}`}>
                <div className="p-3 w-60 flex flex-col h-full">
                  <button
                    type="button"
                    onClick={newChat}
                    className="w-full py-2 px-3 rounded-xl bg-indigo-600 text-white text-sm font-semibold hover:bg-indigo-700 active:scale-95 transition-all mb-4"
                  >
                    + Новый диалог
                  </button>

                  <div className="flex items-center gap-2 px-1 mb-2">
                    <History size={11} className="text-slate-400" />
                    <span className="text-[9px] uppercase tracking-widest text-slate-400 font-bold">История</span>
                  </div>

                  <div className="flex-1 overflow-y-auto scrollbar-thin flex flex-col gap-1.5">
                    {threads.length === 0 ? (
                      <p className="text-[11px] text-slate-400 italic text-center py-6 bg-white/10 rounded-xl border border-dashed border-white/30">
                        История пуста
                      </p>
                    ) : threads.map(t => (
                      <button
                        key={t.id}
                        type="button"
                        onClick={() => loadHistory(t.id)}
                        className={`text-left w-full p-2.5 rounded-xl text-[11px] transition-all border ${threadId === t.id ? 'bg-white/40 border-white/50' : 'hover:bg-white/20 border-transparent'}`}
                      >
                        <p className="text-slate-700 font-medium line-clamp-2 leading-relaxed">{t.preview}</p>
                        <span className="text-[9px] text-slate-400 mt-0.5 block">{t.date}</span>
                      </button>
                    ))}
                  </div>
                </div>
              </aside>

              {/* Messages */}
              <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
                <div className="flex-1 p-4 overflow-y-auto scrollbar-thin flex flex-col gap-3">
                  {messages.length === 0 && !loading && (
                    <div className="flex-1 flex flex-col items-center justify-center gap-3 opacity-40 select-none">
                      <MessageSquare size={44} strokeWidth={1} className="text-slate-500" />
                      <p className="text-sm text-slate-500 font-medium">Чем я могу помочь?</p>
                    </div>
                  )}

                  {messages.map((msg, i) => (
                    <div key={i} className="flex flex-col">
                      <ChatMessage
                        content={msg.content}
                        role={msg.role}
                        isError={msg.isError}
                      />
                      <ActionButtons msg={msg} />
                    </div>
                  ))}

                  {loading && <TypingDots />}
                  <div ref={bottomRef} />
                </div>

                {/* Footer / Input */}
                <footer className="px-3 pb-3 pt-2 shrink-0 border-t border-white/20 bg-white/5">
                  {isListening && <SoundWave />}

                  {attachedFile && (
                    <div className="flex items-center gap-2 mb-2 px-3 py-1.5 bg-white/50 border border-white/50 rounded-xl w-fit text-[11px]">
                      <Paperclip size={13} className="text-indigo-500" />
                      <span className="text-slate-700 font-medium truncate max-w-[180px]">{attachedFile.name}</span>
                      <button
                        type="button"
                        onClick={() => setAttachedFile(null)}
                        className="ml-1 text-slate-400 hover:text-red-500 transition-colors"
                      >
                        <X size={13} />
                      </button>
                    </div>
                  )}

                  <form
                    onSubmit={send}
                    className="flex items-center gap-1 rounded-2xl p-1 bg-white/20 border border-white/30 backdrop-blur-md focus-within:bg-white/35 focus-within:border-indigo-300/50 transition-all shadow-sm"
                  >
                    <button
                      type="button"
                      onClick={() => fileRef.current?.click()}
                      className="p-2 rounded-xl text-slate-400 hover:text-indigo-600 hover:bg-white/30 transition-colors"
                      aria-label="Прикрепить файл"
                    >
                      <Paperclip size={18} />
                    </button>

                    <button
                      type="button"
                      onClick={toggleMic}
                      className={`p-2 rounded-xl transition-all ${isListening ? 'text-red-500 bg-red-50/40 animate-pulse' : 'text-slate-400 hover:text-indigo-600 hover:bg-white/30'}`}
                      aria-label={isListening ? 'Остановить запись' : 'Голосовой ввод'}
                    >
                      {isListening ? <Square size={16} fill="currentColor" /> : <Mic size={18} />}
                    </button>

                    <input
                      type="text"
                      value={input}
                      onChange={e => setInput(e.target.value)}
                      placeholder={isListening ? 'Слушаю...' : 'Спросите AI...'}
                      className="flex-1 bg-transparent border-none outline-none text-sm text-slate-800 placeholder-slate-400 px-1"
                    />

                    {loading ? (
                      <button
                        type="button"
                        onClick={abort}
                        className="p-2 rounded-xl bg-red-500/80 text-white hover:bg-red-600 active:scale-95 transition-all"
                        aria-label="Отменить"
                      >
                        <StopCircle size={17} />
                      </button>
                    ) : (
                      <button
                        type="submit"
                        disabled={!input.trim() && !attachedFile}
                        className="p-2 rounded-xl bg-indigo-600/80 text-white hover:bg-indigo-700 disabled:opacity-30 active:scale-95 transition-all"
                        aria-label="Отправить"
                      >
                        <Send size={17} />
                      </button>
                    )}
                  </form>
                </footer>
              </main>
            </div>
          </div>
        </>
      )}

      <input
        type="file"
        ref={fileRef}
        className="hidden"
        onChange={e => {
          const f = e.target.files?.[0]
          if (!f) return
          const r = new FileReader()
          r.onload = () => setAttachedFile({ path: r.result as string, name: f.name })
          r.readAsDataURL(f)
        }}
      />
    </div>
  )
}