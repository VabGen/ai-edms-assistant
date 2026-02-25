import { useState, useEffect, useRef } from 'react'
import { LiquidGlassFilter } from '../../shared/ui/LiquidGlassFilter'

export function Popup() {
  const [enabled, setEnabled]     = useState(true)
  const [mounted, setMounted]     = useState(false)
  const [ripple,  setRipple]      = useState(false)
  const cardRef                   = useRef<HTMLDivElement>(null)

  useEffect(() => {
    chrome.storage.local.get(['assistantEnabled'], r => {
      if (r.assistantEnabled !== undefined) setEnabled(r.assistantEnabled as boolean)
    })
    // Slide-in on mount
    requestAnimationFrame(() => setMounted(true))
  }, [])

  const toggle = () => {
    const next = !enabled
    setEnabled(next)
    setRipple(true)
    setTimeout(() => setRipple(false), 600)
    chrome.storage.local.set({ assistantEnabled: next })
  }

  return (
    <div
      style={{
        width: '100%',
        minWidth: 260,
        padding: '10px 10px 8px',
        background: 'transparent',
        fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", sans-serif',
      }}
    >
      <LiquidGlassFilter />

      {/* ── Card ── */}
      <div
        ref={cardRef}
        style={{
          position:   'relative',
          borderRadius: 22,
          overflow:   'hidden',
          background: 'rgba(255,255,255,0.18)',
          backdropFilter:         'blur(40px) saturate(180%)',
          WebkitBackdropFilter:   'blur(40px) saturate(180%)',
          border:     '1px solid rgba(255,255,255,0.45)',
          boxShadow:  '0 8px 32px rgba(0,0,0,0.12), 0 1px 0 rgba(255,255,255,0.6) inset, 0 -1px 0 rgba(0,0,0,0.04) inset',
          transform:  mounted ? 'translateY(0) scale(1)' : 'translateY(12px) scale(0.96)',
          opacity:    mounted ? 1 : 0,
          transition: 'transform 0.4s cubic-bezier(0.34,1.56,0.64,1), opacity 0.3s ease',
        }}
      >
        {/* Glass specular top shine */}
        <div style={{
          position:   'absolute',
          top: 0, left: 0, right: 0,
          height: 1,
          background: 'rgba(255,255,255,0.8)',
          borderRadius: '22px 22px 0 0',
        }} />

        {/* Ripple on toggle */}
        {ripple && (
          <div style={{
            position:   'absolute',
            top: '50%', right: 28,
            transform:  'translate(50%, -50%)',
            width: 8, height: 8,
            borderRadius: '50%',
            background:   enabled ? 'rgba(99,102,241,0.3)' : 'rgba(100,116,139,0.2)',
            animation:    'ios-ripple 0.6s ease-out forwards',
            pointerEvents:'none',
          }} />
        )}

        {/* Content */}
        <div style={{ padding: '14px 16px 13px' }}>
          {/* Row: icon + label + toggle */}
          <div style={{ display:'flex', alignItems:'center', gap:10 }}>
            {/* Icon */}
            <div style={{
              width: 34, height: 34,
              borderRadius: 10,
              background:   enabled
                ? 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)'
                : 'rgba(100,116,139,0.15)',
              display:       'flex',
              alignItems:    'center',
              justifyContent:'center',
              flexShrink:    0,
              boxShadow:     enabled
                ? '0 4px 12px rgba(99,102,241,0.4)'
                : 'none',
              transition:    'all 0.35s cubic-bezier(0.34,1.56,0.64,1)',
            }}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                   stroke={enabled ? 'white' : '#94a3b8'}
                   strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"
                   style={{ transition: 'stroke 0.3s' }}>
                <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/>
              </svg>
            </div>

            {/* Label */}
            <div style={{ flex: 1 }}>
              <div style={{
                fontSize:   13,
                fontWeight: 600,
                letterSpacing: -0.2,
                color:      '#1c1c1e',
                lineHeight: 1.2,
              }}>
                EDMS Assistant
              </div>
              <div style={{
                fontSize:   11,
                color:      enabled ? '#6366f1' : '#94a3b8',
                marginTop:  2,
                fontWeight: 500,
                transition: 'color 0.3s',
              }}>
                {enabled ? 'Активен' : 'Выключен'}
              </div>
            </div>

            {/* iOS-style toggle */}
            <div
              onClick={toggle}
              style={{
                position:   'relative',
                width:      48,
                height:     28,
                borderRadius: 14,
                background: enabled
                  ? 'linear-gradient(135deg, #6366f1, #818cf8)'
                  : 'rgba(120,120,128,0.16)',
                cursor:     'pointer',
                transition: 'background 0.3s cubic-bezier(0.4,0,0.2,1)',
                flexShrink: 0,
                boxShadow:  enabled
                  ? '0 2px 8px rgba(99,102,241,0.45), 0 1px 0 rgba(255,255,255,0.3) inset'
                  : '0 1px 3px rgba(0,0,0,0.1) inset',
              }}
              role="switch"
              aria-checked={enabled}
            >
              {/* Knob */}
              <div style={{
                position:     'absolute',
                top:          3,
                left:         enabled ? 23 : 3,
                width:        22,
                height:       22,
                borderRadius: '50%',
                background:   'white',
                boxShadow:    '0 2px 6px rgba(0,0,0,0.20), 0 1px 0 rgba(255,255,255,0.9) inset',
                transition:   'left 0.28s cubic-bezier(0.34,1.56,0.64,1)',
              }} />
            </div>
          </div>

          {/* Divider */}
          <div style={{
            height:     1,
            background: 'rgba(0,0,0,0.06)',
            margin:     '12px -16px',
            marginLeft: '-16px',
          }} />

          {/* Description */}
          <p style={{
            fontSize:   11.5,
            color:      'rgba(60,60,67,0.6)',
            lineHeight: 1.5,
            margin:     0,
            letterSpacing: -0.1,
          }}>
            Управление отображением AI&nbsp;ассистента в интерфейсе системы.
          </p>
        </div>
      </div>

      {/* Keyframes injected inline */}
      <style>{`
        @keyframes ios-ripple {
          0%   { transform: translate(50%,-50%) scale(1);  opacity: 0.6; }
          100% { transform: translate(50%,-50%) scale(18); opacity: 0; }
        }
      `}</style>
    </div>
  )
}
