/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{ts,tsx,html}'],
  theme: {
    extend: {
      keyframes: {
        fadeInScale: {
          '0%':   { opacity: '0', transform: 'scale(0.92) translateY(8px)' },
          '100%': { opacity: '1', transform: 'scale(1) translateY(0)' },
        },
        liquidRipple: {
          '0%':   { transform: 'scale(0.8)', opacity: '0.6' },
          '100%': { transform: 'scale(2.2)', opacity: '0' },
        },
        bounceDot: {
          '0%, 100%': { transform: 'translateY(0)',    opacity: '0.4' },
          '50%':      { transform: 'translateY(-4px)', opacity: '1' },
        },
        slideInLeft: {
          '0%':   { opacity: '0', transform: 'translateX(-8px)' },
          '100%': { opacity: '1', transform: 'translateX(0)' },
        },
        slideInRight: {
          '0%':   { opacity: '0', transform: 'translateX(8px)' },
          '100%': { opacity: '1', transform: 'translateX(0)' },
        },
      },
      animation: {
        'fade-in-scale':  'fadeInScale 0.3s cubic-bezier(0.22, 1, 0.36, 1) forwards',
        'liquid-ripple':  'liquidRipple 3s cubic-bezier(0.4, 0, 0.2, 1) infinite',
        'bounce-dot':     'bounceDot 1.4s ease-in-out infinite',
        'slide-in-left':  'slideInLeft 0.25s ease-out forwards',
        'slide-in-right': 'slideInRight 0.25s ease-out forwards',
      },
    },
  },
  plugins: [],
}
