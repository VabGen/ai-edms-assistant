export function LiquidGlassFilter() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="0" height="0"
      style={{ position: 'absolute', pointerEvents: 'none', visibility: 'hidden' }}
      aria-hidden="true"
    >
      <defs>
        <filter id="liquid-glass" x="-20%" y="-20%" width="140%" height="140%">
          <feTurbulence
            type="fractalNoise"
            baseFrequency="0.005 0.005"
            numOctaves={2}
            seed={92}
            result="noise"
          />
          <feGaussianBlur in="noise" stdDeviation={3} result="blurred" />
          <feDisplacementMap
            in="SourceGraphic" in2="blurred"
            scale={45}
            xChannelSelector="R" yChannelSelector="G"
          />
        </filter>
      </defs>
    </svg>
  )
}
