/// <reference types="wxt/browser" />

declare interface WxtContentScriptContext {
  abort(reason?: string): void
  onInvalidated: (cb: () => void) => void
  isValid: boolean
}

declare interface BackgroundDefinition {
  type?: 'module'
  persistent?: boolean
  main(): void
}

declare interface ContentScriptDefinition {
  matches: string[]
  runAt?: 'document_start' | 'document_end' | 'document_idle'
  allFrames?: boolean
  cssInjectionMode?: 'manifest' | 'manual' | 'ui'
  main(ctx: WxtContentScriptContext): void | Promise<void>
}

declare function defineBackground(def: BackgroundDefinition): BackgroundDefinition
declare function defineContentScript(def: ContentScriptDefinition): ContentScriptDefinition

declare function createShadowRootUi<T>(
  ctx: WxtContentScriptContext,
  options: {
    name: string
    position: 'inline' | 'overlay' | 'modal'
    anchor?: string | Element | (() => Element | null)
    append?: 'first' | 'last' | 'replace' | ((anchor: Element, ui: Element) => void)
    onMount(container: HTMLElement): T
    onRemove?(mounted: T): void
  },
): Promise<{ mount(): void; remove(): void }>
