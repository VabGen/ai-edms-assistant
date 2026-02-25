/** Обёртка над chrome.runtime.sendMessage с Promise */
export function sendMsg<T = unknown>(
  type: string,
  payload: unknown,
): Promise<T> {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage({ type, payload }, (res) => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message))
        return
      }
      if (res?.success) resolve(res.data as T)
      else              reject(new Error(res?.error ?? 'Unknown error'))
    })
  })
}
