let initialization;

/** Best-effort control before Go2 video URLs are assigned; never delay >3s. */
export function initMediaCache() {
  if (initialization) return initialization;
  if (!globalThis.isSecureContext || !("serviceWorker" in navigator)) return Promise.resolve(false);
  const worker = new URL("./media-cache-worker.js?v=go2-master-v7-privacy", import.meta.url);
  const scope = new URL("./", import.meta.url);
  initialization = new Promise((resolve) => {
    const serviceWorker = navigator.serviceWorker;
    let finished = false;
    const finish = (controlled) => {
      if (finished) return;
      finished = true;
      clearTimeout(timer);
      serviceWorker.removeEventListener("controllerchange", check);
      resolve(controlled);
    };
    const check = () => {
      if (serviceWorker.controller?.scriptURL === worker.href) finish(true);
    };
    const timer = setTimeout(() => finish(false), 3000);
    serviceWorker.addEventListener("controllerchange", check);
    try {
      Promise.resolve(serviceWorker.register(worker.href, { scope: scope.href, updateViaCache: "none" }))
        .then(check).catch(() => finish(false));
    } catch { finish(false); }
  });
  return initialization;
}
