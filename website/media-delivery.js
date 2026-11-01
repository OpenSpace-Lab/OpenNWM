// This manifest selects a verified byte-for-byte mirror. Missing configuration
// keeps ordinary relative URLs working for local previews and static hosting.
let assetBase = "";
let localAssetPrefixes = [];
// Large masters can live in the source archive instead of the Pages artifact.
// Their pinned fallback URLs preserve the same bytes as the primary mirror.
const archivedMedia = new Map();
// A failed mirror stays on the original host for this page visit, including
// after offscreen videos release their src or another element uses the asset.
const failedMirrors = new Set();
function applyConfig(config) {
  if (!config || typeof config !== "object" || Array.isArray(config)) return false;
  let base = "";
  if (config.enabled !== false) {
    if (!config.assetBaseUrl) return false;
    const url = new URL(config.assetBaseUrl);
    if (url.protocol !== "https:") return false;
    base = url.href.replace(/\/?$/, "/");
  }
  localAssetPrefixes = Array.isArray(config.localAssetPrefixes)
    ? config.localAssetPrefixes.filter((prefix) => typeof prefix === "string")
      .map((prefix) => prefix.trim().replace(/^\/+/, "").replace(/^assets\//, "").replace(/\/+$/, ""))
      .filter(Boolean).map((prefix) => `${prefix}/`)
    : [];
  assetBase = base;
  return true;
}

let configured = false;
try {
  const inline = document.getElementById("media-delivery-config");
  if (inline) configured = applyConfig(JSON.parse(inline.textContent));
} catch { /* Older or invalid HTML can still use the standalone manifest. */ }
if (!configured) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 3000);
  try {
    const response = await fetch("content/media-delivery.json?v=texture-master-20261030-v6-inline1", { signal: controller.signal });
    if (response.ok) applyConfig(await response.json());
  } catch { /* Use the original host or a registered source archive. */ }
  finally { clearTimeout(timeout); }
}

function assetPath(source) {
  if (!source) return null;
  const value = String(source);
  try {
    const url = new URL(value, document.baseURI);
    const localBase = new URL("assets/", document.baseURI);
    if (url.origin === localBase.origin && url.pathname.startsWith(localBase.pathname)) {
      return url.pathname.slice(localBase.pathname.length) + url.search + url.hash;
    }
    if (assetBase) {
      const mirrorBase = new URL("assets/", assetBase);
      if (url.origin === mirrorBase.origin && url.pathname.startsWith(mirrorBase.pathname)) {
        return url.pathname.slice(mirrorBase.pathname.length) + url.search + url.hash;
      }
    }
  } catch { /* Non-asset inputs are left unchanged. */ }
  return null;
}

export function mediaUrl(source) {
  const relative = assetPath(source);
  // Compare whole directory prefixes, including for already mirrored URLs.
  if (relative !== null && localAssetPrefixes.some((prefix) => relative.startsWith(prefix))) {
    return originalMediaUrl(source);
  }
  const delivered = assetBase && relative !== null ? new URL(`assets/${relative}`, assetBase).href : originalMediaUrl(source);
  return delivered && failedMirrors.has(new URL(delivered, document.baseURI).href)
    ? originalMediaUrl(source) : delivered;
}

export function registerMediaFallback(source, fallback) {
  const relative = assetPath(source);
  if (relative === null || !fallback) return;
  try {
    const url = new URL(fallback);
    if (url.protocol === "https:") archivedMedia.set(relative.split(/[?#]/, 1)[0], url.href);
  } catch { /* Ignore an invalid optional archive URL. */ }
}

export function originalMediaUrl(source) {
  const relative = assetPath(source);
  return relative === null ? source : archivedMedia.get(relative.split(/[?#]/, 1)[0])
    || new URL(`assets/${relative}`, document.baseURI).href;
}

const records = new WeakMap();

function useLocalSource(element, record) {
  if (!record || record.fallback || !record.mirrored) return;
  record.fallback = true;
  failedMirrors.add(new URL(record.delivered, document.baseURI).href);
  const current = element.getAttribute("src");
  if (current && new URL(current, document.baseURI).href === record.local) return;
  element.setAttribute("src", record.local);
  if (element instanceof HTMLVideoElement) {
    element.load();
    element.dispatchEvent(new Event("mediafallback"));
  }
}

/** Apply the mirror, falling back once if that individual resource fails. */
export function setMediaUrl(element, attribute, source) {
  if (!source) return;
  const local = originalMediaUrl(source);
  const delivered = mediaUrl(source);
  let attributes = records.get(element);
  if (!attributes) {
    attributes = new Map();
    records.set(element, attributes);
    element.addEventListener("error", () => useLocalSource(element, attributes.get("src")));
  }
  const mirrored = new URL(delivered, document.baseURI).href !== new URL(local, document.baseURI).href;
  const previous = attributes.get(attribute);
  const record = previous?.local === local && previous.delivered === delivered
    ? previous : { local, delivered, mirrored, fallback: false, posterProbe: false };
  attributes.set(attribute, record);
  const current = element.getAttribute(attribute);
  const unchanged = current && new URL(current, document.baseURI).href === new URL(delivered, document.baseURI).href;
  if (!unchanged) {
    element.setAttribute(attribute, delivered);
  }
  // The hero starts from HTML before modules arrive. Catch an earlier failure
  // as well as future error events without restarting a healthy initial load.
  // An error left by an older src does not describe a newly selected resource.
  if (attribute === "src" && mirrored && unchanged && (
    (element instanceof HTMLVideoElement && element.error) ||
    (element instanceof HTMLImageElement && element.complete && !element.naturalWidth)
  )) useLocalSource(element, record);
  // Poster failures do not produce a video error event. The browser shares this
  // image request/cache entry with the poster; the probe supplies its error hook.
  if (attribute === "poster" && mirrored && !record.posterProbe) {
    record.posterProbe = true;
    const probe = new Image();
    probe.decoding = "async";
    probe.onerror = () => {
      failedMirrors.add(new URL(delivered, document.baseURI).href);
      if (attributes.get(attribute) === record) element.setAttribute(attribute, local);
    };
    probe.src = delivered;
  }
}
