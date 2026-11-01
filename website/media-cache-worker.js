// Cache only these immutable, original-byte masters. Other media streams normally.
const CACHE_PREFIX = "opennwm-go2-master-cache-";
const CACHE_NAME = `${CACHE_PREFIX}v7-privacy`;
const BASE = "https://modelscope.cn/models/LittleBoss/OpenNWM-Website/resolve/site-20261031-v7/";
const MASTERS = [
  ["route_07_clip_01_grass_passage", 83056964, "efdb79d1bf2a07a7d5d85865b56cf60a93a43ebe60510cb98e1415064a19da25"],
  ["route_07_clip_02_gravel_entry_curated", 67303672, "87380e898bcf3afea455dd4de4cd5720cfc55dcf196d01b2c130095ad3edbadb"],
  ["route_07_clip_03_garden_turn", 69424382, "bd31be3662c611bf71d0d4f31ff98eda088da4bb20b53e78baacce522ce51fea"],
  ["route_07_clip_04_stone_wall_bend_curated", 54666144, "02b85d361b4874407fabf4924117d2d6eb716c4a0b4a029cfb8e0397d8206360"],
  ["route_08_clip_01_leaf_grove", 61783191, "be1e2eab1672d751428f4d8d3f59c8bd18756da77c85504f80c3398976576c29"],
  ["route_08_clip_02_tree_slalom", 66279036, "02301dc1cc00f92ed26702934df8cdb8f095cda5825e311519ce9690d636e35e"],
  ["route_08_clip_03_grass_gully", 62402618, "2dfe3e759d9d0752d8ca9100e157845f207845fb6e6573c9a56c80705d95b784"],
  ["route_08_clip_04_paved_exit", 37708893, "a8323df64a9f11bf5b8fee19cb8e6d03aa599776e15993a472e43736bacc8af2"],
].map(([id, bytes, sha256]) => ({
  bytes, sha256, key: `${BASE}assets/go2/${id}/comparison.mp4`,
  cdnPath: `/prod/lfs-objects/${sha256.slice(0, 2)}/${sha256.slice(2, 4)}/${sha256.slice(4)}`,
}));
const pending = new Map();

function findMaster(request) {
  if (request.method !== "GET" || request.mode !== "cors") return null;
  const url = new URL(request.url);
  const canonical = `${url.origin}${url.pathname}`;
  return MASTERS.find((master) => canonical === master.key
    || (url.origin === "https://cdn-lfs-cn-1.modelscope.cn" && url.pathname === master.cdnPath));
}

function parseRange(value, size) {
  if (value === null) return null;
  const match = /^bytes=(\d*)-(\d*)$/i.exec(value.trim());
  if (!match || (!match[1] && !match[2])) return false;
  const first = match[1] ? Number(match[1]) : null;
  const last = match[2] ? Number(match[2]) : null;
  if ((first !== null && !Number.isSafeInteger(first))
    || (last !== null && !Number.isSafeInteger(last))) return false;
  if (first === null) return last > 0 ? [Math.max(0, size - last), size - 1] : false;
  if (first >= size || (last !== null && last < first)) return false;
  return [first, Math.min(last ?? size - 1, size - 1)];
}

function sliceStream(body, start, end) {
  const reader = body.getReader();
  let position = 0;
  let finished = false;
  return new ReadableStream({
    async pull(controller) {
      try {
        while (!finished) {
          const { value, done } = await reader.read();
          if (finished) return;
          if (done) throw new Error("Cached master response is incomplete");
          const offset = position;
          position += value.byteLength;
          if (position <= start) continue;
          controller.enqueue(value.subarray(Math.max(0, start - offset), Math.min(value.byteLength, end + 1 - offset)));
          if (position > end) {
            finished = true;
            controller.close();
            reader.cancel().catch(() => {});
          }
          return;
        }
      } catch (error) {
        if (!finished) {
          finished = true;
          controller.error(error);
          reader.cancel(error).catch(() => {});
        }
      }
    },
    cancel(reason) {
      finished = true;
      return reader.cancel(reason);
    },
  });
}

function fromCache(cached, request, master) {
  const range = parseRange(request.headers.get("Range"), master.bytes);
  const headers = new Headers(cached.headers);
  headers.set("Accept-Ranges", "bytes");
  headers.set("Access-Control-Allow-Origin", "*");
  headers.set("Access-Control-Expose-Headers", "Content-Range, X-OpenNWM-Cache");
  headers.set("X-OpenNWM-Cache", "hit");
  if (range === false) {
    headers.set("Content-Range", `bytes */${master.bytes}`);
    headers.set("Content-Length", "0");
    return new Response(null, { status: 416, headers });
  }
  if (range === null) return new Response(cached.body, { status: 200, headers });
  const [start, end] = range;
  headers.set("Content-Range", `bytes ${start}-${end}/${master.bytes}`);
  headers.set("Content-Length", String(end - start + 1));
  return new Response(sliceStream(cached.body, start, end), { status: 206, headers });
}

function isCompleteResponse(response, request, master) {
  if (!response.body || ![200, 206].includes(response.status)) return false;
  const range = parseRange(request.headers.get("Range"), master.bytes);
  if (range === false || (range && range[0] !== 0)) return false;
  if (Number(response.headers.get("Content-Length")) !== master.bytes) return false;
  const contentRange = response.headers.get("Content-Range");
  // The CDN does not expose Content-Range to CORS readers. In that case its
  // fixed master size and the counted stream below establish completeness.
  return !contentRange || contentRange === `bytes 0-${master.bytes - 1}/${master.bytes}`;
}

function saveComplete(cache, response, master) {
  let received = 0;
  const counted = response.clone().body.pipeThrough(new TransformStream({
    transform(chunk, controller) {
      received += chunk.byteLength;
      if (received > master.bytes) throw new Error("Master response exceeds its verified size");
      controller.enqueue(chunk);
    },
    flush() {
      if (received !== master.bytes) throw new Error("Master response is incomplete");
    },
  }));
  const headers = new Headers({
    "Content-Type": "video/mp4",
    "Content-Length": String(master.bytes),
    "Accept-Ranges": "bytes",
    "Cache-Control": "public, max-age=31536000, immutable",
    "ETag": `"${master.sha256}"`,
    "X-OpenNWM-Complete-Bytes": String(master.bytes),
  });
  // Cache API cannot store 206 responses. Only a fully counted original-byte
  // response is committed as 200; later requests receive the requested slice.
  return cache.put(master.key, new Response(counted, { status: 200, headers }));
}

async function serve(request, master, event) {
  let cache;
  try {
    cache = await caches.open(CACHE_NAME);
    // Seek/probe requests must keep streaming while a full cache write is open.
    const cached = await cache.match(master.key);
    if (cached?.status === 200
      && cached.headers.get("X-OpenNWM-Complete-Bytes") === String(master.bytes)) {
      return await fromCache(cached, request, master);
    }
  } catch { /* Storage failure preserves normal network playback. */ }

  // Leave actual network errors intact so the existing media fallback can act.
  const response = await fetch(request);
  if (cache && !pending.has(master.key) && isCompleteResponse(response, request, master)) {
    try {
      const saving = saveComplete(cache, response, master)
        .catch(async () => { try { await cache.delete(master.key); } catch { /* Storage unavailable. */ } })
        .finally(() => { pending.delete(master.key); });
      pending.set(master.key, saving);
      event.waitUntil(saving);
    } catch { /* A cache setup failure must not consume the network response. */ }
  }
  return response;
}

self.addEventListener("install", (event) => { event.waitUntil(self.skipWaiting()); });
self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    try {
      const names = await caches.keys();
      await Promise.all(names.filter((name) => name.startsWith(CACHE_PREFIX) && name !== CACHE_NAME)
        .map((name) => caches.delete(name)));
    } catch { /* Private browsing or exhausted storage must not block control. */ }
    await self.clients.claim();
  })());
});
self.addEventListener("fetch", (event) => {
  const master = findMaster(event.request);
  if (master) event.respondWith(serve(event.request, master, event));
});
