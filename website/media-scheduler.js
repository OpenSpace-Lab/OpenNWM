import { setMediaUrl } from "./media-delivery.js?v=texture-master-20261030-v6-inline1";

// Share network and decoder capacity across the entire page. Media bytes and
// playback cadence are unchanged; only their lifetime follows the viewport.
const entries = new Set();
const byElement = new WeakMap();
let pending = false;
const nearMargin = 600;
const maxPreloads = 2;
const releaseDelay = 1500;

function queueUpdate() {
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => { pending = false; reconcile(); });
}

const nearObserver = new IntersectionObserver((changes) => {
  for (const change of changes) {
    const entry = byElement.get(change.target);
    if (entry) entry.near = change.isIntersecting;
  }
  queueUpdate();
}, { rootMargin: `${nearMargin}px 0px` });

const visibleObserver = new IntersectionObserver((changes) => {
  for (const change of changes) {
    const entry = byElement.get(change.target);
    if (!entry) continue;
    entry.ratio = change.intersectionRatio;
    entry.visible = change.isIntersecting && change.intersectionRatio >= .08;
  }
  queueUpdate();
}, { threshold: [0, .08, .25, .5, .75, 1] });

function showPosters(entry) {
  entry.videos.forEach((video, index) => {
    const poster = entry.posters[index];
    if (poster && !video.getAttribute("poster")) setMediaUrl(video, "poster", poster);
  });
}

function prepare(entry, preload = "auto") {
  clearTimeout(entry.releaseTimer);
  entry.releaseTimer = null;
  showPosters(entry);
  entry.videos.forEach((video, index) => {
    if (video.preload !== preload) video.preload = preload;
    if (!video.getAttribute("src") && entry.sources[index]) {
      setMediaUrl(video, "src", entry.sources[index]);
      video.load();
    }
  });
  entry.prepared = true;
}

function tryPlay(entry) {
  if (!entry.active || document.hidden || entry.onPlay) return;
  entry.videos.forEach((video) => {
    if (!video.paused || entry.attempts.has(video)) return;
    if (entry.startupBufferSeconds > 0) {
      if (video.readyState < 3 || video.seeking) return;
      // A short cushion absorbs download jitter without changing video quality.
      // Preload is a browser hint, so allow playback if it stops fetching or
      // has already spent ten seconds preparing playable frames.
      if (!hasVideoBuffer(video, entry.startupBufferSeconds)
        && video.networkState !== 1 && performance.now() < entry.startupDeadline) return;
    }
    entry.attempts.add(video);
    Promise.resolve(video.play()).catch(() => {}).finally(() => entry.attempts.delete(video));
  });
}

function resetStartup(entry) {
  clearTimeout(entry.startupTimer);
  entry.startupTimer = null;
  if (entry.startupBufferSeconds > 0) {
    entry.startupDeadline = performance.now() + 10000;
    entry.startupTimer = setTimeout(queueUpdate, 10000);
  }
}

function deactivate(entry) {
  clearTimeout(entry.startupTimer);
  entry.startupTimer = null;
  if (entry.active) {
    entry.active = false;
    entry.onPause?.();
  }
  entry.videos.forEach((video) => { if (!video.paused) video.pause(); });
}

function release(entry) {
  clearTimeout(entry.releaseTimer);
  entry.releaseTimer = null;
  if (!entry.prepared) return;
  deactivate(entry);
  entry.videos.forEach((video, index) => {
    if (video.readyState && !entry.resumePending && Number.isFinite(video.currentTime)) {
      entry.resumeTimes[index] = video.currentTime;
    }
    video.preload = "none";
    video.removeAttribute("src");
    video.load(); // Abort unused transfers and release offscreen decoders.
  });
  entry.prepared = false;
  entry.resumePending = true;
}

function hasVideoBuffer(video, seconds) {
  if (video.readyState < 3) return false;
  const target = Math.min(video.currentTime + seconds, video.duration - .025);
  for (let index = 0; index < video.buffered.length; index++) {
    if (video.buffered.start(index) <= video.currentTime + .025 && video.buffered.end(index) >= target) return true;
  }
  return false;
}

function hasPlaybackBuffer(entry) {
  return entry.videos.every((video) => hasVideoBuffer(video, Math.max(2, entry.startupBufferSeconds)));
}

function rank(entry) {
  const bounds = entry.element.getBoundingClientRect();
  const distance = Math.abs((bounds.top + bounds.bottom) / 2 - innerHeight / 2);
  return entry.priority * 10000 + entry.ratio * 1000 - distance;
}

function reconcile() {
  const candidates = [...entries].filter((entry) => entry.element.isConnected);
  const active = new Set();
  // Small previews are inexpensive; large compositions can reserve extra slots.
  const mobile = matchMedia("(max-width: 680px)").matches;
  let capacity = mobile ? 8 : 16;
  if (!document.hidden) {
    for (const entry of candidates.filter((entry) => entry.visible).sort((a, b) => rank(b) - rank(a))) {
      if (entry.cost > capacity && active.size) continue;
      active.add(entry);
      capacity -= entry.cost;
    }
  }
  const preload = new Set(document.hidden ? [] : candidates
    .filter((entry) => entry.near && !active.has(entry))
    .sort((a, b) => rank(b) - rank(a)).slice(0, mobile ? 1 : maxPreloads));
  // Buffer nearby frames only after visible videos have a playback cushion.
  // The bounded queue keeps large offscreen clips from starving playback.
  const warmPreload = [...active].every(hasPlaybackBuffer) ? "auto" : "metadata";

  for (const entry of candidates) {
    if (entry.near && !document.hidden) showPosters(entry);
    if (active.has(entry)) {
      prepare(entry);
      if (!entry.active) {
        entry.active = true;
        resetStartup(entry);
        entry.onPlay?.(entry.videos.map((video, index) => !entry.resumePending && video.readyState ? video.currentTime : entry.resumeTimes[index]));
        entry.resumePending = false;
      }
      tryPlay(entry);
    } else {
      deactivate(entry);
      if (preload.has(entry)) prepare(entry, warmPreload);
      else {
        entry.videos.forEach((video) => { video.preload = "none"; });
        if (entry.prepared && !entry.releaseTimer) {
          entry.releaseTimer = setTimeout(() => { release(entry); }, releaseDelay);
        }
      }
    }
  }
}

/** Register one video or a synchronized group; callbacks own paired playback. */
export function registerMedia({ element, videos, sources, posters, priority = 0,
  cost = videos.length, startupBufferSeconds = 0, onPlay, onPause }) {
  const entry = {
    element, videos, priority, cost, startupBufferSeconds, onPlay, onPause,
    startupDeadline: 0, startupTimer: null,
    sources: sources || videos.map((video) => video.dataset.src || video.dataset.source || video.getAttribute("src") || ""),
    posters: posters || videos.map((video) => video.dataset.poster || video.getAttribute("poster") || ""),
    visible: false, near: false, ratio: 0, active: false,
    prepared: videos.some((video) => video.hasAttribute("src")),
    resumeTimes: videos.map((video) => video.currentTime || 0),
    resumePending: false,
    attempts: new WeakSet(), releaseTimer: null,
  };
  const handlers = [];
  videos.forEach((video, index) => {
    // Native autoplay can ignore preload="none" and fetch invisible sources.
    // Scripted playback preserves automatic play while respecting this queue.
    video.autoplay = false;
    video.muted = video.defaultMuted = true;
    video.playsInline = true;
    video.controls = false;
    if (video.hasAttribute("src")) setMediaUrl(video, "src", entry.sources[index]);
    if (video.hasAttribute("poster")) setMediaUrl(video, "poster", entry.posters[index]);
    const ready = () => { tryPlay(entry); queueUpdate(); };
    const fallback = () => {
      if (!entry.active) return;
      resetStartup(entry);
      entry.onPause?.();
      entry.onPlay?.(entry.resumeTimes.slice());
      ready();
    };
    const guard = () => { if (!entry.active || document.hidden) video.pause(); };
    const restore = () => {
      const time = entry.resumeTimes[index];
      if (!entry.onPlay && time > .025 && Number.isFinite(video.duration)) {
        video.currentTime = Math.min(time, Math.max(0, video.duration - .05));
      }
      ready();
    };
    video.addEventListener("canplay", ready);
    video.addEventListener("loadeddata", ready);
    video.addEventListener("seeked", ready);
    video.addEventListener("suspend", ready);
    video.addEventListener("progress", queueUpdate);
    video.addEventListener("waiting", queueUpdate);
    video.addEventListener("mediafallback", fallback);
    video.addEventListener("play", guard);
    video.addEventListener("loadedmetadata", restore);
    handlers.push(() => {
      video.removeEventListener("canplay", ready);
      video.removeEventListener("loadeddata", ready);
      video.removeEventListener("seeked", ready);
      video.removeEventListener("suspend", ready);
      video.removeEventListener("progress", queueUpdate);
      video.removeEventListener("waiting", queueUpdate);
      video.removeEventListener("mediafallback", fallback);
      video.removeEventListener("play", guard);
      video.removeEventListener("loadedmetadata", restore);
    });
  });
  entries.add(entry);
  byElement.set(element, entry);
  nearObserver.observe(element);
  visibleObserver.observe(element);
  queueUpdate();
  return {
    refresh: queueUpdate,
    update(next) {
      if (next.sources && next.sources.some((source, index) => source !== entry.sources[index])) {
        release(entry);
        entry.sources = next.sources;
        entry.resumeTimes.fill(0);
      }
      if (next.posters) {
        entry.posters = next.posters;
        videos.forEach((video) => video.removeAttribute("poster"));
      }
      queueUpdate();
    },
    unregister() {
      release(entry);
      handlers.forEach((remove) => remove());
      entries.delete(entry);
      byElement.delete(element);
      nearObserver.unobserve(element);
      visibleObserver.unobserve(element);
    },
  };
}

document.addEventListener("visibilitychange", reconcile);
window.addEventListener("pageshow", queueUpdate);
window.addEventListener("resize", queueUpdate, { passive: true });
document.addEventListener("pointerdown", queueUpdate, { passive: true });
document.addEventListener("keydown", queueUpdate);
