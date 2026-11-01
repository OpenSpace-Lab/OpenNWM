import { registerMedia } from "./media-scheduler.js?v=texture-master-20261030-v6-inline1";
import { registerMediaFallback } from "./media-delivery.js?v=texture-master-20261030-v6-inline1";
import { initMediaCache } from "./media-cache.js?v=privacy-20261031-v7";

const root = document.querySelector("#go2-navigation");
const revision = "privacy-20261031";
const escape = (value) => String(value ?? "").replace(/[&<>"']/g, (character) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
})[character]);
const asset = (path) => typeof path === "string" && /^assets\/go2\/[a-zA-Z0-9_./-]+$/.test(path) && !path.includes("..") ? path : "";
const mediaAsset = (path) => asset(path) ? `${asset(path)}?v=${revision}` : "";

function groupRoutes(manifest, clips) {
  const grouped = new Map();
  for (const clip of clips) {
    const routeId = clip.route_id || clip.id?.match(/^route_\d+/)?.[0];
    // Each source recording is one scene, with all of its excerpts together.
    const key = clip.episode || routeId || clip.id;
    if (!grouped.has(key)) {
      const metadata = (manifest.routes || []).find((route) => clip.episode && route.episode === clip.episode)
        || (manifest.routes || []).find((route) => route.id === routeId);
      grouped.set(key, {
        id: metadata?.id || routeId || key,
        title: metadata?.title || clip.route_title || `Scene ${grouped.size + 1}`,
        clips: [],
      });
    }
    grouped.get(key).clips.push(clip);
  }
  return [...grouped.values()].map((route) => ({ ...route,
    clips: route.clips.sort((a, b) => (a.start_time_s || 0) - (b.start_time_s || 0)),
  }));
}

async function init() {
  if (!root) return;
  const cacheReady = initMediaCache();
  const response = await fetch(`content/go2.json?v=${revision}`);
  if (!response.ok) throw new Error(`Go2 comparison: HTTP ${response.status}`);
  const manifest = await response.json();
  if (manifest.status !== "ready") return;
  const clips = (manifest.clips || []).filter((clip) => clip && asset(clip.video));
  if (!clips.length) return;

  const routes = groupRoutes(manifest, clips);
  root.innerHTML = `<div class="go2-heading"><div><span id="go2-section-number" class="eyebrow section-number">03 / REAL WORLD</span><h2 id="go2-title">Real-world navigation.</h2></div></div>
    <div class="go2-routes">${routes.map((route, index) => `<section class="go2-route" data-route="${escape(route.id)}" aria-labelledby="go2-route-title-${index}">
      <div class="go2-route-heading"><h3 id="go2-route-title-${index}">${escape(route.title)}</h3></div>
      <div class="go2-grid">${route.clips.map((clip, clipIndex) => `<figure class="go2-clip" data-clip="${escape(clip.id)}"><div class="go2-screen"><video muted loop playsinline preload="none" width="1920" height="1080" disablepictureinpicture disableremoteplayback tabindex="-1" aria-label="${escape(route.title)}: clip ${clipIndex + 1}, recorded robot and OpenNWM / NWM predictions and trajectories"></video><p class="go2-media-error" role="status" hidden>Video unavailable</p></div></figure>`).join("")}</div>
    </section>`).join("")}</div>`;
  root.hidden = false;

  await cacheReady;
  const cards = root.querySelectorAll(".go2-clip");
  routes.flatMap((route) => route.clips).forEach((clip, index) => {
    const card = cards[index];
    const video = card.querySelector("video");
    const error = card.querySelector(".go2-media-error");
    video.muted = video.defaultMuted = true;
    video.controls = false;
    video.loop = video.playsInline = true;
    if (clip.fallback_video) video.crossOrigin = "anonymous";
    registerMediaFallback(mediaAsset(clip.video), clip.fallback_video);
    video.addEventListener("playing", () => { error.hidden = true; });
    video.addEventListener("error", () => { error.hidden = false; });
    registerMedia({
      element: card.querySelector(".go2-screen"),
      videos: [video], sources: [mediaAsset(clip.video)], posters: [mediaAsset(clip.poster)],
      cost: 3, priority: 1,
      startupBufferSeconds: clip.fallback_video ? 4 : 0,
    });
  });
}

init().catch((error) => console.error(error));
