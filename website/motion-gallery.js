import { registerMedia } from "./media-scheduler.js?v=texture-master-20261030-v6-inline1";

const escape = (value) => String(value ?? "").replace(/[&<>"']/g, (character) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
})[character]);

// A single scale preserves the input path; up is the initial heading.
function projectTrajectory(positions) {
  const points = positions.map(([forward, left]) => [-left, -forward]);
  const xs = points.map(([x]) => x), ys = points.map(([, y]) => y);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const scale = Math.min(92 / Math.max(maxX - minX, .001), 176 / Math.max(maxY - minY, .001));
  return points.map(([x, y]) => [66 + (x - (minX + maxX) / 2) * scale, 112 + (y - (minY + maxY) / 2) * scale]);
}

/** Synchronized GT/OpenNWM wipe comparisons with their given input paths. */
export function initMotionGallery(demos) {
  const root = document.querySelector("#motion-gallery");
  if (!root) return { render() {} };
  const t = (en) => en;
  const items = [];
  root.innerHTML = '<div class="motion-gallery-grid"></div>';
  const grid = root.querySelector(".motion-gallery-grid");

  function updateTrajectory(item) {
    const time = item.leader.getAttribute("src") ? item.leader.currentTime : item.demo.posterTime || 0;
    const index = Math.min(item.points.length - 1, Math.max(0, Math.floor(time * item.demo.fps + .001)));
    const [x, y] = item.points[index];
    item.cursor.setAttribute("cx", x.toFixed(3));
    item.cursor.setAttribute("cy", y.toFixed(3));
    item.progress.setAttribute("points", item.points.slice(0, index + 1).map((point) => point.join(",")).join(" "));
  }
  function updateWipe(item) {
    const value = Number(item.slider.value);
    item.comparison.style.setProperty("--wipe", `${value}%`);
    item.slider.setAttribute("aria-valuetext", t(`Ground truth ${value}%, OpenNWM ${100 - value}%`, `真实画面 ${value}%，OpenNWM ${100 - value}%`));
  }
  function pause(item) {
    item.generation++;
    item.playing = false;
    item.phase = "paused";
    clearInterval(item.syncTimer);
    item.videos.forEach((video) => { video.pause(); video.playbackRate = 1; });
  }
  function seekPair(item, time) {
    for (const video of item.videos) {
      if (video.readyState >= 1 && Math.abs(video.currentTime - time) > .025) video.currentTime = time;
    }
  }
  function bufferedAhead(video, time = video.currentTime) {
    for (let index = 0; index < video.buffered.length; index++) {
      if (video.buffered.start(index) <= time + .025 && video.buffered.end(index) >= time) return video.buffered.end(index) - time;
    }
    return 0;
  }
  function synchronize(item) {
    if (!item.playing || item.phase !== "running") return;
    // The seeking event supplies the requested target; do not replace it with
    // the other video's old clock before that event has been handled.
    if (item.videos.some((video) => video.seeking)) return;
    if (item.videos.some((video) => video.readyState < 3)) {
      holdPair(item);
      return;
    }
    const prediction = item.videos[1];
    const drift = item.leader.currentTime - prediction.currentTime;
    if (Math.abs(drift) > .12) {
      holdPair(item);
      return;
    }
    // Small clock differences are corrected smoothly, without repeated seeks.
    prediction.playbackRate = Math.abs(drift) > .035 ? 1 + Math.max(-.05, Math.min(.05, drift * .5)) : 1;
  }
  async function startReadyPair(item) {
    if (!item.playing || item.phase !== "buffering" || document.hidden || !item.visible) return;
    for (const video of item.videos) {
      if (!item.pendingSeek.has(video) || video.readyState < 1) continue;
      item.pendingSeek.delete(video);
      if (Math.abs(video.currentTime - item.syncTime) > .025) video.currentTime = item.syncTime;
    }
    // Both decoders must have future data at the same time before either runs.
    // A short shared cushion avoids immediately re-entering waiting on slow links.
    if (!item.videos.every((video) => video.readyState >= 3 && !video.seeking
      && bufferedAhead(video, item.syncTime) >= Math.min(.5, Math.max(0, video.duration - item.syncTime - .025)))) return;
    const generation = item.generation;
    item.phase = "starting";
    const results = await Promise.allSettled(item.videos.map((video) => video.play()));
    if (generation !== item.generation) return;
    if (results.some((result) => result.status === "rejected")) { pause(item); return; }
    item.phase = "running";
    synchronize(item);
    if (item.phase === "running") item.syncTimer = setInterval(() => synchronize(item), 100);
  }
  function holdPair(item, time = Math.min(...item.videos.map((video) => video.currentTime))) {
    if (!item.playing) return;
    item.generation++;
    item.phase = "buffering";
    // Preserve an already aligned pair's clocks when buffering. Rounding the
    // target to a frame would cause an unnecessary backward seek on every wait.
    // Seek only once per barrier: decoders can report the preceding frame, and
    // retrying that seek on each canplay/seeked event can keep it seeking forever.
    item.syncTime = Math.max(0, time);
    item.pendingSeek = new Set(item.videos);
    clearInterval(item.syncTimer);
    item.videos.forEach((video) => { video.pause(); video.playbackRate = 1; });
    updateTrajectory(item);
    startReadyPair(item);
  }
  function play(item, restart = false, resumeTime) {
    if (item.playing || document.hidden || !item.visible) return;
    const time = restart || item.videos.some((video) => video.ended) ? 0 : resumeTime ?? Math.min(...item.videos.map((video) => video.currentTime));
    item.playing = true;
    item.videos.forEach((video) => {
      if (!video.getAttribute("src")) {
        video.src = video.dataset.source;
        video.preload = "auto";
        video.load();
      }
    });
    holdPair(item, time);
  }
  for (const demo of demos.demos) {
    const positions = demo.conditioningTrajectory?.positions;
    if (!Array.isArray(positions) || positions.length !== demo.frameCount) continue;
    const points = projectTrajectory(positions);
    const isMars = demo.environment === "mars";
    const card = document.createElement("article");
    card.className = `motion-card${demo.environment === "lunar" ? " motion-card-lunar" : isMars ? " motion-card-mars" : ""}`;
    card.dataset.demo = demo.id;
    if (demo.environment) card.dataset.environment = demo.environment;
    if (demo.dataset) card.dataset.dataset = demo.dataset;
    const video = (key) => `<video data-model="${key}" muted playsinline preload="none" width="${Number(demo.width) || 224}" height="${Number(demo.height) || 224}" data-source="${escape(demo[key])}" data-poster="${escape(demo[`${key}Poster`] || demo.poster)}"></video>`;
    const lunarMark = demo.environment === "lunar" ? '<svg class="motion-moon-mark" viewBox="0 0 24 24" aria-hidden="true"><path d="M20.5 14.3A8.8 8.8 0 0 1 9.7 3.5a8.8 8.8 0 1 0 10.8 10.8Z"/></svg>' : "";
    const marsMark = isMars ? '<svg class="motion-mars-mark" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8.5"/><path d="m5 8 4 1 2-3 4 1m-11 7 4-1 3 3 5 1m-2-7 2 2 3-1"/></svg>' : "";
    const hasNote = isMars && (demo.conditioningTrajectory?.estimated === true || demo.note);
    card.innerHTML = `<header class="motion-card-heading"><div class="motion-card-title"><h3 id="motion-title-${escape(demo.id)}"></h3>${demo.fineTuned ? '<span class="motion-adaptation-label"></span>' : ""}</div><div class="motion-card-actions">${lunarMark}${marsMark}</div></header><div class="motion-card-media"><div class="motion-comparison" style="--wipe:50%"><div class="motion-video-layer motion-prediction-layer">${video("prediction")}<span class="motion-video-label motion-label-prediction">OpenNWM</span></div><div class="motion-video-layer motion-truth-layer">${video("gt")}<span class="motion-video-label motion-label-gt"></span></div><div class="motion-wipe-handle" aria-hidden="true"><span>‹ ›</span></div><input class="motion-wipe-input" type="range" min="0" max="100" step="1" value="50"></div><figure class="motion-trajectory"><svg viewBox="0 0 132 224" role="img"><path class="motion-path-grid" d="M22 0v224M44 0v224M66 0v224M88 0v224M110 0v224M0 28h132M0 56h132M0 84h132M0 112h132M0 140h132M0 168h132M0 196h132"/><polyline class="motion-path-full" points="${points.map((point) => point.join(",")).join(" ")}"/><polyline class="motion-path-progress"/><circle class="motion-path-start" cx="${points[0][0]}" cy="${points[0][1]}" r="4"/><circle class="motion-path-cursor" r="5"/></svg><figcaption></figcaption></figure></div>${hasNote ? `<p class="motion-card-note" id="motion-note-${escape(demo.id)}"></p>` : ""}`;
    card.setAttribute("aria-labelledby", `motion-title-${demo.id}`);
    if (hasNote) card.setAttribute("aria-describedby", `motion-note-${demo.id}`);
    grid.append(card);
    const leader = card.querySelector('[data-model="gt"]');
    const item = {
      demo, card, points, leader,
      videos: [leader, card.querySelector('[data-model="prediction"]')],
      comparison: card.querySelector(".motion-comparison"), slider: card.querySelector("input"),
      cursor: card.querySelector(".motion-path-cursor"), progress: card.querySelector(".motion-path-progress"),
      visible: false, playing: false, generation: 0,
      phase: "paused", syncTime: 0, syncTimer: null, pendingSeek: new Set(),
    };
    item.videos.forEach((element) => { element.muted = true; });
    item.slider.addEventListener("input", () => updateWipe(item));
    leader.addEventListener("timeupdate", () => {
      updateTrajectory(item);
      synchronize(item);
    });
    item.videos.forEach((video) => {
      // Native autoplay and loop must obey the shared decoder barrier.
      video.addEventListener("play", () => {
        if (!item.playing || !item.visible || document.hidden || !["starting", "running"].includes(item.phase)) video.pause();
      });
      ["loadedmetadata", "canplay", "canplaythrough", "progress"].forEach((event) => video.addEventListener(event, () => startReadyPair(item)));
      video.addEventListener("waiting", () => {
        if (item.playing && (item.phase === "running" || item.phase === "starting") && video.readyState < 3) holdPair(item);
      });
      video.addEventListener("stalled", () => {
        // A stalled download need not interrupt playback if it is buffered.
        if (item.playing && item.phase === "running" && (video.readyState < 3 || bufferedAhead(video) < .25)) holdPair(item);
      });
      video.addEventListener("seeking", () => {
        if (!video.seeking || item.phase === "buffering" || item.phase === "starting") return;
        if (item.playing) holdPair(item, video.currentTime);
        else seekPair(item, video.currentTime);
      });
      video.addEventListener("seeked", () => { updateTrajectory(item); startReadyPair(item); });
      video.addEventListener("ended", () => {
        if (item.playing && video.ended) holdPair(item, 0);
      });
      video.addEventListener("error", () => pause(item));
    });
    updateTrajectory(item);
    items.push(item);
    registerMedia({
      element: item.comparison, videos: item.videos,
      onPlay: (times) => { item.visible = true; play(item, false, Math.min(...times)); },
      onPause: () => { item.visible = false; pause(item); },
    });
  }
  const lunarItems = items.filter(({ demo }) => demo.environment === "lunar" && demo.datasetId === "lusnar" && demo.fineTuned);
  const lunarFootnote = lunarItems.length ? document.createElement("p") : null;
  if (lunarFootnote) {
    lunarFootnote.id = "motion-lunar-method";
    lunarFootnote.className = "motion-lunar-footnote";
    lunarItems[lunarItems.length - 1].card.after(lunarFootnote);
    lunarItems.forEach(({ card }) => card.setAttribute("aria-describedby", lunarFootnote.id));
  }
  const marsItems = items.filter(({ demo }) => demo.datasetId === "marslab" && demo.fineTuned);
  const marsFootnote = marsItems.length ? document.createElement("p") : null;
  if (marsFootnote) {
    marsFootnote.id = "motion-mars-method";
    marsFootnote.className = "motion-lunar-footnote";
    marsItems[marsItems.length - 1].card.after(marsFootnote);
    marsItems.forEach(({ card }) => card.setAttribute("aria-describedby", marsFootnote.id));
  }
  function render() {
    if (lunarFootnote) {
      lunarFootnote.textContent = "LuSNAR simulation: fine-tuned on scenes Moon 1–5, validated on Moon 6, and tested only on Moon 7–9.";
    }
    if (marsFootnote) {
      marsFootnote.textContent = "MarsLab simulation: fine-tuned on six trajectories, validated on one, and tested on three held-out trajectories.";
    }
    items.forEach((item) => {
      const title = t(item.demo.title, item.demo.titleZh);
      const isMars = item.demo.environment === "mars";
      const estimated = item.demo.conditioningTrajectory?.estimated === true;
      item.card.querySelector("h3").textContent = title;
      item.card.querySelector(".motion-label-gt").textContent = t("Ground truth", "真实画面");
      item.card.querySelector("figcaption").textContent = estimated ? t("Estimated path", "估计轨迹") : t("Given path", "给定轨迹");
      item.card.querySelector(".motion-trajectory svg").setAttribute("aria-label", estimated
        ? t(`Visually estimated path for ${title}, with relative scale. The dot follows the current video frame.`, `${title}的视觉估计轨迹，采用相对尺度，圆点对应当前视频帧。`)
        : t(`Given path for ${title}. The dot follows the current video frame.`, `${title}的给定轨迹，圆点对应当前视频帧。`));
      const adaptation = item.card.querySelector(".motion-adaptation-label");
      if (adaptation) {
        adaptation.textContent = isMars ? t("Held-out · fine-tuned", "留出测试 · 微调") : t("Lightly fine-tuned", "少量微调");
        if (isMars) adaptation.title = estimated
          ? t("OpenNWM predicts each future frame from four fixed observed frames and estimated actions. Offline pose estimation uses future reference frames to label those actions; they are not recorded controls.", "OpenNWM 使用固定的四帧观测和估计动作预测各未来帧。动作由包含未来参考帧的完整视频离线估计，并非记录的控制指令。")
          : t("OpenNWM predicts each future frame from four fixed observed frames and planar actions measured from simulator ground-truth body poses. Test trajectories were excluded from fine-tuning.", "OpenNWM 使用固定的四帧观测，以及仿真器真实车体位姿计算的平面动作，预测各未来帧。测试轨迹未参与微调。");
      }
      const note = item.card.querySelector(".motion-card-note");
      if (note) note.textContent = item.demo.note ? t(item.demo.note, item.demo.noteZh) : t("Actions estimated offline from full clips; relative scale.", "动作由完整视频离线估计；尺度为相对值。");
      item.slider.setAttribute("aria-label", t(`Compare Ground truth and OpenNWM: ${title}`, `滑动对比真实画面与 OpenNWM：${title}`));
      item.videos.forEach((element) => element.setAttribute("aria-label", `${title} · ${element.dataset.model === "gt" ? t("Ground truth", "真实画面") : "OpenNWM"}`));
      updateWipe(item);
    });
  }
  render();
  return { render };
}
