const $ = (selector) => document.querySelector(selector);
const escape = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

export async function initCinematic(getLanguage) {
  const t = (en, zh) => getLanguage() === "zh" ? zh || en : en;
  const hero = $("#hero-video");
  const heroScene = $("#hero-scene");
  const section = $(".cinematic-hero");
  const seek = $("#journey-seek");
  let film, journeyScenes = [], chapters = [], frame = 0;
  const mediaUrl = (source) => {
    const url = new URL(source, document.baseURI);
    url.searchParams.set("v", film?.cacheVersion || "fixed-wall-moon-mars-v5");
    return url.href;
  };
  const clock = (seconds) => `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
  const filmDuration = () => Number.isFinite(hero.duration) ? hero.duration : Number(film?.journey?.durationSeconds || journeyScenes.at(-1)?.end || 60);
  const normalizeScene = (scene) => ({
    start: Number(scene.startSeconds ?? scene.start_seconds ?? 0),
    end: Number(scene.endSeconds ?? scene.end_seconds ?? 0),
    planet: scene.planet || "wall",
    labelEn: scene.labelEn || scene.title || scene.label || scene.source || "NavAnywhere",
    labelZh: scene.labelZh || scene.titleZh,
  });
  function updateHeroScene() {
    const time = hero.currentTime || 0;
    const duration = filmDuration();
    const progress = Math.max(0, Math.min(1, time / duration));
    let scene = journeyScenes[0];
    for (const candidate of journeyScenes) {
      if (candidate.start > time + 0.001) break;
      scene = candidate;
    }
    const label = scene ? t(scene.labelEn, scene.labelZh) : "NavAnywhere";
    const planet = scene?.planet || "wall";
    if (heroScene.textContent !== (planet === "wall" ? "" : label)) heroScene.textContent = planet === "wall" ? "" : label;
    section.dataset.planet = planet;
    seek.max = duration;
    seek.value = time;
    seek.setAttribute("aria-valuetext", `${label}, ${clock(time)} ${t("of", "/")} ${clock(duration)}`);
    $("#journey-rover").style.left = `${progress * 100}%`;
    $("#journey-progress").style.width = `${progress * 100}%`;
    const timeText = `${clock(time)} / ${clock(duration)}`;
    if ($("#journey-time").textContent !== timeText) $("#journey-time").textContent = timeText;
    $("#journey-chapters").querySelectorAll("button").forEach((button, i) => {
      const chapter = chapters[i];
      const active = time >= chapter.start - 0.001 && time < (chapter.end || duration) - 0.001;
      button.classList.toggle("active", active);
      button.setAttribute("aria-current", active ? "step" : "false");
    });
  }
  function drawFrame() {
    updateHeroScene();
    frame = !hero.paused && !document.hidden ? requestAnimationFrame(drawFrame) : 0;
  }
  hero.addEventListener("play", () => { cancelAnimationFrame(frame); drawFrame(); });
  hero.addEventListener("pause", () => { cancelAnimationFrame(frame); frame = 0; updateHeroScene(); });
  ["timeupdate", "seeking", "seeked", "loadedmetadata", "emptied"].forEach((event) => hero.addEventListener(event, updateHeroScene));
  seek.addEventListener("input", () => { hero.currentTime = Number(seek.value); updateHeroScene(); });
  function seekTo(time) {
    hero.currentTime = time;
    updateHeroScene();
  }
  $("#journey-chapters").addEventListener("click", (event) => {
    const button = event.target.closest("[data-start]");
    if (button) seekTo(Number(button.dataset.start));
  });
  $("#watch-video-wall")?.addEventListener("click", () => {
    const wall = chapters.find((chapter) => chapter.planet === "wall");
    if (wall) {
      seekTo(wall.start);
      if (hero.paused) $("#hero-pause").click();
    }
  });
  function renderNavigation() {
    const duration = filmDuration();
    const planetNames = { moon: t("MOON", "月球"), mars: t("MARS", "火星"), wall: t("DATASET", "数据集") };
    $("#journey-chapters").innerHTML = chapters.map((chapter, i) => `<button type="button" data-start="${chapter.start}" data-planet="${escape(chapter.planet)}" style="--chapter-span:${Math.max(.01, chapter.end - chapter.start)}" aria-label="${escape(t("Go to ", "跳转到 ") + (planetNames[chapter.planet] || chapter.labelEn))}"><span class="chapter-dot"></span><span>${escape(planetNames[chapter.planet] || t(chapter.labelEn, chapter.labelZh))}</span><span class="chapter-order">${String(i + 1).padStart(2, "0")}</span></button>`).join("");
    $("#journey-stops").innerHTML = journeyScenes.slice(1).map((scene) => `<i class="journey-stop planet-stop" style="left:${scene.start / duration * 100}%"></i>`).join("");
    seek.setAttribute("aria-label", t("Seek through the video wall, Moon and Mars", "跳转视频墙、月球与火星"));
    $(".journey-navigation").setAttribute("aria-label", t("Opening film navigation", "开场影片导航"));
    updateHeroScene();
  }

  function render() {
    renderNavigation();
    if (film?.counts) $("#mosaic-counts").textContent = t(`NAVANYWHERE / ${film.counts.sources} SOURCES / ${film.counts.clips} SEQUENCES`, `NAVANYWHERE / ${film.counts.sources} 个来源 / ${film.counts.clips} 段序列`);
  }
  const response = await fetch("content/cinematic.json?v=main-figure-v8-20261010", { cache: "no-cache" }).catch(() => null);
  if (response?.ok) {
    film = await response.json();
    journeyScenes = (film.journeyScenes || film.journey?.scenes || []).map(normalizeScene).filter((scene) => Number.isFinite(scene.start)).sort((a, b) => a.start - b.start);
    chapters = (film.chapters || []).map(normalizeScene);
    if (!chapters.length) {
      for (const scene of journeyScenes) {
        const previous = chapters.at(-1);
        if (previous?.planet === scene.planet) previous.end = scene.end;
        else chapters.push({ ...scene });
      }
    }
    hero.poster = mediaUrl(film.journey?.poster || "assets/hero/journey-poster.webp");
    hero.src = mediaUrl(film.journey?.src || "assets/hero/journey.mp4");
  } else {
    hero.src = mediaUrl("assets/hero/journey.mp4");
  }
  render();
  return { render };
}
