import { initResearch } from "./research.js?v=main-figure-v8-20261010";
import { initMotionGallery } from "./motion-gallery.js?v=main-figure-v8-20261010";
import { initNavigation } from "./navigation.js?v=main-figure-v8-20261010";
import { initCinematic } from "./cinematic.js?v=main-figure-v8-20261010";
import { initPlayground } from "./playground.js?v=main-figure-v8-20261010";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const motion = matchMedia("(prefers-reduced-motion: reduce)");
let language = "en";
const t = (en, zh) => (language === "zh" ? zh : en);
const icon = (name) => `<svg aria-hidden="true"><use href="#${name}"/></svg>`;
const escape = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (char) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        char
      ],
  );
let research, datasets, demos, paper, cinematic, playground, motionGallery, navigation;
let filter = "all",
  expanded = false,
  currentDataset;

function translate() {
  document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
  $$("[data-en][data-zh]").forEach((node) => {
    node.textContent = node.dataset[language];
  });
  $("#language").textContent = language === "zh" ? "EN" : "中文";
  $("#language").setAttribute(
    "aria-label",
    t("Switch to Chinese", "Switch to English"),
  );
  cinematic?.render();
  playground?.render();
  if (!datasets) return;
  motionGallery?.render();
  navigation?.render();
  renderDatasets();
  renderDistribution();
  research.render();
  updateHeroUI();
  if ($("#dataset-dialog").open && currentDataset) renderDialog(currentDataset);
}
$("#language").addEventListener("click", () => {
  language = language === "en" ? "zh" : "en";
  translate();
});
translate();

// Navigation remains usable before the content finishes loading.
function closeMenu() {
  $(".site-header").classList.remove("menu-open");
  $("#mobile-menu").setAttribute("aria-expanded", "false");
}
$("#mobile-menu").addEventListener("click", () => {
  const open = $(".site-header").classList.toggle("menu-open");
  $("#mobile-menu").setAttribute("aria-expanded", String(open));
});
$$(".desktop-nav a").forEach((link) =>
  link.addEventListener("click", closeMenu),
);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeMenu();
});
const navObserver = new IntersectionObserver(
  (entries) => {
    for (const entry of entries)
      if (entry.isIntersecting) {
        $$(".desktop-nav a").forEach((link) =>
          link.classList.toggle("active", link.hash === `#${entry.target.id}`),
        );
      }
  },
  { rootMargin: "-15% 0px -55% 0px" },
);
$$("main section[id]").forEach((section) => navObserver.observe(section));
if (!motion.matches) {
  document.documentElement.classList.add("js-motion");
  const reveal = new IntersectionObserver(
    (entries, observer) => {
      entries
        .filter((entry) => entry.isIntersecting)
        .forEach((entry) => {
          entry.target.classList.add("visible");
          observer.unobserve(entry.target);
        });
    },
    { threshold: 0.1 },
  );
  $$(".reveal").forEach((node) => reveal.observe(node));
}

// Background footage is a dataset sample, not a model prediction.
const hero = $("#hero-video");
let heroEnabled = !motion.matches,
  heroVisible = true;
function updateHeroUI() {
  $("#hero-pause").innerHTML = icon(hero.paused ? "play" : "pause");
  $("#hero-pause").setAttribute(
    "aria-label",
    hero.paused
      ? t("Play background video", "播放背景视频")
      : t("Pause background video", "暂停背景视频"),
  );
}
function updateHero() {
  if (
    heroEnabled &&
    heroVisible &&
    !document.hidden &&
    !$("#dataset-dialog").open
  )
    hero.play().catch(updateHeroUI);
  else hero.pause();
}
hero.addEventListener("play", updateHeroUI);
hero.addEventListener("pause", updateHeroUI);
hero.addEventListener("loadedmetadata", updateHero);
$("#hero-pause").addEventListener("click", () => {
  heroEnabled = hero.paused;
  updateHero();
});
new IntersectionObserver(
  ([entry]) => {
    heroVisible = entry.isIntersecting;
    updateHero();
  },
  { threshold: 0.05 },
).observe(hero);
motion.addEventListener("change", () => {
  heroEnabled = !motion.matches;
  updateHero();
});

document.addEventListener("visibilitychange", updateHero);

function loadVideo(video, src, poster, signal) {
  return new Promise((resolve, reject) => {
    const finish = (error) => {
      clearTimeout(timeout);
      video.removeEventListener("loadeddata", ready);
      video.removeEventListener("error", failed);
      signal.removeEventListener("abort", aborted);
      error ? reject(error) : resolve();
    };
    const ready = () => finish();
    const failed = () => finish(new Error("Video unavailable"));
    const aborted = () =>
      finish(new DOMException("Scene changed", "AbortError"));
    const timeout = setTimeout(
      () => finish(new Error("Video loading timed out")),
      30000,
    );
    video.addEventListener("loadeddata", ready, { once: true });
    video.addEventListener("error", failed, { once: true });
    signal.addEventListener("abort", aborted, { once: true });
    // Small, self-hosted clips are loaded as blobs so seeking also works on
    // Python's basic HTTP server, which does not implement byte-range requests.
    video.poster = poster;
    video.preload = "auto";
    fetch(src, { signal })
      .then(async (response) => {
        if (!response.ok) throw new Error("Video unavailable");
        const blob = await response.blob();
        if (signal.aborted) return;
        if (video.dataset.objectUrl)
          URL.revokeObjectURL(video.dataset.objectUrl);
        const url = URL.createObjectURL(blob);
        video.dataset.objectUrl = url;
        video.dataset.source = src;
        video.src = url;
        video.load();
      })
      .catch(finish);
  });
}

// A browsable atlas: pretraining sources and action-labeled sources stay distinct.
const descriptionsZh = {
  "botanic-garden": "树冠下蜿蜒的花园石径",
  "casia-nav": "城市街道与地标周边",
  citywalker: "人行道、建筑与街头生活",
  dl3dv: "室内空间与多视角移动",
  egowalk: "第一人称视角下的户外行走",
  krishnacam: "日常通勤与城市穿行",
  lavn: "多样的导航场景与视点",
  rover: "地面机器人眼中的自然环境",
  realestate10k: "住宅与室内空间探索",
  sanpo: "街道与人行道上的导航",
  "great-outdoors": "林间小路与非结构化地形",
  "walking-tours": "夜色中的街头漫步",
  ego4d: "第一人称日常活动与移动",
  "i2nav-robot": "机器人视角的室内外导航",
  "ub-visiogeoloc": "城市与公共空间中的移动",
  recon: "真实机器人导航轨迹",
  sacson: "走廊与室内移动场景",
  scand: "人与机器人共享的校园步道",
  "tartan-drive": "越野驾驶与自然地形",
};
const featured = [
  "walking-tours",
  "botanic-garden",
  "dl3dv",
  "citywalker",
  "realestate10k",
  "rover",
  "i2nav-robot",
  "ego4d",
];
function orderedDatasets() {
  return [...datasets.datasets].sort((a, b) => {
    const rank = (item) =>
      featured.includes(item.id)
        ? featured.indexOf(item.id)
        : featured.length + datasets.datasets.indexOf(item);
    return rank(a) - rank(b);
  });
}
// Load card videos only near the viewport; play only visible previews.
const visibleDatasetVideos = new Set();
function updateDatasetVideos() {
  $$("#dataset-grid video").forEach((video) => {
    const shouldPlay = visibleDatasetVideos.has(video) && !motion.matches && !document.hidden && !$("#dataset-dialog").open;
    if (shouldPlay) video.play().catch(() => {});
    else video.pause();
  });
}
const datasetPreloadObserver = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    if (!entry.isIntersecting) continue;
    const video = entry.target;
    if (!video.src && video.dataset.src) {
      video.src = video.dataset.src;
      video.load();
    }
    datasetPreloadObserver.unobserve(video);
  }
}, { rootMargin: "220px 0px" });
const datasetPlaybackObserver = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    if (entry.isIntersecting) visibleDatasetVideos.add(entry.target);
    else visibleDatasetVideos.delete(entry.target);
  }
  updateDatasetVideos();
}, { threshold: 0.15 });
motion.addEventListener("change", updateDatasetVideos);
document.addEventListener("visibilitychange", updateDatasetVideos);
function renderDatasets() {
  $$("#dataset-grid video").forEach((video) => video.pause());
  datasetPreloadObserver.disconnect();
  datasetPlaybackObserver.disconnect();
  visibleDatasetVideos.clear();
  const matching = orderedDatasets().filter(
    (item) => filter === "all" || item.environments.includes(filter),
  );
  const visible = expanded ? matching : matching.slice(0, 8);
  $("#dataset-grid").innerHTML = visible
    .map(
      (item) =>
        `<button class="dataset-card" type="button" data-dataset="${item.id}" aria-haspopup="dialog" aria-label="${escape(t("Explore ", "查看 ") + item.name)}"><div class="dataset-card-image"><video data-src="${escape(item.video || "")}" poster="${escape(item.image)}" muted loop playsinline preload="none" width="480" height="270" aria-label="${escape(item.name + t(" dataset video", " 数据集视频"))}"></video><span class="dataset-card-index">${String(datasets.datasets.indexOf(item) + 1).padStart(2, "0")}</span><span class="dataset-card-arrow">${icon("arrow-up")}</span>${item.video ? `<span class="dataset-card-play">${icon("play")}</span>` : ""}</div><div class="dataset-card-label"><h3>${escape(item.name)}</h3><span class="source-group">${item.group === "pretraining" ? "PRETRAIN" : "POST-TRAIN"}</span></div><span class="dataset-card-sub">${escape(t(item.title, item.titleZh || descriptionsZh[item.id]))}</span></button>`,
    )
    .join("");
  $$("#dataset-grid video").forEach((video) => {
    video.addEventListener("loadeddata", updateDatasetVideos);
    datasetPreloadObserver.observe(video);
    datasetPlaybackObserver.observe(video);
  });
  $("#dataset-count").textContent = t(
    `${visible.length} / ${matching.length} SOURCES`,
    `${visible.length} / ${matching.length} 个来源`,
  );
  $("#dataset-expand").hidden = matching.length <= 8;
  $("#dataset-expand").setAttribute("aria-expanded", String(expanded));
  $("#dataset-expand").innerHTML =
    `<span>${expanded ? t("Show fewer sources", "收起部分来源") : t("Show all sources", "展开全部来源")}</span><span>${expanded ? "−" : "＋"}</span>`;
  $$("#dataset-filters button").forEach((button) => {
    button.classList.toggle("active", button.dataset.filter === filter);
    button.setAttribute(
      "aria-pressed",
      String(button.dataset.filter === filter),
    );
  });
}
$("#dataset-filters").addEventListener("click", (event) => {
  const button = event.target.closest("[data-filter]");
  if (!button) return;
  filter = button.dataset.filter;
  expanded = false;
  renderDatasets();
});
$("#dataset-expand").addEventListener("click", () => {
  expanded = !expanded;
  renderDatasets();
  if (!expanded)
    $("#dataset-filters").scrollIntoView({
      behavior: motion.matches ? "instant" : "smooth",
      block: "start",
    });
});
const dialog = $("#dataset-dialog"),
  dialogVideo = $("#dialog-video");
let dialogController;
function renderDialog(item) {
  const pretraining = item.group === "pretraining";
  $("#dialog-source").textContent = pretraining
    ? t("NAVANYWHERE / VIDEO PRETRAINING", "NAVANYWHERE / 视频预训练")
    : t("ACTION-LABELED / NWM POST-TRAINING", "动作标注数据 / NWM 后训练");
  $("#dialog-title").textContent = item.name;
  $("#dialog-description").textContent = t(
    item.description,
    descriptionsZh[item.id],
  );
  const stats = (
    pretraining
      ? paper.datasets.navanywhere.sources
      : paper.datasets.action_annotated.rows
  ).find((row) => row.id === item.id);
  const facts = [];
  if (stats) {
    facts.push(
      `${stats.frames.toLocaleString("en-US")} ${t("frames", "帧")}`,
      `${stats.sequences.toLocaleString("en-US")} ${t("sequences", "段序列")}`,
    );
    if (pretraining)
      facts.push(
        `${stats.source_video_hours.toLocaleString("en-US")} h ${t("source video", "源视频")}`,
      );
  }
  facts.push(
    item.video
      ? t(
          `Consecutive source frames · ${item.source?.video?.playback_fps || 12} fps display`,
          `连续源帧 · 以 ${item.source?.video?.playback_fps || 12} fps 展示`,
        )
      : t("A sampled source frame", "源数据中的采样帧"),
  );
  if (!pretraining)
    facts.push(
      t("Excluded from NavAnywhere totals", "不计入 NavAnywhere 统计"),
    );
  $("#dialog-facts").innerHTML = facts
    .map((fact) => `<span>${escape(fact)}</span>`)
    .join("");
}
$("#dataset-grid").addEventListener("click", async (event) => {
  const button = event.target.closest("[data-dataset]");
  if (!button) return;
  currentDataset = datasets.datasets.find(
    (item) => item.id === button.dataset.dataset,
  );
  renderDialog(currentDataset);
  document.querySelectorAll("main video").forEach((video) => video.pause());
  $("#dialog-image").src = currentDataset.image;
  $("#dialog-image").alt = currentDataset.description;
  $("#dialog-image").hidden = Boolean(currentDataset.video);
  dialogVideo.hidden = !currentDataset.video;
  $("#dialog-media-status").hidden = true;
  dialogController?.abort();
  dialogController = new AbortController();
  const controller = dialogController;
  dialog.showModal();
  document.body.classList.add("dialog-open");
  updateHero();
  updateDatasetVideos();
  if (currentDataset.video) {
    try {
      await loadVideo(
        dialogVideo,
        currentDataset.video,
        currentDataset.image,
        controller.signal,
      );
      if (dialog.open && !controller.signal.aborted && !motion.matches)
        await dialogVideo.play();
    } catch (error) {
      if (error.name !== "AbortError") {
        $("#dialog-media-status").textContent = t("Video could not load. Please close and try again.", "视频加载失败，请关闭后重试。");
        $("#dialog-media-status").hidden = false;
      }
    }
  }
});
$("#dialog-close").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", (event) => {
  const box = dialog.getBoundingClientRect();
  if (
    event.target === dialog &&
    (event.clientX < box.left ||
      event.clientX > box.right ||
      event.clientY < box.top ||
      event.clientY > box.bottom)
  )
    dialog.close();
});
dialog.addEventListener("close", () => {
  dialogController?.abort();
  dialogVideo.pause();
  document.body.classList.remove("dialog-open");
  updateHero();
  updateDatasetVideos();
});

const sceneLabels = [
  "住宅空间",
  "公共与商业",
  "城市与交通",
  "公园与花园",
  "自然与越野",
];
const swatches = ["#b6ee65", "#91b96a", "#6f8b63", "#d4e6be", "#a4a7ac"];
function renderDistribution() {
  const categories = paper.datasets.navanywhere.scene_categories;
  $("#distribution-value").textContent = t("5 SCENE TYPES", "5 类环境");
  $("#distribution-bar").innerHTML = categories
    .map(
      (item, i) =>
        `<button type="button" data-category="${i}" style="--share:${item.percent}%;--swatch:${swatches[i]}" aria-label="${escape(t(item.label, sceneLabels[i]))}: ${item.percent}%"></button>`,
    )
    .join("");
  $("#distribution-legend").innerHTML = categories
    .map(
      (item, i) =>
        `<button type="button" data-category="${i}" style="--swatch:${swatches[i]}"><i></i>${escape(t(item.label, sceneLabels[i]))}</button>`,
    )
    .join("");
}
for (const container of [$("#distribution-bar"), $("#distribution-legend")]) {
  const show = (event) => {
    const button = event.target.closest("[data-category]");
    if (!button) return;
    const i = Number(button.dataset.category),
      item = paper.datasets.navanywhere.scene_categories[i];
    $("#distribution-value").textContent =
      `${t(item.label, sceneLabels[i])} · ${item.percent}%`;
  };
  container.addEventListener("pointerover", show);
  container.addEventListener("focusin", show);
  container.addEventListener("click", show);
}
$("#copy-citation").addEventListener("click", async () => {
  const citation = $("#citation").textContent;
  try {
    if (navigator.clipboard && window.isSecureContext)
      await navigator.clipboard.writeText(citation);
    else {
      const textarea = document.createElement("textarea");
      textarea.value = citation;
      textarea.style.cssText = "position:fixed;left:-9999px";
      document.body.append(textarea);
      textarea.select();
      const copied = document.execCommand("copy");
      textarea.remove();
      if (!copied) throw new Error("Copy unavailable");
    }
    $("#citation-status").textContent = t("BibTeX copied.", "BibTeX 已复制。");
  } catch {
    $("#citation-status").textContent = t(
      "Select the citation text to copy it.",
      "请选择引用文本并复制。",
    );
  }
});

async function init() {
  try {
    [datasets, demos, paper] = await Promise.all(
      ["datasets", "demos", "paper"].map(async (name) => {
        const response = await fetch(`content/${name}.json?v=main-figure-v8-20261010`);
        if (!response.ok) throw new Error(`Could not load ${name}`);
        return response.json();
      }),
    );
    research = initResearch(paper, () => language);
    motionGallery = initMotionGallery(demos, () => language);
    navigation = await initNavigation(paper, () => language);
    $("#citation").textContent = paper.citation.bibtex;
    translate();
    updateHero();
  } catch (error) {
    const message = t(
      "Interactive content could not load. Serve this folder with a local HTTP server and reload.",
      "交互内容加载失败。请通过本地 HTTP 服务打开网页并刷新。",
    );
    $("#page-status").textContent = message;
    console.error(error);
  }
}
init();
initCinematic(() => language).then((instance) => { cinematic = instance; });
initPlayground(() => language).then((instance) => { playground = instance; });
