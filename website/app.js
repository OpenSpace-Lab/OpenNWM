import { registerMedia } from "./media-scheduler.js?v=texture-master-20261030-v6-inline1";
import { setMediaUrl } from "./media-delivery.js?v=texture-master-20261030-v6-inline1";
import { initResearch } from "./research.js?v=method-dark-20261031";
import { initMotionGallery } from "./motion-gallery.js?v=texture-master-20261030-v6-inline1";
import { initNavigation } from "./navigation.js?v=texture-master-20261030-v6-inline1";
import { initCinematic } from "./cinematic.js?v=texture-master-20261030-v6-inline1";
import { initPlayground } from "./playground.js?v=texture-master-20261030-v6-inline1";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
$$('img[data-local-src]').forEach((image) => setMediaUrl(image, "src", image.dataset.localSrc));
const motion = matchMedia("(prefers-reduced-motion: reduce)");
const escape = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (char) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        char
      ],
  );
let datasets;

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
          link.classList.toggle("active", link.hash === `#${["in-motion", "navigation"].includes(entry.target.id) ? "simulation" : entry.target.id}`),
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

// Every source is shown directly; the small tag identifies its training stage.
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
// One page-wide scheduler shares bandwidth and decoder capacity.
let datasetMedia = [];
function renderDatasets() {
  datasetMedia.forEach((entry) => entry.unregister());
  datasetMedia = [];
  $("#dataset-grid").innerHTML = orderedDatasets()
    .map(
      (item) =>
        `<article class="dataset-card" data-dataset="${escape(item.id)}"><div class="dataset-card-image"><video data-src="${escape(item.video || "")}" data-poster="${escape(item.image)}" muted loop playsinline preload="none" width="480" height="270" aria-label="${escape(item.name + " dataset video")}"></video></div><div class="dataset-card-label"><h3>${escape(item.name)}</h3><span class="source-group">${item.group === "pretraining" ? "PRETRAIN" : "POST-TRAIN"}</span></div></article>`,
    )
    .join("");
  datasetMedia = $$("#dataset-grid video").map((video) =>
    registerMedia({ element: video, videos: [video], cost: 1 }),
  );
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
    $("#citation-status").textContent = "BibTeX copied.";
  } catch {
    $("#citation-status").textContent = "Select the citation text to copy it.";
  }
});

async function init() {
  try {
    const content = await Promise.all(
      ["datasets", "demos", "paper"].map(async (name) => {
        const revision = name === "paper" ? "paper-tables12-v1-20261029"
          : name === "demos" ? "texture-master-20261030-v6-inline1" : "dataset-timing-v2-20261029";
        const response = await fetch(`content/${name}.json?v=${revision}`);
        if (!response.ok) throw new Error(`Could not load ${name}`);
        return response.json();
      }),
    );
    const [datasetContent, demos, paper] = content;
    datasets = datasetContent;
    initResearch(paper);
    initMotionGallery(demos);
    await initNavigation(paper);
    $("#citation").textContent = paper.citation.bibtex;
    renderDatasets();
  } catch (error) {
    const message = "Interactive content could not load. Serve this folder with a local HTTP server and reload.";
    $("#page-status").textContent = message;
    console.error(error);
  }
}
init();
initCinematic();
initPlayground();
