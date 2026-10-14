const escapeHTML = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (character) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        character
      ],
  );

function trajectoryGraphic(sample, t) {
  const all = [...sample.reference_xy_m, ...sample.planned_xy_m];
  const xs = all.map(([x]) => x);
  const ys = all.map(([, y]) => y);
  const minX = Math.min(...xs),
    maxX = Math.max(...xs);
  const minY = Math.min(...ys),
    maxY = Math.max(...ys);
  const scale = Math.min(
    320 / Math.max(maxX - minX, 0.1),
    124 / Math.max(maxY - minY, 0.1),
  );
  const left = (360 - (maxX - minX) * scale) / 2;
  const top = (152 - (maxY - minY) * scale) / 2;
  const point = ([x, y]) => [
    left + (x - minX) * scale,
    top + (maxY - y) * scale,
  ];
  const points = (path) =>
    path
      .map((p) =>
        point(p)
          .map((n) => n.toFixed(2))
          .join(","),
      )
      .join(" ");
  const start = point(sample.reference_xy_m[0]);
  const goal = point(sample.reference_xy_m.at(-1));
  const length = maxX - minX;
  const barMeters = 10 ** Math.floor(Math.log10(Math.max(length / 3, 0.01)));
  return `<svg class="navigation-path" viewBox="0 0 360 174" role="img" data-points="${escapeHTML(JSON.stringify(sample.planned_xy_m.map(point)))}" aria-label="${escapeHTML(t("OpenNWM planned path and recorded reference, equal scale in meters", "OpenNWM 规划轨迹与记录参考轨迹，等比例米制坐标"))}">
    <polyline class="navigation-reference" points="${points(sample.reference_xy_m)}"/>
    <polyline class="navigation-planned" points="${points(sample.planned_xy_m)}"/>
    <polyline class="navigation-progress" points="${points(sample.planned_xy_m.slice(0, 1))}"/>
    <circle class="navigation-start" cx="${start[0]}" cy="${start[1]}" r="3.5"/>
    <circle class="navigation-goal" cx="${goal[0]}" cy="${goal[1]}" r="4.5"/>
    <circle class="navigation-cursor" cx="${start[0]}" cy="${start[1]}" r="4"/>
    <path class="navigation-scale" d="M20 156v4h${(barMeters * scale).toFixed(2)}v-4"/>
    <text x="20" y="172">${barMeters} m</text>
  </svg>`;
}

export async function initNavigation(paper, getLanguage) {
  const root = document.getElementById("navigation-content");
  const benchmarks = document.getElementById("navigation-benchmarks");
  if (!root) return { render() {} };
  const t = (en, zh) => (getLanguage().startsWith("zh") ? zh : en);
  const response = await fetch(
    "content/navigation.json?v=curated-navigation-v5-20261010",
  );
  if (!response.ok) throw new Error(`Navigation data: HTTP ${response.status}`);
  const manifest = await response.json();
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  let paused = reducedMotion.matches;
  let observer;
  const visible = new Set();
  const benchmarkState = { domain: "recon", metric: "ate" };

  function updatePlayback() {
    root.querySelectorAll("video").forEach((video) => {
      if (!paused && visible.has(video) && !document.hidden)
        video.play().catch(() => {});
      else video.pause();
    });
    const button = root.querySelector(".navigation-play-toggle");
    if (button) {
      button.textContent = paused
        ? t("Play previews", "播放预览")
        : t("Pause previews", "暂停预览");
      button.setAttribute("aria-pressed", String(!paused));
    }
  }

  function benchmarkDomains() {
    return [
      ["recon", "RECON"],
      ["scand", "SCAND"],
      ["office_go2", t("Office-Go2 · OOD", "Office-Go2 · 域外")],
      ["tartan_drive", "TartanDrive"],
    ];
  }

  function renderBenchmarkChart() {
    if (!benchmarks) return;
    const { domain, metric } = benchmarkState;
    const metricName = metric.toUpperCase();
    const key = `${domain}_${metric}`;
    const rows = paper.results.navigation.rows.filter((row) =>
      Number.isFinite(row[key]),
    );
    const values = rows.map((row) => row[key]);
    const maximum = Math.max(...values);
    const axisMaximum = Math.ceil(maximum * 10) / 10;
    const best = Math.min(...values);
    const domainSelect = benchmarks.querySelector("#navigation-result-domain");
    domainSelect.innerHTML = benchmarkDomains()
      .map(
        ([id, label]) =>
          `<option value="${id}"${id === domain ? " selected" : ""}>${escapeHTML(label)}</option>`,
      )
      .join("");
    domainSelect.setAttribute(
      "aria-label",
      t("Navigation evaluation dataset", "导航评测数据集"),
    );
    benchmarks.querySelector("#navigation-chart-title").textContent = t(
      "Navigation benchmark",
      "导航基准",
    );
    const tabs = benchmarks.querySelector("#navigation-metric-tabs");
    tabs.setAttribute("aria-label", t("Navigation metric", "导航指标"));
    tabs.querySelectorAll("[data-metric]").forEach((button) => {
      const active = button.dataset.metric === metric;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
      button.setAttribute(
        "aria-label",
        button.dataset.metric === "ate"
          ? t(
              "Absolute trajectory error, lower is better",
              "绝对轨迹误差，越低越好",
            )
          : t("Relative pose error, lower is better", "相对位姿误差，越低越好"),
      );
    });
    const chart = benchmarks.querySelector("#navigation-result-chart");
    const label = benchmarkDomains().find(([id]) => id === domain)[1];
    chart.setAttribute(
      "aria-label",
      `${label}: ${metricName}, ${t("lower is better", "越低越好")}`,
    );
    chart.innerHTML = rows
      .map((row) => {
        const value = row[key];
        const number = value.toFixed(2);
        const accessible = `${row.model}: ${metricName} ${number}${value === best ? t(", best reported", "，已报告最优") : ""}`;
        return `<div class="chart-row${row.id === "opennwm" ? " ours" : ""}" role="listitem" aria-label="${escapeHTML(accessible)}"><span class="chart-model-label" aria-hidden="true">${escapeHTML(row.model)}</span><div class="chart-track" aria-hidden="true"><div class="chart-bar" style="--value:${(value / axisMaximum) * 100}%"></div></div><span class="chart-number" aria-hidden="true">${number}</span></div>`;
      })
      .join("");
    benchmarks.querySelector("#navigation-chart-direction").textContent = t(
      "LOWER IS BETTER ↓",
      "越低越好 ↓",
    );
  }

  function renderBenchmarks() {
    if (!benchmarks) return;
    benchmarks.innerHTML = `<div class="chart-panel navigation-chart-panel">
      <div class="chart-header"><span id="navigation-chart-title"></span><select id="navigation-result-domain"></select></div>
      <div class="chart-metric-tabs" id="navigation-metric-tabs" role="group"><button type="button" data-metric="ate">ATE ↓</button><button type="button" data-metric="rpe">RPE ↓</button></div>
      <div id="navigation-result-chart" class="result-chart" role="list" aria-live="polite" aria-atomic="true"></div>
      <div class="chart-footnote"><span id="navigation-chart-direction" class="mono"></span></div>
    </div>`;
    renderBenchmarkChart();
  }

  benchmarks?.addEventListener("change", (event) => {
    if (event.target.id !== "navigation-result-domain") return;
    benchmarkState.domain = event.target.value;
    renderBenchmarkChart();
  });
  benchmarks?.addEventListener("click", (event) => {
    const button = event.target.closest(
      "#navigation-metric-tabs [data-metric]",
    );
    if (!button) return;
    benchmarkState.metric = button.dataset.metric;
    renderBenchmarkChart();
  });

  function render() {
    observer?.disconnect();
    visible.clear();
    root.querySelectorAll("video").forEach((video) => video.pause());
    root.innerHTML = `<div class="navigation-gallery-heading"><span>${t("Selected offline local plans", "精选离线局部规划")}</span><div class="navigation-legend"><span><i class="navigation-legend-plan"></i>OpenNWM</span><span><i class="navigation-legend-reference"></i>${t("Reference path", "参考轨迹")}</span></div><button class="navigation-play-toggle" type="button"></button></div>
      <div class="navigation-examples">${manifest.samples
        .map(
          (
            sample,
          ) => `<article class="navigation-example" data-sample-id="${escapeHTML(sample.id)}" data-fps="${sample.fps || manifest.media.fps}">
        <header><h3>${escapeHTML(sample.title[getLanguage().startsWith("zh") ? "zh" : "en"])}</h3><span>${escapeHTML(sample.dataset)}</span></header>
        <div class="navigation-media">
          <figure><img src="${sample.start}" loading="lazy" width="224" height="224" alt="${escapeHTML(t(`${sample.title.en}: recorded start`, `${sample.title.zh}：起点观测`))}"><figcaption>${t("Start", "起点")}</figcaption></figure>
          <figure><video src="${sample.video}" poster="${sample.poster}" width="224" height="224" preload="none" muted loop playsinline aria-label="${escapeHTML(t(`${sample.title.en}: OpenNWM imagined local plan`, `${sample.title.zh}：OpenNWM 规划预测`))}"></video><figcaption>OpenNWM</figcaption></figure>
          <figure><img src="${sample.goal}" loading="lazy" width="224" height="224" alt="${escapeHTML(t(`${sample.title.en}: goal image`, `${sample.title.zh}：目标图像`))}"><figcaption>${t("Goal", "目标")}</figcaption></figure>
        </div>
        <div class="navigation-path-row">${trajectoryGraphic(sample, t)}</div>
      </article>`,
        )
        .join("")}</div>`;
    renderBenchmarks();
    root
      .querySelector(".navigation-play-toggle")
      .addEventListener("click", () => {
        paused = !paused;
        updatePlayback();
      });
    observer = new IntersectionObserver(
      (entries) => {
        entries.forEach(({ target, isIntersecting }) =>
          isIntersecting ? visible.add(target) : visible.delete(target),
        );
        updatePlayback();
      },
      { threshold: 0.1 },
    );
    root.querySelectorAll("video").forEach((video) => {
      video.muted = true;
      const card = video.closest(".navigation-example");
      const graphic = card.querySelector(".navigation-path");
      const points = JSON.parse(graphic.dataset.points);
      const cursor = graphic.querySelector(".navigation-cursor");
      const progress = graphic.querySelector(".navigation-progress");
      const updatePath = () => {
        const frame = Math.min(
          points.length - 1,
          Math.floor(video.currentTime * Number(card.dataset.fps)),
        );
        cursor.setAttribute("cx", points[frame][0]);
        cursor.setAttribute("cy", points[frame][1]);
        progress.setAttribute(
          "points",
          points
            .slice(0, frame + 1)
            .map((point) => point.join(","))
            .join(" "),
        );
        graphic.dataset.frame = String(frame);
      };
      video.addEventListener("timeupdate", updatePath);
      video.addEventListener("seeked", updatePath);
      updatePath();
      observer.observe(video);
    });
    updatePlayback();
  }
  document.addEventListener("visibilitychange", updatePlayback);
  reducedMotion.addEventListener("change", (event) => {
    if (event.matches) {
      paused = true;
      updatePlayback();
    }
  });
  render();
  return { render };
}
