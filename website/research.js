const escapeHTML = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (char) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        char
      ],
  );

export function initResearch(paper, getLanguage) {
  const element = (id) => document.getElementById(id);
  const method = element("method-overview");
  const domainSelect = element("result-domain");
  const metricTabs = element("metric-tabs");
  const chart = element("result-chart");
  const rows = paper.results.direct_prediction.rows;
  const state = { domain: "id", idMetric: "lpips" };
  const t = (en, zh) => (getLanguage().startsWith("zh") ? zh : en);
  function renderMethod() {
    const label = t("Open the original method figure at full resolution", "查看完整尺寸的方法原图");
    const alt = t(
      "OpenNWM architecture from the paper: a latent action model learns motion from video, and a diffusion world model predicts future observations conditioned on actions and visual context.",
      "论文中的 OpenNWM 架构原图：潜在动作模型从视频学习运动，扩散世界模型根据动作与视觉上下文预测未来观测。",
    );
    method.innerHTML = `<figure class="method-paper-figure"><a class="method-figure-link" href="assets/paper/figure-2-framework.png" target="_blank" rel="noopener" aria-label="${escapeHTML(label)}"><img src="assets/paper/figure-2-framework.png" width="2352" height="720" alt="${escapeHTML(alt)}" loading="lazy" decoding="async"></a></figure>`;
  }

  function domains() {
    return [
      ["id", t("In-domain average", "域内平均")],
      ["go_stanford", t("Go Stanford · OOD", "Go Stanford · 域外")],
      ["tum", t("TUM RGB-D · OOD", "TUM RGB-D · 域外")],
      ["planetary_rover", t("Planetary Rover · OOD", "行星探测车 · 域外")],
      ["office_go2", t("Office-Go2 · OOD", "Office-Go2 · 域外")],
    ];
  }

  function renderResults() {
    const isID = state.domain === "id";
    const metric = isID ? state.idMetric : "psnr";
    const metricName = { lpips: "LPIPS", dreamsim: "DreamSim", psnr: "PSNR" }[
      metric
    ];
    const key = `${state.domain}_${metric}`;
    const higher = metric === "psnr";
    const label = domains().find(([id]) => id === state.domain)[1];
    const values = rows.map((row) => row[key]);
    const maximum = Math.max(...values);
    const axisMaximum = higher
      ? Math.ceil(maximum / 2) * 2
      : Math.ceil(maximum * 10) / 10;
    const best = higher ? maximum : Math.min(...values);
    domainSelect.innerHTML = domains()
      .map(
        ([id, name]) =>
          `<option value="${id}"${id === state.domain ? " selected" : ""}>${escapeHTML(name)}</option>`,
      )
      .join("");
    domainSelect.setAttribute(
      "aria-label",
      t("Evaluation dataset", "评测数据集"),
    );
    metricTabs.setAttribute("aria-label", t("Prediction metric", "预测指标"));
    metricTabs.querySelectorAll("[data-metric]").forEach((button) => {
      const active = button.dataset.metric === metric;
      button.hidden = !isID && button.dataset.metric !== "psnr";
      button.disabled = button.hidden;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    chart.setAttribute("role", "list");
    chart.setAttribute(
      "aria-label",
      `${label}: ${metricName}, ${t(higher ? "higher is better" : "lower is better", higher ? "越高越好" : "越低越好")}`,
    );
    chart.innerHTML = rows
      .map((row) => {
        const value = row[key];
        const model = row.id === "nwm-xl-ego4d" ? "NWM†" : row.model;
        const number = value.toFixed(3);
        const accessible = `${model}: ${metricName} ${number}${higher ? " dB" : ""}${value === best ? t(", best reported", "，已报告最优") : ""}`;
        return `<div class="chart-row${row.id === "opennwm" ? " ours" : ""}" role="listitem" aria-label="${escapeHTML(accessible)}"><span class="chart-model-label" aria-hidden="true">${escapeHTML(model)}</span><div class="chart-track" aria-hidden="true"><div class="chart-bar" style="--value:${(value / axisMaximum) * 100}%"></div></div><span class="chart-number" aria-hidden="true">${number}</span></div>`;
      })
      .join("");
    element("chart-footnote").textContent = "NWM†: CDiT-XL + Ego4D";
    element("chart-direction").textContent = higher
      ? t("HIGHER IS BETTER ↑", "越高越好 ↑")
      : t("LOWER IS BETTER ↓", "越低越好 ↓");
  }

  domainSelect.addEventListener("change", () => {
    state.domain = domainSelect.value;
    renderResults();
  });
  metricTabs.addEventListener("click", (event) => {
    const button = event.target.closest("[data-metric]");
    if (!button || button.disabled || state.domain !== "id") return;
    state.idMetric = button.dataset.metric;
    renderResults();
  });
  const render = () => {
    renderMethod();
    renderResults();
  };
  render();
  return { render };
}
