const escapeHTML = (value) => String(value).replace(/[&<>"']/g, (character) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
})[character]);

export function initResearch(paper) {
  const method = document.getElementById("method-overview");
  const results = document.getElementById("paper-results");
  const predictionMetrics = [
    { key: "id_lpips", label: "In-domain · LPIPS ↓", direction: "lower", maximum: 0.5, ticks: [0, 0.1, 0.2, 0.3, 0.4, 0.5] },
    { key: "ood_mean_psnr", label: "Out-of-domain · Mean PSNR (dB) ↑", direction: "higher", maximum: 14, ticks: [0, 3.5, 7, 10.5, 14] },
  ];
  const oodPredictionKeys = ["go_stanford_psnr", "tum_psnr", "planetary_rover_psnr", "office_go2_psnr"];
  const navigationMetrics = [
    { key: "recon_ate", label: "RECON · ATE (m) ↓", direction: "lower", maximum: 3.5, ticks: [0, 1, 2, 3, 3.5] },
    { key: "office_go2_ate", label: "Office-Go2 · ATE (m) ↓", direction: "lower", maximum: 3.5, ticks: [0, 1, 2, 3, 3.5] },
  ];

  function renderChart(rows, metric, decimals, table) {
    const rankings = [...new Set(rows.map((row) => row[metric.key]).filter(Number.isFinite))]
      .sort((a, b) => metric.direction === "higher" ? b - a : a - b);
    const chartId = `metric-${metric.key}`;
    return `<figure class="result-chart" data-table="${table}" data-metric="${metric.key}" aria-labelledby="${chartId}">
      <figcaption id="${chartId}">${escapeHTML(metric.label)}</figcaption>
      <div class="result-chart-scroll" tabindex="0" role="region" aria-labelledby="${chartId}">
      <div class="result-plot" style="--method-count:${rows.length}">
      <div class="result-y-axis" aria-hidden="true">${metric.ticks.map((tick) => `<span style="bottom:${tick / metric.maximum * 100}%">${tick}</span>`).join("")}</div>
      <div class="result-plot-area">
      <div class="result-gridlines" aria-hidden="true">${metric.ticks.map((tick) => `<span style="bottom:${tick / metric.maximum * 100}%"></span>`).join("")}</div>
      <ul class="result-bars">${rows.map((row) => {
        const value = row[metric.key];
        const reported = Number.isFinite(value);
        const rank = reported && value === rankings[0] ? "best" : reported && value === rankings[1] ? "second" : "";
        const number = reported ? value.toFixed(decimals) : "—";
        const model = row.id === "garrido" ? "Garrido et al." : row.model;
        const modelLabel = escapeHTML(model).replace("OpenNWM", "Open<wbr>NWM").replace("CompACT", "Comp<wbr>ACT");
        const params = Number.isFinite(row.params_m) ? `${row.params_m.toLocaleString("en-US")}M` : "—";
        return `<li class="result-bar-row${row.id === "opennwm" ? " result-ours" : ""}${reported ? "" : " result-missing"}" data-model="${row.id}" data-value="${reported ? value : ""}" data-rank="${rank}" style="--bar-height:${reported ? (value / metric.maximum * 100).toFixed(3) : 0}%">
          <span class="result-track">
            ${reported ? '<span class="result-bar" aria-hidden="true"></span>' : ""}
            <span class="result-value"${reported ? "" : ' aria-label="Not reported"'}>${rank === "best" ? `<strong>${number}</strong>` : rank === "second" ? `<span class="result-second">${number}</span>` : number}</span>
          </span>
          <span class="result-model" title="${escapeHTML(row.model)} · ${params} parameters">${modelLabel}</span>
        </li>`;
      }).join("")}</ul>
      </div></div></div>
    </figure>`;
  }

  function renderMethod() {
    if (!method) return;
    const figureLink = method.querySelector(".method-figure-link");
    const figureImage = method.querySelector("img");
    const figureSource = "assets/paper/method-overview-dark-20261031.svg";
    figureLink.href = "assets/paper/method-overview-20261031.pdf";
    figureImage.src = figureSource;
    figureLink.setAttribute("aria-label",
      "Open the original OpenNWM method figure PDF to zoom in");
    figureImage.alt = "OpenNWM method: a latent action encoder and decoder infer actions from observations; a conditional diffusion transformer predicts future frames with frozen VAE encoding and decoding.";
  }

  function renderMainResults() {
    if (!results) return;
    const prediction = paper.results.direct_prediction;
    const navigation = paper.results.navigation;
    const predictionRows = prediction.rows.map((row) => ({
      ...row,
      ood_mean_psnr: oodPredictionKeys.every((key) => Number.isFinite(row[key]))
        ? oodPredictionKeys.reduce((sum, key) => sum + row[key], 0) / oodPredictionKeys.length
        : null,
    }));
    results.innerHTML = `<section class="result-family" aria-labelledby="prediction-results-title">
      <div class="result-family-heading"><h3 id="prediction-results-title">Prediction</h3><span>In-domain &amp; out-of-domain</span></div>
      <div class="result-chart-grid">${predictionMetrics.map((metric) => renderChart(predictionRows, metric, 3, 1)).join("")}</div>
      <p class="results-chart-note">In-domain: ${escapeHTML(prediction.id_datasets.join(", "))}. Out-of-domain: equal-weight mean across ${escapeHTML(prediction.ood_datasets.join(", "))}. NWM†: CDiT-XL + Ego4D.</p>
    </section>
    <section class="result-family" aria-labelledby="navigation-results-title">
      <div class="result-family-heading"><h3 id="navigation-results-title">Navigation</h3><span>Lower is better</span></div>
      <div class="result-chart-grid">${navigationMetrics.map((metric) => renderChart(navigation.rows, metric, 2, 2)).join("")}</div>
      <p class="results-chart-note">RECON: in-domain. Office-Go2: out-of-domain. ATE: absolute trajectory error. —: not reported.</p>
    </section>
    <p class="results-ranking-note"><span class="results-color-key">OpenNWM</span><span>Bold: best · Underlined: second best</span></p>`;
  }

  const render = () => { renderMethod(); renderMainResults(); };
  render();
  return { render };
}
