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
  const steps = element("method-steps");
  const panel = element("method-panel");
  const domainSelect = element("result-domain");
  const metricTabs = element("metric-tabs");
  const chart = element("result-chart");
  const rows = paper.results.direct_prediction.rows;
  const state = { stage: 0, domain: "id", idMetric: "lpips" };
  const t = (en, zh) => (getLanguage().startsWith("zh") ? zh : en);
  const cells = () =>
    `<div class="latent-cells" aria-hidden="true">${Array.from({ length: 32 }, (_, i) => `<i style="--opacity:${0.45 + (i % 6) * 0.1};--delay:${(i % 8) * -0.23}s"></i>`).join("")}</div>`;
  const arrow = '<div class="diagram-arrow" aria-hidden="true">→</div>';
  const thumbnail = (label) =>
    `<div class="diagram-input"><img src="assets/datasets/i2nav-robot.webp" alt="" loading="lazy"><span>${escapeHTML(label)}</span></div>`;

  function stages() {
    return [
      {
        title: t("Learn latent actions", "学习潜在动作"),
        short: t(
          "Visual prediction + physical grounding",
          "视觉预测 + 物理运动约束",
        ),
        kicker: t("LAM · 32-D LATENT SPACE", "LAM · 32 维潜在空间"),
        heading: t("A shared language for motion.", "为运动学习共同的表达。"),
        text: t(
          "A frame pair is encoded into a 32-dimensional variational latent action. Visual reconstruction learns from video; an auxiliary physical predictor grounds the latent on RECON, HuRoN, and SCAND. The 60K-step checkpoint supplies deterministic posterior-mean latents.",
          "将一对图像编码为 32 维变分潜在动作。视觉重建从视频中学习，辅助运动预测器通过 RECON、HuRoN 和 SCAND 的标注建立物理约束。使用 60K 步检查点的后验均值生成确定性潜在动作。",
        ),
        equation: "L = L_RGB + λ_action L_action + β L_KL",
        diagram: `${thumbnail(t("Video frames", "视频帧"))}${arrow}<div class="diagram-model"><span class="eyebrow">LAM</span>${cells()}<strong>${t("Encode motion", "编码运动")}</strong><small>${t("Pixel + Action", "像素 + 动作监督")}</small></div>${arrow}<div class="diagram-output"><div class="action-vector">z ∈ ℝ³²</div><span>${t("Latent action", "潜在动作")}</span></div>`,
      },
      {
        title: t("Pretrain on video", "从视频预训练"),
        short: t(
          "15 sources · 60K world-model steps",
          "15 个来源 · 60K 步世界模型训练",
        ),
        kicker: t("NWM · LATENT PRETRAINING", "NWM · 潜在动作预训练"),
        heading: t(
          "Turn diverse video into dynamics.",
          "从多样视频中学习动态。",
        ),
        text: t(
          "The frozen LAM supplies latent actions for NavAnywhere. A CDiT-B/2 diffusion backbone learns future observations from four context frames, relative time, and local latent actions. Latent conditioning is used for valid offsets within ±8 frames; target offsets extend to ±64.",
          "冻结的 LAM 为 NavAnywhere 提供潜在动作。CDiT-B/2 扩散主干根据四帧上下文、相对时间与局部潜在动作学习未来观测。潜在动作条件用于 ±8 帧内的有效偏移，预测目标偏移可覆盖 ±64 帧。",
        ),
        equation: t(
          "60K steps · 4 context frames · 4 targets / observation",
          "60K 步 · 4 帧上下文 · 每个观测 4 个预测目标",
        ),
        diagram: `<div class="diagram-input"><div class="action-vector">z + Δt</div><span>${t("Latent + time", "潜在动作 + 时间")}</span></div>${arrow}<div class="diagram-model"><span class="eyebrow">CDiT-B/2</span>${cells()}<strong>${t("Learn dynamics", "学习动态")}</strong><small>${t("+ visual context", "+ 视觉上下文")}</small></div>${arrow}<div class="diagram-output"><img src="assets/datasets/i2nav-robot.webp" alt="" loading="lazy"><span>${t("RGB target", "RGB 目标帧")}</span></div>`,
      },
      {
        title: t("Align physical actions", "对齐物理动作"),
        short: t(
          "3K warmup → 100K joint updates",
          "3K 步预热 → 100K 步联合训练",
        ),
        kicker: t("NWM · PHYSICAL POST-TRAINING", "NWM · 物理动作后训练"),
        heading: t(
          "Connect imagination to action.",
          "连接未来预测与实际行动。",
        ),
        text: t(
          "A new physical-action encoder warms up for 3K steps with the CDiT backbone frozen. Both then train jointly for 100K steps on RECON, HuRoN, SCAND, and TartanDrive. At inference, CEM searches physical waypoint actions (Δx, Δy, Δψ) and replans after the first action.",
          "新物理动作编码器先预热 3K 步，此时冻结 CDiT 主干；随后二者在 RECON、HuRoN、SCAND 与 TartanDrive 上联合训练 100K 步。推理时，CEM 搜索物理路点动作 (Δx, Δy, Δψ)，执行第一个动作后重新规划。",
        ),
        equation: t(
          "3K: encoder only → 100K: encoder + backbone",
          "3K：仅编码器 → 100K：编码器 + 主干",
        ),
        diagram: `<div class="diagram-input"><div class="action-vector">Δx\nΔy\nΔψ</div><span>${t("Physical action", "物理动作")}</span></div>${arrow}<div class="diagram-model"><span class="eyebrow">OpenNWM</span>${cells()}<strong>${t("Align & adapt", "对齐与适配")}</strong><small>${t("Encoder + CDiT", "编码器 + CDiT")}</small></div>${arrow}<div class="diagram-output"><img src="assets/datasets/recon.webp" alt="" loading="lazy"><span>${t("RGB target", "RGB 目标帧")}</span></div>`,
      },
    ];
  }

  function renderMethod() {
    const descriptions = stages();
    const selected = descriptions[state.stage];
    steps.setAttribute("aria-label", t("Method stages", "方法阶段"));
    steps.innerHTML = descriptions
      .map(
        (stage, index) =>
          `<button type="button" class="method-step" role="tab" id="method-tab-${index}" aria-controls="method-panel" aria-selected="${index === state.stage}" tabindex="${index === state.stage ? 0 : -1}" data-stage="${index}"><span class="method-step-number" aria-hidden="true">0${index + 1}</span><span><h3>${escapeHTML(stage.title)}</h3><p>${escapeHTML(stage.short)}</p></span></button>`,
      )
      .join("");
    panel.setAttribute("aria-labelledby", `method-tab-${state.stage}`);
    panel.tabIndex = 0;
    const diagram = element("method-diagram");
    diagram.innerHTML = selected.diagram;
    diagram.setAttribute("role", "img");
    diagram.setAttribute(
      "aria-label",
      t(
        "Training schematic; dataset images illustrate the input and target roles. ",
        "训练示意图；数据集图像仅示意输入与目标。",
      ) + selected.heading,
    );
    element("method-kicker").textContent = selected.kicker;
    element("method-subtitle").textContent = selected.heading;
    element("method-description").textContent = selected.text;
    element("method-equation").textContent = selected.equation;
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
        const accessible = `${model}, ${row.parameters_millions}M ${t("parameters", "参数")}: ${metricName} ${number}${higher ? " dB" : ""}${value === best ? t(", best reported", "，本表最优") : ""}`;
        return `<div class="chart-row${row.id === "opennwm" ? " ours" : ""}" role="listitem" aria-label="${escapeHTML(accessible)}"><span class="chart-model-label" aria-hidden="true">${escapeHTML(model)}<small>${row.parameters_millions.toLocaleString("en-US")}M ${t("params", "参数")}</small></span><div class="chart-track" aria-hidden="true"><div class="chart-bar" style="--value:${(value / axisMaximum) * 100}%"></div></div><span class="chart-number" aria-hidden="true">${number}</span></div>`;
      })
      .join("");
    const split = isID
      ? t(
          "ID average: RECON, SCAND, HuRoN, TartanDrive.",
          "域内平均：RECON、SCAND、HuRoN、TartanDrive。",
        )
      : `${label}.`;
    element("chart-footnote").textContent =
      `${split} ${t("Direct prediction at 4 s · Table 1. NWM†: CDiT-XL + Ego4D. Bars start at zero.", "4 秒直接预测 · 表 1。NWM†：CDiT-XL + Ego4D。条形从零起算。")}`;
    element("chart-direction").textContent = higher
      ? t("HIGHER IS BETTER ↑", "越高越好 ↑")
      : t("LOWER IS BETTER ↓", "越低越好 ↓");
  }

  steps.addEventListener("click", (event) => {
    const button = event.target.closest("[data-stage]");
    if (!button) return;
    state.stage = Number(button.dataset.stage);
    renderMethod();
    element(`method-tab-${state.stage}`).focus({ preventScroll: true });
  });
  steps.addEventListener("keydown", (event) => {
    if (!event.target.closest("[data-stage]")) return;
    const offsets = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 };
    if (event.key in offsets)
      state.stage = (state.stage + offsets[event.key] + 3) % 3;
    else if (event.key === "Home") state.stage = 0;
    else if (event.key === "End") state.stage = 2;
    else return;
    event.preventDefault();
    renderMethod();
    element(`method-tab-${state.stage}`).focus({ preventScroll: true });
  });
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
