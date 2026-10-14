const $ = (selector) => document.querySelector(selector);
const escape = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const defaultPaths = {
  forward: [[0, 0], [0, .3], [0, .7], [0, 1.15]],
  left: [[0, 0], [0, .25], [-.12, .55], [-.4, .85], [-.65, 1.05]],
  right: [[0, 0], [0, .25], [.12, .55], [.4, .85], [.65, 1.05]],
};

export async function initPlayground(getLanguage) {
  const t = (en, zh) => getLanguage() === "zh" ? zh || en : en;
  const canvas = $("#trajectory-canvas"), context = canvas.getContext("2d");
  const video = $("#playground-video");
  const motion = matchMedia("(prefers-reduced-motion: reduce)");
  let data, scene, preset = "forward", points = defaultPaths.forward.map((point) => [...point]);
  let live = false, busy = false, drawing = false, activePointer = null, version = 0, pollTimer, controller;
  let healthVersion = 0, serviceChecking = false;
  let mode = "initial", message = "ready", progress = 0, serviceChecked = false, errorDetail = "";
  const toCanvas = ([x, y]) => [270 + x * 222, 385 - y * 220];
  function draw() {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    if (canvas.width !== 540 * ratio) { canvas.width = 540 * ratio; canvas.height = 470 * ratio; }
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, 540, 470);
    context.strokeStyle = "#282e31"; context.lineWidth = 1;
    for (let x = 6; x < 540; x += 44) { context.beginPath(); context.moveTo(x, 0); context.lineTo(x, 470); context.stroke(); }
    for (let y = 33; y < 470; y += 44) { context.beginPath(); context.moveTo(0, y); context.lineTo(540, y); context.stroke(); }
    context.strokeStyle = "#454e52"; context.setLineDash([3, 6]);
    context.beginPath(); context.moveTo(270, 48); context.lineTo(270, 443); context.stroke(); context.setLineDash([]);
    if (points.length > 1) {
      context.beginPath(); points.forEach((p, i) => { const [x, y] = toCanvas(p); i ? context.lineTo(x, y) : context.moveTo(x, y); });
      context.lineWidth = 6; context.lineCap = "round"; context.lineJoin = "round"; context.strokeStyle = "#b6ee65"; context.stroke();
      const end = toCanvas(points.at(-1));
      context.fillStyle = "#161a1c"; context.beginPath(); context.arc(...end, 8, 0, Math.PI * 2); context.fill();
      context.lineWidth = 3; context.strokeStyle = "#b6ee65"; context.stroke();
    }
    context.fillStyle = "#b6ee65"; context.beginPath(); context.arc(270, 385, 19, 0, Math.PI * 2); context.fill();
    context.fillStyle = "#0d0e10"; context.beginPath(); context.moveTo(270, 374); context.lineTo(278, 392); context.lineTo(270, 388); context.lineTo(262, 392); context.closePath(); context.fill();
  }
  const selectedPreset = () => scene?.presets?.find((item) => item.id === preset);
  const pathLength = () => points.slice(1).reduce((sum, point, i) => sum + Math.hypot(point[0] - points[i][0], point[1] - points[i][1]), 0);
  const validPath = () => points.length >= 2 && pathLength() >= .05 && pathLength() <= 3;
  function resetOutput() {
    clearTimeout(pollTimer); controller?.abort(); version++;
    if (activePointer !== null && canvas.hasPointerCapture(activePointer)) canvas.releasePointerCapture(activePointer);
    drawing = false; activePointer = null;
    busy = false; mode = "initial"; progress = 0;
    video.pause(); video.removeAttribute("src"); video.load();
    video.hidden = true; $("#playground-initial").hidden = false;
    $("#prediction-placeholder").hidden = false; $("#generation-overlay").hidden = true;
  }
  function renderStatus() {
    $("#service-status").textContent = serviceChecking || !serviceChecked ? t("Checking live service…", "正在连接实时服务…") : live ? t("Live generation", "实时生成") : t("Preset videos", "预设视频");
    $("#service-status").classList.toggle("connected", live);
    $("#service-retry").hidden = live;
    $("#service-retry").disabled = serviceChecking || busy;
    if (!scene) {
      $("#generate-path").disabled = true;
      $("#play-preset").disabled = true;
      return;
    }
    $("#prediction-mode").textContent = mode === "live" ? t("LIVE MODEL OUTPUT", "实时模型预测") : mode === "preset" ? t("PRECOMPUTED", "预计算结果") : t("STARTING VIEW", "初始视野");
    $("#generate-path").disabled = !live || busy || drawing || !validPath();
    $("#play-preset").disabled = busy || !selectedPreset()?.video;
    $("#path-clear").disabled = busy;
    $("#playground-service-note").textContent = live ? t("Custom generation takes a few minutes.", "自定义生成需要几分钟。") : t("Connect the live service to generate a custom path.", "连接实时服务后可生成自定义路线。");
    const messages = {
      ready: t("Choose a preset or draw your own route.", "选择预设，或绘制你自己的路线。"),
      custom: live
        ? t("Custom path ready. Generate it with the live model.", "自定义路线已就绪，可调用实时模型生成。")
        : t("Custom path ready.", "自定义路线已就绪。"),
      cleared: t("Draw from the starting point, or choose a preset below.", "从起点开始绘制，或选择下方预设。"),
      preset: t("Playing preset prediction.", "正在播放预设预测。"),
      queued: t("Waiting to generate…", "正在等待生成…"),
      running: t("OpenNWM is imagining your path…", "OpenNWM 正在预测这条路线的未来视野…"),
      completed: t("Generated from your path with OpenNWM.", "OpenNWM 已根据你绘制的路线生成结果。"),
      failed: t("Generation could not finish. Please try again.", "生成未能完成，请重试。"),
      unavailable: t("Live generation is unavailable. Choose a preset to explore recorded results.", "实时生成服务不可用。可选择预设探索已有预测。"),
      short: t("Draw a longer route from the starting point.", "请从起点画出一条稍长的路线。"),
      long: t("This route is too long. Clear it and draw a shorter path.", "这条路线过长，请清除后绘制较短的路线。"),
    };
    $("#playground-status").textContent = (messages[message] || messages.ready) + (message === "failed" && errorDetail ? ` ${errorDetail}` : "");
    $("#generation-title").textContent = message === "queued" ? t("Waiting for the GPU…", "等待 GPU…") : t("Imagining your path…", "正在想象前方…");
    $("#generation-progress").value = progress;
    document.querySelectorAll("[data-path]").forEach((button) => { button.classList.toggle("active", button.dataset.path === preset); button.setAttribute("aria-pressed", String(button.dataset.path === preset)); button.disabled = busy; });
  }
  function render() {
    if (!scene) return;
    $("#playground-scenes").innerHTML = data.scenes.map((item) => `<button type="button" data-scene="${escape(item.id)}" class="${item.id === scene.id ? "active" : ""}" aria-pressed="${item.id === scene.id}">${escape(t(item.title, item.titleZh))}</button>`).join("");
    $("#playground-scene-name").textContent = t(scene.title, scene.titleZh);
    canvas.setAttribute("aria-label", t("Draw a route on a top-down plane, or choose a preset route below", "在俯视平面上绘制路线，或选择下方预设路线"));
    renderStatus(); draw();
  }
  function selectPath(id) {
    if (busy) return;
    resetOutput(); preset = id;
    points = (selectedPreset()?.points || defaultPaths[id]).map((point) => [...point]);
    message = "ready"; draw(); renderStatus();
  }
  function selectScene(id) {
    resetOutput(); scene = data.scenes.find((item) => item.id === id) || data.scenes[0];
    $("#playground-initial").src = scene.poster || scene.image;
    $("#playground-initial").alt = t(`${scene.title} — initial observation`, `${scene.titleZh || scene.title} — 初始观测`);
    selectPath("forward"); render();
  }
  $("#playground-scenes").addEventListener("click", (event) => {
    const button = event.target.closest("[data-scene]");
    if (button) { selectScene(button.dataset.scene); $("#playground-scenes").querySelector(`[data-scene="${CSS.escape(scene.id)}"]`)?.focus({ preventScroll: true }); }
  });
  document.querySelectorAll("[data-path]").forEach((button) => button.addEventListener("click", () => selectPath(button.dataset.path)));
  $("#path-clear").addEventListener("click", () => { resetOutput(); points = [[0, 0]]; preset = null; message = "cleared"; renderStatus(); draw(); });
  function pointerPoint(event) {
    const bounds = canvas.getBoundingClientRect();
    const x = (event.clientX - bounds.left) / bounds.width * 540;
    const y = (event.clientY - bounds.top) / bounds.height * 470;
    return [Math.max(-1, Math.min(1, (x - 270) / 222)), Math.max(-.25, Math.min(1.5, (385 - y) / 220))];
  }
  canvas.addEventListener("pointerdown", (event) => {
    if (busy || drawing || !scene || !event.isPrimary || (event.pointerType === "mouse" && event.button !== 0)) return;
    event.preventDefault(); resetOutput(); drawing = true; preset = null; points = [[0, 0]];
    activePointer = event.pointerId; canvas.setPointerCapture(event.pointerId); points.push(pointerPoint(event)); message = "custom"; draw(); renderStatus();
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!drawing || event.pointerId !== activePointer) return;
    const point = pointerPoint(event), last = points.at(-1);
    if (Math.hypot(point[0] - last[0], point[1] - last[1]) > .015 && points.length < 240) points.push(point);
    draw();
  });
  const stopDrawing = (event) => {
    if (event.pointerId !== activePointer) return;
    drawing = false; activePointer = null;
    message = pathLength() < .05 ? "short" : pathLength() > 3 ? "long" : "custom";
    renderStatus();
  };
  canvas.addEventListener("pointerup", stopDrawing); canvas.addEventListener("pointercancel", stopDrawing);
  canvas.addEventListener("lostpointercapture", stopDrawing);
  window.addEventListener("resize", draw);
  function showVideo(src, poster, outputMode) {
    mode = outputMode; video.src = src; video.poster = poster || scene.poster || scene.image;
    video.hidden = false; $("#playground-initial").hidden = true;
    $("#prediction-placeholder").hidden = true; $("#generation-overlay").hidden = true;
    video.load(); if (!motion.matches) video.play().catch(() => {}); renderStatus();
  }
  $("#play-preset").addEventListener("click", () => {
    const item = selectedPreset(); if (!item?.video || busy) return;
    resetOutput(); message = "preset"; showVideo(item.video, item.poster, "preset");
  });
  async function health() {
    const checkVersion = ++healthVersion;
    serviceChecking = true; renderStatus();
    try {
      const response = await fetch("api/health", { signal: AbortSignal.timeout(4500), cache: "no-store" });
      if (!response.ok) throw new Error("Service unavailable");
      const result = await response.json();
      if (checkVersion !== healthVersion) return;
      live = result.status === "ok" && result.live !== false;
    } catch { if (checkVersion === healthVersion) live = false; }
    if (checkVersion !== healthVersion) return;
    serviceChecked = true; serviceChecking = false; renderStatus();
  }
  $("#service-retry").addEventListener("click", health);
  async function poll(jobId, currentVersion) {
    if (version !== currentVersion) return;
    try {
      const response = await fetch(`api/jobs/${encodeURIComponent(jobId)}`, { signal: AbortSignal.any([controller.signal, AbortSignal.timeout(15000)]), cache: "no-store" });
      if (!response.ok) throw new Error(t("Could not retrieve the generation status.", "无法获取生成状态。"));
      const job = await response.json();
      if (version !== currentVersion) return;
      progress = Math.max(0, Math.min(1, Number(job.progress) || 0));
      if (job.status === "completed") {
        if (!job.video) throw new Error(t("The service returned no video.", "服务未返回视频。"));
        busy = false; message = "completed"; showVideo(job.video, job.poster, "live"); return;
      }
      if (job.status === "failed") throw new Error(job.error || t("The model reported an error.", "模型运行失败。"));
      message = job.status === "queued" ? "queued" : "running"; renderStatus();
      pollTimer = setTimeout(() => poll(jobId, currentVersion), 1800);
    } catch (error) {
      if (version !== currentVersion || error.name === "AbortError") return;
      busy = false; message = "failed"; errorDetail = String(error.message).slice(0, 200);
      $("#generation-overlay").hidden = true; renderStatus();
    }
  }
  $("#generate-path").addEventListener("click", async () => {
    if (!live || busy || drawing || !validPath()) return;
    resetOutput(); const currentVersion = version;
    busy = true; message = "queued"; errorDetail = ""; controller = new AbortController();
    $("#generation-overlay").hidden = false; $("#prediction-placeholder").hidden = true; renderStatus();
    try {
      const response = await fetch("api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sceneId: scene.id, points, seed: 0 }), signal: AbortSignal.any([controller.signal, AbortSignal.timeout(20000)]) });
      const job = await response.json();
      if (version !== currentVersion) return;
      if (!response.ok || !job.jobId) throw new Error(job.error || job.detail || t("The service could not start this request.", "服务无法启动该请求。"));
      poll(job.jobId, currentVersion);
    } catch (error) {
      if (version !== currentVersion || error.name === "AbortError") return;
      busy = false; message = "failed"; errorDetail = String(error.message).slice(0, 200);
      $("#generation-overlay").hidden = true; renderStatus(); health();
    }
  });
  video.addEventListener("error", () => {
    if (!video.getAttribute("src")) return;
    message = "failed"; errorDetail = t("The prediction video could not load.", "预测视频加载失败。"); renderStatus();
  });
  new IntersectionObserver(([entry]) => { if (!entry.isIntersecting) video.pause(); }, { threshold: .01 }).observe(video);
  document.addEventListener("visibilitychange", () => { if (document.hidden) video.pause(); else if (!busy) health(); });
  draw();
  try {
    const response = await fetch("content/playground.json?v=earth-space-v3-20260930", { cache: "no-cache" });
    if (!response.ok) throw new Error("Scene manifest unavailable");
    data = await response.json();
    if (!data.scenes?.length) throw new Error("No scenes available");
    selectScene(data.defaultScene || data.scenes[0].id);
  } catch {
    $("#playground-status").textContent = t("Demo scenes could not load. Please reload the page.", "演示场景加载失败，请刷新页面。");
    $("#play-preset").disabled = true;
  }
  await health();
  return { render };
}
