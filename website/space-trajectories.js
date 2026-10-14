const escape = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

// Only held-out LuSNAR records have verified poses. Single-image planetary
// demonstrations elsewhere in space.json use synthetic controls.
export function projectSpaceTrajectory(manifest) {
  const actions = manifest.direct?.metric_actions;
  if (manifest.dataset !== "LuSNAR" || !/^Moon_[789]$/.test(manifest.scene)
      || manifest.pose_source !== `LuSNAR/${manifest.scene}/gt.txt`
      || !Array.isArray(actions) || actions.length !== 16
      || !actions.every((point) => Array.isArray(point) && point.length === 3 && point.every(Number.isFinite))
      || manifest.playback_fps !== 4
      || manifest.prediction_steps?.some((step, index) => step !== index + 1)
      || manifest.prediction_steps?.length !== actions.length) {
    throw new Error("Space trajectory requires verified recorded LuSNAR target poses");
  }
  // metric_actions are already measured in meters: forward, left, yaw.
  // Reflect into an SVG plane with forward up and left left; use ONE scale.
  const meters = [[0, 0], ...actions.map(([forward, left]) => [-left, -forward])];
  const xs = meters.map(([x]) => x), ys = meters.map(([, y]) => y);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const scale = Math.min(288 / Math.max(maxX - minX, .001), 116 / Math.max(maxY - minY, .001));
  const points = meters.map(([x, y]) => [180 + (x - (minX + maxX) / 2) * scale, 74 + (y - (minY + maxY) / 2) * scale]);
  const scaleMeters = 10 ** Math.floor(Math.log10(72 / scale));
  return { points, meters, scale, scaleMeters, fps: manifest.playback_fps, poseSource: manifest.pose_source };
}

export async function loadSpaceTrajectories(scenes) {
  const trajectories = new Map();
  await Promise.all(scenes.filter((scene) => /^lusnar-finetuned-moon-[789]$/.test(scene.id)).map(async (scene) => {
    try {
      const response = await fetch(scene.manifest);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const trajectory = projectSpaceTrajectory(await response.json());
      if (trajectory.poseSource !== `LuSNAR/Moon_${scene.id.at(-1)}/gt.txt`) throw new Error("Scene/pose mismatch");
      trajectories.set(scene.id, trajectory);
    } catch (error) {
      console.warn(`Recorded trajectory unavailable for ${scene.id}: ${error.message}`);
    }
  }));
  return trajectories;
}

export function spaceTrajectoryMarkup(scene, trajectory, t) {
  const label = t("Actual trajectory", "实际轨迹");
  if (!trajectory) return `<figure class="space-recorded-trajectory"><figcaption>${label}</figcaption><img src="${escape(scene.trajectory)}" alt="${label}" loading="lazy" width="224" height="224"></figure>`;
  const { points, scale, scaleMeters, fps, poseSource } = trajectory;
  const coords = (values) => values.map((point) => point.join(",")).join(" ");
  const [start, first] = points, end = points.at(-1);
  return `<figure class="space-recorded-trajectory"><figcaption>${label}</figcaption><svg viewBox="0 0 360 166" role="img" aria-label="${escape(t("Recorded LuSNAR rover path, equal scale in meters; forward is up", "LuSNAR 实际记录轨迹，等比例米制坐标，前方朝上"))}" data-points="${escape(JSON.stringify(points))}" data-fps="${fps}" data-pose-source="${escape(poseSource)}" data-frame="0" data-point="1">
    <path class="space-trajectory-grid" d="M36 16V136M84 16V136M132 16V136M180 16V136M228 16V136M276 16V136M324 16V136M36 16H324M36 56H324M36 96H324M36 136H324"/>
    <polyline class="space-trajectory-full" points="${coords(points)}"/>
    <polyline class="space-trajectory-progress" points="${coords(points.slice(0, 2))}"/>
    <circle class="space-trajectory-start" cx="${start[0]}" cy="${start[1]}" r="3.5"/>
    <circle class="space-trajectory-end" cx="${end[0]}" cy="${end[1]}" r="3.5"/>
    <circle class="space-trajectory-cursor" cx="${first[0]}" cy="${first[1]}" r="5"/>
    <path class="space-trajectory-scale" d="M24 147v4h${scaleMeters * scale}v-4"/>
    <text x="24" y="164">${scaleMeters} m</text>
  </svg></figure>`;
}

export function bindSpaceTrajectories(root) {
  const cleanup = [];
  root.querySelectorAll(".transfer-card").forEach((card) => {
    const video = card.querySelector("video"), svg = card.querySelector(".space-recorded-trajectory svg");
    if (!video || !svg) return;
    const points = JSON.parse(svg.dataset.points), fps = Number(svg.dataset.fps);
    const cursor = svg.querySelector(".space-trajectory-cursor"), progress = svg.querySelector(".space-trajectory-progress");
    let callback = null;
    const update = (time = video.currentTime) => {
      const frame = Math.max(0, Math.min(points.length - 2, Math.floor(time * fps + .0001)));
      // Encoded frame zero is horizon 1. The anchor is drawn but is not a frame.
      const point = frame + 1;
      cursor.setAttribute("cx", points[point][0]); cursor.setAttribute("cy", points[point][1]);
      progress.setAttribute("points", points.slice(0, point + 1).map((p) => p.join(",")).join(" "));
      svg.dataset.frame = String(frame); svg.dataset.point = String(point);
    };
    const stop = () => { if (callback !== null) video.cancelVideoFrameCallback?.(callback); callback = null; };
    const decodedFrame = (_, metadata) => { update(metadata.mediaTime); callback = video.requestVideoFrameCallback(decodedFrame); };
    const start = () => { stop(); if (video.requestVideoFrameCallback) callback = video.requestVideoFrameCallback(decodedFrame); };
    const updateCurrent = () => update();
    ["loadedmetadata", "timeupdate", "seeked", "ended"].forEach((event) => video.addEventListener(event, updateCurrent));
    video.addEventListener("play", start); video.addEventListener("pause", stop); update();
    cleanup.push(() => {
      stop(); video.pause();
      ["loadedmetadata", "timeupdate", "seeked", "ended"].forEach((event) => video.removeEventListener(event, updateCurrent));
      video.removeEventListener("play", start); video.removeEventListener("pause", stop);
    });
  });
  return () => cleanup.forEach((dispose) => dispose());
}
