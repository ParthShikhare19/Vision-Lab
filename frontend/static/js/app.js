/* Vision Lab frontend: plain JavaScript, no build step, no CDN.
 *
 * Flow: grab a frame from <video>/<img> -> send it to /api/detect as JPEG ->
 * draw the returned boxes on the <canvas>. A new frame is only sent after the
 * previous answer arrives, so the frame rate adapts to the speed of the CPU.
 */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const stage = $("stage");
  const sctx = stage.getContext("2d");
  const video = $("video");
  const image = $("image");
  const cap = document.createElement("canvas");      // off-screen frame grabber
  const cctx = cap.getContext("2d");
  const SESSION = Math.random().toString(36).slice(2) + Date.now().toString(36);

  const state = {
    source: "webcam",
    running: false,
    loopActive: false,
    ready: false,
    stream: null,
    result: null,
    line: null,          // [x1, y1, x2, y2] normalised, or null
    drawing: null,       // line being dragged right now
    fps: 0,
    lastAlert: 0,
    audio: null,
  };

  /* ---------------- small helpers ---------------- */
  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function hueFor(label) {
    let h = 0;
    for (let i = 0; i < label.length; i++) h = (h * 31 + label.charCodeAt(i)) % 360;
    return h;
  }
  const colorFor = (label) => "hsl(" + hueFor(label) + ", 85%, 55%)";
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

  async function api(url, options) {
    const res = await fetch(url, options);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || "Request failed.");
    return data;
  }

  function hint(text) { $("hint").textContent = text; }

  function mediaEl() { return state.source === "image" ? image : video; }

  function sourceSize() {
    const m = mediaEl();
    return state.source === "image"
      ? { w: m.naturalWidth, h: m.naturalHeight }
      : { w: m.videoWidth, h: m.videoHeight };
  }

  function resizeStage() {
    const { w, h } = sourceSize();
    if (!w || !h) return;
    const cw = Math.min(w, 960);
    stage.width = cw;
    stage.height = Math.round((cw * h) / w);
  }

  /* ---------------- source handling ---------------- */
  function setButtons() {
    const s = state.source;
    $("btn-start").hidden = s !== "webcam";
    $("lbl-video").hidden = s !== "video";
    $("lbl-image").hidden = s !== "image";
    $("btn-sample").hidden = s !== "image";
    $("btn-start").disabled = state.running;
    $("btn-stop").disabled = !state.running;
  }

  function stopAll() {
    state.running = false;
    if (state.stream) {
      state.stream.getTracks().forEach((t) => t.stop());
      state.stream = null;
    }
    video.pause();
    $("pill-fps").textContent = "– fps";
    setButtons();
  }

  function clearScene() {
    stopAll();
    video.srcObject = null;
    video.removeAttribute("src");
    video.load();
    image.removeAttribute("src");
    state.ready = false;
    state.result = null;
    stage.width = 960;
    stage.height = 540;
    renderLive(null);
  }

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      clearScene();
      state.source = tab.dataset.source;
      setButtons();
      hint({
        webcam: "Press “Start camera”. Tip: drag on the picture to draw a counting line.",
        video: "Choose a video file from your computer. It is processed in your browser tab and never uploaded as a file.",
        image: "Choose a photo, or try the sample. Detection runs once per photo.",
      }[state.source]);
    });
  });

  $("btn-start").addEventListener("click", async () => {
    try {
      state.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    } catch (e) {
      hint("Camera unavailable: " + (e.message || e.name) +
        ". Allow camera access, or use the Video file or Photo tab.");
      return;
    }
    video.srcObject = state.stream;
    await video.play();
    resizeStage();
    state.ready = true;
    state.running = true;
    setButtons();
    hint("Detecting live. Drag on the picture to draw a counting line.");
    startLoop();
  });

  $("btn-stop").addEventListener("click", () => { stopAll(); hint("Stopped."); });

  $("file-video").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    stopAll();
    state.result = null;
    video.srcObject = null;
    video.src = URL.createObjectURL(file);
    video.loop = false;
    video.onloadeddata = async () => {
      resizeStage();
      state.ready = true;
      state.running = true;
      setButtons();
      try { await video.play(); } catch (err) { /* autoplay blocked; user can retry */ }
      hint("Detecting “" + file.name + "”. Drag on the picture to draw a counting line.");
      startLoop();
    };
    video.onended = () => {
      state.running = false;
      setButtons();
      hint("Video finished. Choose another file, or check the analytics below.");
    };
    e.target.value = "";
  });

  function loadImage(src) {
    stopAll();
    state.result = null;
    image.onload = () => {
      resizeStage();
      state.ready = true;
      hint("Photo loaded. Change confidence or the object type to re-run detection.");
      detectOnce().catch((err) => hint(err.message));
    };
    image.src = src;
  }
  $("file-image").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) loadImage(URL.createObjectURL(file));
    e.target.value = "";
  });
  $("btn-sample").addEventListener("click", () => loadImage("/static/samples/bus.jpg"));
  ["lbl-video", "lbl-image"].forEach((id) => {
    $(id).addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        $(id === "lbl-video" ? "file-video" : "file-image").click();
      }
    });
  });

  /* ---------------- detection loop ---------------- */
  function captureBlob() {
    const { w, h } = sourceSize();
    if (!w || !h) return Promise.resolve(null);
    if (state.source !== "image" && video.readyState < 2) return Promise.resolve(null);
    const scale = Math.min(1, 640 / Math.max(w, h));   // the model works at 640 px
    cap.width = Math.round(w * scale);
    cap.height = Math.round(h * scale);
    cctx.drawImage(mediaEl(), 0, 0, cap.width, cap.height);
    return new Promise((resolve) => cap.toBlob(resolve, "image/jpeg", 0.8));
  }

  async function detectOnce() {
    const blob = await captureBlob();
    if (!blob) return;
    const params = new URLSearchParams({
      conf: ($("conf").value / 100).toFixed(2),
      focus: $("focus").value,
      track: state.source !== "image" && $("track").checked ? "1" : "0",
      flip: $("flip").checked ? "1" : "0",
      session: SESSION,
      source: state.source,
    });
    if (state.line) params.set("line", state.line.map((v) => v.toFixed(4)).join(","));
    const data = await api("/api/detect?" + params, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: blob,
    });
    state.result = data;
    renderLive(data);
    checkAlert(data);
  }

  async function startLoop() {
    if (state.loopActive) return;
    state.loopActive = true;
    let last = performance.now();
    while (state.running) {
      try {
        await detectOnce();
      } catch (e) {
        hint("Detection stopped: " + e.message);
        stopAll();
        break;
      }
      const now = performance.now();
      state.fps = state.fps ? state.fps * 0.8 + (1000 / (now - last)) * 0.2 : 1000 / (now - last);
      last = now;
      $("pill-fps").textContent = state.fps.toFixed(1) + " fps";
      await new Promise((r) => setTimeout(r, 0));
    }
    state.loopActive = false;
  }

  /* ---------------- drawing ---------------- */
  function drawBoxes(ctx, W, H) {
    const r = state.result;
    if (!r) return;
    const lw = Math.max(2, W / 380);
    const fs = Math.max(12, Math.round(W / 55));
    ctx.font = "600 " + fs + "px system-ui, sans-serif";
    ctx.textBaseline = "middle";
    r.detections.forEach((d) => {
      const x1 = d.box[0] * W, y1 = d.box[1] * H, x2 = d.box[2] * W, y2 = d.box[3] * H;
      const color = colorFor(d.label);
      ctx.lineWidth = lw;
      ctx.strokeStyle = color;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      const text = d.label + " " + Math.round(d.confidence * 100) + "%" +
        (d.confirmed ? "  #" + d.id : "");
      const tw = ctx.measureText(text).width + 10;
      const th = fs + 6;
      const ty = y1 - th < 0 ? y1 : y1 - th;
      ctx.fillStyle = color;
      ctx.fillRect(x1, ty, tw, th);
      ctx.fillStyle = "#0e1522";
      ctx.fillText(text, x1 + 5, ty + th / 2);
    });
  }

  function drawLine(ctx, W, H) {
    const pts = state.drawing ? [...state.drawing.a, ...state.drawing.b] : state.line;
    if (!pts) return;
    const ax = pts[0] * W, ay = pts[1] * H, bx = pts[2] * W, by = pts[3] * H;
    ctx.save();
    ctx.lineCap = "round";
    ctx.strokeStyle = "#0e1522";
    ctx.lineWidth = Math.max(6, W / 160);
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    ctx.strokeStyle = "#ffd23f";
    ctx.lineWidth = Math.max(3, W / 300);
    ctx.setLineDash([10, 8]);
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    ctx.setLineDash([]);

    // arrow showing the "in" direction (positive side of the line, see tracker.py)
    const dx = bx - ax, dy = by - ay, len = Math.hypot(dx, dy) || 1;
    let nx = -dy / len, ny = dx / len;
    if ($("flip").checked) { nx = -nx; ny = -ny; }
    const mx = (ax + bx) / 2, my = (ay + by) / 2, al = Math.max(30, W / 24);
    ctx.fillStyle = "#ffd23f";
    ctx.strokeStyle = "#ffd23f";
    ctx.lineWidth = Math.max(3, W / 300);
    ctx.beginPath(); ctx.moveTo(mx, my); ctx.lineTo(mx + nx * al, my + ny * al); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(mx + nx * (al + 10), my + ny * (al + 10));
    ctx.lineTo(mx + nx * al - ny * 7, my + ny * al + nx * 7);
    ctx.lineTo(mx + nx * al + ny * 7, my + ny * al - nx * 7);
    ctx.closePath(); ctx.fill();

    const l = state.result && state.result.line;
    if (l) {
      const fs = Math.max(13, Math.round(W / 50));
      ctx.font = "700 " + fs + "px system-ui, sans-serif";
      const text = "in " + l.in + "  ·  out " + l.out;
      const tw = ctx.measureText(text).width + 12;
      const bh = fs + 8;
      // put the label on the opposite side of the arrow so they never overlap
      const lx = mx - nx * (bh + 14), ly = my - ny * (bh + 14);
      ctx.fillStyle = "#ffd23f";
      ctx.fillRect(lx - tw / 2, ly - bh / 2, tw, bh);
      ctx.fillStyle = "#0e1522";
      ctx.textBaseline = "middle";
      ctx.fillText(text, lx - tw / 2 + 6, ly);
    }
    ctx.restore();
  }

  function drawScene() {
    const W = stage.width, H = stage.height;
    if (!state.ready) {
      sctx.fillStyle = "#0e1522";
      sctx.fillRect(0, 0, W, H);
      sctx.fillStyle = "#8c99b0";
      sctx.font = "600 22px system-ui, sans-serif";
      sctx.textAlign = "center";
      sctx.fillText("Choose a source to begin", W / 2, H / 2);
      sctx.textAlign = "start";
      return;
    }
    sctx.drawImage(mediaEl(), 0, 0, W, H);
    drawBoxes(sctx, W, H);
    drawLine(sctx, W, H);
  }
  (function frame() { drawScene(); requestAnimationFrame(frame); })();

  /* ---------------- counting line (drag on the picture) ---------------- */
  function norm(e) {
    const r = stage.getBoundingClientRect();
    return [clamp((e.clientX - r.left) / r.width, 0, 1), clamp((e.clientY - r.top) / r.height, 0, 1)];
  }
  stage.addEventListener("pointerdown", (e) => {
    if (!state.ready) return;
    e.preventDefault();
    stage.setPointerCapture(e.pointerId);
    state.drawing = { a: norm(e), b: norm(e) };
  });
  stage.addEventListener("pointermove", (e) => {
    if (state.drawing) state.drawing.b = norm(e);
  });
  stage.addEventListener("pointerup", () => {
    if (!state.drawing) return;
    const { a, b } = state.drawing;
    state.drawing = null;
    if (Math.hypot(b[0] - a[0], b[1] - a[1]) > 0.06) {
      state.line = [a[0], a[1], b[0], b[1]];
      hint("Counting line set. Objects crossing it are counted as “in” (arrow side) or “out”.");
    }
  });
  $("btn-line").addEventListener("click", () => { state.line = null; hint("Counting line cleared."); });

  /* ---------------- live numbers ---------------- */
  function renderLive(data) {
    const tally = $("tally");
    tally.replaceChildren();
    if (!data) {
      ["n-now", "n-unique", "n-in", "n-out"].forEach((id) => { $(id).textContent = "0"; });
      $("pill-ms").textContent = "– ms";
      $("tally-empty").hidden = false;
      return;
    }
    $("n-now").textContent = data.total;
    $("n-unique").textContent = data.unique_total !== undefined ? data.unique_total : "–";
    $("n-in").textContent = data.line ? data.line.in : "–";
    $("n-out").textContent = data.line ? data.line.out : "–";
    $("pill-ms").textContent = data.inference_ms.toFixed(0) + " ms";

    const labels = Object.keys(data.counts).sort((a, b) => data.counts[b] - data.counts[a]);
    $("tally-empty").hidden = labels.length > 0;
    labels.forEach((label) => {
      const li = el("li");
      const sw = el("span", "swatch");
      sw.style.background = colorFor(label);
      li.appendChild(sw);
      li.appendChild(el("span", "", label));
      li.appendChild(el("span", "num", String(data.counts[label])));
      const u = data.unique && data.unique[label];
      li.appendChild(el("span", "uniq", u ? u + " unique" : ""));
      tally.appendChild(li);
    });
  }

  /* ---------------- alert rule ---------------- */
  function beep() {
    if (!$("rule-sound").checked) return;
    try {
      state.audio = state.audio || new (window.AudioContext || window.webkitAudioContext)();
      const osc = state.audio.createOscillator();
      const gain = state.audio.createGain();
      osc.frequency.value = 880;
      gain.gain.value = 0.15;
      osc.connect(gain); gain.connect(state.audio.destination);
      osc.start(); osc.stop(state.audio.currentTime + 0.25);
    } catch (e) { /* sound is optional */ }
  }

  function checkAlert(data) {
    const label = $("rule-class").value;
    const need = parseInt($("rule-n").value, 10) || 1;
    const have = data.counts[label] || 0;
    const now = Date.now();
    if (have < need || now - state.lastAlert < 8000) return;
    state.lastAlert = now;
    const text = "Alert: " + have + " × " + label + " in view (rule: at least " + need + ")";
    const banner = $("alert-banner");
    banner.textContent = text;
    banner.hidden = false;
    setTimeout(() => { banner.hidden = true; }, 5000);
    beep();
    api("/api/events", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind: "alert", label: label, count: have,
                             detail: have + " in frame, rule ≥ " + need }),
    }).then(refreshPanels).catch(() => {});
  }

  /* ---------------- analytics, events, snapshots ---------------- */
  function drawTimeline(values) {
    const svg = $("timeline");
    svg.replaceChildren();
    if (values.length < 2) return;
    const NS = "http://www.w3.org/2000/svg";
    const max = Math.max(...values, 1);
    const step = 300 / (values.length - 1);
    const pts = values.map((v, i) => (i * step).toFixed(1) + "," + (66 - (v / max) * 60).toFixed(1));
    const area = document.createElementNS(NS, "polygon");
    area.setAttribute("points", "0,70 " + pts.join(" ") + " 300,70");
    area.setAttribute("fill", "rgba(53, 86, 240, .14)");
    svg.appendChild(area);
    const line = document.createElementNS(NS, "polyline");
    line.setAttribute("points", pts.join(" "));
    line.setAttribute("fill", "none");
    line.setAttribute("stroke", "#3556f0");
    line.setAttribute("stroke-width", "2");
    line.setAttribute("vector-effect", "non-scaling-stroke");
    svg.appendChild(line);
  }

  async function refreshStats() {
    const s = await api("/api/stats");
    const k = document.querySelectorAll(".kpi");
    k[0].textContent = s.samples;
    k[1].textContent = s.avg_infer_ms;
    k[2].textContent = s.events;
    drawTimeline(s.timeline);
    const bars = $("bars");
    bars.replaceChildren();
    const max = Math.max(...s.top_classes.map((c) => c.avg_per_sample), 0.01);
    s.top_classes.forEach((c) => {
      const row = el("div", "bar-row");
      row.appendChild(el("span", "", c.label));
      const track = el("div", "bar-track");
      const fill = el("div", "bar-fill");
      fill.style.width = (c.avg_per_sample / max) * 100 + "%";
      fill.style.background = colorFor(c.label);
      track.appendChild(fill);
      row.appendChild(track);
      row.appendChild(el("span", "", c.avg_per_sample.toFixed(1)));
      bars.appendChild(row);
    });
  }

  async function refreshEvents() {
    const rows = await api("/api/events?limit=40");
    const body = $("events");
    body.replaceChildren();
    $("events-empty").hidden = rows.length > 0;
    rows.forEach((r) => {
      const tr = el("tr");
      [r.ts.slice(11), r.kind, r.label, r.detail].forEach((t) => tr.appendChild(el("td", "", t)));
      body.appendChild(tr);
    });
  }

  async function refreshGallery() {
    const rows = await api("/api/snapshots");
    const box = $("gallery");
    box.replaceChildren();
    $("gallery-empty").hidden = rows.length > 0;
    rows.forEach((r) => {
      const wrap = el("div", "shot");
      const link = el("a");
      link.href = "/snapshots/" + r.filename;
      link.target = "_blank";
      link.rel = "noopener";
      const img = el("img");
      img.src = "/snapshots/" + r.filename;
      img.alt = "Snapshot taken " + r.ts + ". " + r.summary;
      link.appendChild(img);
      wrap.appendChild(link);
      const del = el("button", "", "×");
      del.title = "Delete snapshot";
      del.setAttribute("aria-label", "Delete snapshot " + r.id);
      del.addEventListener("click", async () => {
        await api("/api/snapshots/" + r.id, { method: "DELETE" });
        refreshGallery();
      });
      wrap.appendChild(del);
      wrap.appendChild(el("span", "", r.ts.slice(11) + " · " + r.summary));
      box.appendChild(wrap);
    });
  }

  function refreshPanels() {
    refreshStats().catch(() => {});
    refreshEvents().catch(() => {});
  }

  $("btn-snap").addEventListener("click", () => {
    if (!state.ready) { hint("Start a source first, then save a snapshot."); return; }
    drawScene();
    stage.toBlob(async (blob) => {
      const form = new FormData();
      form.append("image", blob, "snapshot.jpg");
      const r = state.result;
      const list = r ? Object.entries(r.counts).map(([k, v]) => v + " " + k).join(", ") : "";
      form.append("summary", list || "no objects");
      try {
        await api("/api/snapshots", { method: "POST", body: form });
        hint("Snapshot saved.");
        refreshGallery();
        refreshStats().catch(() => {});
      } catch (e) { hint(e.message); }
    }, "image/jpeg", 0.9);
  });

  $("btn-reset").addEventListener("click", async () => {
    await api("/api/session/reset?session=" + SESSION, { method: "POST" });
    if (state.result) { delete state.result.unique; delete state.result.line; }
    ["n-unique", "n-in", "n-out"].forEach((id) => { $(id).textContent = "0"; });
    hint("Counts reset.");
  });

  $("btn-clear-log").addEventListener("click", async () => {
    if (!confirm("Clear the event log and analytics samples? Snapshots are kept.")) return;
    await api("/api/events", { method: "DELETE" });
    refreshPanels();
  });

  /* ---------------- controls ---------------- */
  function rerunIfImage() {
    if (state.source === "image" && state.ready) detectOnce().catch((e) => hint(e.message));
  }
  $("conf").addEventListener("input", () => {
    $("conf-out").textContent = $("conf").value + "%";
  });
  $("conf").addEventListener("change", rerunIfImage);
  $("focus").addEventListener("change", rerunIfImage);

  /* ---------------- start-up ---------------- */
  async function init() {
    setButtons();
    try {
      const h = await api("/api/health");
      const pill = $("pill-model");
      pill.textContent = h.model.replace(".onnx", "") + " ready";
      pill.classList.add("ok");
      const c = await api("/api/classes");
      const select = $("rule-class");
      [...c.classes].sort().forEach((name) => {
        const o = el("option", "", name);
        o.value = name;
        if (name === "person") o.selected = true;
        select.appendChild(o);
      });
    } catch (e) {
      const pill = $("pill-model");
      pill.textContent = "Server error";
      pill.classList.add("bad");
    }
    refreshPanels();
    refreshGallery().catch(() => {});
    setInterval(refreshPanels, 3000);
  }
  init();
})();
