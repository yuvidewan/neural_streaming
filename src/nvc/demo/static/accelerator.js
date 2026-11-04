"use strict";

// NVC-ACCEL, the accelerator designed for this codec (hardware/ARCHITECTURE.md), as an
// interactive 3D floorplan. A small canvas renderer rather than a 3D library, so the page
// keeps working with no network: boxes, back-face culling, painter's order, hover picking.
// Every number in the side panel comes from ARCHITECTURE.md and keeps its tag there:
// MEASURED (a result in this repo), DERIVED (arithmetic on measured facts), ASSUMED (an
// estimate for hardware that has not been built).

(function () {
  const canvas = document.getElementById("accel-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const panel = document.getElementById("accel-info");

  const C = { die: "#0e2a33", ctrl: "#3d5f69", io: "#4f7782", mem: "#7b64a8", pe: "#2fa89c",
              peHot: "#7fe0d4", quant: "#e8a33d", lane: "#0f766e", host: "#6b7780", bus: "#e8a33d" };

  // [x0, x1, z0, z1, height]; the die spans x -11..11, z -7..7.
  const BLOCKS = [
    { id: "die", name: "NVC-ACCEL V1 (FPGA prototype)", box: [-11, 11, -7, 7, 0.35], color: C.die, pick: false },
    { id: "host", name: "Host CPU", box: [-17.5, -13, -2.2, 2.2, 1.6], color: C.host, label: "Host CPU" },
    { id: "ctrl", name: "Control & sequencer FSM", box: [-10, -5, -6.2, -3, 1.0], color: C.ctrl, label: "Control FSM" },
    { id: "dma", name: "Frame DMA (AXI4-Stream)", box: [-10, -5, -1.6, 1.6, 1.0], color: C.io, label: "Frame DMA" },
    { id: "wsram", name: "Weight SRAM", box: [-3, 3, -6.2, -4.2, 0.9], color: C.mem, label: "Weight SRAM 580 KB" },
    { id: "cnn", name: "CNN engine: 16×16 INT8 systolic array", box: [-3, 3, -3.2, 3.2, 0.3], color: "#1b4a54",
      label: "CNN engine 16×16 PEs" },
    { id: "asram", name: "Activation SRAM", box: [-3, 3, 4.2, 6.2, 0.9], color: C.mem, label: "Activation SRAM 2 MB" },
    { id: "quant", name: "Quantize / dequantize", box: [4.3, 5.8, -3.2, 3.2, 0.8], color: C.quant, label: "Quant" },
    { id: "tsram", name: "Entropy table SRAM", box: [7, 10.4, -6.2, -4.2, 0.9], color: C.mem, label: "Table SRAM" },
    { id: "pack", name: "Stream packer / unpacker", box: [7, 10.4, 4.2, 6.2, 0.9], color: C.io, label: "Packer" },
  ];
  // Eight range-coder lanes side by side, one pickable group.
  for (let i = 0; i < 8; i++) {
    const z0 = -3.2 + i * 0.8;
    BLOCKS.push({ id: "ent", name: "Entropy coder", box: [7, 10.4, z0 + 0.08, z0 + 0.72, 0.7], color: C.lane,
                  lane: i, label: i === 0 ? "Entropy coder ×8 lanes" : null });
  }

  const INFO = {
    die: { text: "One chip with every stage of the codec on it, as an FPGA prototype first. "
                 + "Designed for this codec's exact shapes: 256×256×3 frames, a 64×16×16 latent, 4/6/8-bit.",
           facts: [["Status", "architecture designed; entropy idea validated in software; not yet synthesised", "MEASURED"],
                   ["INT8 activations", "−0.807 dB PSNR vs float32 (per-channel), full DAVIS TEST", "MEASURED"],
                   ["Built for", "the codec's first network (593k params). Today's 7.2M-param network would need the CNN engine and weight SRAM resized", "DERIVED"]] },
    host: { text: "Sends frames in, picks the bit depth, and loads the weights and entropy tables once at start-up.",
            facts: [["Off-chip traffic", "under 6 MB/s at 30 fps: a 196 KB frame in, a 5–15 KB payload out", "DERIVED"]] },
    ctrl: { text: "Register file and sequencer. Steps the pipeline layer by layer and switches between encode and decode.",
            facts: [["Interface", "AXI4-Lite target", "ASSUMED"], ["Clock", "single clock domain in V1", "ASSUMED"]] },
    dma: { text: "Moves frames in and compressed payloads out. Weights never travel per frame.",
           facts: [["Frame in", "256×256×3 bytes = 196 KB", "DERIVED"],
                   ["Payload out", "about 5–15 KB per frame", "MEASURED"]] },
    cnn: { text: "The neural network runs here: 256 multiply-accumulate units in a 16×16 grid, holding weights in place "
                 + "(weight-stationary). Encoder and decoder share it, because their layers mirror each other exactly.",
           facts: [["Peak", "204.8 GOP/s at 400 MHz", "ASSUMED"],
                   ["Needed for 30 fps", "39.2 GOP/s, about 19% of peak", "DERIVED"],
                   ["Work per frame", "654M MACs encode + decode", "DERIVED"],
                   ["Why it's simple", "every layer is 4×4 stride 2, so one schedule covers all 8 layers", "MEASURED"]] },
    wsram: { text: "All network weights live on chip permanently, so no memory traffic is spent on them.",
             facts: [["Size", "593,411 weights at INT8 = about 580 KB", "DERIVED"]] },
    asram: { text: "Holds the feature maps between layers, double-buffered so one layer writes while the next reads.",
             facts: [["Size", "about 2 MB (largest map 128×128×32)", "DERIVED"]] },
    quant: { text: "Rounds the latent to whole numbers for coding (and back for decoding), fused into the CNN's output "
                   + "stage so it adds no pipeline stage.", facts: [["Operation", "round(z / scale) + zero_point, per channel", "MEASURED"]] },
    ent: { text: "The key design idea. Software codes all 16,384 symbols as one serial stream. The chip splits it into 64 "
                 + "independent per-channel streams, so 8 lanes work in parallel.",
           facts: [["Cost of splitting", "+1.1% bits, bit-exact on a real DAVIS frame", "MEASURED"],
                   ["No divider", "the table total is 2^16, so every division is a 16-bit shift", "MEASURED"],
                   ["Speed (8 lanes)", "10.2 µs encode, 41 µs decode per frame, vs ~71 ms in software", "DERIVED"]] },
    tsram: { text: "The probability tables the range coder uses, one per latent channel. Reloaded only when the bit depth changes.",
             facts: [["Contents", "64 channels × up to 256 cumulative counts", "DERIVED"]] },
    pack: { text: "Writes the file header and the 64 per-channel streams (each with a 2-byte length) into the output, "
                  + "and splits them apart when decoding.", facts: [["Format", ".nvc header + NVC-HW payload", "MEASURED"]] },
  };

  // Data path, at bus height. Encode runs forward; decode runs it backwards.
  const BUS_Y = 1.45;
  const PATH = [[-15.2, 0], [-7.5, 0], [0, 0], [5.05, 0], [8.7, 0], [8.7, 5.2], [8.7, 6.75], [-7.5, 6.75], [-7.5, 1.2], [-15.2, 1.2]];
  const STAGE_AT = [["host", 0], ["dma", 1], ["cnn", 2], ["quant", 3], ["ent", 4], ["pack", 5]];
  const LINKS = [["wsram", [0, -4.2], [0, -3.2]], ["asram", [0, 4.2], [0, 3.2]], ["tsram", [8.7, -4.2], [8.7, -3.2]]];

  const view = { yaw: -0.62, pitch: -0.72, dist: 37, auto: true, mode: "encode", running: true, t: 0 };
  let hovered = null, selected = "ent", picks = [], dragging = null, last = 0;
  let visible = false, frame = 0;      // drawn only while its page is open (window.nvcAccel)

  // ------------------------------------------------------------ projection

  function camera(p) {
    const cy = Math.cos(view.yaw), sy = Math.sin(view.yaw), cp = Math.cos(view.pitch), sp = Math.sin(view.pitch);
    const x1 = p[0] * cy - p[2] * sy, z1 = p[0] * sy + p[2] * cy;
    return [x1, p[1] * cp - z1 * sp, p[1] * sp + z1 * cp];
  }
  function screen(q, w, h) {
    const depth = view.dist + q[2], f = Math.min(w, h) * 1.55;
    return [w / 2 + f * q[0] / depth, h * 0.56 - f * q[1] / depth, depth];
  }
  const shades = new Map();         // a few dozen distinct (colour, light) pairs, reused every frame
  const shade = (hex, k) => {
    const key = hex + k;
    let rgb = shades.get(key);
    if (!rgb) {
      const n = parseInt(hex.slice(1), 16);
      const ch = (v) => Math.max(0, Math.min(255, Math.round(v * k)));
      rgb = `rgb(${ch(n >> 16)},${ch((n >> 8) & 255)},${ch(n & 255)})`;
      shades.set(key, rgb);
    }
    return rgb;
  };

  // The six faces of a box as vertex lists, each with its outward normal and a light level.
  function boxFaces(x0, x1, y0, y1, z0, z1) {
    const v = (x, y, z) => [x, y, z];
    return [
      { n: [0, 1, 0], k: 1.0, pts: [v(x0, y1, z0), v(x1, y1, z0), v(x1, y1, z1), v(x0, y1, z1)] },
      { n: [0, -1, 0], k: 0.4, pts: [v(x0, y0, z0), v(x0, y0, z1), v(x1, y0, z1), v(x1, y0, z0)] },
      { n: [0, 0, 1], k: 0.78, pts: [v(x0, y0, z1), v(x0, y1, z1), v(x1, y1, z1), v(x1, y0, z1)] },
      { n: [0, 0, -1], k: 0.6, pts: [v(x0, y0, z0), v(x1, y0, z0), v(x1, y1, z0), v(x0, y1, z0)] },
      { n: [1, 0, 0], k: 0.68, pts: [v(x1, y0, z0), v(x1, y0, z1), v(x1, y1, z1), v(x1, y1, z0)] },
      { n: [-1, 0, 0], k: 0.86, pts: [v(x0, y0, z0), v(x0, y1, z0), v(x0, y1, z1), v(x0, y0, z1)] },
    ];
  }

  // ------------------------------------------------------------ the scene, rebuilt each frame

  function stageProgress() {
    // Position of the travelling packet along PATH, 0..PATH.length-1.
    const span = PATH.length - 1, s = (view.t * 0.55) % span;
    return view.mode === "encode" ? s : span - s;
  }

  function sceneFaces() {
    const faces = [];
    const s = stageProgress();
    const active = STAGE_AT.reduce((best, [id, at]) => (Math.abs(s - at) < 0.6 ? id : best), null);
    const add = (block, x0, x1, y0, y1, z0, z1, color, pe = false) => {
      for (const face of boxFaces(x0, x1, y0, y1, z0, z1)) faces.push({ ...face, id: block, color, pe });
    };
    for (const b of BLOCKS) {
      const [x0, x1, z0, z1, h] = b.box;
      const base = b.id === "die" || b.id === "host" ? 0 : 0.35;
      let color = b.color;
      if (b.id === active && b.id !== "cnn") color = shade(b.color, 1.35);
      if (b.id === "ent" && active === "ent") color = (Math.floor(view.t * 6) + b.lane) % 2 ? shade(C.lane, 1.5) : C.lane;
      add(b.id, x0, x1, base, base + h, z0, z1, color);
    }
    // The PE grid: 16x16 cubes on the CNN platform, rippling while the CNN stage is busy.
    const busy = active === "cnn";
    for (let i = 0; i < 16; i++) {
      for (let j = 0; j < 16; j++) {
        const x = -2.8 + i * 0.35, z = -2.8 + j * 0.35;
        const wave = busy ? Math.max(0, Math.sin(view.t * 7 - (i + j) * 0.45)) : 0;
        add("cnn", x, x + 0.27, 0.65, 0.95 + wave * 0.45, z, z + 0.27, wave > 0.5 ? C.peHot : C.pe, true);
      }
    }
    return { faces, active, s };
  }

  // ------------------------------------------------------------ drawing

  function draw(now) {
    const dt = last ? Math.min(0.05, (now - last) / 1000) : 0;
    last = now;
    if (view.running) view.t += dt;
    if (view.auto && !dragging) view.yaw += dt * 0.12;

    if (!visible) { frame = 0; return; }
    const dpr = Math.min(2, window.devicePixelRatio || 1), w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const { faces, active, s } = sceneFaces();
    const drawn = [];
    for (const face of faces) {
      const cam = face.pts.map(camera);
      const centre = cam.reduce((a, p) => [a[0] + p[0] / 4, a[1] + p[1] / 4, a[2] + p[2] / 4], [0, 0, 0]);
      const n = camera(face.n);
      // Visible when the normal points back toward the eye at (0, 0, -dist).
      if (n[0] * -centre[0] + n[1] * -centre[1] + n[2] * (-view.dist - centre[2]) <= 0) continue;
      drawn.push({ face, poly: cam.map((p) => screen(p, w, h)), depth: centre[2] });
    }
    drawn.sort((a, b) => b.depth - a.depth);

    picks = [];
    for (const { face, poly } of drawn) {
      const lit = face.id === hovered || face.id === selected ? 1.18 : 1;
      if (face.pe) {
        // Most of the faces in the scene: filled directly, no outline and no pick path
        // (the CNN platform underneath catches clicks on the array).
        ctx.beginPath();
        poly.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
        ctx.fillStyle = shade(face.color, face.k * lit);
        ctx.fill();
        continue;
      }
      const path = new Path2D();
      poly.forEach((p, i) => (i ? path.lineTo(p[0], p[1]) : path.moveTo(p[0], p[1])));
      path.closePath();
      ctx.fillStyle = face.color.startsWith("#") ? shade(face.color, face.k * lit) : face.color;
      ctx.fill(path);
      ctx.strokeStyle = "rgba(255,255,255,0.08)";
      ctx.stroke(path);
      const block = BLOCKS.find((b) => b.id === face.id);
      if (!block || block.pick !== false) picks.push({ path, id: face.id });
    }

    drawLinks(w, h);
    drawBus(w, h, s);
    drawLabels(w, h, active);
    frame = requestAnimationFrame(draw);
  }

  function polyline(points, w, h) {
    const pts = points.map((p) => screen(camera(p), w, h));
    ctx.beginPath();
    pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
    return pts;
  }

  function drawLinks(w, h) {
    ctx.setLineDash([3, 4]);
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "rgba(190,170,230,0.85)";
    for (const [, a, b] of LINKS) {
      polyline([[a[0], 1.1, a[1]], [b[0], 1.1, b[1]]], w, h);
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }

  function drawBus(w, h, s) {
    const pts = PATH.map(([x, z]) => [x, BUS_Y, z]);
    polyline(pts, w, h);
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(232,163,61,0.55)";
    ctx.stroke();
    // Packets: a short train of dots travelling the path in the current direction.
    for (let k = 0; k < 4; k++) {
      const span = PATH.length - 1;
      const u = ((s - k * 0.22 * (view.mode === "encode" ? 1 : -1)) % span + span) % span;
      const i = Math.floor(u), f = u - i, a = pts[i], b = pts[Math.min(i + 1, span)];
      const p = screen(camera([a[0] + (b[0] - a[0]) * f, BUS_Y, a[2] + (b[2] - a[2]) * f]), w, h);
      ctx.beginPath();
      ctx.arc(p[0], p[1], 5.5 - k, 0, Math.PI * 2);
      ctx.fillStyle = k ? `rgba(232,163,61,${0.8 - k * 0.18})` : "#ffd27a";
      ctx.fill();
    }
  }

  function drawLabels(w, h, active) {
    ctx.font = "600 12px 'Segoe UI', system-ui, Arial, sans-serif";
    ctx.textAlign = "center";
    for (const b of BLOCKS) {
      if (!b.label) continue;
      const [x0, x1, z0, z1, hgt] = b.box;
      const zc = b.id === "ent" ? 0 : (z0 + z1) / 2;
      const top = b.id === "cnn" ? 1.6 : (b.id === "host" ? 0 : 0.35) + hgt + 0.35;
      const p = screen(camera([(x0 + x1) / 2, top, zc]), w, h);
      const text = b.label, tw = ctx.measureText(text).width + 12;
      const on = b.id === active || b.id === hovered || b.id === selected;
      ctx.fillStyle = on ? "rgba(255,255,255,0.96)" : "rgba(14,42,51,0.78)";
      ctx.beginPath();
      ctx.roundRect(p[0] - tw / 2, p[1] - 20, tw, 18, 9);
      ctx.fill();
      ctx.fillStyle = on ? "#0e2a33" : "#ffffff";
      ctx.fillText(text, p[0], p[1] - 7);
    }
  }

  // ------------------------------------------------------------ the side panel

  function showInfo(id) {
    selected = id;
    const info = INFO[id], block = BLOCKS.find((b) => b.id === id);
    if (!info || !block) return;
    panel.innerHTML = "";
    const title = document.createElement("h3");
    title.textContent = block.name;
    const text = document.createElement("p");
    text.textContent = info.text;
    const list = document.createElement("dl");
    for (const [label, value, tag] of info.facts) {
      const dt = document.createElement("dt");
      dt.textContent = label;
      const chip = document.createElement("span");
      chip.className = "tag tag-" + tag.toLowerCase();
      chip.textContent = tag;
      dt.appendChild(chip);
      const dd = document.createElement("dd");
      dd.textContent = value;
      list.append(dt, dd);
    }
    panel.append(title, text, list);
  }

  // ------------------------------------------------------------ interaction

  function pickAt(event) {
    const box = canvas.getBoundingClientRect(), x = event.clientX - box.left, y = event.clientY - box.top;
    for (let i = picks.length - 1; i >= 0; i--) {
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      if (ctx.isPointInPath(picks[i].path, x * dpr, y * dpr)) {
        return picks[i].id;
      }
    }
    return null;
  }

  canvas.addEventListener("pointerdown", (event) => {
    dragging = { x: event.clientX, y: event.clientY, moved: false };
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (dragging) {
      const dx = event.clientX - dragging.x, dy = event.clientY - dragging.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) { dragging.moved = true; view.auto = false; }
      view.yaw += dx * 0.008;
      view.pitch = Math.max(-1.45, Math.min(-0.12, view.pitch + dy * 0.006));
      dragging.x = event.clientX; dragging.y = event.clientY;
    } else {
      hovered = pickAt(event);
      canvas.style.cursor = hovered ? "pointer" : "grab";
    }
  });
  canvas.addEventListener("pointerup", (event) => {
    if (dragging && !dragging.moved) {
      const id = pickAt(event);
      if (id) showInfo(id);
    }
    dragging = null;
  });
  canvas.addEventListener("pointerleave", () => { hovered = null; });
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    view.dist = Math.max(24, Math.min(80, view.dist * (event.deltaY > 0 ? 1.08 : 0.93)));
  }, { passive: false });

  const setMode = (mode) => {
    view.mode = mode;
    document.querySelectorAll("[data-accel-mode]").forEach((b) => b.classList.toggle("on", b.dataset.accelMode === mode));
    document.getElementById("accel-mode-note").textContent = mode === "encode"
      ? "Encode: frame in → CNN encoder → quantize → 8 coder lanes → packed file out."
      : "Decode: file in → unpack → 8 decoder lanes → dequantize → CNN decoder → frame out.";
  };
  document.querySelectorAll("[data-accel-mode]").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.accelMode)));
  document.getElementById("accel-pause").addEventListener("click", (event) => {
    view.running = !view.running;
    event.currentTarget.textContent = view.running ? "Pause" : "Play";
  });
  document.getElementById("accel-about").addEventListener("click", () => showInfo("die"));
  document.getElementById("accel-reset").addEventListener("click", () => {
    Object.assign(view, { yaw: -0.62, pitch: -0.72, dist: 37, auto: true });
  });

  setMode("encode");
  showInfo("ent");
  window.nvcAccel = {
    start() {
      visible = true;
      last = 0;
      if (!frame) frame = requestAnimationFrame(draw);
    },
    stop() { visible = false; },
  };
})();
