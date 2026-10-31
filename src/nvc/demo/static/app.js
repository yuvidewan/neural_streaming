"use strict";

const $ = (id) => document.getElementById(id);
const INK = "#0e2a33", TEAL = "#0f766e", AMBER = "#e8a33d", MUTED = "#5f6b73", CARD = "#f1f5f6";
const STEP_SCALE = 2;            // the step canvases are drawn at 2x for sharp arrows

const state = {
  status: null, clips: [], clipId: null, bundleId: null,
  result: null, images: null, frame: 0, playing: false, timer: null,
  token: 0, activityMax: 1, busy: false,
};

// ---------------------------------------------------------------- helpers

async function api(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch (error) { /* not JSON */ }
    const failure = new Error(detail);
    failure.status = response.status;
    throw failure;
  }
  return response.json();
}

const pad = (index) => String(index).padStart(3, "0");
const fixed = (value, digits) => (value === null || value === undefined ? "n/a" : value.toFixed(digits));
function bytes(count) {
  if (count >= 1024 * 1024) return (count / (1024 * 1024)).toFixed(2) + " MB";
  if (count >= 1024) return (count / 1024).toFixed(1) + " KB";
  return count.toLocaleString() + " B";
}

function show(id, visible) { $(id).hidden = !visible; }
function message(text, kind) {
  const box = $("message");
  box.textContent = text || "";
  box.className = "banner" + (kind ? " " + kind : "");
  box.hidden = !text;
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Could not load " + url));
    image.src = url;
  });
}

// ---------------------------------------------------------------- start-up and selection

async function refreshLists() {
  state.status = await api("/api/status");
  state.clips = await api("/api/clips");

  const bundleSelect = $("bundle-select");
  bundleSelect.innerHTML = "";
  for (const bundle of state.status.bundles) {
    const option = document.createElement("option");
    option.value = bundle.id;
    option.textContent = bundle.label;
    bundleSelect.appendChild(option);
  }
  const bundleIds = state.status.bundles.map((bundle) => bundle.id);
  if (!bundleIds.includes(state.bundleId)) state.bundleId = state.status.default_bundle;
  if (state.bundleId) bundleSelect.value = state.bundleId;

  const clipSelect = $("clip-select");
  clipSelect.innerHTML = "";
  for (const clip of state.clips) {
    const option = document.createElement("option");
    option.value = clip.id;
    option.textContent = clip.name + (clip.source === "upload" ? " (uploaded)" : "");
    clipSelect.appendChild(option);
  }
  const clipIds = state.clips.map((clip) => clip.id);
  if (!clipIds.includes(state.clipId)) state.clipId = clipIds[0] || null;
  if (state.clipId) clipSelect.value = state.clipId;
}

function currentBundle() {
  return state.status.bundles.find((bundle) => bundle.id === state.bundleId) || null;
}
function currentClip() {
  return state.clips.find((clip) => clip.id === state.clipId) || null;
}

async function loadView() {
  stop();
  const token = ++state.token;
  message("");
  show("viewer", false);
  show("pending", false);
  show("empty", false);
  state.result = null;
  state.images = null;

  const bundle = currentBundle();
  const banner = $("standin-banner");
  if (bundle && bundle.kind === "standin") {
    banner.textContent = bundle.note + " Copy the trained bundles into outputs/codec_bundles to switch.";
    banner.hidden = false;
  } else {
    banner.hidden = true;
  }

  if (!bundle) {
    $("empty-text").textContent = "No codec bundle was found. Run python scripts/build_demo.py to build a stand-in codec and the example clips.";
    show("empty", true);
    return;
  }
  const clip = currentClip();
  if (!clip) {
    $("empty-text").textContent = "No clips yet. Upload a short video, or run python scripts/build_demo.py for the example clips.";
    show("empty", true);
    return;
  }

  let result;
  try {
    result = await api(`/api/clips/${clip.id}/results/${bundle.id}`);
  } catch (error) {
    if (token !== state.token) return;
    if (error.status === 404) {
      $("pending-text").textContent = `"${clip.name}" has not been encoded with ${bundle.label} yet.`;
      show("pending", !state.busy);
    } else {
      message(error.message, "error");
    }
    return;
  }
  if (token !== state.token) return;

  try {
    state.images = await loadFrames(clip, bundle, result, token);
  } catch (error) {
    if (token === state.token) message(error.message, "error");
    return;
  }
  if (token !== state.token) return;

  state.result = result;
  state.frame = 0;
  state.activityMax = Math.max(1e-6, ...result.nvc.activity.filter(Boolean).flat(2));
  $("scrub").max = result.frames - 1;
  $("scrub").value = 0;
  renderStatic();
  show("viewer", true);
  drawBars();
  render();
}

async function loadFrames(clip, bundle, result, token) {
  const base = `/media/${clip.id}`;
  const stamp = `?v=${result.nvc.total_bytes}-${result.frames}`;
  const haveH264 = result.h264.available;
  const total = result.frames * (haveH264 ? 3 : 2);
  let loaded = 0;
  const track = (promise) => promise.then((image) => {
    loaded += 1;
    if (token === state.token && !state.busy) jobPanel("Loading frames", `${loaded} of ${total}`, loaded / total);
    return image;
  });
  const images = { original: [], nvc: [], h264: [], prediction: [] };
  const work = [];
  for (let index = 0; index < result.frames; index++) {
    const name = pad(index) + ".png";
    work.push(track(loadImage(`${base}/original/${name}`)).then((image) => { images.original[index] = image; }));
    work.push(track(loadImage(`${base}/${bundle.id}/nvc/${name}${stamp}`)).then((image) => { images.nvc[index] = image; }));
    if (haveH264) {
      work.push(track(loadImage(`${base}/${bundle.id}/h264/${name}${stamp}`)).then((image) => { images.h264[index] = image; }));
    }
    if (result.nvc.frame_types[index] === "P") {
      work.push(loadImage(`${base}/${bundle.id}/pred/${name}${stamp}`).then((image) => { images.prediction[index] = image; }));
    }
  }
  await Promise.all(work);
  if (!state.busy) show("job", false);
  return images;
}

// ---------------------------------------------------------------- rendering

function renderStatic() {
  const result = state.result, nvc = result.nvc, h264 = result.h264;
  $("meta-original").textContent = `24.000 bits per pixel · ${bytes(result.raw_bytes)}`;
  $("meta-nvc").textContent = `${fixed(nvc.bpp, 3)} bpp · ${fixed(nvc.psnr, 2)} dB · MS-SSIM ${fixed(nvc.msssim, 4)}`;
  $("meta-h264").textContent = h264.available
    ? `${fixed(h264.bpp, 3)} bpp · ${fixed(h264.psnr, 2)} dB · MS-SSIM ${fixed(h264.msssim, 4)}`
    : h264.reason;

  const cards = [
    { label: "Raw video", value: bytes(result.raw_bytes), note: `${result.frames} frames, ${result.width}×${result.height}` },
    { label: "Neural codec stream", value: bytes(nvc.total_bytes), note: `${(result.raw_bytes / nvc.total_bytes).toFixed(1)}× smaller than raw` },
  ];
  if (h264.available) {
    cards.push({
      label: "H.264 file", value: bytes(h264.total_bytes),
      note: h264.within_target ? `CRF ${h264.crf}, the best quality that fits` : `CRF ${h264.crf}; could not go as low as our stream`,
    });
    if (nvc.psnr !== null && h264.psnr !== null) {
      const gap = h264.psnr - nvc.psnr;
      cards.push({
        label: "PSNR at this bitrate", value: `${gap >= 0 ? "H.264 +" : "Ours +"}${Math.abs(gap).toFixed(2)} dB`,
        note: `${fixed(nvc.psnr, 2)} dB ours, ${fixed(h264.psnr, 2)} dB H.264`,
      });
    }
  }
  cards.push({
    label: "Decoder check", value: nvc.decoder_exact ? "Bit-exact" : "Mismatch",
    note: "Decoded stream against the encoder's frames", kind: nvc.decoder_exact ? "good" : "bad",
  });
  cards.push({
    label: "Speed per frame", value: `${Math.round(nvc.encode_ms_per_frame)} ms encode`,
    note: `${Math.round(nvc.decode_ms_per_frame)} ms decode, on ${result.device}`,
  });
  const stats = $("stats");
  stats.innerHTML = "";
  for (const card of cards) {
    const box = document.createElement("div");
    box.className = "stat" + (card.kind ? " " + card.kind : "");
    for (const [name, text] of [["label", card.label], ["value", card.value], ["note", card.note]]) {
      const line = document.createElement("div");
      line.className = name;
      line.textContent = text;
      box.appendChild(line);
    }
    stats.appendChild(box);
  }
  $("footer-status").textContent =
    `${result.bundle.label} · GOP ${result.bundle.gop_size}, ${result.bundle.block_size}×${result.bundle.block_size} blocks, ±${result.bundle.search_range} px search · ${result.fps.toFixed(0)} fps`;
}

function drawPane(id, image) {
  const context = $(id).getContext("2d");
  context.clearRect(0, 0, 256, 256);
  if (image) context.drawImage(image, 0, 0, 256, 256);
}

function render() {
  const result = state.result;
  if (!result) return;
  const index = state.frame, nvc = result.nvc, images = state.images;
  drawPane("pane-original", images.original[index]);
  drawPane("pane-nvc", images.nvc[index]);
  drawPane("pane-h264", images.h264[index]);

  const type = nvc.frame_types[index];
  const frameBytes = nvc.motion_bytes[index] + nvc.residual_bytes[index];
  $("frame-label").textContent = `Frame ${index + 1} of ${result.frames} · ${type}-frame`;
  $("scrub").value = index;
  $("frame-nvc").textContent = `This frame: ${bytes(frameBytes)}, ${fixed(nvc.psnr_per_frame[index], 2)} dB`;
  $("frame-h264").textContent = result.h264.available
    ? `This frame: ${fixed(result.h264.psnr_per_frame[index], 2)} dB` : "";

  $("inside-title").textContent = `Inside frame ${index + 1}`;
  const raw = result.width * result.height * 3;
  $("inside-summary").textContent = type === "I"
    ? `I-frame: coded on its own, with no prediction. ${bytes(frameBytes)} against ${bytes(raw)} raw.`
    : `P-frame: ${bytes(nvc.motion_bytes[index])} of motion vectors and ${bytes(nvc.residual_bytes[index])} of residual, against ${bytes(raw)} raw.`;

  drawMotion(index);
  drawPrediction(index);
  drawResidual(index);
  drawError(index);
  drawBars();
}

function stepContext(id) {
  const canvas = $(id);
  if (canvas.width !== 256 * STEP_SCALE) { canvas.width = 256 * STEP_SCALE; canvas.height = 256 * STEP_SCALE; }
  const context = canvas.getContext("2d");
  context.setTransform(STEP_SCALE, 0, 0, STEP_SCALE, 0, 0);
  context.clearRect(0, 0, 256, 256);
  return context;
}

function placeholder(context, text) {
  context.fillStyle = CARD;
  context.fillRect(0, 0, 256, 256);
  context.fillStyle = MUTED;
  context.font = "13px 'Segoe UI', system-ui, sans-serif";
  context.textAlign = "center";
  context.fillText(text, 128, 132);
}

function drawMotion(index) {
  const context = stepContext("step-motion");
  const motion = state.result.nvc.motion[index];
  if (!motion) { placeholder(context, "I-frame: no motion search"); return; }
  const block = state.result.bundle.block_size;
  context.drawImage(state.images.original[index], 0, 0, 256, 256);
  context.fillStyle = "rgba(14, 42, 51, 0.6)";
  context.fillRect(0, 0, 256, 256);
  context.strokeStyle = AMBER;
  context.fillStyle = AMBER;
  context.lineWidth = 1.2;
  const [dy, dx] = motion;
  for (let row = 0; row < dy.length; row++) {
    for (let column = 0; column < dy[row].length; column++) {
      const centerX = column * block + block / 2, centerY = row * block + block / 2;
      const offsetX = dx[row][column], offsetY = dy[row][column];
      if (offsetX === 0 && offsetY === 0) {
        context.globalAlpha = 0.45;
        context.fillRect(centerX - 0.75, centerY - 0.75, 1.5, 1.5);
        context.globalAlpha = 1;
        continue;
      }
      // The block was fetched from (center + offset); the arrow shows the content moving here.
      const fromX = centerX + offsetX, fromY = centerY + offsetY;
      context.beginPath();
      context.moveTo(fromX, fromY);
      context.lineTo(centerX, centerY);
      context.stroke();
      const angle = Math.atan2(centerY - fromY, centerX - fromX);
      context.beginPath();
      context.moveTo(centerX, centerY);
      context.lineTo(centerX - 4 * Math.cos(angle - 0.5), centerY - 4 * Math.sin(angle - 0.5));
      context.lineTo(centerX - 4 * Math.cos(angle + 0.5), centerY - 4 * Math.sin(angle + 0.5));
      context.closePath();
      context.fill();
    }
  }
}

function drawPrediction(index) {
  const context = stepContext("step-prediction");
  const image = state.images.prediction[index];
  if (!image) { placeholder(context, "I-frame: nothing to predict from"); return; }
  context.drawImage(image, 0, 0, 256, 256);
}

function heat(value) {
  // ink -> teal -> amber -> near white, for 0..1
  const stops = [[14, 42, 51], [15, 118, 110], [232, 163, 61], [255, 244, 220]];
  const scaled = Math.min(0.9999, Math.max(0, value)) * (stops.length - 1);
  const low = Math.floor(scaled), mix = scaled - low;
  const channel = (k) => Math.round(stops[low][k] + (stops[low + 1][k] - stops[low][k]) * mix);
  return `rgb(${channel(0)}, ${channel(1)}, ${channel(2)})`;
}

function drawResidual(index) {
  const context = stepContext("step-residual");
  const activity = state.result.nvc.activity[index];
  if (!activity) { placeholder(context, "I-frame: no residual"); return; }
  const rows = activity.length, columns = activity[0].length;
  const cellWidth = 256 / columns, cellHeight = 256 / rows;
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      context.fillStyle = heat(Math.sqrt(activity[row][column] / state.activityMax));
      context.fillRect(column * cellWidth, row * cellHeight, cellWidth + 0.5, cellHeight + 0.5);
    }
  }
}

const scratch = document.createElement("canvas");
scratch.width = 256; scratch.height = 256;
const scratchContext = scratch.getContext("2d", { willReadFrequently: true });
function pixels(image) {
  scratchContext.drawImage(image, 0, 0, 256, 256);
  return scratchContext.getImageData(0, 0, 256, 256);
}

function drawError(index) {
  const context = stepContext("step-error");
  const original = pixels(state.images.original[index]).data;
  const decodedData = pixels(state.images.nvc[index]);
  const decoded = decodedData.data;
  for (let offset = 0; offset < decoded.length; offset += 4) {
    const difference = (Math.abs(original[offset] - decoded[offset])
      + Math.abs(original[offset + 1] - decoded[offset + 1])
      + Math.abs(original[offset + 2] - decoded[offset + 2])) / 3;
    const level = Math.min(255, difference * 8);
    decoded[offset] = level; decoded[offset + 1] = level; decoded[offset + 2] = level; decoded[offset + 3] = 255;
  }
  scratchContext.putImageData(decodedData, 0, 0);
  context.drawImage(scratch, 0, 0, 256, 256);
}

function drawBars() {
  const result = state.result;
  if (!result) return;
  const canvas = $("bars");
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth, height = canvas.clientHeight;
  if (canvas.width !== Math.round(width * ratio)) { canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio); }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  const nvc = result.nvc, count = result.frames;
  const totals = nvc.motion_bytes.map((motion, index) => motion + nvc.residual_bytes[index]);
  const peak = Math.max(...totals, 1);
  // The bottom strip is kept for the current-frame marker.
  const marker = 10, floor = height - marker;
  const slot = width / count, barWidth = Math.max(1, slot - 2), usable = floor - 4;
  for (let index = 0; index < count; index++) {
    const x = index * slot + 1;
    const residualHeight = usable * nvc.residual_bytes[index] / peak;
    const motionHeight = usable * nvc.motion_bytes[index] / peak;
    context.globalAlpha = index === state.frame ? 1 : 0.8;
    context.fillStyle = nvc.frame_types[index] === "I" ? INK : TEAL;
    context.fillRect(x, floor - residualHeight, barWidth, residualHeight);
    if (motionHeight > 0) {
      context.fillStyle = AMBER;
      context.fillRect(x, floor - residualHeight - motionHeight, barWidth, motionHeight);
    }
    context.globalAlpha = 1;
    if (index === state.frame) {
      const middle = x + barWidth / 2;
      context.fillStyle = INK;
      context.beginPath();
      context.moveTo(middle, floor + 2);
      context.lineTo(middle - 5, height);
      context.lineTo(middle + 5, height);
      context.closePath();
      context.fill();
    }
  }
}

// ---------------------------------------------------------------- transport

function seek(index) {
  if (!state.result) return;
  const count = state.result.frames;
  state.frame = ((index % count) + count) % count;
  render();
}
function stop() {
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
  state.playing = false;
  $("play").textContent = "Play";
}
function play() {
  if (!state.result) return;
  state.playing = true;
  $("play").textContent = "Pause";
  const interval = 1000 / Math.min(30, Math.max(1, state.result.fps));
  state.timer = setInterval(() => seek(state.frame + 1), interval);
}

// ---------------------------------------------------------------- jobs

const STAGES = {
  queued: ["Waiting for the worker", 0.0, 0.02],
  ingest: ["Reading the video", 0.02, 0.1],
  encode: ["Encoding", 0.1, 0.7],
  decode: ["Decoding the stream and checking it", 0.7, 0.85],
  h264: ["Encoding H.264 at the same bitrate", 0.85, 0.95],
  score: ["Scoring frames", 0.95, 1.0],
  done: ["Done", 1.0, 1.0],
};

function jobPanel(title, detail, fraction) {
  $("job-title").textContent = title;
  $("job-detail").textContent = detail || "";
  $("job-bar").style.width = Math.round(100 * Math.min(1, Math.max(0, fraction))) + "%";
  show("job", true);
}

function setBusy(busy) {
  state.busy = busy;
  $("upload-button").disabled = busy;
  $("pending-button").disabled = busy;
  if (busy) show("pending", false);
}

async function followJob(jobId, heading) {
  setBusy(true);
  try {
    for (;;) {
      const job = await api(`/api/jobs/${jobId}`);
      if (job.status === "error") throw new Error(job.error || "The job failed.");
      const [label, start, end] = STAGES[job.stage] || STAGES.queued;
      const within = job.total > 0 ? job.done / job.total : 0;
      const detail = job.stage === "encode" && job.total ? `${label} frame ${job.done} of ${job.total}` : label;
      jobPanel(heading, detail, start + (end - start) * within);
      if (job.status === "done") return job;
      await new Promise((resolve) => setTimeout(resolve, 400));
    }
  } finally {
    setBusy(false);
    show("job", false);
  }
}

async function encodeCurrent() {
  const clip = currentClip(), bundle = currentBundle();
  if (!clip || !bundle) return;
  try {
    const started = await api(`/api/clips/${clip.id}/results/${bundle.id}`, { method: "POST" });
    await followJob(started.job_id, `Encoding "${clip.name}" with ${bundle.label}`);
    await refreshLists();
    await loadView();
  } catch (error) {
    message(error.message, "error");
  }
}

async function uploadFile(file) {
  message("");
  const form = new FormData();
  form.append("file", file);
  form.append("bundle_id", state.bundleId || "");
  try {
    jobPanel(`Uploading "${file.name}"`, "", 0.01);
    const started = await api("/api/upload", { method: "POST", body: form });
    const job = await followJob(started.job_id, `Encoding "${file.name}"`);
    state.clipId = job.clip_id;
    state.bundleId = job.bundle_id;
    await refreshLists();
    await loadView();
  } catch (error) {
    show("job", false);
    message(error.message, "error");
  }
}

// ---------------------------------------------------------------- wiring

$("clip-select").addEventListener("change", (event) => { state.clipId = event.target.value; loadView(); });
$("bundle-select").addEventListener("change", (event) => { state.bundleId = event.target.value; loadView(); });
$("play").addEventListener("click", () => (state.playing ? stop() : play()));
$("prev").addEventListener("click", () => { stop(); seek(state.frame - 1); });
$("next").addEventListener("click", () => { stop(); seek(state.frame + 1); });
$("scrub").addEventListener("input", (event) => { stop(); seek(Number(event.target.value)); });
$("bars").addEventListener("click", (event) => {
  if (!state.result) return;
  const box = event.currentTarget.getBoundingClientRect();
  stop();
  seek(Math.min(state.result.frames - 1, Math.floor((event.clientX - box.left) / box.width * state.result.frames)));
});
$("pending-button").addEventListener("click", encodeCurrent);
$("upload-button").addEventListener("click", () => $("upload-input").click());
$("upload-input").addEventListener("change", (event) => {
  const file = event.target.files[0];
  event.target.value = "";
  if (file) uploadFile(file);
});
window.addEventListener("resize", drawBars);
document.addEventListener("keydown", (event) => {
  if (["INPUT", "SELECT", "BUTTON"].includes(event.target.tagName) || !state.result) return;
  if (event.key === " ") { event.preventDefault(); state.playing ? stop() : play(); }
  if (event.key === "ArrowLeft") { stop(); seek(state.frame - 1); }
  if (event.key === "ArrowRight") { stop(); seek(state.frame + 1); }
});

(async function start() {
  try {
    await refreshLists();
    await loadView();
  } catch (error) {
    message(error.message, "error");
  }
})();
