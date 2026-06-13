const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay ? overlay.getContext("2d") : null;
const startCameraBtn = document.getElementById("startCamera");
const toggleRecognizeBtn = document.getElementById("toggleRecognize");
const mirrorCameraInput = document.getElementById("mirrorCamera");
const registerBtn = document.getElementById("registerBtn");
const refreshFacesBtn = document.getElementById("refreshFaces");
const nameInput = document.getElementById("nameInput");
const dbStatus = document.getElementById("dbStatus");
const cameraStatus = document.getElementById("cameraStatus");
const faceList = document.getElementById("faceList");
const resultList = document.getElementById("resultList");
const progressBar = document.getElementById("progressBar");
const registerStatus = document.getElementById("registerStatus");
const pipelineStatus = document.getElementById("pipelineStatus");
const stageOriginal = document.getElementById("stageOriginal");
const stageOriginalNote = document.getElementById("stageOriginalNote");
const stageGray = document.getElementById("stageGray");
const stageResize = document.getElementById("stageResize");
const stageResult = document.getElementById("stageResult");
const rgbHistogram = document.getElementById("rgbHistogram");
const histogramStatus = document.getElementById("histogramStatus");
const mathStatus = document.getElementById("mathStatus");
const metricResolution = document.getElementById("metricResolution");
const metricPixels = document.getElementById("metricPixels");
const metricResize = document.getElementById("metricResize");
const metricReduction = document.getElementById("metricReduction");
const metricFps = document.getElementById("metricFps");
const metricFaces = document.getElementById("metricFaces");
const metricAiFps = document.getElementById("metricAiFps");
const metricProcessTime = document.getElementById("metricProcessTime");
const formulaGray = document.getElementById("formulaGray");
const formulaNormalize = document.getElementById("formulaNormalize");
const formulaResize = document.getElementById("formulaResize");
const formulaDistance = document.getElementById("formulaDistance");
const formulaCosine = document.getElementById("formulaCosine");
const formulaDecision = document.getElementById("formulaDecision");

let cameraReady = false;
let recognizing = false;
let recognizeTimer = null;
let recognizeRequestRunning = false;
let pipelineTimer = null;
let lastFaces = [];
let displayFaces = [];
let overlayAnimationId = null;
let lastPipelineAt = 0;
let currentPipelineFps = 0;
let lastRecognitionMs = 0;
let lastAiFps = 0;
let lastProcessTime = 0;
let mirrorCamera = mirrorCameraInput ? mirrorCameraInput.checked : true;
const TRACKING_SMOOTHING = 0.72;
const RECOGNITION_DELAY_MS = 280;
const PIPELINE_DELAY_MS = 500;

function setText(element, value) {
  if (element) {
    element.textContent = value;
  }
}

function setHtml(element, value) {
  if (element) {
    element.innerHTML = value;
  }
}

function setProgress(value, total) {
  const pct = Math.round((value / total) * 100);
  if (progressBar) progressBar.style.width = `${pct}%`;
  setText(registerStatus, `${value}/${total} sample`);
}

function captureFrame(quality = 0.82) {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const captureCtx = canvas.getContext("2d");
  drawVideoFrame(captureCtx, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", quality);
}

function drawVideoFrame(canvasCtx, width, height) {
  canvasCtx.save();
  if (mirrorCamera) {
    canvasCtx.translate(width, 0);
    canvasCtx.scale(-1, 1);
  }
  canvasCtx.drawImage(video, 0, 0, width, height);
  canvasCtx.restore();
}

function updateMirrorMode() {
  if (!mirrorCameraInput || !video) return;
  mirrorCamera = mirrorCameraInput.checked;
  video.classList.toggle("is-mirrored", mirrorCamera);
  setText(stageOriginalNote, mirrorCamera ? "Frame masuk mirror" : "Frame masuk tanpa mirror");
  if (cameraReady) {
    drawFaces(displayFaces);
    drawPipelineStages();
  }
}

function setCanvasSize(canvas, width, height) {
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
}

function formatNumber(value) {
  return new Intl.NumberFormat("id-ID").format(value);
}

function getCenterSample(canvasCtx, width, height) {
  const x = Math.max(0, Math.floor(width / 2));
  const y = Math.max(0, Math.floor(height / 2));
  const data = canvasCtx.getImageData(x, y, 1, 1).data;
  const gray = Math.round(0.299 * data[0] + 0.587 * data[1] + 0.114 * data[2]);
  return { r: data[0], g: data[1], b: data[2], gray };
}

function buildRgbHistogram(imageData) {
  const red = new Array(256).fill(0);
  const green = new Array(256).fill(0);
  const blue = new Array(256).fill(0);

  for (let i = 0; i < imageData.data.length; i += 4) {
    red[imageData.data[i]] += 1;
    green[imageData.data[i + 1]] += 1;
    blue[imageData.data[i + 2]] += 1;
  }

  return { red, green, blue };
}

function drawHistogramChannel(canvasCtx, values, width, height, color, maxValue) {
  canvasCtx.beginPath();
  values.forEach((value, index) => {
    const x = (index / 255) * width;
    const y = height - (value / maxValue) * height;
    if (index === 0) {
      canvasCtx.moveTo(x, y);
      return;
    }
    canvasCtx.lineTo(x, y);
  });
  canvasCtx.strokeStyle = color;
  canvasCtx.lineWidth = 2;
  canvasCtx.stroke();
}

function drawRgbHistogram(imageData, width, height) {
  if (!rgbHistogram) return;

  const histogramCtx = rgbHistogram.getContext("2d");
  const histogram = buildRgbHistogram(imageData);
  const maxValue = Math.max(1, ...histogram.red, ...histogram.green, ...histogram.blue);
  const pixelRatio = window.devicePixelRatio || 1;
  const displayWidth = rgbHistogram.clientWidth || rgbHistogram.width;
  const displayHeight = rgbHistogram.clientHeight || rgbHistogram.height;

  if (rgbHistogram.width !== Math.round(displayWidth * pixelRatio)) {
    rgbHistogram.width = Math.round(displayWidth * pixelRatio);
  }
  if (rgbHistogram.height !== Math.round(displayHeight * pixelRatio)) {
    rgbHistogram.height = Math.round(displayHeight * pixelRatio);
  }

  histogramCtx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  histogramCtx.clearRect(0, 0, displayWidth, displayHeight);
  histogramCtx.fillStyle = "#ffffff";
  histogramCtx.fillRect(0, 0, displayWidth, displayHeight);

  const padding = { top: 14, right: 14, bottom: 26, left: 34 };
  const chartWidth = displayWidth - padding.left - padding.right;
  const chartHeight = displayHeight - padding.top - padding.bottom;

  histogramCtx.save();
  histogramCtx.translate(padding.left, padding.top);
  histogramCtx.strokeStyle = "#d7dde4";
  histogramCtx.lineWidth = 1;
  histogramCtx.beginPath();
  for (let i = 0; i <= 4; i += 1) {
    const y = (chartHeight / 4) * i;
    histogramCtx.moveTo(0, y);
    histogramCtx.lineTo(chartWidth, y);
  }
  histogramCtx.stroke();

  histogramCtx.globalAlpha = 0.9;
  drawHistogramChannel(histogramCtx, histogram.red, chartWidth, chartHeight, "#d73535", maxValue);
  drawHistogramChannel(histogramCtx, histogram.green, chartWidth, chartHeight, "#159447", maxValue);
  drawHistogramChannel(histogramCtx, histogram.blue, chartWidth, chartHeight, "#2468d8", maxValue);
  histogramCtx.restore();

  histogramCtx.fillStyle = "#627080";
  histogramCtx.font = "12px Arial";
  histogramCtx.fillText("0", padding.left, displayHeight - 8);
  histogramCtx.fillText("128", padding.left + chartWidth / 2 - 10, displayHeight - 8);
  histogramCtx.fillText("255", padding.left + chartWidth - 20, displayHeight - 8);
  histogramCtx.fillText("Piksel", 6, 18);

  setText(histogramStatus, `${width}x${height} | 256 bin`);
}

function updateMathPanel(sample, stageWidth, stageHeight, halfWidth, halfHeight) {
  const fullPixels = video.videoWidth * video.videoHeight;
  const processPixels = Math.round(fullPixels * 0.5 * 0.5);
  const reduction = fullPixels ? Math.round((1 - processPixels / fullPixels) * 100) : 0;
  const bestFace = lastFaces.find((face) => face.distance !== null) || lastFaces[0];
  const distance = bestFace && bestFace.distance !== null ? bestFace.distance : null;
  const cosine = bestFace && bestFace.cosine_similarity !== null ? bestFace.cosine_similarity : null;
  const threshold = bestFace && bestFace.threshold ? bestFace.threshold : 0.5;
  const decision = distance === null ? "menunggu wajah" : distance <= threshold ? "dikenal" : "tidak dikenal";
  const normalizedGray = sample.gray / 255;

  setText(metricResolution, `${video.videoWidth} x ${video.videoHeight}`);
  setText(metricPixels, formatNumber(fullPixels));
  setText(metricResize, `${stageWidth}x${stageHeight} -> ${halfWidth}x${halfHeight}`);
  setText(metricReduction, `${reduction}% lebih ringan`);
  setText(metricFps, `${currentPipelineFps.toFixed(1)} fps`);
  setText(metricFaces, String(lastFaces.length));
  setText(metricAiFps, lastAiFps ? `${lastAiFps.toFixed(2)} fps` : "-");
  setText(metricProcessTime, lastProcessTime ? `${lastProcessTime.toFixed(4)}s` : "-");
  setText(formulaGray, `Y = 0.299(${sample.r}) + 0.587(${sample.g}) + 0.114(${sample.b}) = ${sample.gray}`);
  setText(formulaNormalize, `I' = ${sample.gray} / 255 = ${normalizedGray.toFixed(4)}`);
  setText(formulaResize, `W' = 0.5W, H' = 0.5H, piksel proses = ${formatNumber(processPixels)}`);
  setText(formulaDistance, distance === null ? "d = sqrt(sum((encoding_db - encoding_frame)^2))" : `d terbaik = ${distance}`);
  setText(formulaCosine, cosine === null ? "cos(theta) = (A . B) / (|A| |B|)" : `cos(theta) terbaik = ${Number(cosine).toFixed(4)}`);
  setText(formulaDecision, distance === null ? `d <= ${threshold} -> dikenal` : `${distance} <= ${threshold} -> ${decision}`);
  setText(mathStatus, lastRecognitionMs ? `AI ${lastRecognitionMs} ms` : "Realtime");
}

function drawLabel(canvasCtx, text, x, y, color = "#0b7a75") {
  canvasCtx.font = "15px Arial";
  const labelWidth = canvasCtx.measureText(text).width + 14;
  canvasCtx.fillStyle = color;
  canvasCtx.fillRect(x, Math.max(0, y - 27), labelWidth, 27);
  canvasCtx.fillStyle = "#fff";
  canvasCtx.fillText(text, x + 7, Math.max(18, y - 8));
}

function drawDetectedFaces(canvasCtx, canvas, faces) {
  const scaleX = canvas.width / video.videoWidth;
  const scaleY = canvas.height / video.videoHeight;

  faces.forEach((face) => {
    const box = face.box;
    const left = box.left * scaleX;
    const top = box.top * scaleY;
    const width = (box.right - box.left) * scaleX;
    const height = (box.bottom - box.top) * scaleY;
    const known = face.name !== "Tidak Dikenal";
    const color = known ? "#0b7a75" : "#bf2f36";
    const distance = face.distance === null ? "-" : Number(face.distance).toFixed(3);
    const label = known ? `${face.name} | d=${distance}` : `${face.status} | d=${distance}`;

    canvasCtx.strokeStyle = color;
    canvasCtx.lineWidth = 3;
    canvasCtx.strokeRect(left, top, width, height);
    drawLabel(canvasCtx, label, left, top, color);
  });
}

function centerOf(face) {
  const box = face.box;
  return {
    x: (box.left + box.right) / 2,
    y: (box.top + box.bottom) / 2,
  };
}

function updateTrackedFaces(faces) {
  const now = performance.now();
  const used = new Set();

  faces.forEach((face) => {
    const faceCenter = centerOf(face);
    let bestIndex = -1;
    let bestDistance = Infinity;

    displayFaces.forEach((tracked, index) => {
      if (used.has(index)) return;
      const trackedCenter = centerOf(tracked);
      const distance = Math.hypot(faceCenter.x - trackedCenter.x, faceCenter.y - trackedCenter.y);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestIndex = index;
      }
    });

    if (bestIndex >= 0 && bestDistance < 180) {
      const tracked = displayFaces[bestIndex];
      tracked.targetBox = { ...face.box };
      tracked.name = face.name;
      tracked.status = face.status;
      tracked.confidence = face.confidence;
      tracked.distance = face.distance;
      tracked.cosine_similarity = face.cosine_similarity;
      tracked.threshold = face.threshold;
      tracked.lastSeen = now;
      used.add(bestIndex);
      return;
    }

    displayFaces.push({
      ...face,
      box: { ...face.box },
      targetBox: { ...face.box },
      lastSeen: now,
    });
  });

  displayFaces = displayFaces.filter((face) => now - face.lastSeen < 1400);
}

function animateOverlay() {
  if (!cameraReady) return;
  const now = performance.now();
  displayFaces = displayFaces.filter((face) => now - face.lastSeen < 1400);
  displayFaces.forEach((face) => {
    ["top", "right", "bottom", "left"].forEach((key) => {
      face.box[key] += (face.targetBox[key] - face.box[key]) * TRACKING_SMOOTHING;
    });
  });
  drawFaces(displayFaces);
  overlayAnimationId = requestAnimationFrame(animateOverlay);
}

function drawPipelineStages() {
  if (!cameraReady || !video.videoWidth || !video.videoHeight) return;

  const now = performance.now();
  if (lastPipelineAt) {
    currentPipelineFps = 1000 / Math.max(1, now - lastPipelineAt);
  }
  lastPipelineAt = now;

  const stageWidth = 320;
  const stageHeight = Math.round(stageWidth * (video.videoHeight / video.videoWidth));
  const halfWidth = Math.max(1, Math.round(stageWidth * 0.5));
  const halfHeight = Math.max(1, Math.round(stageHeight * 0.5));

  [stageOriginal, stageGray, stageResize, stageResult].forEach((canvas) => {
    setCanvasSize(canvas, stageWidth, stageHeight);
  });

  const originalCtx = stageOriginal.getContext("2d");
  drawVideoFrame(originalCtx, stageWidth, stageHeight);
  const centerSample = getCenterSample(originalCtx, stageWidth, stageHeight);
  const originalFrame = originalCtx.getImageData(0, 0, stageWidth, stageHeight);
  drawRgbHistogram(originalFrame, stageWidth, stageHeight);

  const grayCtx = stageGray.getContext("2d");
  drawVideoFrame(grayCtx, stageWidth, stageHeight);
  const grayFrame = grayCtx.getImageData(0, 0, stageWidth, stageHeight);
  for (let i = 0; i < grayFrame.data.length; i += 4) {
    const r = grayFrame.data[i];
    const g = grayFrame.data[i + 1];
    const b = grayFrame.data[i + 2];
    const y = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
    grayFrame.data[i] = y;
    grayFrame.data[i + 1] = y;
    grayFrame.data[i + 2] = y;
  }
  grayCtx.putImageData(grayFrame, 0, 0);

  const resizeCtx = stageResize.getContext("2d");
  const temp = document.createElement("canvas");
  temp.width = halfWidth;
  temp.height = halfHeight;
  const tempCtx = temp.getContext("2d");
  drawVideoFrame(tempCtx, halfWidth, halfHeight);
  resizeCtx.imageSmoothingEnabled = false;
  resizeCtx.clearRect(0, 0, stageWidth, stageHeight);
  resizeCtx.drawImage(temp, 0, 0, halfWidth, halfHeight, 0, 0, stageWidth, stageHeight);
  resizeCtx.imageSmoothingEnabled = true;

  const resultCtx = stageResult.getContext("2d");
  drawVideoFrame(resultCtx, stageWidth, stageHeight);
  drawDetectedFaces(resultCtx, stageResult, displayFaces.length ? displayFaces : lastFaces);
  setText(pipelineStatus, `CCTV 30 FPS | ${video.videoWidth}x${video.videoHeight} -> ${halfWidth}x${halfHeight}`);
  updateMathPanel(centerSample, stageWidth, stageHeight, halfWidth, halfHeight);
}

function resizeOverlay() {
  overlay.width = video.clientWidth;
  overlay.height = video.clientHeight;
}

function drawFaces(faces) {
  if (!ctx) return;
  resizeOverlay();
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  drawDetectedFaces(ctx, overlay, faces);
}

function renderResults(faces) {
  if (!faces.length) {
    setHtml(resultList, `<div class="empty-state">Belum ada wajah terdeteksi.</div>`);
    return;
  }

  setHtml(resultList, faces
    .map((face) => {
      const confidence = face.confidence ? `${face.confidence}%` : "-";
      const distance = face.distance === null ? "-" : Number(face.distance).toFixed(4);
      const threshold = Number(face.threshold).toFixed(2);
      const cosine = face.cosine_similarity === null ? "-" : Number(face.cosine_similarity).toFixed(4);
      return `
        <div class="result-row">
          <div>
            <strong>Nama: ${face.name}</strong><br>
            <small>Status: ${face.status} | Distance: ${distance} | Threshold: ${threshold}</small><br>
            <small>Cosine: ${cosine} | Akurasi ${confidence}</small>
          </div>
        </div>
      `;
    })
    .join(""));
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok || data.ok === false) {
    throw new Error(data.error || "Request gagal.");
  }
  return data;
}

async function refreshStatus() {
  try {
    const data = await api("/api/status");
    setText(dbStatus, `Database ${data.database} aktif di ${data.host} | ${data.count} encoding`);
  } catch (error) {
    setText(dbStatus, `Database belum aktif: ${error.message}`);
  }
}

async function refreshFaces() {
  try {
    const data = await api("/api/faces");
    if (!data.faces.length) {
      setHtml(faceList, `<div class="empty-state">Database kosong.</div>`);
      return;
    }
    setHtml(faceList, data.faces
      .map(
        (face) => `
          <div class="face-row">
            <div>
              <strong>${face.name}</strong><br>
              <small>${face.samples} encoding</small>
            </div>
            <button class="delete-btn" type="button" data-name="${face.name}">Hapus</button>
          </div>
        `
      )
      .join(""));
  } catch (error) {
    setHtml(faceList, `<div class="error-state">${error.message}</div>`);
  }
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 640 },
      height: { ideal: 480 },
      frameRate: { ideal: 30, max: 30 },
      facingMode: "user",
    },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  cameraReady = true;
  setText(cameraStatus, "Kamera aktif");
  if (startCameraBtn) startCameraBtn.disabled = true;
  if (toggleRecognizeBtn) toggleRecognizeBtn.disabled = false;
  if (registerBtn) registerBtn.disabled = false;
  resizeOverlay();
  drawPipelineStages();
  pipelineTimer = setInterval(drawPipelineStages, PIPELINE_DELAY_MS);
  if (!overlayAnimationId) {
    overlayAnimationId = requestAnimationFrame(animateOverlay);
  }
}

async function recognizeOnce() {
  if (!cameraReady || !recognizing || recognizeRequestRunning) return;
  recognizeRequestRunning = true;
  try {
    const startedAt = performance.now();
    const data = await api("/api/recognize", {
      method: "POST",
      body: JSON.stringify({ image: captureFrame() }),
    });
    lastRecognitionMs = Math.round(performance.now() - startedAt);
    lastAiFps = data.fps || 0;
    lastProcessTime = data.process_time || 0;
    lastFaces = data.faces;
    updateTrackedFaces(data.faces);
    renderResults(data.faces);
    drawPipelineStages();
  } catch (error) {
    setHtml(resultList, `<div class="error-state">${error.message}</div>`);
  } finally {
    recognizeRequestRunning = false;
  }
}

async function recognizeLoop() {
  if (!recognizing) return;
  await recognizeOnce();
  recognizeTimer = setTimeout(recognizeLoop, RECOGNITION_DELAY_MS);
}

function toggleRecognize() {
  recognizing = !recognizing;
  setText(toggleRecognizeBtn, recognizing ? "Stop Deteksi" : "Mulai Deteksi");
  if (recognizing) {
    recognizeLoop();
  } else {
    clearTimeout(recognizeTimer);
    lastFaces = [];
    displayFaces = [];
    if (ctx) ctx.clearRect(0, 0, overlay.width, overlay.height);
    drawPipelineStages();
  }
}

async function registerFace() {
  const name = nameInput ? nameInput.value.trim() : "";
  if (!name || !cameraReady) return;

  if (registerBtn) registerBtn.disabled = true;
  setProgress(0, 5);
  const images = [];
  for (let i = 0; i < 5; i += 1) {
    await new Promise((resolve) => setTimeout(resolve, 500));
    images.push(captureFrame(0.9));
    setProgress(i + 1, 5);
  }

  try {
    const data = await api("/api/register", {
      method: "POST",
      body: JSON.stringify({ name, images }),
    });
    setText(registerStatus, `${data.name} tersimpan dari ${data.samples} sample`);
    if (nameInput) nameInput.value = "";
    await refreshStatus();
    await refreshFaces();
  } catch (error) {
    setText(registerStatus, error.message);
  } finally {
    if (registerBtn) registerBtn.disabled = false;
  }
}

if (startCameraBtn) startCameraBtn.addEventListener("click", startCamera);
if (toggleRecognizeBtn) toggleRecognizeBtn.addEventListener("click", toggleRecognize);
if (registerBtn) registerBtn.addEventListener("click", registerFace);
if (refreshFacesBtn) refreshFacesBtn.addEventListener("click", refreshFaces);
if (mirrorCameraInput) mirrorCameraInput.addEventListener("change", updateMirrorMode);
window.addEventListener("resize", resizeOverlay);

if (faceList) {
  faceList.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-name]");
    if (!button) return;
    await api(`/api/faces/${encodeURIComponent(button.dataset.name)}`, { method: "DELETE" });
    await refreshStatus();
    await refreshFaces();
  });
}

refreshStatus();
refreshFaces();
updateMirrorMode();
