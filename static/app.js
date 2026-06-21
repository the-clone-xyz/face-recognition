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

// State utama aplikasi: kamera, loop recognition, data wajah terakhir, dan overlay tracking.
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
let previousOverlayAt = 0;
const UNKNOWN_FACE_NAME = "Tidak Dikenal";
const TRACKING_EASE = 0.22;
const TRACKING_TARGET_BLEND = 0.6;
const TRACKING_DEADZONE_PX = 4;
const TRACKING_KEEP_MS = 6000;
const TRACKING_FADE_AFTER_MS = 3800;
const TRACKING_MAX_PREDICT_MS = 1100;
const TRACKING_MAX_VELOCITY = 6;
const TRACKING_LABEL_STICKY_MS = 1800;
const RECOGNITION_DELAY_MS = 80;
const PIPELINE_DELAY_MS = 500;

// Helper DOM kecil agar update teks/HTML tidak perlu berulang mengecek null.
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
  // Frame kamera digambar ke canvas sementara, lalu dikirim ke Flask sebagai JPEG base64.
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const captureCtx = canvas.getContext("2d");
  drawVideoFrame(captureCtx, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", quality);
}

function drawVideoFrame(canvasCtx, width, height) {
  // Fungsi pusat untuk menggambar video; mode mirror diterapkan konsisten di preview dan capture.
  canvasCtx.save();
  if (mirrorCamera) {
    canvasCtx.translate(width, 0);
    canvasCtx.scale(-1, 1);
  }
  canvasCtx.drawImage(video, 0, 0, width, height);
  canvasCtx.restore();
}

function updateMirrorMode() {
  // Checkbox Mirror hanya mengubah arah tampilan, bukan struktur data database.
  if (!mirrorCameraInput || !video) return;
  mirrorCamera = mirrorCameraInput.checked;
  video.classList.toggle("is-mirrored", mirrorCamera);
  setText(
    stageOriginalNote,
    mirrorCamera ? "Frame masuk mirror" : "Frame masuk tanpa mirror",
  );
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
  // Histogram menghitung jumlah piksel untuk setiap intensitas 0-255 pada kanal R, G, dan B.
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

function drawHistogramChannel(
  canvasCtx,
  values,
  width,
  height,
  color,
  maxValue,
) {
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
  const maxValue = Math.max(
    1,
    ...histogram.red,
    ...histogram.green,
    ...histogram.blue,
  );
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
  drawHistogramChannel(
    histogramCtx,
    histogram.red,
    chartWidth,
    chartHeight,
    "#d73535",
    maxValue,
  );
  drawHistogramChannel(
    histogramCtx,
    histogram.green,
    chartWidth,
    chartHeight,
    "#159447",
    maxValue,
  );
  drawHistogramChannel(
    histogramCtx,
    histogram.blue,
    chartWidth,
    chartHeight,
    "#2468d8",
    maxValue,
  );
  histogramCtx.restore();

  histogramCtx.fillStyle = "#627080";
  histogramCtx.font = "12px Arial";
  histogramCtx.fillText("0", padding.left, displayHeight - 8);
  histogramCtx.fillText(
    "128",
    padding.left + chartWidth / 2 - 10,
    displayHeight - 8,
  );
  histogramCtx.fillText(
    "255",
    padding.left + chartWidth - 20,
    displayHeight - 8,
  );
  histogramCtx.fillText("Piksel", 6, 18);

  setText(histogramStatus, `${width}x${height} | 256 bin`);
}

function updateMathPanel(
  sample,
  stageWidth,
  stageHeight,
  halfWidth,
  halfHeight,
) {
  // Panel rumus menghubungkan nilai realtime dengan materi PCD yang dipresentasikan.
  const fullPixels = video.videoWidth * video.videoHeight;
  const processPixels = Math.round(fullPixels * 0.5 * 0.5);
  const reduction = fullPixels
    ? Math.round((1 - processPixels / fullPixels) * 100)
    : 0;
  const bestFace =
    displayFaces.find((face) => face.distance !== null) ||
    lastFaces.find((face) => face.distance !== null) ||
    displayFaces[0] ||
    lastFaces[0];
  const distance =
    bestFace && bestFace.distance !== null ? bestFace.distance : null;
  const cosine =
    bestFace && bestFace.cosine_similarity !== null
      ? bestFace.cosine_similarity
      : null;
  const threshold = bestFace && bestFace.threshold ? bestFace.threshold : 0.5;
  const decision =
    distance === null
      ? "menunggu wajah"
      : distance <= threshold
        ? "dikenal"
        : "tidak dikenal";
  const normalizedGray = sample.gray / 255;

  setText(metricResolution, `${video.videoWidth} x ${video.videoHeight}`);
  setText(metricPixels, formatNumber(fullPixels));
  setText(
    metricResize,
    `${stageWidth}x${stageHeight} -> ${halfWidth}x${halfHeight}`,
  );
  setText(metricReduction, `${reduction}% lebih ringan`);
  setText(metricFps, `${currentPipelineFps.toFixed(1)} fps`);
  const visibleTrackedFaces = displayFaces.filter(
    (face) => (face.opacity === undefined ? 1 : face.opacity) > 0.05,
  );
  setText(metricFaces, String(visibleTrackedFaces.length || lastFaces.length));
  setText(metricAiFps, lastAiFps ? `${lastAiFps.toFixed(2)} fps` : "-");
  setText(
    metricProcessTime,
    lastProcessTime ? `${lastProcessTime.toFixed(4)}s` : "-",
  );
  setText(
    formulaGray,
    `Y = 0.299(${sample.r}) + 0.587(${sample.g}) + 0.114(${sample.b}) = ${sample.gray}`,
  );
  setText(
    formulaNormalize,
    `I' = ${sample.gray} / 255 = ${normalizedGray.toFixed(4)}`,
  );
  setText(
    formulaResize,
    `W' = 0.5W, H' = 0.5H, piksel proses = ${formatNumber(processPixels)}`,
  );
  setText(
    formulaDistance,
    distance === null
      ? "d = sqrt(sum((encoding_db - encoding_frame)^2))"
      : `d terbaik = ${distance}`,
  );
  setText(
    formulaCosine,
    cosine === null
      ? "cos(theta) = (A . B) / (|A| |B|)"
      : `cos(theta) terbaik = ${Number(cosine).toFixed(4)}`,
  );
  setText(
    formulaDecision,
    distance === null
      ? `d <= ${threshold} -> dikenal`
      : `${distance} <= ${threshold} -> ${decision}`,
  );
  setText(
    mathStatus,
    lastRecognitionMs ? `AI ${lastRecognitionMs} ms` : "Realtime",
  );
}

function drawLabel(canvasCtx, text, x, y, color = "#0b7a75") {
  canvasCtx.font = "15px Arial";
  const labelWidth = canvasCtx.measureText(text).width + 14;
  const labelX = clamp(x, 0, Math.max(0, canvasCtx.canvas.width - labelWidth));
  const labelY = Math.max(0, y - 27);
  canvasCtx.fillStyle = color;
  canvasCtx.fillRect(labelX, labelY, labelWidth, 27);
  canvasCtx.fillStyle = "#fff";
  canvasCtx.fillText(text, labelX + 7, Math.max(18, labelY + 19));
}

function getVideoRenderRect(canvas) {
  // Video memakai object-fit: cover, jadi area render asli bisa lebih besar dari canvas.
  // Perhitungan ini menjaga kotak wajah tetap sejajar dengan wajah di layar.
  const canvasWidth = canvas.width;
  const canvasHeight = canvas.height;
  if (!video.videoWidth || !video.videoHeight || !canvasWidth || !canvasHeight) {
    return {
      x: 0,
      y: 0,
      scaleX: 1,
      scaleY: 1,
    };
  }

  const videoRatio = video.videoWidth / video.videoHeight;
  const canvasRatio = canvasWidth / canvasHeight;
  if (canvasRatio > videoRatio) {
    const renderHeight = canvasWidth / videoRatio;
    return {
      x: 0,
      y: (canvasHeight - renderHeight) / 2,
      scaleX: canvasWidth / video.videoWidth,
      scaleY: renderHeight / video.videoHeight,
    };
  }

  const renderWidth = canvasHeight * videoRatio;
  return {
    x: (canvasWidth - renderWidth) / 2,
    y: 0,
    scaleX: renderWidth / video.videoWidth,
    scaleY: canvasHeight / video.videoHeight,
  };
}

function drawDetectedFaces(canvasCtx, canvas, faces) {
  // Satu fungsi dipakai untuk overlay kamera utama dan preview hasil di pipeline.
  const renderRect = getVideoRenderRect(canvas);

  faces.forEach((face) => {
    canvasCtx.save();
    canvasCtx.globalAlpha = face.opacity === undefined ? 1 : face.opacity;
    const box = face.box;
    const left = renderRect.x + box.left * renderRect.scaleX;
    const top = renderRect.y + box.top * renderRect.scaleY;
    const width = (box.right - box.left) * renderRect.scaleX;
    const height = (box.bottom - box.top) * renderRect.scaleY;
    const known = face.name !== UNKNOWN_FACE_NAME;
    const color = known ? "#0b7a75" : "#bf2f36";
    const distance =
      face.distance === null ? "-" : Number(face.distance).toFixed(3);
    const label = known
      ? `${face.name} | d=${distance}`
      : `${face.status} | d=${distance}`;

    canvasCtx.strokeStyle = color;
    canvasCtx.lineWidth = 3;
    canvasCtx.strokeRect(left, top, width, height);
    drawLabel(canvasCtx, label, left, top, color);
    canvasCtx.restore();
  });
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function copyBox(box) {
  return {
    top: box.top,
    right: box.right,
    bottom: box.bottom,
    left: box.left,
  };
}

function centerOfBox(box) {
  return {
    x: (box.left + box.right) / 2,
    y: (box.top + box.bottom) / 2,
  };
}

function centerOf(face) {
  return centerOfBox(face.targetBox || face.box);
}

function boxSize(box) {
  return {
    width: Math.max(1, box.right - box.left),
    height: Math.max(1, box.bottom - box.top),
  };
}

function smoothTargetBox(previousBox, nextBox) {
  // Target box dihaluskan agar hasil deteksi yang naik-turun tidak terlihat patah-patah.
  const output = {};
  ["top", "right", "bottom", "left"].forEach((key) => {
    const diff = nextBox[key] - previousBox[key];
    // Smoother blending with reduced deadzone for better continuity
    output[key] =
      Math.abs(diff) < TRACKING_DEADZONE_PX
        ? previousBox[key]
        : previousBox[key] + diff * TRACKING_TARGET_BLEND;
  });
  return output;
}

function calculateVelocity(previousBox, nextBox, elapsedMs) {
  // Velocity dipakai untuk memperkirakan posisi saat backend belum mengirim hasil frame baru.
  const velocity = {};
  const frameTime = Math.max(8, elapsedMs);
  ["top", "right", "bottom", "left"].forEach((key) => {
    const rawVelocity = (nextBox[key] - previousBox[key]) / (frameTime / 16.67);
    velocity[key] = clamp(
      rawVelocity,
      -TRACKING_MAX_VELOCITY,
      TRACKING_MAX_VELOCITY,
    );
  });
  return velocity;
}

function blendVelocity(previousVelocity, nextVelocity) {
  const velocity = {};
  ["top", "right", "bottom", "left"].forEach((key) => {
    const previousValue = previousVelocity ? previousVelocity[key] || 0 : 0;
    velocity[key] = previousValue * 0.65 + nextVelocity[key] * 0.35;
  });
  return velocity;
}

function isKnownFace(face) {
  return face && face.name && face.name !== UNKNOWN_FACE_NAME;
}

function applyTrackedMetadata(tracked, face, now) {
  // Label wajah yang sudah dikenali dibuat sticky sebentar agar tidak berkedip
  // ketika satu frame berubah menjadi "Tidak Dikenal".
  if (isKnownFace(face)) {
    tracked.name = face.name;
    tracked.status = face.status;
    tracked.confidence = face.confidence;
    tracked.distance = face.distance;
    tracked.cosine_similarity = face.cosine_similarity;
    tracked.threshold = face.threshold;
    tracked.lastKnownSeen = now;
    return;
  }

  const shouldKeepKnownLabel =
    isKnownFace(tracked) &&
    now - (tracked.lastKnownSeen || tracked.lastSeen || now) <
      TRACKING_LABEL_STICKY_MS;

  if (!shouldKeepKnownLabel) {
    tracked.name = face.name;
    tracked.status = face.status;
    tracked.confidence = face.confidence;
    tracked.distance = face.distance;
    tracked.cosine_similarity = face.cosine_similarity;
    tracked.threshold = face.threshold;
  }
}

function updateTrackedFaces(faces) {
  // Mencocokkan hasil deteksi baru dengan track lama berdasarkan jarak pusat wajah.
  // Tujuannya agar kotak wajah tetap menjadi object yang sama antar frame.
  const now = performance.now();
  const used = new Set();

  faces.forEach((face) => {
    const rawBox = copyBox(face.box);
    const faceCenter = centerOfBox(rawBox);
    const faceSize = boxSize(rawBox);
    let bestIndex = -1;
    let bestDistance = Infinity;

    displayFaces.forEach((tracked, index) => {
      if (used.has(index)) return;
      const trackedCenter = centerOf(tracked);
      const trackedSize = boxSize(tracked.targetBox || tracked.box);
      const maxDim = Math.max(
        faceSize.width,
        faceSize.height,
        trackedSize.width,
        trackedSize.height,
      );
      const matchLimit = Math.max(120, Math.min(380, maxDim * 2.2));

      let identityPenalty = 0;
      if (tracked.name !== UNKNOWN_FACE_NAME && face.name !== UNKNOWN_FACE_NAME) {
        identityPenalty = tracked.name === face.name ? 0 : 15;
      }

      const distance =
        Math.hypot(
          faceCenter.x - trackedCenter.x,
          faceCenter.y - trackedCenter.y,
        ) + identityPenalty;
      if (distance < bestDistance) {
        bestDistance = distance;
        bestIndex = index;
        tracked.matchLimit = matchLimit;
      }
    });

    if (bestIndex >= 0 && bestDistance < displayFaces[bestIndex].matchLimit) {
      const tracked = displayFaces[bestIndex];
      const previousTarget = tracked.targetBox || tracked.box;
      const elapsedMs = Math.max(16, now - (tracked.lastUpdate || now));
      const nextTarget = smoothTargetBox(previousTarget, rawBox);
      const nextVelocity = calculateVelocity(
        previousTarget,
        nextTarget,
        elapsedMs,
      );
      tracked.targetBox = nextTarget;
      tracked.velocity = blendVelocity(tracked.velocity, nextVelocity);
      applyTrackedMetadata(tracked, face, now);
      tracked.lastSeen = now;
      tracked.lastUpdate = now;
      tracked.opacity = 1;
      used.add(bestIndex);
      return;
    }

    displayFaces.push({
      ...face,
      box: rawBox,
      targetBox: copyBox(rawBox),
      velocity: { top: 0, right: 0, bottom: 0, left: 0 },
      lastSeen: now,
      lastUpdate: now,
      lastKnownSeen: isKnownFace(face) ? now : 0,
      opacity: 1,
      matchCount: 0,
      noDetectCount: 0,
    });
  });

  displayFaces = displayFaces.filter(
    (face) => now - face.lastSeen < TRACKING_KEEP_MS,
  );
}

function animateOverlay() {
  // Overlay digambar dengan requestAnimationFrame agar gerakan kotak tetap halus
  // walaupun request AI ke backend berjalan lebih lambat dari refresh layar.
  if (!cameraReady) return;
  const now = performance.now();
  const elapsedMs = previousOverlayAt
    ? Math.min(50, now - previousOverlayAt)
    : 16.67;
  previousOverlayAt = now;
  const ease =
    1 - Math.pow(1 - TRACKING_EASE, Math.max(0.5, elapsedMs) / 16.67);

  displayFaces = displayFaces.filter(
    (face) => now - face.lastSeen < TRACKING_KEEP_MS,
  );
  displayFaces.forEach((face) => {
    const missingMs = now - face.lastSeen;
    const predictMs = Math.min(TRACKING_MAX_PREDICT_MS, Math.max(0, missingMs));
    const predictFrames = predictMs / 16.67;
    const velocity = face.velocity || { top: 0, right: 0, bottom: 0, left: 0 };
    const targetBox = face.targetBox || face.box;

    ["top", "right", "bottom", "left"].forEach((key) => {
      const predicted = targetBox[key] + velocity[key] * predictFrames;
      face.box[key] += (predicted - face.box[key]) * ease;
    });

    if (missingMs <= TRACKING_FADE_AFTER_MS) {
      face.opacity = 1;
    } else {
      const fadeProgress =
        (missingMs - TRACKING_FADE_AFTER_MS) /
        (TRACKING_KEEP_MS - TRACKING_FADE_AFTER_MS);
      face.opacity = Math.max(0, 1 - Math.pow(fadeProgress, 1.5));
    }
  });
  drawFaces(displayFaces);
  overlayAnimationId = requestAnimationFrame(animateOverlay);
}

function drawPipelineStages() {
  // Preview pipeline PCD diperbarui periodik: RGB asli, grayscale, resize, dan hasil deteksi.
  if (!cameraReady || !video.videoWidth || !video.videoHeight) return;

  const now = performance.now();
  if (lastPipelineAt) {
    currentPipelineFps = 1000 / Math.max(1, now - lastPipelineAt);
  }
  lastPipelineAt = now;

  const stageWidth = 320;
  const stageHeight = Math.round(
    stageWidth * (video.videoHeight / video.videoWidth),
  );
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
  resizeCtx.drawImage(
    temp,
    0,
    0,
    halfWidth,
    halfHeight,
    0,
    0,
    stageWidth,
    stageHeight,
  );
  resizeCtx.imageSmoothingEnabled = true;

  const resultCtx = stageResult.getContext("2d");
  drawVideoFrame(resultCtx, stageWidth, stageHeight);
  drawDetectedFaces(
    resultCtx,
    stageResult,
    displayFaces.length ? displayFaces : lastFaces,
  );
  setText(
    pipelineStatus,
    `CCTV 30 FPS | ${video.videoWidth}x${video.videoHeight} -> ${halfWidth}x${halfHeight}`,
  );
  updateMathPanel(centerSample, stageWidth, stageHeight, halfWidth, halfHeight);
}

function resizeOverlay() {
  // Canvas hanya di-resize saat ukuran berubah; resize setiap frame akan membuat overlay berkedip.
  if (!overlay || !video) return false;
  const width = Math.round(video.clientWidth);
  const height = Math.round(video.clientHeight);
  if (!width || !height) return false;
  let changed = false;
  if (overlay.width !== width) {
    overlay.width = width;
    changed = true;
  }
  if (overlay.height !== height) {
    overlay.height = height;
    changed = true;
  }
  return changed;
}

function drawFaces(faces) {
  if (!ctx) return;
  resizeOverlay();
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  drawDetectedFaces(ctx, overlay, faces);
}

function renderResults(faces) {
  if (!faces.length) {
    setHtml(
      resultList,
      `<div class="empty-state">Belum ada wajah terdeteksi.</div>`,
    );
    return;
  }

  setHtml(
    resultList,
    faces
      .map((face) => {
        const confidence = face.confidence ? `${face.confidence}%` : "-";
        const distance =
          face.distance === null ? "-" : Number(face.distance).toFixed(4);
        const threshold = Number(face.threshold).toFixed(2);
        const cosine =
          face.cosine_similarity === null
            ? "-"
            : Number(face.cosine_similarity).toFixed(4);
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
      .join(""),
  );
}

async function api(path, options = {}) {
  // Wrapper fetch untuk semua endpoint Flask agar error JSON ditampilkan seragam di UI.
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
  // Mengecek koneksi MySQL dan jumlah encoding tersimpan.
  try {
    const data = await api("/api/status");
    setText(
      dbStatus,
      `Database ${data.database} aktif di ${data.host} | ${data.count} encoding`,
    );
  } catch (error) {
    setText(dbStatus, `Database belum aktif: ${error.message}`);
  }
}

async function refreshFaces() {
  // Mengambil daftar nama wajah dari database untuk panel samping.
  try {
    const data = await api("/api/faces");
    if (!data.faces.length) {
      setHtml(faceList, `<div class="empty-state">Database kosong.</div>`);
      return;
    }
    setHtml(
      faceList,
      data.faces
        .map(
          (face) => `
          <div class="face-row">
            <div>
              <strong>${face.name}</strong><br>
              <small>${face.samples} encoding</small>
            </div>
            <button class="delete-btn" type="button" data-name="${face.name}">Hapus</button>
          </div>
        `,
        )
        .join(""),
    );
  } catch (error) {
    setHtml(faceList, `<div class="error-state">${error.message}</div>`);
  }
}

async function startCamera() {
  // Browser meminta akses webcam, lalu video stream dipakai untuk capture dan preview.
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
  // Satu siklus recognition: capture frame -> POST ke backend -> update track dan hasil.
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

    if (data.faces && data.faces.length > 0) {
      updateTrackedFaces(data.faces);
    }

    const visibleFaces = displayFaces.filter(
      (face) => (face.opacity === undefined ? 1 : face.opacity) > 0.05,
    );
    renderResults(visibleFaces.length ? visibleFaces : lastFaces);
    drawPipelineStages();
  } catch (error) {
    setHtml(resultList, `<div class="error-state">${error.message}</div>`);
  } finally {
    recognizeRequestRunning = false;
  }
}

async function recognizeLoop() {
  // Loop recognition berjalan dengan setTimeout agar request backend tidak saling menumpuk.
  if (!recognizing) return;
  await recognizeOnce();
  recognizeTimer = setTimeout(recognizeLoop, RECOGNITION_DELAY_MS);
}

function toggleRecognize() {
  // Tombol mulai/stop hanya mengatur loop recognition dan membersihkan overlay saat stop.
  recognizing = !recognizing;
  setText(toggleRecognizeBtn, recognizing ? "Stop Deteksi" : "Mulai Deteksi");
  if (recognizing) {
    previousOverlayAt = 0;
    recognizeLoop();
  } else {
    clearTimeout(recognizeTimer);
    lastFaces = [];
    displayFaces = [];
    previousOverlayAt = 0;
    if (ctx) ctx.clearRect(0, 0, overlay.width, overlay.height);
    drawPipelineStages();
  }
}

async function registerFace() {
  // Registrasi mengambil 5 sample frame, lalu backend menyimpan rata-rata encoding ke MySQL.
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
    setText(
      registerStatus,
      `${data.name} tersimpan dari ${data.samples} sample`,
    );
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
if (toggleRecognizeBtn)
  toggleRecognizeBtn.addEventListener("click", toggleRecognize);
if (registerBtn) registerBtn.addEventListener("click", registerFace);
if (refreshFacesBtn) refreshFacesBtn.addEventListener("click", refreshFaces);
if (mirrorCameraInput)
  mirrorCameraInput.addEventListener("change", updateMirrorMode);
window.addEventListener("resize", resizeOverlay);

if (faceList) {
  faceList.addEventListener("click", async (event) => {
    // Event delegation: tombol hapus dibuat dinamis setelah data /api/faces diterima.
    const button = event.target.closest("[data-name]");
    if (!button) return;
    await api(`/api/faces/${encodeURIComponent(button.dataset.name)}`, {
      method: "DELETE",
    });
    await refreshStatus();
    await refreshFaces();
  });
}

refreshStatus();
refreshFaces();
updateMirrorMode();
