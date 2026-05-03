const state = {
  currentFile: null,
  pdfDoc: null,
  currentPage: 1,
  pageCount: 0,
  selectedCitation: null,
  citations: [],
  jobStartedAt: null,
  jobElapsedTimer: null,
};

const PDFJS_URL = "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.min.mjs";
const PDFJS_WORKER_URL = "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.worker.min.mjs";
let pdfjsModulePromise = null;

const JOB_STAGES = [
  ["queued", "Queued"],
  ["parsing", "Parsing document"],
  ["chunking", "Creating chunks"],
  ["embedding_dense", "Embedding dense vectors"],
  ["embedding_sparse", "Embedding sparse vectors"],
  ["indexing", "Indexing in Qdrant"],
  ["done", "Ready"],
];

const JOB_DETAIL = {
  queued: "Queued for background ingestion",
  parsing: "Azure Document Intelligence is extracting layout and text",
  chunking: "Building structure-aware chunks with page lineage",
  embedding_dense: "Creating dense embeddings through Bifrost",
  embedding_sparse: "Creating SPLADE sparse embeddings",
  indexing: "Writing chunks and vectors into Qdrant",
  done: "Ingestion complete. You can query this document now.",
  failed: "Ingestion failed. Check the error message below.",
};

const el = {
  healthStatus: document.querySelector("#healthStatus"),
  serviceLinks: document.querySelector("#serviceLinks"),
  uploadForm: document.querySelector("#uploadForm"),
  fileInput: document.querySelector("#fileInput"),
  docIdInput: document.querySelector("#docIdInput"),
  uploadButton: document.querySelector("#uploadButton"),
  jobState: document.querySelector("#jobState"),
  jobStage: document.querySelector("#jobStage"),
  jobProgress: document.querySelector("#jobProgress"),
  jobElapsed: document.querySelector("#jobElapsed"),
  jobBar: document.querySelector("#jobBar"),
  jobDetail: document.querySelector("#jobDetail"),
  jobTimeline: document.querySelector("#jobTimeline"),
  jobMeta: document.querySelector("#jobMeta"),
  queryForm: document.querySelector("#queryForm"),
  queryInput: document.querySelector("#queryInput"),
  queryDocIds: document.querySelector("#queryDocIds"),
  queryVersionIds: document.querySelector("#queryVersionIds"),
  topKInput: document.querySelector("#topKInput"),
  topNInput: document.querySelector("#topNInput"),
  askButton: document.querySelector("#askButton"),
  answerOutput: document.querySelector("#answerOutput"),
  requestMeta: document.querySelector("#requestMeta"),
  cacheStatus: document.querySelector("#cacheStatus"),
  warningOutput: document.querySelector("#warningOutput"),
  timingOutput: document.querySelector("#timingOutput"),
  citationList: document.querySelector("#citationList"),
  citationCount: document.querySelector("#citationCount"),
  sourceDetail: document.querySelector("#sourceDetail"),
  copyCitationButton: document.querySelector("#copyCitationButton"),
  pdfMessage: document.querySelector("#pdfMessage"),
  pdfViewport: document.querySelector("#pdfViewport"),
  pdfCanvas: document.querySelector("#pdfCanvas"),
  overlayLayer: document.querySelector("#overlayLayer"),
  prevPageButton: document.querySelector("#prevPageButton"),
  nextPageButton: document.querySelector("#nextPageButton"),
  pageIndicator: document.querySelector("#pageIndicator"),
};

const sleep = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));

function setHealth(ok, text) {
  el.healthStatus.textContent = text;
  el.healthStatus.className = `status ${ok ? "status-ok" : "status-error"}`;
  el.uploadButton.disabled = !ok;
  el.askButton.disabled = !ok;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function parseCsv(value) {
  const items = value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  return items.length ? items : null;
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.message || payload.detail || `Request failed: ${response.status}`);
  }
  return payload;
}

async function checkHealth() {
  try {
    await requestJson("/health");
    setHealth(true, "Healthy");
  } catch (error) {
    setHealth(false, "Offline");
    el.jobMeta.textContent = error.message;
  }
}

async function loadDemoConfig() {
  try {
    const config = await requestJson("/demo/config");
    renderServiceLinks(config);
  } catch {
    el.serviceLinks.textContent = "";
  }
}

function renderServiceLinks(config) {
  const links = [
    ["Bifrost", config.bifrost_url],
    ["Logfire", config.logfire_project_url],
  ].filter(([, href]) => href);

  el.serviceLinks.innerHTML = "";
  for (const [label, href] of links) {
    const link = document.createElement("a");
    link.href = href;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.textContent = label;
    el.serviceLinks.appendChild(link);
  }
}

function updateJob(job) {
  el.jobState.textContent = job.state ?? "-";
  el.jobStage.textContent = job.stage ?? "-";
  el.jobProgress.textContent = `${job.progress ?? 0}%`;
  el.jobBar.value = job.progress ?? 0;
  el.jobElapsed.textContent = state.jobStartedAt ? formatElapsed(Date.now() - state.jobStartedAt) : "-";
  renderJobTimeline(job.stage, job.state);
  el.jobDetail.textContent = JOB_DETAIL[job.stage] || JOB_DETAIL[job.state] || "Working";
  el.jobDetail.classList.toggle("running", job.state === "pending" || job.state === "running");
  el.jobMeta.textContent = `doc_id=${job.doc_id} · version_id=${job.version_id}`;
}

function startJobClock() {
  stopJobClock();
  state.jobElapsedTimer = window.setInterval(() => {
    if (state.jobStartedAt) {
      el.jobElapsed.textContent = formatElapsed(Date.now() - state.jobStartedAt);
    }
  }, 1000);
}

function stopJobClock() {
  if (state.jobElapsedTimer) {
    window.clearInterval(state.jobElapsedTimer);
    state.jobElapsedTimer = null;
  }
}

function renderJobTimeline(stage, stateValue) {
  const currentIndex = Math.max(0, JOB_STAGES.findIndex(([key]) => key === stage));
  el.jobTimeline.innerHTML = "";
  for (const [key, label] of JOB_STAGES) {
    const index = JOB_STAGES.findIndex(([candidate]) => candidate === key);
    const item = document.createElement("li");
    item.textContent = label;
    item.className = index < currentIndex || stateValue === "done" ? "done" : "";
    if (index === currentIndex && stateValue !== "done") {
      item.className = "active";
    }
    el.jobTimeline.appendChild(item);
  }
}

function formatElapsed(ms) {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return minutes ? `${minutes}m ${seconds}s` : `${seconds}s`;
}

async function pollJob(jobId) {
  for (; ;) {
    const job = await requestJson(`/jobs/${encodeURIComponent(jobId)}`);
    updateJob(job);
    if (job.state === "done") {
      el.queryDocIds.value = job.doc_id;
      window.localStorage.setItem("ragDemo.docId", job.doc_id);
      window.localStorage.setItem("ragDemo.versionId", job.version_id);
      stopJobClock();
      return;
    }
    if (job.state === "failed") {
      stopJobClock();
      throw new Error(job.error || "Ingestion failed");
    }
    await sleep(1400);
  }
}

async function uploadDocument(event) {
  event.preventDefault();
  const file = el.fileInput.files?.[0];
  if (!file) return;

  state.currentFile = file;
  state.jobStartedAt = Date.now();
  startJobClock();
  updateJob({
    doc_id: el.docIdInput.value.trim() || file.name,
    version_id: "pending",
    state: "pending",
    stage: "queued",
    progress: 0,
  });
  loadPdf(file).catch(() => {
    el.pdfMessage.textContent = "PDF preview could not load. Ingestion can continue.";
  });

  const form = new FormData();
  form.append("file", file);
  if (el.docIdInput.value.trim()) {
    form.append("doc_id", el.docIdInput.value.trim());
  }

  el.uploadButton.disabled = true;
  try {
    const response = await requestJson("/ingest", {
      method: "POST",
      body: form,
    });
    updateJob({ ...response, state: "pending", stage: "queued", progress: 0 });
    await pollJob(response.job_id);
  } catch (error) {
    el.jobMeta.textContent = error.message;
    el.jobState.textContent = "Error";
    el.jobStage.textContent = "failed";
    el.jobDetail.textContent = JOB_DETAIL.failed;
    el.jobDetail.classList.remove("running");
    stopJobClock();
  } finally {
    el.uploadButton.disabled = false;
  }
}

function renderTimings(timings) {
  el.timingOutput.innerHTML = "";
  for (const [key, value] of Object.entries(timings ?? {})) {
    const pill = document.createElement("span");
    pill.className = "pill";
    pill.textContent = `${key}: ${value}ms`;
    el.timingOutput.appendChild(pill);
  }
}

function renderWarnings(warnings) {
  el.warningOutput.innerHTML = "";
  for (const warning of warnings ?? []) {
    const item = document.createElement("div");
    item.textContent = warning;
    el.warningOutput.appendChild(item);
  }
}

async function runQuery(event) {
  event.preventDefault();
  el.askButton.disabled = true;
  el.answerOutput.classList.remove("empty");
  el.answerOutput.textContent = "Thinking...";
  el.requestMeta.textContent = "";
  el.cacheStatus.hidden = true;
  try {
    const payload = {
      query: el.queryInput.value.trim(),
      doc_ids: parseCsv(el.queryDocIds.value),
      version_ids: parseCsv(el.queryVersionIds.value),
      top_k: Number(el.topKInput.value || 20),
      top_n: Number(el.topNInput.value || 5),
    };
    const response = await requestJson("/api/v1/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    el.answerOutput.textContent = response.answer || "";
    el.requestMeta.textContent = response.request_id ? `request ${response.request_id}` : "";
    const cacheHit = Boolean(response.cache_hit || response.timings_ms?.cache_hit);
    el.cacheStatus.hidden = !cacheHit;
    el.cacheStatus.textContent = cacheHit ? "Served from cache" : "";
    renderWarnings(response.warnings);
    renderTimings(response.timings_ms);
    renderCitations(response.citations || []);
  } catch (error) {
    el.answerOutput.textContent = error.message;
    el.cacheStatus.hidden = true;
  } finally {
    el.askButton.disabled = false;
  }
}

function citationTitle(citation) {
  const section = citation.section_path?.length
    ? ` · ${citation.section_path.join(" > ")}`
    : "";
  return `${citation.doc_id} · page ${citation.page}${section}`;
}

function snippet(text, maxLength = 220) {
  if (!text) return "";
  return text.length > maxLength ? `${text.slice(0, maxLength)}...` : text;
}

function renderCitations(citations) {
  state.citations = citations;
  state.selectedCitation = null;
  el.citationCount.textContent = String(citations.length);
  el.citationList.innerHTML = "";
  el.sourceDetail.textContent = "Select a citation to inspect its lineage and chunk text.";
  el.sourceDetail.className = "source-detail empty";
  clearOverlay();

  if (!citations.length) {
    el.citationList.className = "citation-list empty";
    el.citationList.textContent = "No citations returned.";
    return;
  }

  el.citationList.className = "citation-list";
  citations.forEach((citation, index) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "citation-card";
    card.innerHTML = `
      <div class="citation-title">${escapeHtml(citationTitle(citation))}</div>
      <div class="citation-meta">score ${Number(citation.score || 0).toFixed(3)} · ${escapeHtml(citation.chunk_type || "text")} · ${escapeHtml(citation.version_id)}</div>
      <div class="citation-snippet">${escapeHtml(snippet(citation.chunk_text))}</div>
    `;
    card.addEventListener("click", () => selectCitation(index));
    el.citationList.appendChild(card);
  });
  selectCitation(0);
}

async function selectCitation(index) {
  const citation = state.citations[index];
  state.selectedCitation = citation;
  for (const [i, card] of [...el.citationList.querySelectorAll(".citation-card")].entries()) {
    card.classList.toggle("selected", i === index);
  }
  renderSourceDetail(citation);
  if (citation?.page) {
    await renderPage(citation.page, citation);
  }
}

function renderSourceDetail(citation) {
  el.sourceDetail.className = "source-detail";
  el.sourceDetail.innerHTML = `
    <div class="detail-grid">
      <span>Document</span><strong>${escapeHtml(citation.doc_id)}</strong>
      <span>Version</span><strong>${escapeHtml(citation.version_id)}</strong>
      <span>Page</span><strong>${escapeHtml(citation.page)}</strong>
      <span>Section</span><strong>${escapeHtml((citation.section_path || []).join(" > ") || "-")}</strong>
      <span>Type</span><strong>${escapeHtml(citation.chunk_type)}</strong>
      <span>Score</span><strong>${Number(citation.score || 0).toFixed(3)}</strong>
      <span>BBox</span><strong>${escapeHtml(JSON.stringify(citation.bbox || []))}</strong>
    </div>
    <div class="chunk-text">${escapeHtml(citation.chunk_text || "")}</div>
  `;
}

async function copyCitation() {
  if (!state.selectedCitation) return;
  await navigator.clipboard.writeText(JSON.stringify(state.selectedCitation, null, 2));
  el.copyCitationButton.textContent = "Copied";
  window.setTimeout(() => {
    el.copyCitationButton.textContent = "Copy";
  }, 1200);
}

async function loadPdf(file) {
  let pdfjsLib;
  try {
    pdfjsLib = await getPdfJs();
  } catch {
    el.pdfMessage.textContent = "PDF.js did not load. Citation metadata is still available.";
    return;
  }

  const bytes = await file.arrayBuffer();
  state.pdfDoc = await pdfjsLib.getDocument({ data: bytes }).promise;
  state.pageCount = state.pdfDoc.numPages;
  state.currentPage = 1;
  await renderPage(1);
}

async function getPdfJs() {
  pdfjsModulePromise ||= import(PDFJS_URL).then((module) => {
    module.GlobalWorkerOptions.workerSrc = PDFJS_WORKER_URL;
    return module;
  });
  return pdfjsModulePromise;
}

async function renderPage(pageNumber, citation = state.selectedCitation) {
  if (!state.pdfDoc) {
    el.pdfMessage.textContent = "Upload a PDF in this browser session to enable source preview.";
    el.pdfViewport.hidden = true;
    return;
  }

  const boundedPage = Math.min(Math.max(Number(pageNumber) || 1, 1), state.pageCount);
  state.currentPage = boundedPage;
  const page = await state.pdfDoc.getPage(boundedPage);
  const containerWidth = el.pdfViewport.parentElement.clientWidth - 30;
  const naturalViewport = page.getViewport({ scale: 1 });
  const scale = Math.min(1.6, Math.max(0.7, containerWidth / naturalViewport.width));
  const viewport = page.getViewport({ scale });
  const context = el.pdfCanvas.getContext("2d");

  el.pdfCanvas.width = Math.floor(viewport.width);
  el.pdfCanvas.height = Math.floor(viewport.height);
  el.pdfCanvas.style.width = `${Math.floor(viewport.width)}px`;
  el.pdfCanvas.style.height = `${Math.floor(viewport.height)}px`;
  el.overlayLayer.setAttribute("width", String(Math.floor(viewport.width)));
  el.overlayLayer.setAttribute("height", String(Math.floor(viewport.height)));
  el.overlayLayer.style.width = `${Math.floor(viewport.width)}px`;
  el.overlayLayer.style.height = `${Math.floor(viewport.height)}px`;

  el.pdfViewport.hidden = false;
  el.pdfMessage.textContent = "";
  el.pageIndicator.textContent = `${boundedPage} / ${state.pageCount}`;
  el.prevPageButton.disabled = boundedPage <= 1;
  el.nextPageButton.disabled = boundedPage >= state.pageCount;

  await page.render({ canvasContext: context, viewport }).promise;
  drawOverlay(citation, {
    renderedWidth: viewport.width,
    renderedHeight: viewport.height,
    pdfPointWidth: naturalViewport.width,
    pdfPointHeight: naturalViewport.height,
  });
}

function clearOverlay() {
  el.overlayLayer.innerHTML = "";
}

function drawOverlay(citation, pageMetrics) {
  clearOverlay();
  const bbox = citation?.bbox || [];
  if (Number(citation?.page) !== state.currentPage) {
    return;
  }

  if (!bbox.length) {
    el.pdfMessage.textContent = "Selected citation has no bbox. Re-ingest this PDF to generate overlay-ready citations.";
    return;
  }

  const normalizedBbox = normalizeOverlayBbox(bbox, pageMetrics);
  if (!normalizedBbox.length) {
    el.pdfMessage.textContent = "Selected citation has bbox data that cannot be mapped to this PDF page.";
    return;
  }
  el.pdfMessage.textContent = "";

  if (normalizedBbox.length === 4) {
    const [x0, y0, x1, y1] = normalizedBbox;
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", String(x0 * pageMetrics.renderedWidth));
    rect.setAttribute("y", String(y0 * pageMetrics.renderedHeight));
    rect.setAttribute("width", String(Math.max(2, (x1 - x0) * pageMetrics.renderedWidth)));
    rect.setAttribute("height", String(Math.max(2, (y1 - y0) * pageMetrics.renderedHeight)));
    rect.setAttribute("fill", "rgba(11, 107, 203, 0.18)");
    rect.setAttribute("stroke", "#0b6bcb");
    rect.setAttribute("stroke-width", "2");
    el.overlayLayer.appendChild(rect);
    return;
  }

  if (normalizedBbox.length >= 8 && normalizedBbox.length % 2 === 0) {
    const points = [];
    for (let i = 0; i < normalizedBbox.length; i += 2) {
      points.push(
        `${normalizedBbox[i] * pageMetrics.renderedWidth},${normalizedBbox[i + 1] * pageMetrics.renderedHeight}`,
      );
    }
    const polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    polygon.setAttribute("points", points.join(" "));
    polygon.setAttribute("fill", "rgba(11, 107, 203, 0.18)");
    polygon.setAttribute("stroke", "#0b6bcb");
    polygon.setAttribute("stroke-width", "2");
    el.overlayLayer.appendChild(polygon);
  }
}

function normalizeOverlayBbox(bbox, pageMetrics) {
  const values = bbox.map(Number);
  if (!values.every(Number.isFinite)) {
    return [];
  }

  const maxValue = Math.max(...values);
  if (maxValue <= 1.05) {
    return values.map(clampUnit);
  }

  const pageInches = {
    width: pageMetrics.pdfPointWidth / 72,
    height: pageMetrics.pdfPointHeight / 72,
  };
  const looksLikeInches =
    maxValue <= Math.max(pageInches.width, pageInches.height) * 1.2;
  const basis = looksLikeInches
    ? pageInches
    : { width: pageMetrics.pdfPointWidth, height: pageMetrics.pdfPointHeight };

  if (values.length === 4) {
    const [x0, y0, x1, y1] = values;
    return [
      clampUnit(Math.min(x0, x1) / basis.width),
      clampUnit(Math.min(y0, y1) / basis.height),
      clampUnit(Math.max(x0, x1) / basis.width),
      clampUnit(Math.max(y0, y1) / basis.height),
    ];
  }

  if (values.length >= 8 && values.length % 2 === 0) {
    const normalized = [];
    for (let i = 0; i < values.length; i += 2) {
      normalized.push(clampUnit(values[i] / basis.width));
      normalized.push(clampUnit(values[i + 1] / basis.height));
    }
    return normalized;
  }

  return [];
}

function clampUnit(value) {
  return Math.min(1, Math.max(0, value));
}

async function changePage(delta) {
  await renderPage(state.currentPage + delta);
}

function restoreInputs() {
  el.queryDocIds.value = window.localStorage.getItem("ragDemo.docId") || "";
  el.queryVersionIds.value = "";
}

el.uploadForm.addEventListener("submit", uploadDocument);
el.queryForm.addEventListener("submit", runQuery);
el.copyCitationButton.addEventListener("click", copyCitation);
el.prevPageButton.addEventListener("click", () => changePage(-1));
el.nextPageButton.addEventListener("click", () => changePage(1));
el.fileInput.addEventListener("change", async () => {
  const file = el.fileInput.files?.[0];
  if (file) {
    state.currentFile = file;
    await loadPdf(file);
  }
});

restoreInputs();
renderJobTimeline("queued", "idle");
loadDemoConfig();
checkHealth();
