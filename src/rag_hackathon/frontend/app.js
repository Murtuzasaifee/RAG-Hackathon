const state = {
  currentFile: null,
  pdfDoc: null,
  currentPage: 1,
  pageCount: 0,
  selectedCitation: null,
  citations: [],
};

const PDFJS_URL = "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.min.mjs";
const PDFJS_WORKER_URL = "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.worker.min.mjs";
let pdfjsModulePromise = null;

const el = {
  healthStatus: document.querySelector("#healthStatus"),
  uploadForm: document.querySelector("#uploadForm"),
  fileInput: document.querySelector("#fileInput"),
  docIdInput: document.querySelector("#docIdInput"),
  uploadButton: document.querySelector("#uploadButton"),
  jobState: document.querySelector("#jobState"),
  jobStage: document.querySelector("#jobStage"),
  jobProgress: document.querySelector("#jobProgress"),
  jobBar: document.querySelector("#jobBar"),
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

function updateJob(job) {
  el.jobState.textContent = job.state ?? "-";
  el.jobStage.textContent = job.stage ?? "-";
  el.jobProgress.textContent = `${job.progress ?? 0}%`;
  el.jobBar.value = job.progress ?? 0;
  el.jobMeta.textContent = `doc_id=${job.doc_id} · version_id=${job.version_id}`;
}

async function pollJob(jobId) {
  for (;;) {
    const job = await requestJson(`/jobs/${encodeURIComponent(jobId)}`);
    updateJob(job);
    if (job.state === "done") {
      el.queryDocIds.value = job.doc_id;
      window.localStorage.setItem("ragDemo.docId", job.doc_id);
      window.localStorage.setItem("ragDemo.versionId", job.version_id);
      return;
    }
    if (job.state === "failed") {
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
  await loadPdf(file);

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
    renderWarnings(response.warnings);
    renderTimings(response.timings_ms);
    renderCitations(response.citations || []);
  } catch (error) {
    el.answerOutput.textContent = error.message;
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
  drawOverlay(citation, viewport.width, viewport.height);
}

function clearOverlay() {
  el.overlayLayer.innerHTML = "";
}

function drawOverlay(citation, width, height) {
  clearOverlay();
  const bbox = citation?.bbox || [];
  if (!bbox.length || citation.page !== state.currentPage) {
    return;
  }

  const normalized = bbox.every((value) => Number.isFinite(value) && value >= 0 && value <= 1.05);
  if (!normalized) {
    el.pdfMessage.textContent = "Selected citation has non-normalized bbox; showing metadata only.";
    return;
  }

  if (bbox.length === 4) {
    const [x0, y0, x1, y1] = bbox;
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", String(x0 * width));
    rect.setAttribute("y", String(y0 * height));
    rect.setAttribute("width", String(Math.max(2, (x1 - x0) * width)));
    rect.setAttribute("height", String(Math.max(2, (y1 - y0) * height)));
    rect.setAttribute("fill", "rgba(11, 107, 203, 0.18)");
    rect.setAttribute("stroke", "#0b6bcb");
    rect.setAttribute("stroke-width", "2");
    el.overlayLayer.appendChild(rect);
    return;
  }

  if (bbox.length >= 8 && bbox.length % 2 === 0) {
    const points = [];
    for (let i = 0; i < bbox.length; i += 2) {
      points.push(`${bbox[i] * width},${bbox[i + 1] * height}`);
    }
    const polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    polygon.setAttribute("points", points.join(" "));
    polygon.setAttribute("fill", "rgba(11, 107, 203, 0.18)");
    polygon.setAttribute("stroke", "#0b6bcb");
    polygon.setAttribute("stroke-width", "2");
    el.overlayLayer.appendChild(polygon);
  }
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
checkHealth();
