const state = {
  apiKey: "",
  role: "",
  currentFile: null,
  pdfDoc: null,
  currentPage: 1,
  pageCount: 0,
  selectedCitation: null,
  citations: [],
  jobStartedAt: null,
  jobElapsedTimer: null,
  rolePresets: [],
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

const ROLE_DESC = {
  reader: "Query documents",
  editor1: "Upload & query",
  editor2: "Upload & query",
  admin: "Full access",
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const el = {
  loginView: $("#loginView"),
  appView: $("#appView"),
  roleCards: $("#roleCards"),
  manualLoginForm: $("#manualLoginForm"),
  manualApiKey: $("#manualApiKey"),
  healthStatus: $("#healthStatus"),
  serviceLinks: $("#serviceLinks"),
  userRole: $("#userRole"),
  logoutButton: $("#logoutButton"),
  uploadForm: $("#uploadForm"),
  fileInput: $("#fileInput"),
  docIdInput: $("#docIdInput"),
  uploadButton: $("#uploadButton"),
  dropZone: $("#dropZone"),
  fileName: $("#fileName"),
  openIngest: $("#openIngest"),
  closeIngest: $("#closeIngest"),
  ingestOverlay: $("#ingestOverlay"),
  ingestBackdrop: $("#ingestBackdrop"),
  jobState: $("#jobState"),
  jobStage: $("#jobStage"),
  jobProgress: $("#jobProgress"),
  jobElapsed: $("#jobElapsed"),
  jobBar: $("#jobBar"),
  jobDetail: $("#jobDetail"),
  jobTimeline: $("#jobTimeline"),
  jobMeta: $("#jobMeta"),
  queryForm: $("#queryForm"),
  queryInput: $("#queryInput"),
  queryDocIds: $("#queryDocIds"),
  queryVersionIds: $("#queryVersionIds"),
  topKInput: $("#topKInput"),
  topNInput: $("#topNInput"),
  askButton: $("#askButton"),
  answerOutput: $("#answerOutput"),
  requestMeta: $("#requestMeta"),
  cacheStatus: $("#cacheStatus"),
  warningOutput: $("#warningOutput"),
  timingOutput: $("#timingOutput"),
  citationList: $("#citationList"),
  citationCount: $("#citationCount"),
  sourceDetail: $("#sourceDetail"),
  copyCitationButton: $("#copyCitationButton"),
  pdfMessage: $("#pdfMessage"),
  pdfViewport: $("#pdfViewport"),
  pdfCanvas: $("#pdfCanvas"),
  overlayLayer: $("#overlayLayer"),
  prevPageButton: $("#prevPageButton"),
  nextPageButton: $("#nextPageButton"),
  pageIndicator: $("#pageIndicator"),
  docsList: $("#docsList"),
  refreshDocsButton: $("#refreshDocsButton"),
};

function getAuthHeaders() {
  return state.apiKey ? { "X-API-Key": state.apiKey } : {};
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function escapeHtml(v) {
  return String(v ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}

function parseCsv(v) {
  const items = v.split(",").map((s) => s.trim()).filter(Boolean);
  return items.length ? items : null;
}

async function requestJson(url, opts = {}) {
  const res = await fetch(url, opts);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.message || body.detail || `Request failed: ${res.status}`);
  return body;
}

function setHealth(ok, text) {
  el.healthStatus.className = `health-dot ${ok ? "health-ok" : "health-error"}`;
  el.healthStatus.title = text;
  el.uploadButton.disabled = !ok;
  el.askButton.disabled = !ok;
}

function login(apiKey, role) {
  state.apiKey = apiKey;
  state.role = role || "custom";
  window.localStorage.setItem("ragDemo.apiKey", apiKey);
  el.userRole.textContent = state.role;
  el.loginView.classList.add("hidden");
  el.appView.classList.remove("hidden");
  el.appView.style.animation = "fadeIn .3s ease-out";
  checkHealth();
  loadDocuments();
  restoreInputs();
  renderJobTimeline("queued", "idle");
}

function logout() {
  state.apiKey = "";
  window.localStorage.removeItem("ragDemo.apiKey");
  el.appView.classList.add("hidden");
  el.loginView.classList.remove("hidden");
  el.loginView.style.animation = "fadeIn .3s ease-out";
}

async function checkHealth() {
  try {
    await requestJson("/health");
    setHealth(true, "Healthy");
  } catch (e) {
    setHealth(false, "Offline");
    el.jobMeta.textContent = e.message;
  }
}

async function loadDemoConfig() {
  try {
    const config = await requestJson("/demo/config");
    renderServiceLinks(config);
    state.rolePresets = config.role_presets || [];
    renderRoleCards(state.rolePresets);

    const savedKey = window.localStorage.getItem("ragDemo.apiKey") || "";
    if (savedKey) el.manualApiKey.value = savedKey;
  } catch { el.serviceLinks.textContent = ""; }
}

function renderRoleCards(presets) {
  el.roleCards.innerHTML = "";
  for (const { label, key } of presets) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "role-card";
    btn.innerHTML = `<div class="role-card-name">${escapeHtml(label)}</div><div class="role-card-desc">${escapeHtml(ROLE_DESC[label] || "Query documents")}</div>`;
    btn.addEventListener("click", () => login(key, label));
    el.roleCards.appendChild(btn);
  }
}

function renderServiceLinks(config) {
  const obsLabel = config.otel_backend === "langfuse" ? "Langfuse" : "Logfire";
  const links = [["Bifrost", config.bifrost_url], [obsLabel, config.observability_url]].filter(([, h]) => h);
  el.serviceLinks.innerHTML = "";
  for (const [label, href] of links) {
    const a = document.createElement("a");
    a.href = href;
    a.target = "_blank";
    a.rel = "noreferrer";
    a.textContent = label;
    el.serviceLinks.appendChild(a);
  }
}

function updateJob(job) {
  el.jobState.textContent = job.state ?? "-";
  el.jobStage.textContent = job.stage ?? "-";
  el.jobProgress.textContent = `${job.progress ?? 0}%`;
  el.jobBar.value = job.progress ?? 0;
  el.jobElapsed.textContent = state.jobStartedAt ? fmtElapsed(Date.now() - state.jobStartedAt) : "-";
  renderJobTimeline(job.stage, job.state);
  el.jobDetail.textContent = JOB_DETAIL[job.stage] || JOB_DETAIL[job.state] || "Working";
  el.jobDetail.classList.toggle("running", job.state === "pending" || job.state === "running");
  el.jobMeta.textContent = `doc_id=${job.doc_id} \u00b7 version_id=${job.version_id}`;
}

function startJobClock() {
  stopJobClock();
  state.jobElapsedTimer = setInterval(() => {
    if (state.jobStartedAt) el.jobElapsed.textContent = fmtElapsed(Date.now() - state.jobStartedAt);
  }, 1000);
}

function stopJobClock() {
  if (state.jobElapsedTimer) { clearInterval(state.jobElapsedTimer); state.jobElapsedTimer = null; }
}

function renderJobTimeline(stage, sv) {
  const cur = Math.max(0, JOB_STAGES.findIndex(([k]) => k === stage));
  el.jobTimeline.innerHTML = "";
  for (const [key, label] of JOB_STAGES) {
    const idx = JOB_STAGES.findIndex(([k]) => k === key);
    const li = document.createElement("li");
    li.textContent = label;
    li.className = idx < cur || sv === "done" ? "done" : idx === cur && sv !== "done" ? "active" : "";
    el.jobTimeline.appendChild(li);
  }
}

function fmtElapsed(ms) {
  const s = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(s / 60);
  return m ? `${m}m ${s % 60}s` : `${s}s`;
}

async function pollJob(jobId) {
  for (;;) {
    const job = await requestJson(`/jobs/${encodeURIComponent(jobId)}`, { headers: getAuthHeaders() });
    updateJob(job);
    if (job.state === "done") {
      el.queryDocIds.value = job.doc_id;
      window.localStorage.setItem("ragDemo.docId", job.doc_id);
      window.localStorage.setItem("ragDemo.versionId", job.version_id);
      stopJobClock();
      return;
    }
    if (job.state === "failed") { stopJobClock(); throw new Error(job.error || "Ingestion failed"); }
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
  updateJob({ doc_id: el.docIdInput.value.trim() || file.name, version_id: "pending", state: "pending", stage: "queued", progress: 0 });
  loadPdf(file).catch(() => { el.pdfMessage.textContent = "PDF preview could not load."; });
  const form = new FormData();
  form.append("file", file);
  if (el.docIdInput.value.trim()) form.append("doc_id", el.docIdInput.value.trim());
  el.uploadButton.disabled = true;
  try {
    const res = await requestJson("/ingest", { method: "POST", headers: getAuthHeaders(), body: form });
    updateJob({ ...res, state: "pending", stage: "queued", progress: 0 });
    await pollJob(res.job_id);
  } catch (e) {
    el.jobMeta.textContent = e.message;
    el.jobState.textContent = "Error";
    el.jobStage.textContent = "failed";
    el.jobDetail.textContent = JOB_DETAIL.failed;
    el.jobDetail.classList.remove("running");
    stopJobClock();
  } finally { el.uploadButton.disabled = false; }
}

function renderTimings(timings) {
  el.timingOutput.innerHTML = "";
  for (const [key, value] of Object.entries(timings ?? {})) {
    const span = document.createElement("span");
    span.className = "badge";
    span.style.background = "var(--sand)";
    span.style.color = "var(--text-2)";
    span.style.fontFamily = "var(--font-mono)";
    span.style.fontSize = "11px";
    span.textContent = `${key}: ${value}ms`;
    el.timingOutput.appendChild(span);
  }
}

function renderWarnings(warnings) {
  el.warningOutput.innerHTML = "";
  for (const w of warnings ?? []) { const d = document.createElement("div"); d.textContent = w; el.warningOutput.appendChild(d); }
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
    const res = await requestJson("/api/v1/query", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...getAuthHeaders() },
      body: JSON.stringify(payload),
    });
    el.answerOutput.textContent = res.answer || "";
    el.requestMeta.textContent = res.request_id ? `request ${res.request_id}` : "";
    const ch = Boolean(res.cache_hit || res.timings_ms?.cache_hit);
    el.cacheStatus.hidden = !ch;
    renderWarnings(res.warnings);
    renderTimings(res.timings_ms);
    renderCitations(res.citations || []);
    switchTab("answer");
  } catch (e) {
    el.answerOutput.textContent = e.message;
    el.cacheStatus.hidden = true;
  } finally { el.askButton.disabled = false; }
}

function citationTitle(c) {
  const sec = c.section_path?.length ? ` \u00b7 ${c.section_path.join(" > ")}` : "";
  return `${c.doc_id} \u00b7 page ${c.page}${sec}`;
}

function snippet(text, max = 200) {
  return text?.length > max ? text.slice(0, max) + "..." : (text || "");
}

function renderCitations(citations) {
  state.citations = citations;
  state.selectedCitation = null;
  el.citationCount.textContent = String(citations.length);
  el.citationList.innerHTML = "";
  el.sourceDetail.textContent = "Select a citation to inspect its lineage.";
  el.sourceDetail.className = "source-detail empty";
  clearOverlay();
  if (!citations.length) {
    el.citationList.className = "citation-list empty";
    el.citationList.textContent = "No citations returned.";
    return;
  }
  el.citationList.className = "citation-list";
  citations.forEach((c, i) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "citation-card";
    btn.innerHTML = `<div class="citation-title">${escapeHtml(citationTitle(c))}</div><div class="citation-meta">score ${Number(c.score || 0).toFixed(3)} \u00b7 ${escapeHtml(c.chunk_type || "text")} \u00b7 ${escapeHtml(c.version_id)}</div><div class="citation-snippet">${escapeHtml(snippet(c.chunk_text))}</div>`;
    btn.addEventListener("click", () => selectCitation(i));
    el.citationList.appendChild(btn);
  });
  selectCitation(0);
}

async function selectCitation(i) {
  const c = state.citations[i];
  state.selectedCitation = c;
  [...el.citationList.querySelectorAll(".citation-card")].forEach((card, j) => card.classList.toggle("selected", j === i));
  renderSourceDetail(c);
  if (c?.page) await renderPage(c.page, c);
}

function renderSourceDetail(c) {
  el.sourceDetail.className = "source-detail";
  el.sourceDetail.innerHTML = `<div class="detail-grid"><span>Document</span><strong>${escapeHtml(c.doc_id)}</strong><span>Version</span><strong>${escapeHtml(c.version_id)}</strong><span>Page</span><strong>${escapeHtml(c.page)}</strong><span>Section</span><strong>${escapeHtml((c.section_path || []).join(" > ") || "-")}</strong><span>Type</span><strong>${escapeHtml(c.chunk_type)}</strong><span>Score</span><strong>${Number(c.score || 0).toFixed(3)}</strong><span>BBox</span><strong>${escapeHtml(JSON.stringify(c.bbox || []))}</strong></div><div class="chunk-text">${escapeHtml(c.chunk_text || "")}</div>`;
}

async function copyCitation() {
  if (!state.selectedCitation) return;
  await navigator.clipboard.writeText(JSON.stringify(state.selectedCitation, null, 2));
  el.copyCitationButton.textContent = "Copied";
  setTimeout(() => { el.copyCitationButton.textContent = "Copy JSON"; }, 1200);
}

async function loadDocuments() {
  if (!state.apiKey) {
    el.docsList.innerHTML = '<span class="text-dim">Sign in to see documents.</span>';
    el.docsList.classList.add("empty");
    return;
  }
  try {
    const docs = await requestJson("/documents", { headers: getAuthHeaders() });
    renderDocuments(docs);
  } catch (e) {
    el.docsList.innerHTML = `<span class="text-dim">${escapeHtml(e.message)}</span>`;
    el.docsList.classList.add("empty");
  }
}

function renderDocuments(docs) {
  el.docsList.innerHTML = "";
  if (!docs.length) {
    el.docsList.className = "docs-grid empty";
    el.docsList.textContent = "No documents found.";
    return;
  }
  el.docsList.className = "docs-grid";
  for (const doc of docs) {
    const card = document.createElement("div");
    card.className = "doc-card";
    card.innerHTML = `<div class="doc-card-info"><div class="doc-card-title">${escapeHtml(doc.doc_id)}</div><div class="doc-card-meta">${doc.total_chunks} chunks \u00b7 ${escapeHtml(doc.active_version_id || "no active version")}</div></div><div class="doc-card-actions"><button type="button" class="btn btn-outline btn-2xs" data-action="select" data-doc-id="${escapeHtml(doc.doc_id)}">Select</button><button type="button" class="btn btn-danger" data-action="soft-delete" data-doc-id="${escapeHtml(doc.doc_id)}">Soft</button><button type="button" class="btn btn-danger" data-action="hard-delete" data-doc-id="${escapeHtml(doc.doc_id)}">Hard</button></div>`;
    card.querySelectorAll("button[data-action]").forEach((btn) => {
      btn.addEventListener("click", () => handleDocAction(btn.dataset.action, btn.dataset.docId));
    });
    el.docsList.appendChild(card);
  }
}

async function handleDocAction(action, docId) {
  if (action === "select") {
    el.queryDocIds.value = docId;
    window.localStorage.setItem("ragDemo.docId", docId);
    return;
  }
  const mode = action === "soft-delete" ? "soft" : "hard";
  if (!confirm(`${mode === "hard" ? "Permanently" : "Soft"} delete "${docId}"?`)) return;
  try {
    await requestJson(`/documents/${encodeURIComponent(docId)}?mode=${mode}`, { method: "DELETE", headers: getAuthHeaders() });
    await loadDocuments();
  } catch (e) { alert(e.message); }
}

async function loadPdf(file) {
  let lib;
  try { lib = await getPdfJs(); } catch {
    el.pdfMessage.textContent = "PDF.js did not load.";
    return;
  }
  const bytes = await file.arrayBuffer();
  state.pdfDoc = await lib.getDocument({ data: bytes }).promise;
  state.pageCount = state.pdfDoc.numPages;
  state.currentPage = 1;
  await renderPage(1);
}

async function getPdfJs() {
  pdfjsModulePromise ||= import(PDFJS_URL).then((m) => { m.GlobalWorkerOptions.workerSrc = PDFJS_WORKER_URL; return m; });
  return pdfjsModulePromise;
}

async function renderPage(pageNumber, citation = state.selectedCitation) {
  if (!state.pdfDoc) {
    el.pdfMessage.textContent = "Upload a PDF to preview it here.";
    el.pdfViewport.hidden = true;
    return;
  }
  const pg = Math.min(Math.max(Number(pageNumber) || 1, 1), state.pageCount);
  state.currentPage = pg;
  const page = await state.pdfDoc.getPage(pg);
  const cw = el.pdfViewport.parentElement.clientWidth - 24;
  const nv = page.getViewport({ scale: 1 });
  const scale = Math.min(1.6, Math.max(0.7, cw / nv.width));
  const vp = page.getViewport({ scale });
  const ctx = el.pdfCanvas.getContext("2d");
  el.pdfCanvas.width = Math.floor(vp.width);
  el.pdfCanvas.height = Math.floor(vp.height);
  el.pdfCanvas.style.width = `${Math.floor(vp.width)}px`;
  el.pdfCanvas.style.height = `${Math.floor(vp.height)}px`;
  el.overlayLayer.setAttribute("width", String(Math.floor(vp.width)));
  el.overlayLayer.setAttribute("height", String(Math.floor(vp.height)));
  el.overlayLayer.style.width = `${Math.floor(vp.width)}px`;
  el.overlayLayer.style.height = `${Math.floor(vp.height)}px`;
  el.pdfViewport.hidden = false;
  el.pdfMessage.textContent = "";
  el.pageIndicator.textContent = `${pg} / ${state.pageCount}`;
  el.prevPageButton.disabled = pg <= 1;
  el.nextPageButton.disabled = pg >= state.pageCount;
  await page.render({ canvasContext: ctx, viewport: vp }).promise;
  drawOverlay(citation, { renderedWidth: vp.width, renderedHeight: vp.height, pdfPointWidth: nv.width, pdfPointHeight: nv.height });
}

function clearOverlay() { el.overlayLayer.innerHTML = ""; }

function drawOverlay(citation, pm) {
  clearOverlay();
  const bbox = citation?.bbox || [];
  if (Number(citation?.page) !== state.currentPage) return;
  if (!bbox.length) { el.pdfMessage.textContent = "No bbox for this citation."; return; }
  const norm = normalizeBbox(bbox, pm);
  if (!norm.length) { el.pdfMessage.textContent = "Bbox cannot be mapped to page."; return; }
  el.pdfMessage.textContent = "";
  if (norm.length === 4) {
    const [x0, y0, x1, y1] = norm;
    const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    r.setAttribute("x", String(x0 * pm.renderedWidth));
    r.setAttribute("y", String(y0 * pm.renderedHeight));
    r.setAttribute("width", String(Math.max(2, (x1 - x0) * pm.renderedWidth)));
    r.setAttribute("height", String(Math.max(2, (y1 - y0) * pm.renderedHeight)));
    r.setAttribute("fill", "rgba(194, 65, 12, 0.15)");
    r.setAttribute("stroke", "#c2410c");
    r.setAttribute("stroke-width", "2");
    el.overlayLayer.appendChild(r);
    return;
  }
  if (norm.length >= 8 && norm.length % 2 === 0) {
    const pts = [];
    for (let i = 0; i < norm.length; i += 2) pts.push(`${norm[i] * pm.renderedWidth},${norm[i + 1] * pm.renderedHeight}`);
    const p = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    p.setAttribute("points", pts.join(" "));
    p.setAttribute("fill", "rgba(194, 65, 12, 0.15)");
    p.setAttribute("stroke", "#c2410c");
    p.setAttribute("stroke-width", "2");
    el.overlayLayer.appendChild(p);
  }
}

function normalizeBbox(bbox, pm) {
  const vals = bbox.map(Number);
  if (!vals.every(Number.isFinite)) return [];
  const mx = Math.max(...vals);
  if (mx <= 1.05) return vals.map(clamp);
  const pi = { width: pm.pdfPointWidth / 72, height: pm.pdfPointHeight / 72 };
  const inches = mx <= Math.max(pi.width, pi.height) * 1.2;
  const basis = inches ? pi : { width: pm.pdfPointWidth, height: pm.pdfPointHeight };
  if (vals.length === 4) {
    const [x0, y0, x1, y1] = vals;
    return [clamp(Math.min(x0, x1) / basis.width), clamp(Math.min(y0, y1) / basis.height), clamp(Math.max(x0, x1) / basis.width), clamp(Math.max(y0, y1) / basis.height)];
  }
  if (vals.length >= 8 && vals.length % 2 === 0) {
    const out = [];
    for (let i = 0; i < vals.length; i += 2) { out.push(clamp(vals[i] / basis.width)); out.push(clamp(vals[i + 1] / basis.height)); }
    return out;
  }
  return [];
}

function clamp(v) { return Math.min(1, Math.max(0, v)); }

async function changePage(d) { await renderPage(state.currentPage + d); }

function restoreInputs() {
  el.queryDocIds.value = window.localStorage.getItem("ragDemo.docId") || "";
  el.queryVersionIds.value = "";
}

function switchTab(name) {
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  $$(".pane").forEach((p) => p.classList.toggle("active", p.id === `pane${name.charAt(0).toUpperCase() + name.slice(1)}`));
}

function autoResizeTextarea() {
  const ta = el.queryInput;
  ta.style.height = "auto";
  ta.style.height = Math.min(ta.scrollHeight, 80) + "px";
}

function setupDropZone() {
  const zone = el.dropZone;
  if (!zone) return;
  zone.addEventListener("click", () => el.fileInput.click());
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("drag-over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    const file = e.dataTransfer?.files?.[0];
    if (file && file.type === "application/pdf") {
      const dt = new DataTransfer();
      dt.items.add(file);
      el.fileInput.files = dt.files;
      el.fileName.textContent = file.name;
      state.currentFile = file;
      loadPdf(file).catch(() => {});
    }
  });
}

el.manualLoginForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const key = el.manualApiKey.value.trim();
  if (key) login(key, "custom");
});

el.logoutButton.addEventListener("click", logout);
el.openIngest.addEventListener("click", () => el.ingestOverlay.classList.remove("hidden"));
el.closeIngest.addEventListener("click", () => el.ingestOverlay.classList.add("hidden"));
el.ingestBackdrop.addEventListener("click", () => el.ingestOverlay.classList.add("hidden"));

el.uploadForm.addEventListener("submit", async (e) => { await uploadDocument(e); loadDocuments(); });
el.queryForm.addEventListener("submit", runQuery);
el.copyCitationButton.addEventListener("click", copyCitation);
el.prevPageButton.addEventListener("click", () => changePage(-1));
el.nextPageButton.addEventListener("click", () => changePage(1));
el.refreshDocsButton.addEventListener("click", loadDocuments);
el.fileInput.addEventListener("change", async () => {
  const f = el.fileInput.files?.[0];
  if (f) { el.fileName.textContent = f.name; state.currentFile = f; await loadPdf(f); }
});

el.queryInput.addEventListener("input", autoResizeTextarea);

$$(".tab").forEach((tab) => {
  tab.addEventListener("click", () => switchTab(tab.dataset.tab));
});

setupDropZone();
loadDemoConfig();
