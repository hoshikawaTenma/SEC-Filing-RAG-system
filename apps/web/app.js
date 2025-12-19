const API_BASE = "http://localhost:8000";

const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const uploadStatus = document.getElementById("uploadStatus");
const docList = document.getElementById("docList");
const askBtn = document.getElementById("askBtn");
const questionInput = document.getElementById("questionInput");
const answerPanel = document.getElementById("answerPanel");
const sourcesList = document.getElementById("sourcesList");
const evidencePanel = document.getElementById("evidencePanel");
const existingSelect = document.getElementById("existingSelect");
const existingBtn = document.getElementById("existingBtn");
let statusTimer = null;

uploadBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) {
    uploadStatus.textContent = "Choose a file first.";
    return;
  }
  const formData = new FormData();
  formData.append("file", file);
  uploadStatus.textContent = "Uploading...";
  setAskEnabled(false);
  const res = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    uploadStatus.textContent = "Upload failed.";
    return;
  }
  await res.json();
  uploadStatus.textContent = "Upload received. Starting ingestion.";
  startStatusPolling();
});

existingBtn.addEventListener("click", async () => {
  const filename = existingSelect.value;
  if (!filename) {
    uploadStatus.textContent = "Select a file to ingest.";
    return;
  }
  uploadStatus.textContent = "Starting ingestion from existing file...";
  setAskEnabled(false);
  const res = await fetch(`${API_BASE}/ingest_existing`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename }),
  });
  if (!res.ok) {
    uploadStatus.textContent = "Failed to ingest existing file.";
    return;
  }
  await res.json();
  startStatusPolling();
});

askBtn.addEventListener("click", async () => {
  const question = questionInput.value.trim();
  if (!question) {
    answerPanel.textContent = "Ask a question first.";
    return;
  }
  answerPanel.textContent = "Thinking...";
  sourcesList.innerHTML = "";
  evidencePanel.textContent = "";
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) {
    answerPanel.textContent = "No answer yet. Check ingestion.";
    return;
  }
  const data = await res.json();
  renderAnswer(data.answer);
});

async function refreshDocs() {
  const res = await fetch(`${API_BASE}/docs`);
  if (!res.ok) return;
  const docs = await res.json();
  docList.innerHTML = "";
  docs.forEach((doc) => {
    const li = document.createElement("li");
    li.textContent = `${doc.title} (${doc.doc_id})`;
    docList.appendChild(li);
  });
}

async function refreshUploads() {
  const res = await fetch(`${API_BASE}/uploads`);
  if (!res.ok) return;
  const files = await res.json();
  existingSelect.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = files.length ? "Select a file" : "No uploaded files";
  existingSelect.appendChild(placeholder);
  files.forEach((file) => {
    const option = document.createElement("option");
    option.value = file;
    option.textContent = file;
    existingSelect.appendChild(option);
  });
}

async function pollStatus() {
  const res = await fetch(`${API_BASE}/status`);
  if (!res.ok) return;
  const status = await res.json();
  uploadStatus.textContent = status.message || status.stage;
  setAskEnabled(Boolean(status.ready));
  if (status.ready) {
    stopStatusPolling();
    await refreshDocs();
  }
}

function setAskEnabled(enabled) {
  askBtn.disabled = !enabled;
  askBtn.textContent = enabled ? "Ask" : "Ingesting...";
}

function startStatusPolling() {
  if (statusTimer) return;
  pollStatus();
  statusTimer = setInterval(pollStatus, 1500);
}

function stopStatusPolling() {
  if (!statusTimer) return;
  clearInterval(statusTimer);
  statusTimer = null;
}

function renderAnswer(answer) {
  answerPanel.textContent = answer.answer || "";
  sourcesList.innerHTML = "";
  answer.citations.forEach((citation, index) => {
    const div = document.createElement("div");
    div.className = "source-item";
    div.textContent = `Source ${index + 1} · ${citation.section_path || "Parent"} ${
      citation.page_number ? `p.${citation.page_number}` : ""
    }`;
    div.addEventListener("click", async () => {
      const res = await fetch(`${API_BASE}/parents/${citation.parent_id}`);
      if (!res.ok) return;
      const parent = await res.json();
      evidencePanel.textContent = parent.text;
    });
    sourcesList.appendChild(div);
  });
}

refreshDocs();
pollStatus();
refreshUploads();
