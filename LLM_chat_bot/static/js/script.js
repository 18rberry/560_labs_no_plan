let pollInterval = null;

const fileInput = document.getElementById('file-input');
const uploadBtn = document.getElementById('upload-btn');
const resetBtn = document.getElementById('reset-btn');
const pdfList = document.getElementById('pdf-list');
const statusMsg = document.getElementById('status-msg');
const chatWindow = document.getElementById('chat-window');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

// ── File selection preview ──────────────────────────────────
fileInput.addEventListener('change', () => {
  pdfList.innerHTML = '';
  for (const file of fileInput.files) {
    const li = document.createElement('li');
    li.textContent = file.name;
    pdfList.appendChild(li);
  }
});

// ── Upload ──────────────────────────────────────────────────
uploadBtn.addEventListener('click', async () => {
  const files = fileInput.files;
  if (!files.length) {
    statusMsg.textContent = 'Please select at least one PDF file.';
    return;
  }

  const backend = document.querySelector('input[name="backend"]:checked').value;
  const formData = new FormData();
  for (const file of files) formData.append('files', file);
  formData.append('backend', backend);

  setUploadControls(true);
  statusMsg.textContent = 'Uploading...';
  setChat(false);

  try {
    const res = await fetch('/upload', { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json();
      statusMsg.textContent = 'Upload error: ' + (err.error || res.statusText);
      setUploadControls(false);
      return;
    }
    startPolling();
  } catch (e) {
    statusMsg.textContent = 'Network error: ' + e.message;
    setUploadControls(false);
  }
});

// ── Status polling ──────────────────────────────────────────
function startPolling() {
  if (pollInterval) clearInterval(pollInterval);
  pollInterval = setInterval(pollStatus, 2000);
  pollStatus(); // immediate first check
}

async function pollStatus() {
  try {
    const res = await fetch('/status');
    const data = await res.json();
    statusMsg.textContent = data.message;

    if (data.pdf_names && data.pdf_names.length) {
      pdfList.innerHTML = '';
      for (const name of data.pdf_names) {
        const li = document.createElement('li');
        li.textContent = name;
        pdfList.appendChild(li);
      }
    }

    if (data.status === 'ready') {
      clearInterval(pollInterval);
      pollInterval = null;
      setChat(true);
      setUploadControls(false);
    } else if (data.status === 'error') {
      clearInterval(pollInterval);
      pollInterval = null;
      setUploadControls(false);
    }
  } catch (e) {
    // network blip — keep polling
  }
}

// ── Chat ────────────────────────────────────────────────────
sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

async function sendMessage() {
  const text = userInput.value.trim();
  if (!text) return;

  renderMessage('user', text);
  userInput.value = '';
  setChat(false);

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });
    const data = await res.json();
    if (res.ok) {
      renderMessage('bot', data.answer);
    } else {
      renderMessage('bot', 'Error: ' + (data.error || res.statusText));
    }
  } catch (e) {
    renderMessage('bot', 'Network error: ' + e.message);
  } finally {
    setChat(true);
  }
}

// ── Reset ───────────────────────────────────────────────────
resetBtn.addEventListener('click', async () => {
  if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
  await fetch('/reset', { method: 'POST' });
  chatWindow.innerHTML = '';
  pdfList.innerHTML = '';
  fileInput.value = '';
  statusMsg.textContent = 'Upload PDFs to begin.';
  setChat(false);
  setUploadControls(false);
});

// ── Helpers ─────────────────────────────────────────────────
function renderMessage(role, text) {
  const div = document.createElement('div');
  div.className = 'message ' + role;
  div.textContent = text;

  const ts = document.createElement('span');
  ts.className = 'timestamp';
  ts.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  div.appendChild(ts);

  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function setChat(enabled) {
  userInput.disabled = !enabled;
  sendBtn.disabled = !enabled;
  if (enabled) userInput.focus();
}

function setUploadControls(disabled) {
  uploadBtn.disabled = disabled;
  fileInput.disabled = disabled;
}
