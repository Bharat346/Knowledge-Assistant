const queryForm = document.querySelector('#query-form');
const questionInput = document.querySelector('#question');
const conversation = document.querySelector('#conversation');
const uploadForm = document.querySelector('#upload-form');
const fileInput = document.querySelector('#file-input');
const toast = document.querySelector('#toast');
let toastTimer;

function showToast(message) {
  toast.textContent = message;
  toast.classList.add('visible');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('visible'), 2600);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
  return data;
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function setError(id, message = '') {
  const node = document.querySelector(id);
  node.textContent = message;
  node.hidden = !message;
}

async function loadDocuments() {
  const documents = await api('/api/documents');
  const list = document.querySelector('#document-list');
  document.querySelector('#document-count').textContent = documents.length;
  document.querySelector('#index-status').textContent = documents.length ? 'Sources indexed' : 'No sources';
  document.querySelector('#source-index-label').textContent = documents.length
    ? `${documents.length} source${documents.length === 1 ? '' : 's'} indexed`
    : 'Index ready';
  list.replaceChildren();
  if (!documents.length) {
    const empty = document.createElement('div');
    empty.className = 'empty-documents';
    empty.textContent = 'Add a document to begin.';
    list.append(empty);
    return;
  }
  for (const documentInfo of documents) {
    const row = document.createElement('article');
    row.className = 'document-row';
    const type = document.createElement('span');
    type.className = 'file-type';
    type.textContent = documentInfo.file_type;
    const info = document.createElement('div');
    info.className = 'document-info';
    const name = document.createElement('strong');
    name.textContent = documentInfo.filename;
    name.title = documentInfo.filename;
    const meta = document.createElement('span');
    meta.textContent = `${documentInfo.chunk_count} chunks · added ${new Date(documentInfo.created_at).toLocaleDateString()}`;
    info.append(name, meta);
    const size = document.createElement('span');
    size.className = 'document-size';
    size.textContent = formatBytes(documentInfo.size_bytes);
    const remove = document.createElement('button');
    remove.className = 'delete-button';
    remove.type = 'button';
    remove.title = `Delete ${documentInfo.filename}`;
    remove.setAttribute('aria-label', remove.title);
    remove.innerHTML = '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 6h12M8 6V4h4v2m2 0-.7 9.5H6.7L6 6m2.5 3v4m3-4v4"/></svg>';
    remove.addEventListener('click', async () => {
      remove.disabled = true;
      try {
        await api(`/api/documents/${encodeURIComponent(documentInfo.document_id)}`, { method: 'DELETE' });
        await loadDocuments();
        showToast('Document removed from the index');
      } catch (error) {
        remove.disabled = false;
        showToast(error.message);
      }
    });
    row.append(type, info, size, remove);
    list.append(row);
  }
}

function appendUserMessage(question) {
  document.querySelector('#empty-state')?.remove();
  const message = document.createElement('article');
  message.className = 'user-message';
  message.textContent = question;
  conversation.append(message);
}

function appendInlineMarkdown(parent, text) {
  const pattern = /(\*\*.+?\*\*|\*[^*]+\*|`[^`]+`|\[[^\]]+\]\(https?:\/\/[^\s)]+\))/g;
  let cursor = 0;
  for (const match of text.matchAll(pattern)) {
    const token = match[0];
    const start = match.index;
    parent.append(document.createTextNode(text.slice(cursor, start)));
    let node;
    if (token.startsWith('**')) {
      node = document.createElement('strong');
      node.textContent = token.slice(2, -2);
    } else if (token.startsWith('*')) {
      node = document.createElement('em');
      node.textContent = token.slice(1, -1);
    } else if (token.startsWith('`')) {
      node = document.createElement('code');
      node.textContent = token.slice(1, -1);
    } else {
      const parts = token.match(/^\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)$/);
      node = document.createElement('a');
      node.textContent = parts[1];
      node.href = parts[2];
      node.rel = 'noopener noreferrer';
      node.target = '_blank';
    }
    parent.append(node);
    cursor = start + token.length;
  }
  parent.append(document.createTextNode(text.slice(cursor)));
}

function renderMarkdown(parent, text) {
  const lines = text.split(/\r?\n/);
  let paragraph = [];
  let list = null;
  let listType = '';

  const flushParagraph = () => {
    if (!paragraph.length) return;
    const block = document.createElement('p');
    paragraph.forEach((line, index) => {
      if (index) block.append(document.createElement('br'));
      appendInlineMarkdown(block, line);
    });
    parent.append(block);
    paragraph = [];
  };

  for (const line of lines) {
    const unordered = line.match(/^\s*[-*]\s+(.+)$/);
    const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/);
    const item = unordered || ordered;
    if (item) {
      flushParagraph();
      const type = unordered ? 'ul' : 'ol';
      if (listType !== type) {
        list = document.createElement(type);
        parent.append(list);
        listType = type;
      }
      const listItem = document.createElement('li');
      appendInlineMarkdown(listItem, item[1]);
      list.append(listItem);
      continue;
    }
    list = null;
    listType = '';
    if (!line.trim()) {
      flushParagraph();
      continue;
    }
    const heading = line.match(/^\s{0,3}(#{1,3})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      const block = document.createElement(`h${heading[1].length + 2}`);
      appendInlineMarkdown(block, heading[2]);
      parent.append(block);
      continue;
    }
    paragraph.push(line);
  }
  flushParagraph();
}

function appendAssistantMessage({ answer, sources = [], requestId = '', error = false, pending = false, markdown = false }) {
  const message = document.createElement('article');
  message.className = `assistant-message${error ? ' assistant-error' : ''}${pending ? ' assistant-pending' : ''}`;
  const header = document.createElement('div');
  header.className = 'message-header';
  const label = document.createElement('span');
  label.className = 'assistant-label';
  label.textContent = 'K.  ASSISTANT';
  header.append(label);
  if (requestId) {
    const request = document.createElement('span');
    request.className = 'request-id';
    request.textContent = `ID ${requestId.slice(0, 8)}`;
    header.append(request);
  }
  const content = document.createElement('div');
  content.className = 'message-answer';
  if (markdown) renderMarkdown(content, answer);
  else content.textContent = answer;
  message.append(header, content);

  if (sources.length) {
    const sourceHeader = document.createElement('div');
    sourceHeader.className = 'answer-source-heading';
    sourceHeader.textContent = `SOURCES  ${sources.length}`;
    const citations = document.createElement('ol');
    citations.className = 'answer-citations';
    for (const source of sources) {
      const item = document.createElement('li');
      const name = document.createElement('strong');
      name.textContent = source.document;
      const details = document.createElement('span');
      details.textContent = `page ${source.page ?? 'n/a'} · ${source.chunk_id}`;
      item.append(name, details);
      citations.append(item);
    }
    message.append(sourceHeader, citations);
  }
  conversation.append(message);
  message.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  return message;
}

function renderAnswer(result) {
  appendAssistantMessage({ answer: result.answer, sources: result.sources, requestId: result.request_id, markdown: true });
  document.querySelector('#dense-count').textContent = result.dense_count;
  document.querySelector('#bm25-count').textContent = result.bm25_count;
  document.querySelector('#fused-count').textContent = result.fused_count;
  document.querySelector('#reranked-count').textContent = result.reranked_count;
  document.querySelector('#total-latency').textContent = result.total_latency_ms.toFixed(0);
  const stages = document.querySelector('#latency-stages');
  stages.replaceChildren();
  for (const [stage, milliseconds] of Object.entries(result.latency)) {
    const item = document.createElement('span');
    item.textContent = `${stage.replaceAll('_', ' ')} ${milliseconds.toFixed(1)} ms`;
    stages.append(item);
  }
}

queryForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;
  const button = queryForm.querySelector('button[type="submit"]');
  button.disabled = true;
  button.querySelector('span').textContent = 'Thinking';
  appendUserMessage(question);
  questionInput.value = '';
  const pendingMessage = appendAssistantMessage({ answer: 'Searching your indexed sources…', pending: true });
  try {
    const result = await api('/api/query', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    pendingMessage.remove();
    renderAnswer(result);
  } catch (error) {
    pendingMessage.remove();
    appendAssistantMessage({ answer: error.message, error: true });
  } finally {
    button.disabled = false;
    button.querySelector('span').textContent = 'Ask';
  }
});

questionInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    queryForm.requestSubmit();
  }
});

async function uploadFile(file) {
  if (!file) return;
  setError('#upload-error');
  const form = new FormData();
  form.append('file', file);
  const choose = document.querySelector('#choose-file');
  choose.disabled = true;
  choose.textContent = 'Indexing';
  try {
    const result = await api('/api/documents/upload', { method: 'POST', body: form });
    await loadDocuments();
    showToast(result.duplicate ? 'This file is already indexed' : `${result.filename} indexed`);
  } catch (error) {
    setError('#upload-error', error.message);
  } finally {
    choose.disabled = false;
    choose.textContent = 'Browse files';
    fileInput.value = '';
  }
}

document.querySelector('#choose-file').addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => uploadFile(fileInput.files[0]));
uploadForm.addEventListener('submit', (event) => event.preventDefault());
uploadForm.addEventListener('dragover', (event) => { event.preventDefault(); uploadForm.classList.add('dragging'); });
uploadForm.addEventListener('dragleave', () => uploadForm.classList.remove('dragging'));
uploadForm.addEventListener('drop', (event) => {
  event.preventDefault();
  uploadForm.classList.remove('dragging');
  uploadFile(event.dataTransfer.files[0]);
});

loadDocuments().catch((error) => {
  document.querySelector('#index-status').textContent = 'Unavailable';
  document.querySelector('#source-index-label').textContent = 'Index unavailable';
  showToast(error.message);
});