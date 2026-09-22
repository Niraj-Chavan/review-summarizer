// Trust-Aware Review Summarizer — Premium Dashboard
const SHIELD_ICON = `<svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/></svg>`;
const STAR_ICON = `<svg width="16" height="16" fill="currentColor" viewBox="0 0 20 20"><path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/></svg>`;
const ALERT_ICON = `<svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>`;
const CHEVRON_ICON = `<svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M19 9l-7 7-7-7"/></svg>`;
const CLOSE_ICON = `<svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/></svg>`;
const MINUS_ICON = `<svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M20 12H4"/></svg>`;
const CHECK_ICON = `<svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2.4" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/></svg>`;

let isMinimized = false;
let lastData = null;
let expandedBreakdown = true;

function extractAmazonProductId() {
  const m1 = window.location.pathname.match(/\/dp\/([A-Z0-9]{10})/i);
  if (m1) return m1[1];
  const m2 = window.location.pathname.match(/\/product\/([A-Z0-9]{10})/i);
  if (m2) return m2[1];
  const m3 = window.location.search.match(/asin=([A-Z0-9]{10})/i);
  if (m3) return m3[1];
  return null;
}

function extractProductTitle() {
  const el = document.querySelector('#productTitle');
  if (el && el.innerText.trim().length > 5) return el.innerText.trim().slice(0,120);
  const h1 = document.querySelector('h1');
  if (h1 && h1.innerText.trim().length > 5) return h1.innerText.trim().slice(0,120);
  return document.title.slice(0,120);
}

function createBadgeContainer() {
  const c = document.createElement('div');
  c.id = 'trust-badge-container';
  document.body.appendChild(c);
  return c;
}

function esc(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function getScoreMeta(score) {
  if (score > 0.2) return { label: 'Positive', cls: 'positive', dot: '#10b981' };
  if (score < -0.2) return { label: 'Negative', cls: 'negative', dot: '#ef4444' };
  return { label: 'Neutral', cls: 'neutral', dot: '#94a3b8' };
}

function starRow(rating) {
  const filled = Math.round(rating);
  let html = '';
  for (let i = 1; i <= 5; i++) {
    const cls = i <= filled ? 'star-filled' : 'star-empty';
    html += `<svg class="${cls}" fill="currentColor" viewBox="0 0 20 20"><path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/></svg>`;
  }
  return `<div class="stars-row">${html}</div>`;
}

function renderHeader() {
  return `
    <div id="trust-badge-header">
      <div class="header-left">
        <div class="header-icon-wrap">${SHIELD_ICON}</div>
        <div class="header-titles">
          <span id="trust-badge-title">Trust-Aware Summary</span>
          <span class="header-subtitle"><span class="live-dot"></span> Live • Explainable AI</span>
        </div>
      </div>
      <div class="header-actions">
        <button class="icon-btn" id="trust-minimize" title="Minimize">${MINUS_ICON}</button>
        <button class="icon-btn" id="trust-close" title="Close">${CLOSE_ICON}</button>
      </div>
    </div>
  `;
}

function renderFAB(data) {
  const rating = data ? data.trust_adjusted_rating.toFixed(1) : '…';
  return `
    <div class="trust-fab" id="trust-fab">
      <span class="fab-icon">${SHIELD_ICON}</span>
      <span>Trust Summary</span>
      <span class="fab-rating">${STAR_ICON} ${rating}</span>
      <span class="fab-chevron">${CHEVRON_ICON}</span>
    </div>
  `;
}

function renderLoading(container) {
  container.classList.remove('minimized');
  container.innerHTML = `
    ${renderHeader()}
    <div class="trust-scroll">
      <div class="trust-badge-loading">
        <div class="spinner"></div>
        <div>
          <div class="loading-title">Analyzing reviews…</div>
          <div class="loading-sub">Detecting fake reviews, scoring trust &<br>building aspect sentiment</div>
        </div>
        <div style="width:100%; display:grid; gap:8px; margin-top:4px;">
          <div class="shimmer"></div>
          <div class="shimmer" style="width:78%"></div>
          <div class="shimmer" style="width:92%"></div>
        </div>
      </div>
    </div>
  `;
  bindHeaderActions(container);
}

function renderError(container, message, canRetry = true) {
  container.classList.remove('minimized');
  container.innerHTML = `
    ${renderHeader()}
    <div class="trust-scroll">
      <div class="trust-badge-error">
        <strong>Connection failed</strong><br>
        <span style="opacity:0.9">${esc(message)}</span>
        ${canRetry ? `<button class="retry-btn" id="trust-retry">↻ Retry</button>` : ''}
        <div style="margin-top:10px; font-size:11px; opacity:0.7; line-height:1.4;">
          Make sure the FastAPI backend is running:<br>
          <code style="background:white; padding:2px 6px; border-radius:6px; border:1px solid #fecaca;">uvicorn src.api.main:app --port 8000</code>
          &amp; <code style="background:white; padding:2px 6px; border-radius:6px; border:1px solid #fecaca;">ollama serve</code>
        </div>
      </div>
    </div>
  `;
  bindHeaderActions(container);
  const retry = container.querySelector('#trust-retry');
  if (retry) retry.addEventListener('click', () => init(true));
}

function bindHeaderActions(container) {
  const minBtn = container.querySelector('#trust-minimize');
  const closeBtn = container.querySelector('#trust-close');
  if (minBtn) minBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    setMinimized(true);
  });
  if (closeBtn) closeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    container.style.display = 'none';
    // Re-show after 8s as gentle nudge
    setTimeout(() => { container.style.display = 'flex'; }, 8000);
  });
  const fab = container.querySelector('#trust-fab');
  if (fab) fab.addEventListener('click', () => setMinimized(false));
}

function setMinimized(min) {
  isMinimized = min;
  const container = document.getElementById('trust-badge-container');
  if (!container) return;
  if (min && lastData) {
    container.classList.add('minimized');
    container.innerHTML = renderFAB(lastData);
    bindHeaderActions(container);
  } else if (lastData) {
    renderSuccess(container, lastData);
  } else {
    renderLoading(container);
  }
}

function renderSuccess(container, data) {
  lastData = data;
  if (isMinimized) {
    container.classList.add('minimized');
    container.innerHTML = renderFAB(data);
    bindHeaderActions(container);
    return;
  }
  container.classList.remove('minimized');

  const raw = data.raw_rating ?? 0;
  const adj = data.trust_adjusted_rating ?? 0;
  const delta = adj - raw;
  const deltaAbs = Math.abs(delta).toFixed(1);
  let deltaCls = 'flat', deltaArrow = '→', deltaText = `No change`;
  if (delta < -0.05) { deltaCls = 'down'; deltaArrow = '↓'; deltaText = `${deltaArrow} ${deltaAbs} vs raw`; }
  else if (delta > 0.05) { deltaCls = 'up'; deltaArrow = '↑'; deltaText = `${deltaArrow} ${deltaAbs} vs raw`; }

  const aspects = Array.isArray(data.aspects) ? data.aspects : [];
  const flagged = data.down_weighted_count ?? 0;
  const totalReviews = data.total_reviews ?? (flagged + Math.max(4 - flagged, 2));
  const trusted = Math.max(totalReviews - flagged, 1);
  const trustPct = Math.round((trusted / Math.max(totalReviews,1)) * 100);
  const flaggedPct = 100 - trustPct;
  const latency = data.latency_seconds ? `${data.latency_seconds}s` : '—';
  const model = data.model || 'llama3.1:8b';

  // Gauge degree: trustPct -> 360deg
  const gaugeDeg = Math.round(trustPct * 3.6);

  let aspectsHtml = '';
  if (aspects.length > 0) {
    aspectsHtml = `
      <div class="trust-badge-aspects">
        <div class="aspects-head">
          <span class="aspects-title">Aspect breakdown</span>
          <span class="aspects-count">${aspects.length} aspects</span>
        </div>
        <div class="aspect-list">
          ${aspects.map(a => {
            const m = getScoreMeta(a.sentiment_score);
            // Map -1..1 to 0..100% width, center at 50%
            const pct = Math.round(((a.sentiment_score + 1) / 2) * 100);
            const barWidth = Math.max(8, Math.min(100, pct));
            return `
              <div class="aspect-item">
                <div class="aspect-row">
                  <span class="aspect-name"><span class="aspect-dot" style="background:${m.dot}"></span>${esc(a.name)}</span>
                  <span class="aspect-score aspect-score-${m.cls}">${m.label} ${(a.sentiment_score > 0 ? '+' : '')}${a.sentiment_score.toFixed(1)}</span>
                </div>
                <div class="aspect-bar-track">
                  <div class="aspect-bar-fill ${m.cls}" style="width:${barWidth}%"></div>
                </div>
              </div>
            `;
          }).join('')}
        </div>
      </div>
    `;
  }

  container.innerHTML = `
    ${renderHeader()}
    <div class="trust-scroll">
      <div class="trust-hero">
        <div class="rating-main">
          <span class="rating-label">Trust-adjusted rating</span>
          <div class="rating-row">
            <div class="trust-badge-rating"><span class="star">${STAR_ICON}</span> ${adj.toFixed(1)}</div>
            <span class="rating-delta ${deltaCls}">${deltaText}</span>
          </div>
          <div class="trust-badge-raw">Raw average ${raw.toFixed(1)} / 5</div>
          ${starRow(adj)}
          <div class="meta-row">
            <span class="meta-chip">${CHECK_ICON} ${trusted} trusted</span>
            <span class="meta-chip" style="color:#b45309; background:#fffbeb; border-color:#fde68a;">${ALERT_ICON} ${flagged} flagged</span>
          </div>
        </div>
        <div class="trust-gauge-wrap">
          <div class="gauge" style="--deg:${gaugeDeg}deg">
            <div class="gauge-inner">
              <div>
                <div class="gauge-value">${trustPct}%</div>
                <div class="gauge-label">Trusted</div>
              </div>
            </div>
          </div>
          <div class="gauge-caption">${flagged} reviews<br>down-weighted</div>
        </div>
      </div>

      <div class="trust-badge-summary">
        <span class="summary-quote">AI Summary</span>
        <p>${esc(data.summary_text || 'No summary available.')}</p>
      </div>

      ${aspectsHtml}

      <div class="trust-breakdown">
        <div class="breakdown-header" id="breakdown-toggle">
          <span class="breakdown-title">${ALERT_ICON} Trust breakdown</span>
          <span class="breakdown-chevron ${expandedBreakdown ? 'open' : ''}">${CHEVRON_ICON}</span>
        </div>
        <div class="breakdown-body" id="breakdown-body" style="${expandedBreakdown ? '' : 'display:none'}">
          <div class="trust-bar">
            <div class="trust-bar-trusted" style="width:${trustPct}%"></div>
            <div class="trust-bar-flagged" style="width:${flaggedPct}%"></div>
          </div>
          <div class="breakdown-stats">
            <div class="stat-card">
              <div class="stat-value" style="color:var(--success)">${trusted}</div>
              <div class="stat-label">Trusted</div>
            </div>
            <div class="stat-card">
              <div class="stat-value" style="color:var(--danger)">${flagged}</div>
              <div class="stat-label">Flagged</div>
            </div>
          </div>
          <div class="breakdown-list">
            <div class="breakdown-item">${CHECK_ICON} Verified purchase & burstiness analyzed</div>
            <div class="breakdown-item">${CHECK_ICON} Near-duplicate & templated phrase detection</div>
            <div class="breakdown-item">${ALERT_ICON} Collusion graph (Louvain) risk scored</div>
          </div>
        </div>
      </div>

      <div class="trust-footer">
        <div class="trust-badge-footer">
          <span style="display:grid; place-items:center; background:white; border-radius:50%; width:22px; height:22px; border:1px solid #fecaca; flex-shrink:0;">${ALERT_ICON}</span>
          <span><strong>${flagged} reviews</strong> down-weighted for low trust — weighted rating reflects only credible opinions.</span>
        </div>
        <div class="footer-meta">
          <span>⚡ ${latency} • <span class="model-badge">${esc(model)}</span></span>
          <span style="opacity:0.9">Trust-Aware RAG • FAISS + BM25 + XGBoost</span>
        </div>
      </div>
    </div>
  `;
  bindHeaderActions(container);
  const toggle = container.querySelector('#breakdown-toggle');
  const body = container.querySelector('#breakdown-body');
  const chev = container.querySelector('.breakdown-chevron');
  if (toggle && body && chev) {
    toggle.addEventListener('click', () => {
      expandedBreakdown = !expandedBreakdown;
      body.style.display = expandedBreakdown ? 'grid' : 'none';
      chev.classList.toggle('open', expandedBreakdown);
    });
  }
}

async function fetchWithFallback(productId) {
  const title = extractProductTitle();
  const endpoints = [
    'http://localhost:8000/api/summarize',
    'http://127.0.0.1:8000/api/summarize',
    'http://localhost:8001/api/summarize',
    'http://127.0.0.1:8001/api/summarize',
    'http://localhost:8080/api/summarize',
    'http://127.0.0.1:8080/api/summarize'
  ];
  let lastErr = null;
  for (const url of endpoints) {
    try {
      const resp = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ product_id: productId, platform: 'amazon', product_title: title })
      });
      if (!resp.ok) {
        if (resp.status === 503) throw new Error("Backend or Ollama unreachable. Is 'ollama serve' running?");
        if (resp.status === 504) throw new Error("Request timed out — model is warming up. Try again.");
        throw new Error(`Server returned ${resp.status}`);
      }
      return await resp.json();
    } catch (e) {
      lastErr = e;
      if (e.message && e.message.includes('503')) throw e;
      // try next endpoint only on network failure
      if (e.message !== 'Failed to fetch' && !e.message.includes('NetworkError')) {
        // if it's an HTTP error, don't fallback
        if (e.message.startsWith('Server returned') || e.message.includes('timed out')) throw e;
      }
      continue;
    }
  }
  throw lastErr || new Error('Failed to fetch');
}

async function init(isRetry = false) {
  const productId = extractAmazonProductId();
  if (!productId) return;
  const container = document.getElementById('trust-badge-container') || createBadgeContainer();
  if (!isRetry) lastData = null;
  renderLoading(container);
  try {
    const data = await fetchWithFallback(productId);
    renderSuccess(container, data);
  } catch (err) {
    if (err.message === 'Failed to fetch') {
      renderError(container, 'Cannot connect to the local API. The backend appears to be offline.');
    } else {
      renderError(container, err.message || 'Unexpected error');
    }
  }
}

// Handle SPA navigation on Amazon
let lastUrl = location.href;
setInterval(() => {
  if (location.href !== lastUrl) {
    lastUrl = location.href;
    const old = document.getElementById('trust-badge-container');
    if (old) old.remove();
    isMinimized = false;
    lastData = null;
    setTimeout(() => init(), 800);
  }
}, 1000);

setTimeout(init, 1100);
