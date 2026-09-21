function extractAmazonProductId() {
    // Attempt to match ASIN in the URL
    const match = window.location.pathname.match(/\/dp\/([A-Z0-9]{10})/i);
    if (match) return match[1];
    
    const match2 = window.location.pathname.match(/\/product\/([A-Z0-9]{10})/i);
    if (match2) return match2[1];

    return null;
}

function createBadgeContainer() {
    const container = document.createElement('div');
    container.id = 'trust-badge-container';
    document.body.appendChild(container);
    return container;
}

function renderLoading(container) {
    container.innerHTML = `
        <div id="trust-badge-header">
            <span id="trust-badge-title">Trust-Aware Summary</span>
        </div>
        <div class="trust-badge-loading">
            Analyzing reviews & detecting fakes...<br>
            <small>(This may take a minute if models are cold starting)</small>
        </div>
    `;
}

function renderError(container, message) {
    container.innerHTML = `
        <div id="trust-badge-header">
            <span id="trust-badge-title">Trust-Aware Summary</span>
        </div>
        <div class="trust-badge-error">
            <strong>Error:</strong> ${message}
        </div>
    `;
}

function getScoreClass(score) {
    if (score > 0.2) return 'aspect-score-positive';
    if (score < -0.2) return 'aspect-score-negative';
    return 'aspect-score-neutral';
}

function renderSuccess(container, data) {
    let aspectsHtml = '';
    if (data.aspects && data.aspects.length > 0) {
        aspectsHtml = `
            <div class="trust-badge-aspects">
                <strong>Aspect Breakdown:</strong>
                ${data.aspects.map(a => `
                    <div class="aspect-item">
                        <span>${a.name}</span>
                        <span class="${getScoreClass(a.sentiment_score)}">${a.sentiment_score.toFixed(2)}</span>
                    </div>
                `).join('')}
            </div>
        `;
    }

    container.innerHTML = `
        <div id="trust-badge-header">
            <span id="trust-badge-title">Trust-Aware Summary</span>
        </div>
        <div class="trust-badge-rating">
            ★ ${data.trust_adjusted_rating.toFixed(2)}
        </div>
        <div class="trust-badge-raw">
            Raw Average: ★ ${data.raw_rating.toFixed(2)}
        </div>
        <div class="trust-badge-summary">
            ${data.summary_text}
        </div>
        ${aspectsHtml}
        <div class="trust-badge-footer">
            ⚠️ ${data.down_weighted_count} reviews down-weighted for low trust.
        </div>
    `;
}

async function init() {
    const productId = extractAmazonProductId();
    if (!productId) {
        console.log("Trust Summarizer: No product ID found on this page.");
        return; // Not a product page
    }

    const container = createBadgeContainer();
    renderLoading(container);

    try {
        const response = await fetch('http://localhost:8000/api/summarize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ product_id: productId, platform: 'amazon' })
        });

        if (!response.ok) {
            if (response.status === 503) {
                throw new Error("Local backend or Ollama is unreachable. Is 'ollama serve' running?");
            }
            if (response.status === 504) {
                throw new Error("Request timed out. The model took too long to respond.");
            }
            throw new Error(`Server returned ${response.status}`);
        }

        const data = await response.json();
        renderSuccess(container, data);

    } catch (err) {
        console.error("Trust Summarizer Error:", err);
        // Better message if fetch totally fails (CORS or server down)
        if (err.message === "Failed to fetch") {
            renderError(container, "Cannot connect to the FastAPI server. Is it running on port 8000?");
        } else {
            renderError(container, err.message);
        }
    }
}

// Wait a bit to ensure URL is stable (SPA navigation handling if needed)
setTimeout(init, 1000);
