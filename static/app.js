// Brain Tumor Detection — Frontend Logic
let modelsLoaded = [];

document.addEventListener('DOMContentLoaded', () => {
    fetchModelInfo();
    setupDragDrop('upload-area-classify', 'file-classify', handleClassifySelect);
    setupDragDrop('upload-area-batch', 'file-batch', handleBatchSelect);
    setupDragDrop('upload-area-compare', 'file-compare', handleCompareSelect);
});

async function fetchModelInfo() {
    try {
        const res = await fetch('/api/model-info');
        const data = await res.json();

        const badge = document.getElementById('model-badge');
        const text = document.getElementById('model-status-text');

        if (data.status === 'ready') {
            badge.querySelector('.badge-dot').classList.add('ready');
            text.textContent = `${data.models.join(' & ')} Ready`;
            modelsLoaded = data.models;
        }
    } catch (e) {
        document.getElementById('model-status-text').textContent = "Offline";
    }
}

// ---- Tabs ----
function switchTab(tabId) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');

    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.getElementById(`page-${tabId}`).classList.add('active');

    if (tabId === 'architecture') loadArchitecture();
    if (tabId === 'dataset') loadDataset();
    if (tabId === 'history') loadHistory();
}

// ---- File Upload Setup ----
function setupDragDrop(areaId, inputId, callback) {
    const area = document.getElementById(areaId);
    const input = document.getElementById(inputId);
    if (!area) return;

    area.onclick = () => input.click();
    input.onchange = (e) => callback(e.target.files);

    area.ondragover = (e) => { e.preventDefault(); area.style.borderColor = "var(--accent-indigo)"; };
    area.ondragleave = () => area.style.borderColor = "";
    area.ondrop = (e) => {
        e.preventDefault();
        area.style.borderColor = "";
        input.files = e.dataTransfer.files;
        callback(e.dataTransfer.files);
    };
}

// ---- 1. Single Classify ----
let classifyFile = null;
function handleClassifySelect(files) {
    if (!files.length) return;
    classifyFile = files[0];

    document.getElementById('upload-area-classify').style.display = 'none';
    document.getElementById('classify-preview').style.display = 'block';

    const reader = new FileReader();
    reader.onload = e => document.getElementById('classify-img').src = e.target.result;
    reader.readAsDataURL(classifyFile);

    document.getElementById('classify-fname').textContent = classifyFile.name;
    document.getElementById('classify-results').style.display = 'none';
}

function resetClassify() {
    classifyFile = null;
    document.getElementById('upload-area-classify').style.display = 'block';
    document.getElementById('classify-preview').style.display = 'none';
    document.getElementById('classify-results').style.display = 'none';
}

async function classifySingle() {
    if (!classifyFile) return;
    document.getElementById('scan-overlay').style.display = 'flex';
    document.getElementById('classify-btn').disabled = true;

    const formData = new FormData();
    formData.append('file', classifyFile);

    const res = await fetch('/api/predict', { method: 'POST', body: formData });
    const data = await res.json();

    setTimeout(() => {
        document.getElementById('scan-overlay').style.display = 'none';
        document.getElementById('classify-btn').disabled = false;

        if (data.error) return alert(data.error);

        const isTumor = data.prediction.includes("Tumor");
        const results = document.getElementById('classify-results');
        results.style.display = 'block';

        let probsHtml = '<div class="prob-container">';
        if (data.probabilities) {
            Object.entries(data.probabilities).forEach(([cls, prob]) => {
                const isT = cls.includes("Tumor");
                probsHtml += `
                    <div class="prob-row">
                        <div class="prob-label"><span>${isT ? '🔴' : '🟢'} ${cls}</span><span style="font-family:monospace;">${prob}%</span></div>
                        <div class="prob-bar-bg"><div class="prob-bar-fill ${isT ? 'prob-fill-tumor' : 'prob-fill-healthy'}" style="width: 0%;" data-target="${prob}%"></div></div>
                    </div>
                `;
            });
        }
        probsHtml += '</div>';

        results.innerHTML = `
            <div class="result-grid">
                <div class="card ${isTumor ? 'tumor' : 'healthy'}">
                    <div class="pred-icon">${isTumor ? '⚠️' : '✅'}</div>
                    <div class="pred-text">${data.prediction}</div>
                    <div class="muted">Analyzed by ${data.model_type}</div>
                </div>
                <div class="card">
                    <div class="muted" style="margin-bottom:10px;">Class Probabilities</div>
                    ${probsHtml}
                </div>
            </div>
        `;

        // Trigger animations for bars
        setTimeout(() => {
            document.querySelectorAll('.prob-bar-fill').forEach(bar => {
                bar.style.width = bar.getAttribute('data-target');
            });
        }, 100);
    }, 1000);
}

// ---- 2. Batch Predict ----
async function handleBatchSelect(files) {
    if (!files.length) return;
    document.getElementById('upload-area-batch').innerHTML = `<h3>Processing ${files.length} images...</h3>`;

    const formData = new FormData();
    for (let f of files) formData.append('files', f);

    const res = await fetch('/api/batch-predict', { method: 'POST', body: formData });
    const data = await res.json();

    let html = `
        <div class="card" style="margin-bottom:20px; display:flex; justify-content:space-around;">
            <div><div>Total Scans</div><div style="font-size:2rem; font-weight:bold;">${data.summary.total}</div></div>
            <div><div>Tumor Detected</div><div style="font-size:2rem; font-weight:bold; color:var(--accent-red);">${data.summary.tumor}</div></div>
            <div><div>Healthy</div><div style="font-size:2rem; font-weight:bold; color:var(--accent-emerald);">${data.summary.healthy}</div></div>
        </div>
        <div class="batch-list">
    `;

    data.results.forEach(r => {
        if (r.error) {
            html += `<div class="batch-item"><span>${r.filename}</span><span class="text-red">Error</span></div>`;
        } else {
            const isT = r.prediction.includes("Tumor");
            html += `
                <div class="batch-item ${isT ? 'Tumor' : 'Healthy'}">
                    <span style="font-family:monospace;">${r.filename}</span>
                    <span><span class="${isT ? 'text-red' : 'text-green'}">${r.prediction}</span> (${r.confidence}%)</span>
                </div>`;
        }
    });

    html += `</div><button class="btn btn-ghost" style="margin-top:20px; width:100%;" onclick="location.reload()">Upload New Batch</button>`;

    document.getElementById('upload-area-batch').style.display = 'none';
    document.getElementById('batch-results').style.display = 'block';
    document.getElementById('batch-results').innerHTML = html;
}

// ---- 3. Compare Models ----
async function handleCompareSelect(files) {
    if (!files.length) return;
    document.getElementById('upload-area-compare').innerHTML = `<h3>Analyzing with multiple models...</h3>`;

    const formData = new FormData();
    formData.append('file', files[0]);

    const res = await fetch('/api/predict-compare', { method: 'POST', body: formData });
    const data = await res.json();

    let html = `<div class="compare-grid">`;
    Object.entries(data.comparisons).forEach(([modelName, r]) => {
        if (r.error) {
            html += `<div class="card"><div class="pred-text">${modelName}</div><div class="muted text-red">Failed</div></div>`;
        } else {
            const isT = r.prediction.includes("Tumor");
            html += `
                <div class="card ${isT ? 'tumor' : 'healthy'}">
                    <div style="font-size: 1.2rem; font-weight:bold; margin-bottom:10px; border-bottom:1px solid var(--border-subtle); padding-bottom:10px;">${modelName} Model</div>
                    <div class="pred-icon">${isT ? '⚠️' : '✅'}</div>
                    <div class="pred-text">${r.prediction}</div>
                    <div class="muted" style="margin-top:10px; font-family:monospace;">Confidence: ${r.confidence}%</div>
                </div>
            `;
        }
    });
    html += `</div><button class="btn btn-ghost" style="margin-top:20px; width:100%;" onclick="location.reload()">Compare Another</button>`;

    document.getElementById('upload-area-compare').style.display = 'none';
    document.getElementById('compare-results').style.display = 'block';
    document.getElementById('compare-results').innerHTML = html;
}

// ---- 4. Architecture ----
async function loadArchitecture() {
    const el = document.getElementById('arch-content');
    if (el.innerHTML !== '<p class="muted">Loading architecture...</p>') return;

    try {
        const res = await fetch('/api/model-architecture');
        const data = await res.json();

        let html = '';
        Object.entries(data).forEach(([name, info]) => {
            html += `
                <div class="card" style="margin-bottom:30px; text-align:left;">
                    <h3 style="font-size:1.5rem; margin-bottom:10px;">► ${name} Architecture</h3>
                    <p class="muted" style="margin-bottom:20px;">${info.description}</p>
                    
                    <div class="arch-header">
                        <div class="arch-stat">Total Parameters <span>${info.total_params}</span></div>
                        <div class="arch-stat">Trainable Params <span>${info.trainable_params}</span></div>
                    </div>
                    
                    <h4 style="margin-bottom:10px; color:var(--text-secondary);">Layer Pipeline</h4>
                    <div class="arch-layers">
            `;
            info.layers.forEach(l => {
                html += `<div class="arch-layer"><span style="color:var(--accent-cyan)">${l.name}</span><span style="color:var(--accent-amber)">${l.type}</span><span class="muted">${l.details}</span></div>`;
            });
            html += `</div></div>`;
        });
        el.innerHTML = html;
    } catch (e) { el.innerHTML = "Error loading architecture info."; }
}

// ---- 5. Dataset ----
async function loadDataset() {
    const el = document.getElementById('dataset-content');
    if (el.innerHTML !== '<p class="muted">Loading dataset info...</p>') return;

    try {
        const res = await fetch('/api/dataset-stats');
        const data = await res.json();

        if (data.total_train === 0) {
            el.innerHTML = `<div class="card"><p class="muted">Cannot find dataset directory locally. Are the images extracted?</p></div>`;
            return;
        }

        let html = `
            <div class="dataset-grid">
                <div class="card">
                    <h3 style="margin-bottom:10px;">Training Set</h3>
                    <div style="font-size:2.5rem; font-weight:bold; color:var(--accent-indigo);">${data.total_train}</div>
                    <div class="muted">Images</div>
                </div>
                <div class="card">
                    <h3 style="margin-bottom:10px;">Validation Set</h3>
                    <div style="font-size:2.5rem; font-weight:bold; color:var(--accent-cyan);">${data.total_val}</div>
                    <div class="muted">Images</div>
                </div>
            </div>
            <h3 style="margin: 30px 0 10px;">Sample Images</h3>
            <div class="img-grid">
        `;

        data.sample_images.forEach(img => {
            const isT = img.label.includes("Tumor");
            html += `
                <div style="position:relative;">
                    <img src="${img.data}" alt="${img.label}">
                    <div style="position:absolute; bottom:5px; left:5px; right:5px; background:rgba(0,0,0,0.8); padding:4px; border-radius:4px; font-size:0.75rem; text-align:center; color:${isT ? 'var(--accent-red)' : 'var(--accent-emerald)'}">${img.label}</div>
                </div>
            `;
        });
        el.innerHTML = html + `</div>`;
    } catch (e) { el.innerHTML = "Error loading dataset info."; }
}

// ---- 6. History ----
async function loadHistory() {
    const el = document.getElementById('history-content');
    if (el.innerHTML !== '<p class="muted">Loading training history...</p>') return;

    try {
        const res = await fetch('/api/training-history');
        const data = await res.json();

        if (!data.available) {
            el.innerHTML = `<div class="card"><p class="muted">No training history plots found. Try completing the training process first.</p></div>`;
            return;
        }

        let html = `<div class="history-grid">`;
        Object.entries(data.plots).forEach(([name, img]) => {
            const isCM = name.includes('Confusion');
            html += `<div class="${isCM ? '' : 'full-width-img'}">
                        <h4 style="margin-bottom:10px; text-transform:capitalize;">${name.replace(/_/g, ' ')}</h4>
                        <img src="${img}" alt="${name}">
                    </div>`;
        });
        el.innerHTML = html + `</div>`;
    } catch (e) { el.innerHTML = "Error loading training history."; }
}
