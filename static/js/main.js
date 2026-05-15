/* ═══════════════════════════
   ALGOSENSE – MAIN JS
═══════════════════════════ */

// ── State ──────────────────────────────────────────────────
let currentFile   = null;
let analysisData  = null;
let barChartInst  = null;
let radarChartInst= null;
let timingInst    = null;
let importanceInst= null;

const CHART_COLORS = [
  '#7c3aed','#06b6d4','#10b981','#f59e0b','#ef4444','#a855f7','#22d3ee'
];
const ALGO_ICONS = {
  'Random Forest':       '🌲',
  'SVM':                 '🔵',
  'KNN':                 '📍',
  'Decision Tree':       '🌿',
  'Logistic Regression': '📈',
  'Gradient Boosting':   '🚀',
  'Naive Bayes':         '🔔',
};

// ── DOM Refs ───────────────────────────────────────────────
const dropZone     = document.getElementById('dropZone');
const fileInput    = document.getElementById('fileInput');
const fileInfo     = document.getElementById('fileInfo');
const fileNameEl   = document.getElementById('fileName');
const removeFileBtn= document.getElementById('removeFile');
const browseBtn    = document.getElementById('browseBtn');
const targetRow    = document.getElementById('targetRow');
const targetSelect = document.getElementById('targetSelect');
const analyzeBtn   = document.getElementById('analyzeBtn');
const analyzeBtnTxt= document.getElementById('analyzeBtnText');
const btnSpinner   = document.getElementById('btnSpinner');
const resultsSection = document.getElementById('resultsSection');
const uploadCard   = document.getElementById('uploadCard');

// ── File Handling ──────────────────────────────────────────
browseBtn.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('click', (e) => {
  if (e.target !== browseBtn && e.target !== removeFileBtn) fileInput.click();
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f && f.name.endsWith('.csv')) handleFile(f);
  else showToast('Please drop a CSV file', true);
});

removeFileBtn.addEventListener('click', (e) => { e.stopPropagation(); resetFile(); });

async function handleFile(file) {
  currentFile = file;
  fileNameEl.textContent = file.name;
  fileInfo.style.display = 'block';
  analyzeBtn.disabled = false;
  analyzeBtnTxt.textContent = 'Analyze Dataset';

  // Fetch columns
  const fd = new FormData(); fd.append('file', file);
  try {
    const res = await fetch('/columns', { method: 'POST', body: fd });
    const data = await res.json();
    if (data.columns) {
      targetSelect.innerHTML = data.columns.map(c => `<option value="${c}">${c}</option>`).join('');
      // Pre-select last column (common convention for target)
      targetSelect.selectedIndex = data.columns.length - 1;
      targetRow.style.display = 'flex';
    }
  } catch (_) {}
}

function resetFile() {
  currentFile = null; fileInput.value = '';
  fileInfo.style.display = 'none'; targetRow.style.display = 'none';
  analyzeBtn.disabled = true; analyzeBtnTxt.textContent = 'Select a file first';
}

// ── Sample Datasets ────────────────────────────────────────
async function loadSample(name) {
  showToast(`⏳ Loading ${name} dataset…`);
  try {
    const res = await fetch(`/sample/${name}`);
    if (!res.ok) throw new Error('fetch failed');
    const blob = await res.blob();
    const file = new File([blob], `${name}.csv`, { type: 'text/csv' });
    await handleFile(file);
    showToast(`✅ ${name} loaded! Click Analyze.`);
  } catch (e) {
    showToast('❌ Could not load sample. Try uploading your own CSV.', true);
  }
}

// ── Analyze ────────────────────────────────────────────────
analyzeBtn.addEventListener('click', runAnalysis);

async function runAnalysis() {
  if (!currentFile) return;
  setLoading(true);

  const fd = new FormData();
  fd.append('file', currentFile);
  fd.append('target_col', targetSelect.value || '');

  try {
    const res = await fetch('/analyze', { method: 'POST', body: fd });
    const data = await res.json();

    if (!res.ok || !data.success) {
      showToast('Error: ' + (data.error || 'Unknown error'), true);
      setLoading(false); return;
    }

    analysisData = data;
    renderResults(data);
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (e) {
    showToast('Network error. Is the server running?', true);
  }
  setLoading(false);
}

function setLoading(on) {
  analyzeBtn.disabled = on;
  analyzeBtnTxt.textContent = on ? 'Analyzing…' : 'Analyze Dataset';
  btnSpinner.style.display = on ? 'block' : 'none';
}

// ── Render Results ─────────────────────────────────────────
function renderResults(data) {
  const { meta_features: meta, results, feature_importances: fi,
          feature_names, confusion_matrices: cms, target_col,
          dataset_preview: preview } = data;

  const best = results[0];

  // Winner Banner
  document.getElementById('winnerName').textContent =
    `${ALGO_ICONS[best.name] || '🏆'} ${best.name}`;
  document.getElementById('winnerScore').textContent =
    `F1: ${best.f1}%  ·  Accuracy: ${best.accuracy}%  ·  Train Time: ${best.train_time}ms`;

  renderMeta(meta);
  renderLeaderboard(results);
  renderBarChart(results);
  renderRadarChart(results);
  renderTimingChart(results);
  renderImportanceChart(fi, feature_names);
  renderConfusionSelector(cms);
  renderExplain(meta, best);
  renderPreview(preview);

  // Reset to first tab
  switchTab('leaderboard');
}

// ── Meta Features ──────────────────────────────────────────
const META_CONFIG = {
  n_samples:             { label:'Samples',          color:'#7c3aed', desc:'Total rows in dataset' },
  n_features:            { label:'Features',         color:'#06b6d4', desc:'Number of input columns' },
  n_classes:             { label:'Classes',          color:'#10b981', desc:'Unique target categories' },
  class_balance:         { label:'Class Balance',    color:'#f59e0b', desc:'Min/Max class ratio (1=perfect)' },
  mean_variance:         { label:'Mean Variance',    color:'#a855f7', desc:'Average feature spread' },
  mean_skewness:         { label:'Mean Skewness',    color:'#ef4444', desc:'Average distribution asymmetry' },
  feature_to_sample_ratio:{ label:'Feature/Sample', color:'#22d3ee', desc:'Dimensionality ratio' },
  mean_correlation:      { label:'Mean Correlation', color:'#f97316', desc:'Avg inter-feature correlation' },
};

function renderMeta(meta) {
  const grid = document.getElementById('metaGrid');
  grid.innerHTML = '';
  for (const [key, cfg] of Object.entries(META_CONFIG)) {
    const val = meta[key];
    if (val === undefined) continue;
    const card = document.createElement('div');
    card.className = 'meta-card';
    card.innerHTML = `
      <span class="meta-label">${cfg.label}</span>
      <span class="meta-value" style="color:${cfg.color}">${val}</span>
      <span class="meta-desc">${cfg.desc}</span>
    `;
    grid.appendChild(card);
  }
}

// ── Leaderboard ────────────────────────────────────────────
function renderLeaderboard(results) {
  const tbody = document.getElementById('leaderboardBody');
  tbody.innerHTML = '';
  results.forEach((r, i) => {
    const rank = i + 1;
    const rankClass = rank <= 3 ? `rank-${rank}` : 'rank-n';
    const rankLabel = rank === 1 ? '🥇' : rank === 2 ? '🥈' : rank === 3 ? '🥉' : rank;
    const tr = document.createElement('tr');
    if (r.is_best) tr.className = 'best-row';
    tr.innerHTML = `
      <td><span class="rank-badge ${rankClass}">${rankLabel}</span></td>
      <td class="algo-name-cell">
        ${ALGO_ICONS[r.name]||''} ${r.name}
        ${r.is_best ? '<span class="best-tag">BEST</span>' : ''}
      </td>
      <td>${scoreCell(r.accuracy)}</td>
      <td>${scoreCell(r.f1)}</td>
      <td>${scoreCell(r.precision)}</td>
      <td>${scoreCell(r.recall)}</td>
      <td>${r.train_time} ms</td>
      <td>${r.pred_time} ms</td>
    `;
    tbody.appendChild(tr);
  });
}

function scoreCell(val) {
  return `<div class="score-bar">
    <span style="min-width:46px;font-weight:600">${val}%</span>
    <div class="score-fill-wrap">
      <div class="score-fill" style="width:${val}%"></div>
    </div>
  </div>`;
}

// ── Bar Chart ──────────────────────────────────────────────
function renderBarChart(results) {
  if (barChartInst) barChartInst.destroy();
  const ctx = document.getElementById('barChart').getContext('2d');
  const labels = results.map(r => r.name);
  const mkDataset = (label, key, color) => ({
    label, data: results.map(r => r[key]),
    backgroundColor: color + 'cc', borderColor: color,
    borderWidth: 2, borderRadius: 6,
  });
  barChartInst = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        mkDataset('Accuracy (%)',  'accuracy',  '#7c3aed'),
        mkDataset('F1 Score (%)',  'f1',        '#06b6d4'),
        mkDataset('Precision (%)', 'precision', '#10b981'),
        mkDataset('Recall (%)',    'recall',    '#f59e0b'),
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#8b90b8', font:{size:13} } } },
      scales: {
        x: { ticks:{color:'#8b90b8'}, grid:{color:'rgba(255,255,255,.04)'} },
        y: { min:0, max:100, ticks:{color:'#8b90b8',callback:v=>v+'%'}, grid:{color:'rgba(255,255,255,.06)'} },
      },
    },
  });
}

// ── Radar Chart ────────────────────────────────────────────
function renderRadarChart(results) {
  if (radarChartInst) radarChartInst.destroy();
  const ctx = document.getElementById('radarChart').getContext('2d');
  const top5 = results.slice(0, 5);
  radarChartInst = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: ['Accuracy','F1 Score','Precision','Recall'],
      datasets: top5.map((r, i) => ({
        label: r.name,
        data: [r.accuracy, r.f1, r.precision, r.recall],
        borderColor: CHART_COLORS[i], backgroundColor: CHART_COLORS[i]+'22',
        pointBackgroundColor: CHART_COLORS[i], borderWidth: 2, pointRadius: 4,
      })),
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels:{ color:'#8b90b8', font:{size:13} } } },
      scales: {
        r: {
          min:0, max:100, angleLines:{color:'rgba(255,255,255,.1)'},
          grid:{color:'rgba(255,255,255,.08)'},
          pointLabels:{color:'#e8eaf6', font:{size:13, weight:'600'}},
          ticks:{color:'#8b90b8', backdropColor:'transparent', callback:v=>v+'%'},
        },
      },
    },
  });
}

// ── Timing Scatter ─────────────────────────────────────────
function renderTimingChart(results) {
  if (timingInst) timingInst.destroy();
  const ctx = document.getElementById('timingChart').getContext('2d');
  timingInst = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: results.map((r, i) => ({
        label: r.name,
        data: [{ x: r.train_time, y: r.accuracy }],
        backgroundColor: CHART_COLORS[i] + 'cc',
        borderColor: CHART_COLORS[i], borderWidth: 2,
        pointRadius: 10, pointHoverRadius: 14,
      })),
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend:{ labels:{ color:'#8b90b8', font:{size:13} } },
        tooltip:{ callbacks:{ label: ctx => `${ctx.dataset.label}: Train ${ctx.parsed.x}ms, Acc ${ctx.parsed.y}%` } }
      },
      scales: {
        x: { title:{ display:true, text:'Training Time (ms)', color:'#8b90b8' }, ticks:{color:'#8b90b8'}, grid:{color:'rgba(255,255,255,.06)'} },
        y: { title:{ display:true, text:'Accuracy (%)', color:'#8b90b8' }, ticks:{color:'#8b90b8',callback:v=>v+'%'}, grid:{color:'rgba(255,255,255,.06)'} },
      },
    },
  });
}

// ── Feature Importance ─────────────────────────────────────
function renderImportanceChart(fi, names) {
  if (importanceInst) importanceInst.destroy();
  const ctx = document.getElementById('importanceChart').getContext('2d');
  if (!fi || !names) {
    ctx.fillStyle = '#8b90b8'; ctx.font = '16px Inter'; ctx.textAlign = 'center';
    ctx.fillText('Feature importance not available', ctx.canvas.width/2, ctx.canvas.height/2);
    return;
  }
  // Sort descending, take top 15
  const pairs = names.map((n, i) => ({ name:n, val:fi[i] })).sort((a,b)=>b.val-a.val).slice(0,15);
  importanceInst = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: pairs.map(p => p.name),
      datasets: [{
        label: 'Feature Importance (Random Forest)',
        data: pairs.map(p => (p.val*100).toFixed(2)),
        backgroundColor: pairs.map((_, i) => `hsl(${260+i*12},70%,60%)`),
        borderRadius: 6,
      }],
    },
    options: {
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      plugins: { legend:{ display:false } },
      scales: {
        x: { ticks:{color:'#8b90b8',callback:v=>v+'%'}, grid:{color:'rgba(255,255,255,.06)'} },
        y: { ticks:{color:'#e8eaf6', font:{size:13}}, grid:{color:'rgba(255,255,255,.04)'} },
      },
    },
  });
}

// ── Confusion Matrix ───────────────────────────────────────
function renderConfusionSelector(cms) {
  const sel = document.getElementById('confusionSelect');
  sel.innerHTML = Object.keys(cms).map(n => `<option value="${n}">${n}</option>`).join('');
  sel.value = Object.keys(cms)[0];
  sel.addEventListener('change', () => renderCM(cms[sel.value]));
  renderCM(cms[sel.value]);
}

function renderCM(matrix) {
  const wrap = document.getElementById('confusionMatrixWrap');
  if (!matrix || matrix.length === 0) { wrap.innerHTML = '<p style="color:var(--text-dim)">No data</p>'; return; }
  const n = matrix.length;
  const maxVal = Math.max(...matrix.flat());

  let html = '<table class="cm-table"><thead><tr><th class="cm-label" style="border:none"></th>';
  for (let j = 0; j < n; j++) html += `<th class="cm-header">Pred ${j}</th>`;
  html += '</tr></thead><tbody>';
  for (let i = 0; i < n; i++) {
    html += `<tr><td class="cm-label">Actual ${i}</td>`;
    for (let j = 0; j < n; j++) {
      const v = matrix[i][j];
      const intensity = maxVal > 0 ? v / maxVal : 0;
      const isDiag = i === j;
      const bg = isDiag
        ? `rgba(124,58,237,${0.15 + intensity*0.7})`
        : `rgba(239,68,68,${intensity*0.5})`;
      const color = intensity > 0.5 ? '#fff' : 'var(--text-dim)';
      html += `<td style="background:${bg};color:${color}">${v}</td>`;
    }
    html += '</tr>';
  }
  html += '</tbody></table>';
  wrap.innerHTML = html;
}

// ── Explain ────────────────────────────────────────────────
function renderExplain(meta, best) {
  const grid = document.getElementById('explainGrid');
  const insights = getInsights(meta, best);
  grid.innerHTML = insights.map(ins => `
    <div class="explain-card">
      <span class="explain-icon">${ins.icon}</span>
      <div class="explain-text">
        <h4>${ins.title}</h4>
        <p>${ins.text}</p>
      </div>
    </div>
  `).join('');
}

function getInsights(meta, best) {
  const ins = [];
  ins.push({ icon:'🏆', title:`${best.name} Wins`, text:`Achieved the highest F1 score of ${best.f1}% and accuracy of ${best.accuracy}% on your actual data — not a guess.` });

  if (meta.n_samples < 200) ins.push({ icon:'📦', title:'Small Dataset', text:`With only ${meta.n_samples} samples, simpler models like KNN or Decision Tree can sometimes avoid overfitting that plagues deep ensembles.` });
  else if (meta.n_samples > 5000) ins.push({ icon:'📦', title:'Large Dataset', text:`${meta.n_samples} samples gives ensemble methods like Random Forest and Gradient Boosting plenty of data to learn robust patterns.` });
  else ins.push({ icon:'📦', title:'Medium Dataset', text:`${meta.n_samples} rows is a solid size — most algorithms perform reliably, making the benchmark results highly trustworthy.` });

  if (meta.class_balance < 0.5) ins.push({ icon:'⚖️', title:'Class Imbalance Detected', text:`Balance ratio is ${meta.class_balance} — one class dominates. F1 score is more meaningful than accuracy here, which is why we ranked by F1.` });
  else ins.push({ icon:'⚖️', title:'Balanced Classes', text:`Class balance of ${meta.class_balance} is good. All metrics are reliable, and accuracy is a fair measure.` });

  if (meta.mean_variance > 1) ins.push({ icon:'📐', title:'High Feature Variance', text:`Mean variance of ${meta.mean_variance} suggests features have very different scales. Scaled models like SVM and LR benefit from our automatic standardization.` });
  else ins.push({ icon:'📐', title:'Low Variance Features', text:`Features are relatively tight in scale. Tree-based models (RF, DT, GB) are naturally scale-invariant and tend to do well here.` });

  if (meta.n_features > meta.n_samples * 0.1) ins.push({ icon:'📊', title:'High Dimensionality', text:`Feature/sample ratio of ${meta.feature_to_sample_ratio} is high. Regularized models (SVM, LR) handle this better than non-parametric ones.` });

  if (meta.mean_correlation > 0.7) ins.push({ icon:'🔗', title:'High Feature Correlation', text:`Correlation of ${meta.mean_correlation} means features carry redundant information. PCA pre-processing could improve performance further.` });

  ins.push({ icon:'🔬', title:'Meta-Learning Verdict', text:`These insights mirror what a meta-learning model learns across thousands of datasets — your data's characteristics pointed directly to ${best.name}.` });

  return ins;
}

// ── Preview Table ──────────────────────────────────────────
function renderPreview(rows) {
  if (!rows || rows.length === 0) return;
  const cols = Object.keys(rows[0]);
  const table = document.getElementById('previewTable');
  table.innerHTML = `
    <thead><tr>${cols.map(c=>`<th>${c}</th>`).join('')}</tr></thead>
    <tbody>${rows.map(r=>`<tr>${cols.map(c=>`<td>${r[c]??''}</td>`).join('')}</tr>`).join('')}</tbody>
  `;
}

// ── Tabs ───────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tabId));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === `tab-${tabId}`));
}

// ── Reset ──────────────────────────────────────────────────
function resetAnalysis() {
  resultsSection.style.display = 'none';
  resetFile();
  document.getElementById('uploadCard').scrollIntoView({ behavior:'smooth' });
}

// ── Toast ──────────────────────────────────────────────────
let toastTimer;
function showToast(msg, isError = false) {
  const t = document.getElementById('toast');
  t.textContent = msg; t.className = 'toast show' + (isError ? ' error' : '');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.className = 'toast', 3200);
}

// ── Scroll animations ──────────────────────────────────────
const observer = new IntersectionObserver((entries) => {
  entries.forEach(e => { if (e.isIntersecting) { e.target.style.opacity=1; e.target.style.transform='translateY(0)'; } });
}, { threshold: 0.1 });

document.querySelectorAll('.step-card, .vs-card, .meta-card').forEach(el => {
  el.style.opacity = 0; el.style.transform = 'translateY(24px)';
  el.style.transition = 'opacity .5s ease, transform .5s ease';
  observer.observe(el);
});
