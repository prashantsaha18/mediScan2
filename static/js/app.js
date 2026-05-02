// ── Toggle buttons ──────────────────────────
function setToggle(field, val, btn) {
  document.getElementById(field).value = val;
  btn.closest('.toggle-group').querySelectorAll('.toggle-btn')
     .forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}

// ── Collect form data ────────────────────────
function collectData() {
  const fields = [
    'age','gender','bmi','systolic_bp','diastolic_bp',
    'glucose','hba1c','cholesterol','hdl','ldl','triglycerides',
    'heart_rate','smoking','exercise_days','family_history',
    'alcohol_consumption','stress_level','sleep_hours',
    'waist_cm','creatinine'
  ];
  const payload = {};
  fields.forEach(f => {
    payload[f] = parseFloat(document.getElementById(f).value) || 0;
  });
  return payload;
}

// ── Main analyze ─────────────────────────────
async function analyze() {
  const btn = document.getElementById('analyzeBtn');
  const txt = document.getElementById('btn-text');
  btn.classList.add('loading');
  txt.innerHTML = '<span class="spinner"></span>Analyzing…';

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(collectData())
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    renderResults(data);
  } catch(e) {
    alert('Error: ' + e.message + '\n\nMake sure app.py is running on port 5000 and open http://127.0.0.1:5000');
  } finally {
    btn.classList.remove('loading');
    txt.textContent = 'Analyze My Health';
  }
}

// ── Render all results ───────────────────────
function renderResults(data) {
  renderOverall(data);
  renderDiseaseGrid(data.diseases);
  renderRiskFactors(data.risk_factors);
  renderRecommendations(data.recommendations);
  renderContributors(data.top_contributors);

  const panel = document.getElementById('result-panel');
  panel.style.display = 'block';
  setTimeout(() => panel.scrollIntoView({ behavior: 'smooth', block: 'start' }), 80);
}

// ── Overall verdict ──────────────────────────
function renderOverall(data) {
  const isDisease = data.prediction === 1;
  const prob      = Math.round(data.probability * 100);

  const hero  = document.getElementById('result-hero');
  const badge = document.getElementById('result-badge');
  const title = document.getElementById('result-title');
  const sub   = document.getElementById('result-subtitle');
  const pct   = document.getElementById('prob-pct');
  const bar   = document.getElementById('prob-bar');
  const pill  = document.getElementById('risk-pill');

  hero.className  = 'result-hero ' + (isDisease ? 'disease' : 'healthy');
  badge.className = 'result-badge ' + (isDisease ? 'disease' : 'healthy');
  badge.textContent = isDisease ? '⚠ Elevated Risk Detected' : '✓ Low Overall Risk';

  title.textContent = isDisease ? 'Health Risks Identified' : 'You Appear Healthy';
  sub.textContent   = isDisease
    ? `${data.diseases_detected.length} condition(s) flagged. Review the breakdown and recommendations below.`
    : 'Your biomarkers are within acceptable ranges. Maintain your healthy habits!';

  pct.textContent  = prob + '%';
  pill.className   = 'risk-pill ' + data.risk_level;
  pill.textContent = data.risk_level + ' Risk';

  setTimeout(() => { bar.style.width = prob + '%'; }, 60);
}

// ── Disease cards ────────────────────────────
const DISEASE_META = {
  diabetes:     { icon: '🩸', label: 'Diabetes' },
  heart:        { icon: '❤️', label: 'Heart Disease' },
  hypertension: { icon: '💉', label: 'Hypertension' },
  kidney:       { icon: '🫘', label: 'Kidney Risk' },
};

function renderDiseaseGrid(diseases) {
  const grid = document.getElementById('disease-grid');
  grid.innerHTML = '';

  Object.entries(diseases).forEach(([key, info]) => {
    const meta     = DISEASE_META[key] || { icon: '🔬', label: key };
    const detected = info.detected;
    const pct      = Math.round(info.probability * 100);

    const card = document.createElement('div');
    card.className = 'disease-card ' + (detected ? 'detected' : 'safe');
    card.innerHTML = `
      <div class="disease-icon">${meta.icon}</div>
      <div class="disease-name">${meta.label}</div>
      <div class="disease-pct ${detected ? 'detected' : 'safe'}">${pct}%</div>
      <span class="disease-status ${detected ? 'detected' : 'safe'}">
        ${detected ? '⚠ Detected' : '✓ Normal'}
      </span>
      <div style="margin-top:8px">
        <div style="height:4px;background:rgba(255,255,255,0.06);border-radius:100px;overflow:hidden">
          <div style="height:100%;width:${pct}%;border-radius:100px;background:${detected ? 'var(--danger)' : 'var(--good)'};transition:width .9s ease"></div>
        </div>
      </div>`;
    grid.appendChild(card);

    // Animate bars
    setTimeout(() => {
      const bar = card.querySelector('div[style*="height:4px"] > div');
      if (bar) bar.style.width = pct + '%';
    }, 100);
  });
}

// ── Risk factors ─────────────────────────────
let allRiskFlags = [];

function renderRiskFactors(flags) {
  allRiskFlags = flags;
  document.getElementById('flag-count').textContent = flags.length;
  buildFilterRow(flags);
  renderFlagList(flags);
}

function buildFilterRow(flags) {
  const row = document.getElementById('risk-filter-row');
  const counts = { critical: 0, high: 0, borderline: 0 };
  flags.forEach(f => { if (counts[f.status] !== undefined) counts[f.status]++; });

  row.innerHTML = `
    <button class="filter-btn all active" onclick="filterFlags('all',this)">All (${flags.length})</button>
    ${counts.critical ? `<button class="filter-btn critical" onclick="filterFlags('critical',this)">🔴 Critical (${counts.critical})</button>` : ''}
    ${counts.high     ? `<button class="filter-btn high"     onclick="filterFlags('high',this)">🟠 High (${counts.high})</button>` : ''}
    ${counts.borderline ? `<button class="filter-btn borderline" onclick="filterFlags('borderline',this)">🟡 Borderline (${counts.borderline})</button>` : ''}
  `;
}

function filterFlags(status, btn) {
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const filtered = status === 'all' ? allRiskFlags : allRiskFlags.filter(f => f.status === status);
  renderFlagList(filtered);
}

function renderFlagList(flags) {
  const list = document.getElementById('risk-list');
  list.innerHTML = '';

  if (!flags.length) {
    list.innerHTML = '<li class="no-flags">No risk flags for this filter.</li>';
    return;
  }

  flags.forEach(f => {
    const li = document.createElement('li');
    li.className = 'risk-item';

    const valDisplay = typeof f.value === 'number'
      ? `${f.value} ${f.unit}` : `${f.value} ${f.unit}`;
    const deltaText = f.delta_pct > 0 ? `+${f.delta_pct}% above threshold` : '';

    li.innerHTML = `
      <div class="risk-dot-wrap">
        <span class="risk-dot ${f.status}"></span>
      </div>
      <div class="risk-content">
        <div class="risk-label">${f.label}</div>
        <div class="risk-value-row">
          <span class="risk-value">${valDisplay}</span>
          ${deltaText ? `<span class="risk-delta ${f.status}">${deltaText}</span>` : ''}
        </div>
      </div>`;
    list.appendChild(li);
  });
}

// ── Recommendations ──────────────────────────
function renderRecommendations(recs) {
  const list = document.getElementById('rec-list');
  list.innerHTML = '';

  recs.forEach(r => {
    const li = document.createElement('li');
    li.className = `rec-item ${r.urgency || 'medium'}`;
    li.innerHTML = `
      <span class="rec-urgency ${r.urgency}">${r.urgency}</span>
      <div class="rec-title">${r.title}</div>
      <div class="rec-detail">${r.detail}</div>`;
    list.appendChild(li);
  });
}

// ── Feature importance ───────────────────────
function renderContributors(contributors) {
  const container = document.getElementById('contrib-list');
  container.innerHTML = '';

  if (!contributors || !contributors.length) return;

  const maxImp = contributors[0].importance;
  contributors.forEach(c => {
    const pct = maxImp > 0 ? Math.round((c.importance / maxImp) * 100) : 0;
    const label = c.feature.replace(/_/g, ' ');
    const div = document.createElement('div');
    div.className = 'contrib-item';
    div.innerHTML = `
      <div class="contrib-name">${label}</div>
      <div class="contrib-track">
        <div class="contrib-bar" style="width:0%" data-width="${pct}%"></div>
      </div>
      <div class="contrib-pct">${(c.importance * 100).toFixed(1)}%</div>`;
    container.appendChild(div);
  });

  // Animate bars after render
  setTimeout(() => {
    document.querySelectorAll('.contrib-bar').forEach(b => {
      b.style.width = b.dataset.width;
    });
  }, 120);
}
