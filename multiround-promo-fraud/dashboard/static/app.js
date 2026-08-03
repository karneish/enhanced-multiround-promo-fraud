"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  schema: null,
  datasets: [],
  runs: [],
  activeRunId: null,
  live: { chart: null, log: '', timeline: [], trial: 0, round: 0, lastLineCount: 0 },
  results: { chart: null },
  graph: null,
};

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------
document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tabpane').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    $(`tab-${btn.dataset.tab}`).classList.add('active');
    if (btn.dataset.tab === 'results') loadExperiments();
    if (btn.dataset.tab === 'graph') initGraph();
  });
});

// ---------------------------------------------------------------------------
// System / schema
// ---------------------------------------------------------------------------
async function init() {
  try {
    const health = await api('/api/health');
    const badge = $('sysbadge');
    badge.textContent = health.import_error ? `⚠ import: ${health.import_error}` : `torch ${health.torch} · dgl ${health.dgl}`;
    badge.classList.add(health.import_error ? '' : 'ok');
  } catch (e) {
    $('sysbadge').textContent = '⚠ backend unreachable';
  }
  try {
    state.schema = await api('/api/schema');
    state.datasets = await api('/api/datasets');
    fillSelects();
    applyGenDefaults();
    toggleGenOpts();
    renderDatasetChips();
    renderAbout();
    updateConfigPreview();
    updateLaunchState();
  } catch (e) {
    console.error(e);
  }
}

function fillSelects() {
  const s = state.schema;
  const fill = (id, items) => {
    const sel = $(id);
    sel.innerHTML = '';
    items.forEach(it => {
      const o = document.createElement('option');
      o.value = it; o.textContent = it;
      sel.appendChild(o);
    });
  };
  fill('cfg-model', s.models);
  fill('cfg-augment', s.augments);
  fill('cfg-adverchoose', s.adver_choose);
  fill('cfg-advermod', s.adver_mod);
  toggleGenOpts();
  $('about-models').textContent = s.models.join(', ');
  $('about-augments').textContent = s.augments.join(', ');
  $('about-adverchoose').textContent = s.adver_choose.join(', ');
  $('about-advermod').textContent = s.adver_mod.join(', ');
  $('about-datasets').textContent = s.datasets.join(', ');
  $('about-backend').textContent = s.import_error ? `⚠ ${s.import_error}` : 'OK';
}

function renderDatasetChips() {
  const names = state.schema.datasets;
  ['cfg-dsets', 'cfg-traindsets'].forEach(id => {
    const wrap = $(id);
    wrap.innerHTML = '';
    names.forEach((n, i) => {
      const chip = document.createElement('span');
      chip.className = 'chip' + (id === 'cfg-dsets' && i === 0 ? ' on' : '');
      chip.textContent = n;
      chip.dataset.name = n;
      chip.addEventListener('click', () => chip.classList.toggle('on'));
      wrap.appendChild(chip);
    });
  });
}

function selectedChips(id) {
  return [...document.querySelectorAll(`#${id} .chip.on`)].map(c => c.dataset.name);
}

// ---------------------------------------------------------------------------
// Config building
// ---------------------------------------------------------------------------
function csvList(v) {
  return v.split(',').map(x => x.trim()).filter(x => x !== '').map(x => {
    if (/^-?\d+$/.test(x)) return Number(x);
    if (/^[-+]?(\d+\.?\d*|\.\d+)$/.test(x)) return Number(x);
    return x;
  });
}

function buildConfig() {
  const dsetList = selectedChips('cfg-dsets');
  if (dsetList.length === 0) throw new Error('Select at least one dataset');
  const raw = $('cfg-json').value.trim();
  let overrides = {};
  if (raw) {
    try { overrides = JSON.parse(raw); } catch (e) { throw new Error('Raw JSON override is invalid: ' + e.message); }
  }
  const cfg = {
    TRIAL_NUM: Math.max(1, parseInt($('cfg-trials').value) || 1),
    FAILURE_LIMIT: Math.max(0, parseInt($('cfg-failure').value) || 0),
    EXPERIMENT_DESC: $('cfg-desc').value || 'Dashboard experiment',
    LIST_DSET: dsetList,
    LIST_TRAIN_DSET: dsetList.map((d, i) => (selectedChips('cfg-traindsets').includes(d) ? d : 'NONE')),
    EXP_DICT: {
      device: [$('cfg-device').value],
      exp_type: [$('cfg-exptype').value],
      round_num: [Math.max(1, parseInt($('cfg-rounds').value) || 1)],
      model_name: [$('cfg-model').value],
      round_reset_model: [false],
      embed_type: [$('cfg-embed').value],
      h_feats: [Math.max(4, parseInt($('cfg-hfeats').value) || 64)],
      num_layers: [Math.max(1, parseInt($('cfg-layers').value) || 2)],
      round_window: [Math.max(1, parseInt($('cfg-window').value) || 7)],
      num_epoch: [Math.max(1, parseInt($('cfg-epoch').value) || 3)],
      num_round_epoch: [Math.max(1, parseInt($('cfg-repoch').value) || 3)],
      early_stopping: [Math.max(1, parseInt($('cfg-es').value) || 3)],
      loss_type: [$('cfg-losstype').value || 'ndist'],
      norm_name: [$('cfg-norm').value],
      temporal_agg: [$('cfg-tagg').value],
      alpha: csvList($('cfg-alpha').value),
      beta: csvList($('cfg-beta').value),
      augment_name: [$('cfg-augment').value],
      adver_choose_name: [$('cfg-adverchoose').value],
      adver_mod_name: [$('cfg-advermod').value],
      ...(genConfig()),
    },
    ...overrides,
  };
  if (overrides.EXP_DICT) {
    cfg.EXP_DICT = { ...cfg.EXP_DICT, ...overrides.EXP_DICT };
  }
  return cfg;
}

function genConfig() {
  if ($('cfg-advermod').value !== 'INTELLIGENT') return {};
  return {
    adver_gen_type: [$('cfg-gentype').value],
    adver_gen_epochs: [Math.max(1, parseInt($('cfg-genepochs').value) || 300)],
    adver_gen_feat_coef: csvList($('cfg-genfeatcoef').value),
    adver_gen_conn_coef: csvList($('cfg-genconncoef').value),
    adver_gen_ring_ratio: csvList($('cfg-genringratio').value),
    adver_gen_round_window: [Math.max(1, parseInt($('cfg-genwindow').value) || 5)],
  };
}

function toggleGenOpts() {
  const on = $('cfg-advermod').value === 'INTELLIGENT';
  $('gen-opts').classList.toggle('hidden', !on);
}

function applyGenDefaults() {
  const d = state.schema && state.schema.defaults ? state.schema.defaults.adver : null;
  if (!d) return;
  const map = {
    'cfg-gentype': 'adver_gen_type',
    'cfg-genepochs': 'adver_gen_epochs',
    'cfg-genfeatcoef': 'adver_gen_feat_coef',
    'cfg-genconncoef': 'adver_gen_conn_coef',
    'cfg-genringratio': 'adver_gen_ring_ratio',
    'cfg-genwindow': 'adver_gen_round_window',
  };
  Object.entries(map).forEach(([elId, key]) => {
    const el = $(elId);
    if (!el || d[key] === undefined) return;
    if (el.tagName === 'SELECT') {
      const v = String(d[key]);
      if ([...el.options].some(o => o.value === v)) el.value = v;
    } else {
      el.value = d[key];
    }
  });
}

function updateConfigPreview() {
  try {
    const cfg = buildConfig();
    $('cfg-preview').textContent = JSON.stringify(cfg, null, 2);
  } catch (e) {
    $('cfg-preview').textContent = '// ' + e.message;
  }
}

['cfg-desc', 'cfg-trials', 'cfg-failure', 'cfg-rounds', 'cfg-hfeats', 'cfg-layers', 'cfg-window',
 'cfg-epoch', 'cfg-repoch', 'cfg-es', 'cfg-alpha', 'cfg-beta', 'cfg-losstype', 'cfg-json',
 'cfg-device', 'cfg-exptype', 'cfg-model', 'cfg-embed', 'cfg-augment', 'cfg-adverchoose',
 'cfg-advermod', 'cfg-tagg', 'cfg-norm', 'cfg-gentype', 'cfg-genepochs', 'cfg-genfeatcoef',
 'cfg-genconncoef', 'cfg-genringratio', 'cfg-genwindow'
].forEach(id => {
  const el = $(id);
  if (el) el.addEventListener('input', updateConfigPreview);
});
$('cfg-advermod').addEventListener('change', () => { toggleGenOpts(); updateConfigPreview(); });
document.querySelectorAll('#cfg-dsets .chip, #cfg-traindsets .chip').forEach(c => c.addEventListener('click', updateConfigPreview));

$('btn-jsonview').addEventListener('click', () => {
  try {
    const cfg = buildConfig();
    $('cfg-json').value = JSON.stringify(cfg.EXP_DICT, null, 2).slice(0, 0) + JSON.stringify(cfg.EXP_DICT, null, 2);
    $('cfg-json').value = JSON.stringify({ EXP_DICT: cfg.EXP_DICT }, null, 2);
    updateConfigPreview();
    $('launch-msg').textContent = 'EXP_DICT copied into the JSON override box. Adjust if needed.';
    $('launch-msg').className = 'msg ok';
  } catch (e) {
    $('launch-msg').textContent = e.message;
    $('launch-msg').className = 'msg err';
  }
});

$('btn-launch').addEventListener('click', async () => {
  let cfg;
  try {
    cfg = buildConfig();
  } catch (e) {
    $('launch-msg').textContent = e.message;
    $('launch-msg').className = 'msg err';
    return;
  }
  $('btn-launch').disabled = true;
  $('launch-msg').textContent = 'launching…';
  $('launch-msg').className = 'msg';
  try {
    const res = await api('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cfg),
    });
    $('launch-msg').textContent = `Launched run ${res.run_id}`;
    $('launch-msg').className = 'msg ok';
    state.activeRunId = res.run_id;
    resetLive();
    document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tabpane').forEach(p => p.classList.remove('active'));
    document.querySelector('.tab[data-tab="dashboard"]').classList.add('active');
    $('tab-dashboard').classList.add('active');
    connectStream(res.run_id);
  } catch (e) {
    $('launch-msg').textContent = 'Launch failed: ' + e.message;
    $('launch-msg').className = 'msg err';
  } finally {
    $('btn-launch').disabled = false;
  }
});

// ---------------------------------------------------------------------------
// Live dashboard
// ---------------------------------------------------------------------------
function updateLaunchState() {
  if (state.activeRunId && state.runs.some(r => r.id === state.activeRunId)) return;
}

function resetLive() {
  state.live = { log: '', timeline: [], trial: 0, round: 0, lastLineCount: 0 };
  $('live-log').textContent = '';
  $('timeline').innerHTML = '<div class="muted">waiting for events…</div>';
  $('dash-trial').textContent = '– / –';
  $('dash-lines').textContent = '0';
  $('dash-failures').textContent = '0';
  $('dash-bestf1').textContent = '–';
  $('dash-rundesc').textContent = '';
  $('dash-elapsed').textContent = '0s';
  const sb = $('stream-status');
  sb.textContent = 'connecting';
  sb.className = 'pill starting';
  if (state.live.chart) { state.live.chart.destroy(); state.live.chart = null; }
}

function connectStream(runId) {
  const url = `/api/run/${runId}/stream`;
  const es = new EventSource(url);
  state.live.lastLineCount = 0;
  let roundSeries = {};

  es.addEventListener('state', ev => {
    const data = JSON.parse(ev.data);
    applyParsed(data.parsed);
    const sb = $('stream-status');
    sb.textContent = data.state;
    sb.className = 'pill ' + (data.state === 'done' ? 'done' : data.state === 'running' ? 'running' : 'failed');
    $('dash-state').querySelector('#dash-state-body').textContent =
      data.state === 'done' ? 'COMPLETE ✓' : data.state === 'failed' ? 'FAILED ✗' : data.state.toUpperCase();
    $('dash-state-body').className = 'big ' + (data.state === 'done' ? 'ok' : data.state === 'failed' ? 'err' : '');
    if (data.state !== 'running') {
      es.close();
      loadExperiments(true);
      updateRunsList();
    }
  });

  es.onmessage = ev => {
    const data = JSON.parse(ev.data);
    if (data.type === 'log') {
      appendLogLine(data.line);
      parseLogLine(data.line, roundSeries);
      $('dash-lines').textContent = String(++state.live.lastLineCount);
    }
  };

  es.onerror = () => {
    $('stream-status').textContent = 'reconnecting…';
    $('stream-status').className = 'pill starting';
  };

  updateRunsList();
  // also poll status as a fallback for parsed metrics
  const poll = setInterval(async () => {
    if (state.activeRunId !== runId) { clearInterval(poll); return; }
    try {
      const st = await api(`/api/run/${runId}`);
      if (st.state !== 'running') { clearInterval(poll); }
      applyParsed(st.parsed);
    } catch (e) { /* ignore */ }
  }, 3000);
}

function applyParsed(parsed) {
  if (!parsed) return;
  if (parsed.trial !== undefined) state.live.trial = parsed.trial;
  if (parsed.round !== undefined) state.live.round = parsed.round;
  $('dash-trial').textContent = `${state.live.trial} / ${state.live.round}`;
  if (parsed.failures !== undefined && parsed.failures !== null) $('dash-failures').textContent = parsed.failures;
  if (parsed.best_f1 !== undefined && parsed.best_f1 !== null) $('dash-bestf1').textContent = (parsed.best_f1 * 100).toFixed(1) + '%';
}

function appendLogLine(line) {
  const logEl = $('live-log');
  let html = line
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/TRIAL NUMBER \d+/g, m => `<span class="hl-trial">${m}</span>`)
    .replace(/Starting round \d+\.\.\./g, m => `<span class="hl-round">${m}</span>`);
  state.live.log += html + '\n';
  const parts = state.live.log.split('\n');
  if (parts.length > 300) { parts.splice(0, parts.length - 300); state.live.log = parts.join('\n'); }
  logEl.innerHTML = state.live.log;
  logEl.scrollTop = logEl.scrollHeight;
}

function parseLogLine(line, roundSeries) {
  let m;
  m = line.match(/TRIAL NUMBER (\d+)/);
  if (m) { state.live.trial = parseInt(m[1]); addTimeline('trial', `Trial ${m[1]}`, 'tl-trial'); }
  m = line.match(/Starting round (\d+)\.\.\./);
  if (m) { state.live.round = parseInt(m[1]); addTimeline('round', `Round ${m[1]}`, 'tl-round'); }
  m = line.match(/Best Val: REC ([0-9.]+) PRE ([0-9.]+) MF1 ([0-9.]+) AUC ([0-9.]+)/);
  if (m) {
    const rec = parseFloat(m[1]), pre = parseFloat(m[2]), f1 = parseFloat(m[3]), auc = parseFloat(m[4]);
    const r = state.live.round;
    if (!roundSeries[r]) roundSeries[r] = { rec: [], pre: [], f1: [], auc: [] };
    roundSeries[r].rec.push(rec); roundSeries[r].pre.push(pre); roundSeries[r].f1.push(f1); roundSeries[r].auc.push(auc);
    addTimeline('metric', `R${r} val  REC ${(rec*100).toFixed(1)}  PRE ${(pre*100).toFixed(1)}  F1 ${(f1*100).toFixed(1)}  AUC ${(auc*100).toFixed(1)}`, 'tl-metric');
    updateLiveChart(roundSeries);
  }
  m = line.match(/Dataset - (Overall|Train|Val|Test|Round \d+): REC ([0-9.]+) PRE ([0-9.]+) MF1 ([0-9.]+) AUC ([0-9.]+)/);
  if (m) {
    addTimeline('eval', `${m[1]}: REC ${m[2]} F1 ${m[4]} AUC ${m[5]}`, 'tl-eval');
  }
}

function addTimeline(type, text, cls) {
  state.live.timeline.push({ t: new Date(), text, cls });
  if (state.live.timeline.length > 120) state.live.timeline.shift();
  const tl = $('timeline');
  if (tl.querySelector('.muted')) tl.innerHTML = '';
  const div = document.createElement('div');
  div.className = 'tl-item';
  const time = document.createElement('span');
  time.className = 'tl-time';
  time.textContent = new Date().toLocaleTimeString('en-GB', { hour12: false });
  const txt = document.createElement('span');
  txt.className = cls || '';
  txt.textContent = text;
  div.appendChild(time); div.appendChild(txt);
  tl.appendChild(div);
  tl.scrollTop = tl.scrollHeight;
}

function updateLiveChart(roundSeries) {
  const labels = Object.keys(roundSeries).sort((a, b) => a - b);
  const avg = k => labels.map(r => {
    const v = roundSeries[r][k];
    return v.reduce((s, x) => s + x, 0) / v.length;
  });
  const datasets = [
    { label: 'F1', data: avg('f1'), borderColor: '#3fb950', tension: .2 },
    { label: 'AUC', data: avg('auc'), borderColor: '#58a6ff', tension: .2 },
    { label: 'Recall', data: avg('rec'), borderColor: '#d29922', tension: .2 },
    { label: 'Precision', data: avg('pre'), borderColor: '#bc8cff', tension: .2 },
  ];
  const ctx = $('live-chart');
  if (state.live.chart) state.live.chart.destroy();
  state.live.chart = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { y: { min: 0, max: 1, ticks: { callback: v => (v * 100).toFixed(0) + '%' } } },
      plugins: { legend: { labels: { color: '#8b949e' } } },
    },
  });
}

// ---------------------------------------------------------------------------
// Runs list (dashboard header area)
// ---------------------------------------------------------------------------
async function updateRunsList() {
  try {
    state.runs = await api('/api/runs');
  } catch (e) { /* ignore */ }
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------
async function loadExperiments(silent) {
  const wrap = $('exp-list');
  if (!silent) wrap.innerHTML = '<div class="muted">loading…</div>';
  let exps;
  try { exps = await api('/api/experiments'); }
  catch (e) { wrap.innerHTML = '<div class="muted">failed to load</div>'; return; }
  if (!exps.length) { wrap.innerHTML = '<div class="muted">no experiments yet — launch one from the Launch tab</div>'; return; }
  wrap.innerHTML = '';
  exps.slice().reverse().forEach(e => {
    const item = document.createElement('div');
    item.className = 'exp-item';
    item.innerHTML = `
      <div>
        <div class="desc">${escapeHtml(e.desc || 'Untitled')}</div>
        <div class="meta2">${e.cname}/${e.ts} · ${e.n_csvs} runs · ${e.has_combined ? 'combined ✓' : ''}</div>
      </div>
      <div class="meta2">open →</div>`;
    item.addEventListener('click', () => openExperiment(e));
    wrap.appendChild(item);
  });
}

async function openExperiment(exp) {
  const meta = await api(`/api/experiments/${exp.cname}/${exp.ts}/meta`);
  $('exp-meta').textContent = meta.meta;
  $('exp-meta').classList.add('hidden');
  $('exp-detail').classList.remove('hidden');
  $('exp-table-wrap').classList.remove('hidden');

  const target = exp.has_combined ? 'combined_result.csv' : (exp.csvs[0]);
  try {
    const data = await api(`/api/experiments/${exp.cname}/${exp.ts}/${target}`);
    renderResultChart(data);
    renderResultTable(data);
  } catch (e) {
    $('exp-table').innerHTML = '<div class="muted">no csv rows</div>';
  }
}

function renderResultChart(data) {
  const rows = (data.rows || []).filter(r => r.eval_type === 'val_set_best');
  if (!rows.length) { if (state.results.chart) { state.results.chart.destroy(); state.results.chart = null; } return; }
  const groupKey = r => `${r.model_name || ''}-a${r.alpha}-b${r.beta}`;
  const groups = {};
  rows.forEach(r => { const k = groupKey(r); (groups[k] = groups[k] || []).push(r); });
  const labels = [...new Set(rows.map(r => r.round))];
  const datasets = [];
  const colors = ['#58a6ff', '#3fb950', '#d29922', '#bc8cff', '#f85149', '#79c0ff', '#f0883e', '#a5d6ff'];
  let ci = 0;
  for (const [k, g] of Object.entries(groups)) {
    const byRound = {};
    g.forEach(r => byRound[r.round] = r);
    const col = colors[ci++ % colors.length];
    ['f1', 'auc'].forEach(metric => {
      datasets.push({
        label: `${k} · ${metric.toUpperCase()}`,
        data: labels.map(l => byRound[l] ? byRound[l][metric] : null),
        borderColor: metric === 'f1' ? col : 'transparent',
        backgroundColor: metric === 'f1' ? 'transparent' : col + '44',
        borderDash: metric === 'auc' ? [4, 3] : [],
        borderWidth: metric === 'f1' ? 2 : 1.5,
        tension: .2,
      });
    });
  }
  const ctx = $('res-chart');
  if (state.results.chart) state.results.chart.destroy();
  state.results.chart = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { y: { min: 0, max: 1, ticks: { callback: v => (v * 100).toFixed(0) + '%' } } },
      plugins: { legend: { labels: { color: '#8b949e', font: { size: 10 } } } },
    },
  });
}

function renderResultTable(data) {
  const cols = (data.columns || []).filter(c => ['round', 'eval_type', 'rec', 'prec', 'f1', 'auc', 'tp', 'fp', 'tn', 'fn', 'alpha', 'beta', 'model_name', 'trial', 'time', 'gen_type', 'gen_seeds', 'gen_feat_div', 'gen_feat_shift', 'gen_new_edges', 'gen_ext_edges', 'gen_ring_edges', 'gen_ring_ratio', 'gen_missed_conf'].includes(c));
  const table = document.createElement('table');
  table.className = 'data';
  const head = document.createElement('thead');
  const hr = document.createElement('tr');
  cols.forEach(c => { const th = document.createElement('th'); th.textContent = c; hr.appendChild(th); });
  head.appendChild(hr); table.appendChild(head);
  const body = document.createElement('tbody');
  (data.rows || []).slice(0, 500).forEach(r => {
    const tr = document.createElement('tr');
    cols.forEach(c => {
      const td = document.createElement('td');
      let v = r[c];
      if (['rec', 'prec', 'f1', 'auc'].includes(c) && typeof v === 'number') v = (v * 100).toFixed(1) + '%';
      if (['gen_feat_div', 'gen_feat_shift', 'gen_ring_ratio', 'gen_missed_conf'].includes(c) && typeof v === 'number') v = (v * 100).toFixed(1) + '%';
      if (['gen_seeds', 'gen_new_edges', 'gen_ext_edges', 'gen_ring_edges'].includes(c) && v === '') v = '–';
      td.textContent = v ?? '';
      tr.appendChild(td);
    });
    body.appendChild(tr);
  });
  table.appendChild(body);
  $('exp-table').innerHTML = '';
  $('exp-table').appendChild(table);
}

$('btn-meta-toggle').addEventListener('click', () => $('exp-meta').classList.toggle('hidden'));
$('btn-table-toggle').addEventListener('click', () => $('exp-table-wrap').classList.toggle('hidden'));

// ---------------------------------------------------------------------------
// Graph explorer
// ---------------------------------------------------------------------------
async function initGraph() {
  const sel = $('graph-dset');
  if (!sel.options.length) {
    state.schema.datasets.forEach(d => {
      const o = document.createElement('option');
      o.value = d; o.textContent = d;
      sel.appendChild(o);
    });
  }
  if (state.graph) return;
  await loadGraph();
}

async function loadGraph() {
  const dset = $('graph-dset').value;
  const n = $('graph-n').value;
  $('graph-info').textContent = `loading ${dset}…`;
  try {
    state.graph = await api(`/api/datasets/${encodeURIComponent(dset)}/graph?n=${n}`);
    drawGraph();
  } catch (e) {
    $('graph-info').textContent = 'failed to load graph';
  }
}

$('btn-graph-reload').addEventListener('click', () => { state.graph = null; loadGraph(); });
$('graph-dset').addEventListener('change', () => { state.graph = null; loadGraph(); });
$('graph-n').addEventListener('change', () => { state.graph = null; loadGraph(); });

function drawGraph() {
  const g = state.graph;
  const canvas = $('graph-canvas');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.width, H = canvas.height;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);

  const npos = g.sampled_nodes.map(n => ({
    x: (n.x + 1) / 2 * W,
    y: (n.y + 1) / 2 * H,
    label: n.label,
    degree: n.degree,
    id: n.id,
  }));

  ctx.strokeStyle = 'rgba(88,166,255,0.10)';
  ctx.lineWidth = 1;
  g.edges.forEach(([a, b]) => {
    const pa = npos[a], pb = npos[b];
    if (!pa || !pb) return;
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
  });

  for (const p of npos) {
    const r = 2 + Math.min(5, Math.log10(p.degree + 1) * 2);
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fillStyle = p.label === 1 ? '#f85149' : '#58a6ff';
    ctx.fill();
  }

  const n1 = g.sampled_nodes.filter(n => n.label === 1).length;
  const n0 = g.sampled_nodes.length - n1;
  $('graph-info').textContent =
    `${g.name} · total ${g.num_nodes_total.toLocaleString()} nodes / ${g.num_edges_total.toLocaleString()} edges · ` +
    `shown ${g.sampled_nodes.length} nodes (${n0} neg, ${n1} fraud) · ${g.edges.length} edges`;
}

// ---------------------------------------------------------------------------
// utils
// ---------------------------------------------------------------------------
function escapeHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// periodic elapsed ticker
setInterval(() => {
  if (state.activeRunId && state.runs.some(r => r.id === state.activeRunId)) {
    const run = state.runs.find(r => r.id === state.activeRunId);
    if (run && run.state === 'running') {
      const el = Date.now() / 1000 - run.started_at;
      $('dash-elapsed').textContent = el.toFixed(0) + 's';
    }
  }
}, 1000);

init();
