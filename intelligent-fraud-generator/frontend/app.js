/* =====================================================================
   Intelligent Fraud Generator — dashboard logic (self-contained, no CDN).
   Animated live charts, SSE streaming, canvas graph explorer, toasts.
   ===================================================================== */

const API = '/api';

// ---------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------
const state = {
  schema: null,
  datasets: [],
  sims: {},                 // id -> {meta, report, events, rounds}
  selected: null,           // active live run id
  sse: null,
  graphData: null,
  graphAnim: null,          // rAF id
};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}

function fmtNum(v, d = 3) {
  if (v === null || v === undefined || isNaN(v)) return '—';
  return Number(v).toFixed(d);
}

function shortId(id) { return String(id).slice(0, 8); }

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

function stateBadge(s) {
  const map = { running: 'running', done: 'done', error: 'error', stopped: 'stopped', pending: 'pending' };
  return `<span class="badge badge-${map[s] || 'pending'}">${s}</span>`;
}

function toast(msg, kind = 'info', ms = 3600) {
  const wrap = $('#toasts');
  const t = el('div', `toast ${kind}`, msg);
  wrap.appendChild(t);
  setTimeout(() => { t.classList.add('out'); setTimeout(() => t.remove(), 450); }, ms);
}

// smooth count-up
function countUp(node, target, opts = {}) {
  const decimals = opts.decimals != null ? opts.decimals : (Number.isInteger(target) ? 0 : 2);
  const dur = opts.dur || 700;
  if (!node) return;
  const start = performance.now();
  const from = 0;
  function tick(now) {
    const p = Math.min(1, (now - start) / dur);
    const e = 1 - Math.pow(1 - p, 3); // easeOutCubic
    const val = from + (target - from) * e;
    node.textContent = decimals ? val.toFixed(decimals) : Math.round(val).toLocaleString();
    if (p < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

// ---------------------------------------------------------------------------
// fetch
// ---------------------------------------------------------------------------
async function apiGet(path) {
  const r = await fetch(API + path);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function apiPost(path, body) {
  const r = await fetch(API + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

// ---------------------------------------------------------------------------
// tabs (with sliding indicator)
// ---------------------------------------------------------------------------
function moveIndicator() {
  const active = $('.tab.active');
  const ind = $('#tabIndicator');
  if (active && ind) {
    ind.style.width = active.offsetWidth + 'px';
    ind.style.left = active.offsetLeft + 'px';
  }
}

function switchTab(name) {
  $$('.tab').forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
  $$('.panel').forEach((p) => p.classList.toggle('active', p.id === 'tab-' + name));
  moveIndicator();
  if (name === 'graph') {
    refreshGraphSelect();
    requestAnimationFrame(() => resizeGraphCanvas());
  }
}

$$('.tab').forEach((btn) => {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});
window.addEventListener('resize', () => { moveIndicator(); resizeGraphCanvas(); });

// ---------------------------------------------------------------------------
// schema + form
// ---------------------------------------------------------------------------
const CONFIG_FIELDS = [
  ['rounds', 'Rounds', 'number', 'how many attack/defend rounds'],
  ['base_accounts', 'Base accounts', 'number', 'genuine accounts in round 0'],
  ['initial_fraud', 'Initial fraud', 'number', 'attacker accounts in round 0'],
  ['genuine_per_round', 'Genuine / round', 'number', 'new normal users each round'],
  ['fraud_per_round', 'Fraud / round', 'number', 'new fraud accounts each round'],
  ['seed', 'Seed', 'number', 'random seed (reproducibility)'],
  ['supervised_ratio', 'Supervised ratio', 'number', 'fraction labelled in round 0'],
  ['budget_pos', 'Review budget +', 'number', 'max fraud accounts revealed/round'],
  ['budget_neg', 'Review budget −', 'number', 'max genuine accounts revealed/round'],
  ['gan_epochs', 'GAN epochs', 'number', 'generator training steps'],
  ['gan_noise_dim', 'GAN noise dim', 'number', 'latent space size'],
  ['gan_hidden', 'GAN hidden units', 'number', 'MLP hidden width'],
  ['diversity', 'Diversity', 'number', 'minimum distance from known fraud'],
  ['conn_coef', 'Connection coeff', 'number', 'fraction of victims linked'],
  ['ring_ratio', 'Ring ratio', 'number', 'share of fraud wired into rings'],
  ['profile_window', 'Memory window', 'number', 'rounds of misses remembered'],
];

function buildForm() {
  const d = state.schema.defaults;
  const wrap = $('#configForm');
  wrap.innerHTML = '';
  for (const [key, label, type, hint] of CONFIG_FIELDS) {
    const cell = el('div', 'field');
    const lab = el('label', '', label);
    lab.title = hint || '';
    const input = el('input');
    input.type = type;
    input.className = 'text-input';
    input.step = 'any';
    input.value = d[key];
    input.dataset.key = key;
    cell.appendChild(lab);
    cell.appendChild(input);
    wrap.appendChild(cell);
  }
}

function readConfig() {
  const cfg = { describe: $('#cfgDescribe').value || 'Untitled experiment' };
  $$('#configForm input').forEach((i) => {
    const key = i.dataset.key;
    const def = state.schema.defaults[key];
    cfg[key] = (typeof def === 'number') ? Number(i.value) : i.value;
  });
  cfg.generator_mode = $('#modeSeg .seg-btn.active').dataset.val;
  cfg.gen_type = $('#typeSeg .seg-btn.active').dataset.val;
  return cfg;
}

function applyPreset(cfg) {
  $$('#configForm input').forEach((i) => {
    if (cfg[i.dataset.key] !== undefined) i.value = cfg[i.dataset.key];
  });
  if (cfg.describe) $('#cfgDescribe').value = cfg.describe;
}

function wireSeg(segId, hintId, hints) {
  const seg = $(segId);
  seg.querySelectorAll('.seg-btn').forEach((b) => {
    b.addEventListener('click', () => {
      seg.querySelectorAll('.seg-btn').forEach((x) => x.classList.toggle('active', x === b));
      if (hintId && hints) $(hintId).innerHTML = hints[b.dataset.val] || '';
    });
  });
}

// ---------------------------------------------------------------------------
// launch
// ---------------------------------------------------------------------------
async function launch() {
  $('#launchError').textContent = '';
  const btn = $('#launchBtn');
  btn.disabled = true;
  btn.classList.add('loading');
  try {
    const cfg = readConfig();
    const res = await apiPost('/run', cfg);
    state.sims[res.id] = { meta: res, events: [], rounds: [], report: null };
    toast('Simulation launched: ' + shortId(res.id), 'ok');
    await loadHistory();
    await populateRunSelects();
    selectLiveRun(res.id);
    switchTab('live');
  } catch (err) {
    $('#launchError').textContent = 'Launch failed: ' + err.message;
    toast('Launch failed: ' + err.message, 'err');
  } finally {
    btn.disabled = false;
    btn.classList.remove('loading');
  }
}

// ---------------------------------------------------------------------------
// run selects
// ---------------------------------------------------------------------------
async function populateRunSelects() {
  const ids = Object.keys(state.sims);
  for (const sel of ['#liveRunSelect', '#graphRunSelect']) {
    const node = $(sel);
    if (!node) continue;
    const cur = node.value;
    const keep = node.querySelector('option[value=""]');
    node.innerHTML = '';
    if (keep) node.appendChild(keep);
    for (const id of ids) {
      const m = state.sims[id].meta || {};
      const o = el('option');
      o.value = id;
      o.textContent = `${shortId(id)} · ${(m.describe || '—').slice(0, 30)}`;
      node.appendChild(o);
    }
    if (cur && ids.includes(cur)) node.value = cur;
  }
}

// ---------------------------------------------------------------------------
// live tab
// ---------------------------------------------------------------------------
function selectLiveRun(id) {
  state.selected = id;
  const sel = $('#liveRunSelect');
  if (sel) sel.value = id;
  $('#liveRunLabel').textContent = '';
  const m = state.sims[id] && state.sims[id].meta;
  if (m) $('#liveRunLabel').textContent = (m.describe || id);
  openStream(id);
}

function openStream(id) {
  if (state.sse) state.sse.close();
  const sse = new EventSource(`${API}/stream/${id}`);
  state.sse = sse;
  sse.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    state.sims[id] = state.sims[id] || { meta: {}, events: [], rounds: [], report: null };
    state.sims[id].events.push(ev);
    if (ev.type === 'round_result') {
      state.sims[id].rounds.push(ev);
      renderLive();
    } else if (ev.type === 'state' && ev.finished) {
      sse.close();
      state.sse = null;
      loadReport(id).then(() => { renderLive(); loadHistory(); populateRunSelects(); });
      toast('Run finished: ' + shortId(id), 'ok');
    }
    if (ev.type === 'log') renderLog(id);
  };
  sse.onerror = () => { /* server closed / transient */ };
}

function renderRoundTracker(sim) {
  const box = $('#roundTracker');
  if (!sim) { box.innerHTML = ''; return; }
  const total = (sim.report && sim.report.config && sim.report.config.rounds) ||
                (sim.meta && sim.meta.config && sim.meta.config.rounds) ||
                (sim.rounds.length + 1);
  const label = el('span', 'rt-label', 'rounds');
  box.innerHTML = '';
  box.appendChild(label);
  for (let r = 0; r < total; r++) {
    const rec = sim.rounds.find((x) => x.round === r);
    const chip = el('div', 'round-chip');
    const ball = el('span', 'ball');
    chip.appendChild(ball);
    chip.appendChild(document.createTextNode('r' + r));
    if (rec) chip.classList.add('done');
    else if (r === sim.rounds.length) chip.classList.add('active');
    box.appendChild(chip);
  }
}

function renderMetrics(sim) {
  const r = sim.rounds[sim.rounds.length - 1];
  const m = r.metrics || {};
  const g = r.gen || {};
  const cards = [
    ['Round', `#${r.round}`, 'r', null, null],
    ['Nodes', r.num_nodes, 'n', null, 'accounts'],
    ['Missed fraud', r.missed, r.missed > 0 ? 'bad' : 'good', null, 'escaped'],
    ['Macro-F1', m.f1, 'm', 3, null],
    ['AUC', m.auc, 'm', 3, null],
    ['Recall', m.rec, 'm', 3, null],
    ['Precision', m.prec, 'm', 3, null],
    ['Threshold', m.threshold, 'm', 3, null],
    ['Feat. diversity', g.gen_feat_div, 'p', 3, 'closer is better'],
    ['Feat. shift', g.gen_feat_shift, 'p', 3, 'vs missed seeds'],
    ['Ring ratio', g.gen_ring_ratio, 'a', 2, '0.5 = mixed'],
    ['New edges', g.gen_new_edges ?? '—', 'm', null, 'injected'],
  ];
  const wrap = $('#liveMetrics');
  wrap.innerHTML = '';
  cards.forEach(([k, v, c, dec, sub], i) => {
    const card = el('div', `metric ${c}`);
    card.style.animationDelay = (i * 0.05) + 's';
    card.appendChild(el('div', 'k', k));
    const vNode = el('div', 'v', '0');
    card.appendChild(vNode);
    if (sub) card.appendChild(el('div', 's', sub));
    wrap.appendChild(card);
    if (typeof v === 'number' && !isNaN(v)) countUp(vNode, v, { decimals: dec, dur: 650 });
    else vNode.textContent = v;
  });
}

// ---------------------------------------------------------------------------
// animated svg charts
// ---------------------------------------------------------------------------
function lineChart(points, w, h, opts = {}) {
  if (!points || points.length < 1) return '<div class="empty">no data yet</div>';
  const pad = { l: 36, r: 12, t: 14, b: 24 };
  const iw = w - pad.l - pad.r;
  const ih = h - pad.t - pad.b;
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const yMin = Math.min(0, ...ys);
  const yMax = Math.max(0.05, ...ys);
  const X = (x) => pad.l + (maxX === minX ? 0 : (x - minX) / (maxX - minX)) * iw;
  const Y = (y) => pad.t + (1 - (y - yMin) / (yMax - yMin)) * ih;
  const path = points.map((p, i) => `${i ? 'L' : 'M'}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(' ');
  const grid = [0, 0.25, 0.5, 0.75, 1].map((v) =>
    `<line x1="${pad.l}" y1="${Y(v).toFixed(1)}" x2="${w - pad.r}" y2="${Y(v).toFixed(1)}" class="grid-line"/>`).join('');
  const dots = points.map((p, i) =>
    `<circle cx="${X(p[0]).toFixed(1)}" cy="${Y(p[1]).toFixed(1)}" r="3" fill="${opts.color || '#74c69d'}"
      style="animation-delay:${0.5 + i * 0.12}s" class="dot-anim"/>`).join('');
  const xlabels = points.map((p, i) =>
    `<text x="${X(p[0]).toFixed(1)}" y="${h - 8}" class="axis" text-anchor="middle">${p[0]}</text>`).join('');
  const last = points[points.length - 1];
  const value = `<text x="${X(last[0]).toFixed(1)}" y="${(Y(last[1]) - 8).toFixed(1)}"
    class="val-label" text-anchor="middle">${fmtNum(last[1], opts.decimals || 2)}</text>`;
  const gid = 'g' + Math.random().toString(36).slice(2, 8);
  const area = opts.fill ? `<path d="${path} L${X(last[0]).toFixed(1)},${pad.t + ih} L${X(points[0][0]).toFixed(1)},${pad.t + ih} Z"
    fill="url(#${gid})" class="chart-area-gradient" opacity="0.25"/>` : '';
  return `<svg viewBox="0 0 ${w} ${h}" class="chart-svg">
    <defs><linearGradient id="${gid}" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="${opts.color || '#74c69d'}" stop-opacity="0.9"/>
      <stop offset="1" stop-color="${opts.color || '#74c69d'}" stop-opacity="0"/>
    </linearGradient></defs>
    ${grid}${area}
    <path d="${path}" fill="none" stroke="${opts.color || '#74c69d'}" stroke-width="2.2"
      stroke-linecap="round" stroke-linejoin="round" class="path-anim" style="filter: drop-shadow(0 0 6px ${opts.glow || opts.color || '#74c69d'})"/>
    ${dots}${xlabels}${value}</svg>`;
}

function barChart(items, w, h, opts = {}) {
  const sorted = Object.entries(items).sort((a, b) => b[1] - a[1]).slice(0, 12);
  if (!sorted.length) return '<div class="empty">no data yet</div>';
  const pad = { l: 8, r: 8, t: 18, b: 46 };
  const iw = w - pad.l - pad.r;
  const ih = h - pad.t - pad.b;
  const max = Math.max(...sorted.map(([, v]) => v), 1);
  const bw = Math.max(8, Math.min(30, (iw / sorted.length) - 8));
  const bwTotal = bw + 8;
  const startX = pad.l + Math.max(0, (iw - bwTotal * sorted.length) / 2);
  const bars = sorted.map(([k, v], i) => {
    const bh = (v / max) * ih;
    const x = startX + i * bwTotal;
    const y = pad.t + ih - bh;
    const label = k.length > 22 ? k.slice(0, 20) + '…' : k;
    const rotate = k.length > 14 ? 'transform="rotate(-24)"' : '';
    const color = opts.colors ? (opts.colors[k] || opts.color || '#95d5b2') : (opts.color || '#95d5b2');
    return `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bw}" height="${bh.toFixed(1)}"
      rx="4" fill="${color}" class="bar-anim" style="animation-delay:${i * 0.06}s">
      <title>${esc(k)}: ${v}</title></rect>
      <text x="${(x + bw / 2).toFixed(1)}" y="${(y - 4).toFixed(1)}" class="val-label" text-anchor="middle">${v}</text>
      <text x="${(x + bw / 2).toFixed(1)}" y="${h - 10}" class="axis" text-anchor="end" ${rotate}>${label}</text>`;
  }).join('');
  return `<svg viewBox="0 0 ${w} ${h}" class="chart-svg">${bars}</svg>`;
}

// ---------------------------------------------------------------------------
// live charts
// ---------------------------------------------------------------------------
const PERF = [
  ['F1', 'metrics.f1', '#52b788', true],
  ['AUC', 'metrics.auc', '#95d5b2', false],
  ['Recall', 'metrics.rec', '#b7e4c7', false],
  ['Prec', 'metrics.prec', '#fbbf24', false],
];
const GEN = [
  ['feat. diversity', 'gen_feat_div', '#b7e4c7', true],
  ['feat. shift', 'gen_feat_shift', '#e89b7a', true],
  ['ring ratio', 'gen_ring_ratio', '#fbbf24', false],
];

function renderPerfChart(sim) {
  const rounds = sim.rounds;
  const w = 600, h = 190;
  $('#perfLegend').innerHTML = PERF.map(([name, , color]) =>
    `<span><i class="swatch" style="background:${color}"></i>${name}</span>`).join('');
  $('#livePerfChart').innerHTML = '<div class="stack">' + PERF.map(([name, key, color]) =>
    `<div class="chart-inline"><div class="mini-label">${name}</div>` +
    lineChart(rounds.map((r) => [r.round, getPath(r, key)]), w, h, { color }) + '</div>'
  ).join('') + '</div>';
}

function renderGenChart(sim) {
  const rounds = sim.rounds;
  const w = 600, h = 190;
  $('#genLegend').innerHTML = GEN.map(([name, , color]) =>
    `<span><i class="swatch" style="background:${color}"></i>${name}</span>`).join('');
  $('#liveGenChart').innerHTML = '<div class="stack">' + GEN.map(([name, key, color]) =>
    `<div class="chart-inline"><div class="mini-label">${name}</div>` +
    lineChart(rounds.map((r) => [r.round, getPath(r, 'gen.' + key) ?? 0]), w, h, { color }) + '</div>'
  ).join('') + '</div>' +
  `<p class="muted">New referral edges: ${rounds.map((r) => `r${r.round}: ${r.gen?.gen_new_edges ?? 0}`).join(' · ')}</p>`;
}

function getPath(obj, path) {
  return path.split('.').reduce((o, k) => (o == null ? o : o[k]), obj);
}

function renderStrategies(sim) {
  const last = sim.rounds[sim.rounds.length - 1];
  const counts = last.gen?.gen_strategies || {};
  const colors = {
    'fake_identity': '#52b788', 'referral_farming': '#e89b7a',
    'device_spray': '#fbbf24', 'vpn_hop': '#95d5b2', 'quiet_sampler': '#b7e4c7',
  };
  const w = 560, h = 240;
  $('#liveStrategies').innerHTML = barChart(counts, w, h, { colors });
}

function renderLog(id) {
  const sim = state.sims[id];
  if (!sim) return;
  const box = $('#liveLog');
  const logs = sim.events.filter((e) => e.type === 'log').slice(-220);
  if (!logs.length) { box.innerHTML = '<div class="empty">waiting for logs…</div>'; return; }
  box.innerHTML = logs.map((e) => {
    const cls = /\[error\]/.test(e.text) ? 'err' : /\[warn\]/.test(e.text) ? 'warn' : /\[done\]/.test(e.text) ? 'ok' : '';
    return `<div class="log-line ${cls}"><span class="t">${fmtNum(e.t, 1)}s</span><span class="tx">${esc(e.text)}</span></div>`;
  }).join('');
  if ($('#liveAutoScroll').checked) box.scrollTop = box.scrollHeight;
  $('#liveLogCount').textContent = `· ${logs.length} lines`;
}

function renderLive() {
  const sim = state.sims[state.selected];
  if (!sim || !sim.rounds.length) return;
  renderRoundTracker(sim);
  renderMetrics(sim);
  renderPerfChart(sim);
  renderGenChart(sim);
  renderStrategies(sim);
  renderLog(state.selected);
}

// ---------------------------------------------------------------------------
// reports
// ---------------------------------------------------------------------------
async function loadReport(id, force) {
  if (state.sims[id] && state.sims[id].report && !force) return state.sims[id].report;
  const rep = await apiGet(`/report/${id}`);
  state.sims[id].report = rep;
  state.sims[id].rounds = [];
  for (const ev of rep.rounds || []) state.sims[id].rounds.push(ev);
  return rep;
}

// ---------------------------------------------------------------------------
// history
// ---------------------------------------------------------------------------
async function loadHistory() {
  const h = await apiGet('/history');
  const tbody = $('#historyTable tbody');
  tbody.innerHTML = h.map((m) => {
    const meta = (state.sims[m.id] && state.sims[m.id].meta) || m;
    const mode = (meta && meta.config && meta.config.generator_mode) || meta.generator_mode || '—';
    const type = (meta && meta.config && meta.config.gen_type) || meta.gen_type || '—';
    return `<tr>
      <td>${shortId(m.id)}</td>
      <td>${esc(m.describe || '—')}</td>
      <td>${esc(mode)}</td>
      <td>${esc(type)}</td>
      <td>${m.rounds ?? '—'}</td>
      <td>${stateBadge(m.state)}</td>
      <td>${new Date(m.started_at * 1000).toLocaleTimeString()}</td>
      <td><button class="btn mini" data-load="${m.id}">report</button>
          <button class="btn mini" data-live="${m.id}">live</button></td>
    </tr>`;
  }).join('') || '<tr><td colspan="8" class="empty">no runs yet</td></tr>';

  const runsBody = $('#runsTable tbody');
  runsBody.innerHTML = h.map((m) => {
    const s = state.sims[m.id];
    const meta = (state.sims[m.id] && state.sims[m.id].meta) || m;
    const mode = (meta && meta.config && meta.config.generator_mode) || meta.generator_mode || '—';
    const type = (meta && meta.config && meta.config.gen_type) || meta.gen_type || '—';
    const done = s && s.report ? s.report.rounds.length : (m.state === 'done' ? m.rounds : '…');
    return `<tr>
      <td>${shortId(m.id)}</td>
      <td>${esc(m.describe || '—')}</td>
      <td>${esc(mode)}</td>
      <td>${esc(type)}</td>
      <td>${m.rounds ?? '—'}</td>
      <td>${done}</td>
      <td>${stateBadge(m.state)}</td>
      <td><button class="btn mini" data-live="${m.id}">open</button></td>
    </tr>`;
  }).join('') || '<tr><td colspan="8" class="empty">no runs yet</td></tr>';

  tbody.querySelectorAll('[data-load]').forEach((b) => b.addEventListener('click', async () => {
    await loadReport(b.dataset.load, true);
    renderCompare();
    toast('Report loaded', 'info');
  }));
  tbody.querySelectorAll('[data-live]').forEach((b) => b.addEventListener('click', () => goLive(b.dataset.live)));
  runsBody.querySelectorAll('[data-live]').forEach((b) => b.addEventListener('click', () => goLive(b.dataset.live)));
}

function goLive(id) {
  selectLiveRun(id);
  switchTab('live');
  renderLive();
}

function renderCompare() {
  const rows = [];
  for (const id of Object.keys(state.sims)) {
    const sim = state.sims[id];
    if (!sim.report) continue;
    const rounds = sim.rounds;
    if (!rounds.length) continue;
    const last = rounds[rounds.length - 1].metrics;
    rows.push({ id, describe: sim.report.config.describe || id, mode: sim.report.config.generator_mode, type: sim.report.config.gen_type, f1: last.f1, auc: last.auc });
  }
  if (!rows.length) { $('#compareChart').innerHTML = '<div class="empty">no finished runs yet</div>'; return; }
  const w = 620, h = 190;
  const f1 = rows.map((r, i) => [i + 1, r.f1]);
  const auc = rows.map((r, i) => [i + 1, r.auc]);
  $('#compareChart').innerHTML =
    `<div class="legend"><span><i class="swatch" style="background:#52b788"></i>F1</span>
      <span><i class="swatch" style="background:#95d5b2"></i>AUC</span></div>
     <div class="chart-inline">${lineChart(f1, w, h, { color: '#52b788' })}</div>
     <div class="chart-inline">${lineChart(auc, w, h, { color: '#95d5b2' })}</div>
     <table class="table"><thead><tr><th>id</th><th>describe</th><th>mode</th><th>model</th><th>F1</th><th>AUC</th></tr></thead><tbody>` +
     rows.map((r, i) => `<tr><td>${shortId(r.id)}</td><td>${esc(r.describe)}</td><td>${esc(r.mode)}</td><td>${esc(r.type)}</td>
       <td>${fmtNum(r.f1)}</td><td>${fmtNum(r.auc)}</td></tr>`).join('') + '</tbody></table>';
}

// ---------------------------------------------------------------------------
// graph explorer (canvas force layout)
// ---------------------------------------------------------------------------
let graphSim = null;
const GRAPH_NODES = [];
const GRAPH_LINKS = [];

function resizeGraphCanvas() {
  const cv = $('#graphCanvas');
  if (!cv) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = cv.getBoundingClientRect();
  cv.width = (rect.width || 1100) * dpr;
  cv.height = 640 * dpr;
  const ctx = cv.getContext('2d');
  ctx.scale(dpr, dpr);
  cv.style.height = Math.round(rect.width * 640 / 1100) + 'px';
}

async function refreshGraphSelect() {
  await populateRunSelects();
  const has = Object.keys(state.sims).length > 0;
  $('#graphRunSelect').disabled = !has;
  $('#graphLoadBtn').disabled = !has;
}

async function loadGraph() {
  const id = $('#graphRunSelect').value;
  if (!id) return;
  try {
    const g = await apiGet(`/graph/${id}`);
    state.graphData = g;
    buildGraphSim();
    toast('Graph loaded: ' + g.nodes.length + ' nodes', 'info');
  } catch (err) {
    toast('Graph load failed: ' + err.message, 'err');
  }
}

function buildGraphSim() {
  const g = state.graphData;
  GRAPH_NODES.length = 0;
  GRAPH_LINKS.length = 0;
  if (!g || !g.nodes.length) return;

  const W = 1100, H = 640;
  const referralSet = new Set(g.edges.referral.map((e) => e.join('|')));

  g.nodes.forEach((n, i) => {
    GRAPH_NODES.push({
      i, id: n.id, label: n.label, round: n.round, strategy: n.strategy || n.base || '',
      attrs: n.attrs, r: n.label === 1 ? 7 : 3.6,
      x: Math.random() * W, y: Math.random() * H,
      vx: 0, vy: 0,
    });
  });

  if ($('#graphShowReferral').checked) {
    for (const [a, b] of g.edges.referral) {
      const i = GRAPH_NODES.findIndex((n) => n.id === a);
      const j = GRAPH_NODES.findIndex((n) => n.id === b);
      if (i >= 0 && j >= 0) GRAPH_LINKS.push({ a: i, b: j, kind: referralSet.has(a + '|' + b) ? 'referral' : 'referral2' });
    }
  }
  if ($('#graphShowShared').checked) {
    const devMap = new Map(), ipMap = new Map();
    for (const [a, d] of g.edges.device) { if (!devMap.has(d)) devMap.set(d, []); devMap.get(d).push(a); }
    for (const [a, ip] of g.edges.ip) { if (!ipMap.has(ip)) ipMap.set(ip, []); ipMap.get(ip).push(a); }
    const linkShared = (map, seen) => {
      for (const list of map.values()) {
        if (list.length < 2) continue;
        for (let x = 0; x < list.length; x++) for (let y = x + 1; y < list.length; y++) {
          if (list[y] - list[x] > 40) continue;
          const key = list[x] + '|' + list[y];
          if (seen.has(key)) continue;
          seen.add(key);
          const i = GRAPH_NODES.findIndex((n) => n.id === list[x]);
          const j = GRAPH_NODES.findIndex((n) => n.id === list[y]);
          if (i >= 0 && j >= 0) GRAPH_LINKS.push({ a: i, b: j, kind: 'shared' });
        }
      }
    };
    linkShared(devMap, new Set());
    linkShared(ipMap, new Set());
  }
}

function graphLoop() {
  const cv = $('#graphCanvas');
  if (!cv) return;
  const ctx = cv.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const W = cv.width / dpr, H = cv.height / dpr;
  ctx.clearRect(0, 0, W, H);

  const showG = $('#graphShowGenuine').checked;
  const showF = $('#graphShowFraud').checked;
  const colorRound = $('#graphColorRound').checked;
  const t = performance.now() / 1000;

  if (!GRAPH_NODES.length) {
    ctx.fillStyle = '#52b788';
    ctx.font = '13px Cascadia Code, monospace';
    ctx.textAlign = 'center';
    ctx.fillText('select a run and press “Load graph”', W / 2, H / 2);
    state.graphAnim = requestAnimationFrame(graphLoop);
    return;
  }

  const maxRound = Math.max(...GRAPH_NODES.map((n) => n.round));
  const roundColor = (r) => {
    const k = maxRound <= 1 ? 0 : r / maxRound;
    const hue = 150 - 100 * k;
    return `hsl(${hue}, 70%, 52%)`;
  };

  // --- physics ---
  const n = GRAPH_NODES.length;
  const repulsion = 900 / Math.max(20, Math.sqrt(n));
  for (let i = 0; i < n; i++) {
    const a = GRAPH_NODES[i];
    a.vx *= 0.82; a.vy *= 0.82;
    // repulsion
    for (let j = i + 1; j < n; j++) {
      const b = GRAPH_NODES[j];
      let dx = a.x - b.x, dy = a.y - b.y;
      let d = Math.sqrt(dx * dx + dy * dy) || 0.001;
      const f = Math.min(2.2, repulsion / (d * d));
      a.vx += (dx / d) * f; a.vy += (dy / d) * f;
      b.vx -= (dx / d) * f; b.vy -= (dy / d) * f;
    }
    // gentle centering + drift
    a.vx += (W / 2 - a.x) * 0.0006 + Math.sin(t + i) * 0.02;
    a.vy += (H / 2 - a.y) * 0.0006 + Math.cos(t * 1.3 + i) * 0.02;
  }
  for (const l of GRAPH_LINKS) {
    const a = GRAPH_NODES[l.a], b = GRAPH_NODES[l.b];
    const dx = b.x - a.x, dy = b.y - a.y;
    const d = Math.sqrt(dx * dx + dy * dy) || 0.001;
    const target = l.kind === 'shared' ? 70 : 50;
    const f = Math.max(-0.6, Math.min(0.6, (d - target) * 0.012));
    a.vx += (dx / d) * f; a.vy += (dy / d) * f;
    b.vx -= (dx / d) * f; b.vy -= (dy / d) * f;
  }
  for (const a of GRAPH_NODES) {
    a.x += a.vx; a.y += a.vy;
    a.x = Math.max(18, Math.min(W - 18, a.x));
    a.y = Math.max(18, Math.min(H - 18, a.y));
  }

  // --- edges ---
  for (const l of GRAPH_LINKS) {
    const a = GRAPH_NODES[l.a], b = GRAPH_NODES[l.b];
    const style = l.kind === 'referral' ? `rgba(255,143,107,${0.35 + 0.1 * Math.sin(t + l.a)})`
      : l.kind === 'referral2' ? `rgba(116,198,157,0.18)`
      : `rgba(251,191,36,0.12)`;
    ctx.strokeStyle = style;
    ctx.lineWidth = l.kind === 'referral' ? 1.2 : 0.7;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  // --- nodes ---
  for (const node of GRAPH_NODES) {
    if (node.label === 1 && !showF) continue;
    if (node.label === 0 && !showG) continue;
    const isFraud = node.label === 1;
    const pulse = isFraud ? 1 + Math.sin(t * 2.4 + node.i) * 0.22 : 1 + Math.sin(t * 1.4 + node.i) * 0.1;
    const r = node.r * pulse;
    let fill = isFraud ? '#ef6a62' : (colorRound ? roundColor(node.round) : '#52b788');
    ctx.beginPath();
    ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
    if (isFraud) {
      ctx.shadowColor = '#ef6a62';
      ctx.shadowBlur = 14 + Math.sin(t * 3 + node.i) * 6;
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.shadowBlur = 0;
    } else {
      ctx.fillStyle = fill;
      ctx.globalAlpha = 0.9;
      ctx.fill();
      ctx.globalAlpha = 1;
    }
  }

  state.graphAnim = requestAnimationFrame(graphLoop);
}

// hover tooltip
let hoverPos = null;
function graphMouse(e) {
  const cv = $('#graphCanvas');
  const rect = cv.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  const scale = (rect.width) / cv.width * (window.devicePixelRatio || 1);
  const sx = x / scale, sy = y / scale;
  let hit = null;
  for (const node of GRAPH_NODES) {
    const dx = node.x - sx, dy = node.y - sy;
    if (dx * dx + dy * dy < (node.r + 6) * (node.r + 6)) { hit = node; break; }
  }
  const tip = $('#graphTip');
  if (hit) {
    const roundTxt = hit.round === undefined ? '?' : hit.round;
    const attr = Object.entries(hit.attrs || {}).slice(0, 4)
      .map(([k, v]) => `${k}=${v}`).join(' ');
    tip.innerHTML = `<div class="gt-t">${hit.label === 1 ? 'FRAUD' : 'GENUINE'} · acct ${hit.id} · r${roundTxt}</div>` +
      (hit.strategy ? `strategy: ${esc(hit.strategy)}<br>` : '') + esc(attr);
    tip.style.display = 'block';
    tip.style.left = Math.min(rect.width - 280, x + 16) + 'px';
    tip.style.top = (y + 12) + 'px';
  } else {
    tip.style.display = 'none';
  }
}

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------
async function init() {
  const status = $('#serverStatus');
  status.className = 'status checking';
  status.querySelector('.status-text').textContent = 'checking…';
  try {
    const [schema, datasets, history] = await Promise.all([
      apiGet('/schema'), apiGet('/datasets'), apiGet('/history'),
    ]);
    state.schema = schema;
    state.datasets = datasets;
    buildForm();
    status.className = 'status online';
    status.querySelector('.status-text').textContent = 'online · ' + (schema.defaults.rounds) + ' rounds default';
    history.forEach((m) => { if (!state.sims[m.id]) state.sims[m.id] = { meta: m, events: [], rounds: [], report: null }; });
    await populateRunSelects();

    const presetsWrap = $('#presets');
    datasets.forEach((ds) => {
      const b = el('button', 'preset-btn', ds.label);
      b.addEventListener('click', () => { applyPreset(ds.config); toast('Preset applied: ' + ds.label, 'info'); });
      presetsWrap.appendChild(b);
    });

    renderStrategiesAbout();
    loadHistory();
    renderCompare();
    moveIndicator();
    resizeGraphCanvas();
  } catch (err) {
    status.className = 'status offline';
    status.querySelector('.status-text').textContent = 'offline (' + err.message + ')';
    toast('Backend unreachable: ' + err.message, 'err');
  }

  wireSeg('#modeSeg', '#modeHint', {
    intelligent: '<b>intelligent</b> — the generator learns from the fraud the detector missed and synthesises new strategies (new devices, VPN IPs, shifted amounts/timing, rings &amp; referral chains); it never copies a miss.',
    replay: '<b>replay (baseline)</b> — next round\'s fraud is an exact clone of the missed fraud, mirroring the original framework behaviour.',
  });
  $('#modeHint').innerHTML = '<b>intelligent</b> — the generator learns from the fraud the detector missed and synthesises new strategies (new devices, VPN IPs, shifted amounts/timing, rings &amp; referral chains); it never copies a miss.';
  wireSeg('#typeSeg', null, null);

  $('#launchBtn').addEventListener('click', launch);
  $('#liveRefreshBtn').addEventListener('click', () => renderLive());
  $('#liveRunSelect').addEventListener('change', (e) => { if (e.target.value) selectLiveRun(e.target.value); });
  $('#graphLoadBtn').addEventListener('click', loadGraph);
  $('#graphRunSelect').addEventListener('change', () => {});
  ['graphShowGenuine', 'graphShowFraud', 'graphShowReferral', 'graphShowShared', 'graphColorRound']
    .forEach((id) => $(`#${id}`).addEventListener('change', () => { buildGraphSim(); }));

  const cv = $('#graphCanvas');
  cv.addEventListener('mousemove', graphMouse);
  cv.addEventListener('mouseleave', () => { $('#graphTip').style.display = 'none'; });
  state.graphAnim = requestAnimationFrame(graphLoop);

  setInterval(async () => {
    try {
      const h = await apiGet('/history');
      const key = JSON.stringify(h);
      if (key !== state._lastHistory) {
        state._lastHistory = key;
        h.forEach((m) => { if (!state.sims[m.id]) state.sims[m.id] = { meta: m, events: [], rounds: [], report: null }; });
        await populateRunSelects();
        loadHistory();
      }
    } catch { /* offline */ }
  }, 4000);
}

function renderStrategiesAbout() {
  const wrap = $('#strategyCards');
  const st = state.schema && state.schema.strategies;
  if (!st || !Object.keys(st).length) return;
  const maxRing = Math.max(...Object.values(st).map((v) => v.ring_affinity));
  wrap.innerHTML = Object.entries(st).map(([name, v]) => `
    <div class="strategy-chip">
      <div class="sc-name">${esc(name)}</div>
      <div class="sc-bar"><span>ring</span><div class="track"><div class="fill" style="width:${(v.ring_affinity / maxRing) * 100}%"></div></div><span>${v.ring_affinity}</span></div>
      <div class="sc-bar"><span>device spray</span><div class="track"><div class="fill" style="width:${v.device_spray * 100}%; animation-delay:.1s"></div></div><span>${v.device_spray}</span></div>
      <div class="sc-bar"><span>ip reuse</span><div class="track"><div class="fill" style="width:${v.ip_reuse * 100}%; animation-delay:.2s"></div></div><span>${v.ip_reuse}</span></div>
    </div>`).join('');
}

document.addEventListener('DOMContentLoaded', init);
