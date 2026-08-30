/* =====================================================================
   Adaptive Defensive Layer — dashboard logic (self-contained, no CDN).
   Animated live charts, SSE streaming, canvas graph explorer, defense tab.
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
  selectedDefense: null,    // active defense run id
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

function pct(v, d = 1) {
  if (v === null || v === undefined || isNaN(v)) return '—';
  return (Number(v) * 100).toFixed(d) + '%';
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

function countUp(node, target, opts = {}) {
  const decimals = opts.decimals != null ? opts.decimals : (Number.isInteger(target) ? 0 : 2);
  const dur = opts.dur || 700;
  if (!node) return;
  const start = performance.now();
  const from = 0;
  function tick(now) {
    const p = Math.min(1, (now - start) / dur);
    const e = 1 - Math.pow(1 - p, 3);
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
  if (name === 'defense') renderDefenseTab();
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

const ADL_FIELDS = [
  ['t1', 'Threshold T1', 'number', 'risk below T1 -> Allow'],
  ['t2', 'Threshold T2', 'number', 'risk above T2 -> Block'],
  ['threshold_alpha', 'Adaptation α', 'number', 'threshold learning rate'],
  ['review_catch_rate', 'Review catch rate', 'number', 'share of reviewed fraud that is caught'],
  ['w_pf', 'Weight · P_f', 'number', 'fraud probability weight'],
  ['w_centrality', 'Weight · C', 'number', 'centrality weight'],
  ['w_ring', 'Weight · S', 'number', 'ring participation weight'],
  ['w_velocity', 'Weight · V', 'number', 'velocity weight'],
  ['w_trust', 'Weight · A', 'number', 'trust weight'],
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

function buildAdlForm() {
  const d = state.schema.defaults;
  const wrap = $('#adlForm');
  wrap.innerHTML = '';
  for (const [key, label, type, hint] of ADL_FIELDS) {
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
  $$('#adlForm input').forEach((i) => {
    const key = i.dataset.key;
    const def = state.schema.defaults[key];
    cfg[key] = (typeof def === 'number') ? Number(i.value) : i.value;
  });
  cfg.generator_mode = $('#modeSeg .seg-btn.active').dataset.val;
  cfg.gen_type = $('#typeSeg .seg-btn.active').dataset.val;
  cfg.adl_enabled = $('#adlEnabled').checked;
  cfg.threshold_policy = $('#policySeg .seg-btn.active').dataset.val;
  return cfg;
}

function applyPreset(cfg) {
  $$('#configForm input').forEach((i) => {
    if (cfg[i.dataset.key] !== undefined) i.value = cfg[i.dataset.key];
  });
  $$('#adlForm input').forEach((i) => {
    if (cfg[i.dataset.key] !== undefined) i.value = cfg[i.dataset.key];
  });
  if (cfg.adl_enabled !== undefined) $('#adlEnabled').checked = !!cfg.adl_enabled;
  if (cfg.threshold_policy !== undefined) {
    $('#policySeg .seg-btn').forEach((b) => b.classList.toggle('active', b.dataset.val === cfg.threshold_policy));
  }
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
    selectDefenseRun(res.id);
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
  for (const sel of ['#liveRunSelect', '#defenseRunSelect', '#graphRunSelect']) {
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
      o.textContent = `${shortId(id)} · ${(m.describe || '—').slice(0, 34)}`;
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
      loadReport(id).then(() => { renderLive(); renderDefenseTab(); loadHistory(); populateRunSelects(); });
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
  const d = r.defense || null;
  const cards = [
    ['Round', `#${r.round}`, 'r', null, null],
    ['Nodes', r.num_nodes, 'n', null, 'all accounts'],
    ['Blocked total', d ? d.n_block : 0, d ? 'a' : 'm', null, 'ADL block decisions'],
    ['Missed by detector', r.missed, r.missed > 0 ? 'bad' : 'good', null, 'FN on active'],
    ['Fraud escaped', d ? d.fraud_escaped : '—', d && d.fraud_escaped > 0 ? 'bad' : 'good', null, 'survived defense'],
    ['Escape rate', d ? d.escape_rate : null, 'm', 3, d ? '↓ better' : 'ADL off'],
    ['False-block rate', d ? d.false_block_rate : null, 'm', 3, d ? '↓ better' : 'ADL off'],
    ['Defense recall', d ? d.defense_recall : null, 'm', 3, d ? 'fraud caught' : 'ADL off'],
    ['Avg risk', d ? d.avg_risk : null, 'a', 3, d ? 'system risk' : 'ADL off'],
    ['Decision latency', d ? d.decision_latency_ms : null, 'a', 2, d ? 'ms / txn' : 'ADL off'],
    ['Macro-F1', m.f1, 'm', 3, null],
    ['AUC', m.auc, 'm', 3, null],
    ['Threshold', m.threshold, 'm', 3, null],
    ['Feat. diversity', g.gen_feat_div, 'p', 3, 'closer is better'],
    ['Feat. shift', g.gen_feat_shift, 'p', 3, 'vs missed seeds'],
    ['Ring ratio', g.gen_ring_ratio, 'a', 2, '0.5 = mixed'],
  ];
  const wrap = $('#liveMetrics');
  wrap.innerHTML = '';
  cards.forEach(([k, v, c, dec, sub], i) => {
    const card = el('div', `metric ${c}`);
    card.style.animationDelay = (i * 0.04) + 's';
    card.appendChild(el('div', 'k', k));
    const vNode = el('div', 'v', '0');
    card.appendChild(vNode);
    if (sub) card.appendChild(el('div', 's', sub));
    wrap.appendChild(card);
    if (typeof v === 'number' && !isNaN(v)) countUp(vNode, v, { decimals: dec, dur: 600 });
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
  const yMin = opts.yMin != null ? opts.yMin : Math.min(0, ...ys);
  const yMax = opts.yMax != null ? opts.yMax : Math.max(0.05, ...ys);
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

// stacked bars: rows = [{round, allow, review, block, ...}]
function stackedBarChart(rows, w, h, opts = {}) {
  if (!rows || !rows.length) return '<div class="empty">no data yet</div>';
  const keys = opts.keys || ['allow', 'review', 'block'];
  const colors = opts.colors || { allow: '#52b788', review: '#e0a458', block: '#ef6a62' };
  const pad = { l: 44, r: 8, t: 18, b: 30 };
  const iw = w - pad.l - pad.r;
  const ih = h - pad.t - pad.b;
  const totals = rows.map((r) => keys.reduce((s, k) => s + (r[k] || 0), 0));
  const max = Math.max(...totals, 1);
  const bw = Math.max(10, Math.min(36, iw / rows.length - 6));
  const startX = pad.l + Math.max(0, (iw - (bw + 6) * rows.length) / 2);
  const Y = (v) => pad.t + ih - (v / max) * ih;
  const bars = rows.map((r, i) => {
    const x = startX + i * (bw + 6);
    let yAcc = pad.t + ih;
    let seg = '';
    for (const k of keys) {
      const v = r[k] || 0;
      if (v <= 0) continue;
      const hh = (v / max) * ih;
      const y = yAcc - hh;
      yAcc = y;
      seg += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bw}" height="${hh.toFixed(1)}"
        rx="2" fill="${colors[k] || '#52b788'}" class="bar-anim" style="animation-delay:${i * 0.05}s">
        <title>r${r.round} ${k}: ${v}</title></rect>`;
    }
    const tot = totals[i];
    return seg + `<text x="${(x + bw / 2).toFixed(1)}" y="${(Y(tot) - 5).toFixed(1)}" class="val-label" text-anchor="middle">${tot}</text>`
      + `<text x="${(x + bw / 2).toFixed(1)}" y="${h - 10}" class="axis" text-anchor="middle">r${r.round}</text>`;
  }).join('');
  return `<svg viewBox="0 0 ${w} ${h}" class="chart-svg">${bars}</svg>`;
}

// horizontal risk-distribution histogram with T1/T2 markers
function riskHistogram(hist, t1, t2, w, h, opts = {}) {
  const pad = { l: 34, r: 10, t: 18, b: 26 };
  const iw = w - pad.l - pad.r;
  const ih = h - pad.t - pad.b;
  const nb = hist.nbins || 20;
  const total = Math.max(...(hist.allow || []).map((_, i) => (hist.allow[i] || 0) + (hist.review[i] || 0) + (hist.block[i] || 0)), 1);
  const bw = iw / nb;
  const X = (frac) => pad.l + frac * iw;
  const bars = [];
  const colors = { allow: 'rgba(82,183,136,.85)', review: 'rgba(224,164,88,.9)', block: 'rgba(239,106,98,.95)' };
  for (let i = 0; i < nb; i++) {
    const a = hist.allow[i] || 0, rv = hist.review[i] || 0, b = hist.block[i] || 0;
    const tot = a + rv + b;
    if (!tot) continue;
    const x = X(i / nb);
    let yAcc = pad.t + ih;
    for (const [k, v] of [['allow', a], ['review', rv], ['block', b]]) {
      if (v <= 0) continue;
      const hh = (v / total) * ih;
      yAcc -= hh;
      bars.push(`<rect x="${x.toFixed(1)}" y="${yAcc.toFixed(1)}" width="${bw.toFixed(1)}" height="${hh.toFixed(1)}"
        fill="${colors[k]}" class="bar-anim" style="animation-delay:${i * 0.02}s">
        <title>bin ${(i / nb).toFixed(2)}-${((i + 1) / nb).toFixed(2)} ${k}: ${v}</title></rect>`);
    }
  }
  // threshold markers
  const mk = (frac, label, color) => `
    <line x1="${X(frac).toFixed(1)}" y1="${pad.t}" x2="${X(frac).toFixed(1)}" y2="${pad.t + ih}"
      stroke="${color}" stroke-width="2" stroke-dasharray="5 4" class="thresh-line">
    <title>${label} = ${frac.toFixed(2)}</title></line>
    <text x="${X(frac).toFixed(1)}" y="${pad.t - 4}" class="thresh-label" fill="${color}" text-anchor="middle">${label}</text>`;
  const grid = [0, 0.25, 0.5, 0.75, 1].map((v) =>
    `<line x1="${pad.l}" y1="${(pad.t + ih - v * ih).toFixed(1)}" x2="${w - pad.r}" y2="${(pad.t + ih - v * ih).toFixed(1)}" class="grid-line"/>`).join('');
  const xlab = [0, 0.25, 0.5, 0.75, 1].map((v) =>
    `<text x="${X(v).toFixed(1)}" y="${h - 8}" class="axis" text-anchor="middle">${v.toFixed(2)}</text>`).join('');
  const t1m = (t1 != null && t1 <= 1) ? mk(t1, 'T1', '#e0a458') : '';
  const t2m = (t2 != null && t2 <= 1) ? mk(t2, 'T2', '#ef6a62') : '';
  return `<svg viewBox="0 0 ${w} ${h}" class="chart-svg">${grid}${bars}${t1m}${t2m}${xlab}</svg>`;
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
const DEF_LIVE = [
  ['escape', 'defense.escape_rate', '#ef6a62', true],
  ['false-block', 'defense.false_block_rate', '#e0a458', true],
  ['dRecall', 'defense.defense_recall', '#52b788', false],
  ['dPrec', 'defense.defense_precision', '#95d5b2', false],
];
const GEN = [
  ['feat. diversity', 'gen_feat_div', '#b7e4c7', true],
  ['feat. shift', 'gen_feat_shift', '#e89b7a', true],
  ['ring ratio', 'gen_ring_ratio', '#fbbf24', false],
];

function getPath(obj, path) {
  return path.split('.').reduce((o, k) => (o == null ? o : o[k]), obj);
}

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

function renderDefChart(sim) {
  const rounds = sim.rounds;
  const w = 600, h = 190;
  const hasDef = rounds.some((r) => r.defense);
  $('#defLegend').innerHTML = DEF_LIVE.map(([name, , color]) =>
    `<span><i class="swatch" style="background:${color}"></i>${name}</span>`).join('');
  if (!hasDef) {
    $('#liveDefChart').innerHTML = '<div class="empty">ADL disabled &mdash; no defence data</div>';
    return;
  }
  $('#liveDefChart').innerHTML = '<div class="stack">' + DEF_LIVE.map(([name, key, color]) =>
    `<div class="chart-inline"><div class="mini-label">${name}</div>` +
    lineChart(rounds.filter((r) => r.defense).map((r) => [r.round, getPath(r, key)]), w, h, { color }) + '</div>'
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
  renderDefChart(sim);
  renderGenChart(sim);
  renderStrategies(sim);
  renderLog(state.selected);
}

// ---------------------------------------------------------------------------
// defense tab
// ---------------------------------------------------------------------------
function selectDefenseRun(id) {
  state.selectedDefense = id;
  const sel = $('#defenseRunSelect');
  if (sel) sel.value = id;
  const m = state.sims[id] && state.sims[id].meta;
  $('#defenseRunLabel').textContent = m ? (m.describe || id) : '';
  renderDefenseTab();
}

function defenseRounds(sim) {
  return (sim.rounds || []).filter((r) => r.defense);
}

function renderAdlState(sim, rounds) {
  const box = $('#adlState');
  const rep = sim.report;
  if (!rounds.length) { box.innerHTML = '<div class="empty">no defense rounds yet</div>'; return; }
  const last = rounds[rounds.length - 1].defense;
  const st = (rep && rep.adl_state) || {};
  const items = [
    ['policy', st.policy || last.policy || '—'],
    ['T1', fmtNum(last.t1, 3)],
    ['T2', fmtNum(last.t2, 3)],
    ['α (alpha)', fmtNum(st.alpha, 3)],
    ['review catch', pct(last.reviewed_fraud_caught / (last.reviewed_fraud_caught + last.reviewed_fraud_survived || 1)) + ' →',
     'configured ' + (st.review_catch_rate != null ? st.review_catch_rate : '?')],
    ['avg risk', fmtNum(last.avg_risk, 3)],
    ['escape', pct(last.escape_rate)],
    ['false block', pct(last.false_block_rate)],
    ['defense recall', pct(last.defense_recall)],
  ];
  box.innerHTML = items.map(([k, v, sub]) => `
    <div class="adl-kv"><span class="adl-k">${esc(k)}</span><span class="adl-v">${esc(v)}</span>${sub ? `<span class="adl-s">${esc(sub)}</span>` : ''}</div>`).join('');
}

function renderAdlWeights(rounds) {
  const box = $('#adlWeights');
  if (!rounds.length) { box.innerHTML = '<div class="empty">no data yet</div>'; return; }
  const w = rounds[rounds.length - 1].defense.weights || {};
  const keys = [['w_pf', 'P_f'], ['w_centrality', 'C'], ['w_ring', 'S'], ['w_velocity', 'V'], ['w_trust', 'A']];
  const max = Math.max(...keys.map(([k]) => w[k] || 0), 1);
  box.innerHTML = keys.map(([k, name], i) => {
    const v = w[k] || 0;
    return `<div class="w-row">
      <span class="w-name">w&middot;${name}</span>
      <div class="w-track"><div class="w-fill" style="width:${(v / max) * 100}%; animation-delay:${i * .08}s"></div></div>
      <span class="w-val">${fmtNum(v, 3)}</span>
    </div>`;
  }).join('');
}

function renderRiskDist(rounds) {
  const box = $('#riskDistChart');
  if (!rounds.length) { box.innerHTML = '<div class="empty">no data yet</div>'; return; }
  const last = rounds[rounds.length - 1].defense;
  $('#riskLegend').innerHTML =
    `<span><i class="swatch" style="background:#52b788"></i>allow</span>` +
    `<span><i class="swatch" style="background:#e0a458"></i>review</span>` +
    `<span><i class="swatch" style="background:#ef6a62"></i>block</span>` +
    `<span class="thresh-note">T1 = ${fmtNum(last.t1, 2)} · T2 = ${fmtNum(last.t2, 2)}</span>`;
  box.innerHTML = riskHistogram(last.risk_hist, last.t1, last.t2, 600, 210);
}

function renderDecisionChart(rounds) {
  const box = $('#decisionChart');
  if (!rounds.length) { box.innerHTML = '<div class="empty">no data yet</div>'; return; }
  const rows = rounds.map((r) => ({ round: r.round, allow: r.defense.n_allow, review: r.defense.n_review, block: r.defense.n_block }));
  box.innerHTML = stackedBarChart(rows, 600, 220, {});
}

function renderDefenseTrade(rounds) {
  const box = $('#defenseChart');
  if (!rounds.length) { box.innerHTML = '<div class="empty">no data yet</div>'; return; }
  const w = 600, h = 200;
  const series = [
    ['escape rate', 'escape_rate', '#ef6a62', true],
    ['false-block', 'false_block_rate', '#e0a458', true],
    ['defense recall', 'defense_recall', '#52b788', false],
    ['defense precision', 'defense_precision', '#95d5b2', false],
    ['block rate', 'block_rate', '#b7e4c7', false],
  ];
  $('#defenseLegend').innerHTML = series.map(([n, , c]) =>
    `<span><i class="swatch" style="background:${c}"></i>${n}</span>`).join('');
  box.innerHTML = '<div class="stack">' + series.map(([n, key, c]) =>
    `<div class="chart-inline"><div class="mini-label">${n}</div>` +
    lineChart(rounds.map((r) => [r.round, r.defense[key]]), w, h, { color: c }) + '</div>'
  ).join('') + '</div>';
}

function renderThresholdChart(rounds, threshHist) {
  const box = $('#thresholdChart');
  if (!rounds.length) { box.innerHTML = '<div class="empty">no data yet</div>'; return; }
  const w = 600, h = 200;
  // prefer report threshold history; fall back to per-round thresholds used
  let data;
  if (threshHist && threshHist.length) {
    data = threshHist;
  } else {
    data = rounds.map((r) => ({ round: r.round, t1: r.defense.t1, t2: r.defense.t2 }));
  }
  $('#threshLegend').innerHTML =
    `<span><i class="swatch" style="background:#e0a458"></i>T1 (allow/review)</span>` +
    `<span><i class="swatch" style="background:#ef6a62"></i>T2 (review/block)</span>`;
  box.innerHTML = '<div class="stack">' +
    `<div class="chart-inline"><div class="mini-label">T1</div>` + lineChart(data.map((d) => [d.round, d.t1]), w, h, { color: '#e0a458', yMin: 0, yMax: 1 }) + '</div>' +
    `<div class="chart-inline"><div class="mini-label">T2</div>` + lineChart(data.map((d) => [d.round, d.t2]), w, h, { color: '#ef6a62', yMin: 0, yMax: 1 }) + '</div>' +
    '</div>';
}

function renderRiskComponents(rounds) {
  const box = $('#riskCompChart');
  if (!rounds.length) { box.innerHTML = '<div class="empty">no data yet</div>'; return; }
  const w = 600, h = 190;
  const series = [
    ['P_f', 'pf', '#ef6a62'],
    ['centrality C', 'centrality', '#e0a458'],
    ['ring S', 'ring', '#fbbf24'],
    ['velocity V', 'velocity', '#95d5b2'],
    ['trust A', 'trust', '#b7e4c7'],
  ];
  $('#riskCompLegend').innerHTML = series.map(([n, , c]) =>
    `<span><i class="swatch" style="background:${c}"></i>${n}</span>`).join('');
  box.innerHTML = '<div class="stack">' + series.map(([n, key, c]) =>
    `<div class="chart-inline"><div class="mini-label">${n}</div>` +
    lineChart(rounds.map((r) => [r.round, r.defense.components ? r.defense.components[key] : 0]), w, h, { color: c }) + '</div>'
  ).join('') + '</div>';
}

function renderDefenseTable(rounds) {
  const body = $('#defenseTable tbody');
  if (!rounds.length) { body.innerHTML = '<tr><td colspan="12" class="empty">no defense rounds yet</td></tr>'; return; }
  body.innerHTML = rounds.map((r) => {
    const d = r.defense;
    return `<tr>
      <td>${r.round}</td>
      <td>${d.n_allow}</td>
      <td>${d.n_review}</td>
      <td>${d.n_block}</td>
      <td>${d.fraud_blocked}</td>
      <td>${d.fraud_escaped}</td>
      <td>${d.legit_blocked}</td>
      <td>${pct(d.escape_rate)}</td>
      <td>${pct(d.false_block_rate)}</td>
      <td>${fmtNum(d.defense_precision, 2)}</td>
      <td>${fmtNum(d.defense_recall, 2)}</td>
      <td>${fmtNum(d.decision_latency_ms, 2)}ms</td>
    </tr>`;
  }).join('');
}

function renderDefenseTab() {
  const id = state.selectedDefense;
  const sim = id && state.sims[id];
  if (!sim) {
    ['adlState', 'adlWeights', 'riskDistChart', 'decisionChart',
     'defenseChart', 'thresholdChart', 'riskCompChart'].forEach((s) => { $(`#${s}`).innerHTML = '<div class="empty">select a run</div>'; });
    renderDefenseTable([]);
    return;
  }
  const rounds = defenseRounds(sim);
  const threshHist = sim.report ? sim.report.threshold_history : null;
  if (!rounds.length) {
    ['adlState', 'adlWeights', 'riskDistChart', 'decisionChart',
     'defenseChart', 'thresholdChart', 'riskCompChart'].forEach((s) => { $(`#${s}`).innerHTML = '<div class="empty">ADL disabled or no rounds yet</div>'; });
    renderDefenseTable([]);
    return;
  }
  renderAdlState(sim, rounds);
  renderAdlWeights(rounds);
  renderRiskDist(rounds);
  renderDecisionChart(rounds);
  renderDefenseTrade(rounds);
  renderThresholdChart(rounds, threshHist);
  renderRiskComponents(rounds);
  renderDefenseTable(rounds);
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
    const adl = (meta && meta.config && meta.config.adl_enabled);
    const pol = (meta && meta.config && meta.config.threshold_policy) || '';
    return `<tr>
      <td>${shortId(m.id)}</td>
      <td>${esc(m.describe || '—')}</td>
      <td>${esc(mode)}</td>
      <td>${esc(type)}</td>
      <td>${adl ? `<span class="badge badge-done">${esc(pol)}</span>` : '<span class="badge badge-stopped">off</span>'}</td>
      <td>${m.rounds ?? '—'}</td>
      <td>${stateBadge(m.state)}</td>
      <td>${new Date(m.started_at * 1000).toLocaleTimeString()}</td>
      <td><button class="btn mini" data-load="${m.id}">report</button>
          <button class="btn mini" data-live="${m.id}">live</button>
          <button class="btn mini" data-def="${m.id}">defense</button></td>
    </tr>`;
  }).join('') || '<tr><td colspan="9" class="empty">no runs yet</td></tr>';

  const runsBody = $('#runsTable tbody');
  runsBody.innerHTML = h.map((m) => {
    const s = state.sims[m.id];
    const meta = (state.sims[m.id] && state.sims[m.id].meta) || m;
    const mode = (meta && meta.config && meta.config.generator_mode) || meta.generator_mode || '—';
    const type = (meta && meta.config && meta.config.gen_type) || meta.gen_type || '—';
    const adl = (meta && meta.config && meta.config.adl_enabled);
    const done = s && s.report ? s.report.rounds.length : (m.state === 'done' ? m.rounds : '…');
    return `<tr>
      <td>${shortId(m.id)}</td>
      <td>${esc(m.describe || '—')}</td>
      <td>${esc(mode)}</td>
      <td>${esc(type)}</td>
      <td>${adl ? 'on' : 'off'}</td>
      <td>${m.rounds ?? '—'}</td>
      <td>${done}</td>
      <td>${stateBadge(m.state)}</td>
      <td><button class="btn mini" data-live="${m.id}">open</button></td>
    </tr>`;
  }).join('') || '<tr><td colspan="9" class="empty">no runs yet</td></tr>';

  tbody.querySelectorAll('[data-load]').forEach((b) => b.addEventListener('click', async () => {
    await loadReport(b.dataset.load, true);
    renderCompare();
    toast('Report loaded', 'info');
  }));
  tbody.querySelectorAll('[data-live]').forEach((b) => b.addEventListener('click', () => goLive(b.dataset.live)));
  runsBody.querySelectorAll('[data-live]').forEach((b) => b.addEventListener('click', () => goLive(b.dataset.live)));
  tbody.querySelectorAll('[data-def]').forEach((b) => b.addEventListener('click', async () => {
    await loadReport(b.dataset.def);
    selectDefenseRun(b.dataset.def);
    switchTab('defense');
  }));
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
    const d = rounds.filter((r) => r.defense).pop();
    rows.push({
      id, describe: sim.report.config.describe || id,
      mode: sim.report.config.generator_mode, type: sim.report.config.gen_type,
      adl: sim.report.config.adl_enabled,
      f1: last.f1, auc: last.auc,
      escape: d ? d.defense.escape_rate : null,
      falseBlock: d ? d.defense.false_block_rate : null,
      dRecall: d ? d.defense.defense_recall : null,
    });
  }
  if (!rows.length) { $('#compareChart').innerHTML = '<div class="empty">no finished runs yet</div>'; return; }
  const w = 620, h = 190;
  const chart = (key, color, label) => lineChart(rows.map((r, i) => [i + 1, r[key]]), w, h, { color });
  $('#compareChart').innerHTML =
    `<div class="legend">
      <span><i class="swatch" style="background:#52b788"></i>F1</span>
      <span><i class="swatch" style="background:#95d5b2"></i>AUC</span>
      <span><i class="swatch" style="background:#ef6a62"></i>escape</span>
      <span><i class="swatch" style="background:#e0a458"></i>false-block</span>
      <span><i class="swatch" style="background:#b7e4c7"></i>defense recall</span></div>
     <div class="chart-inline">${chart('f1', '#52b788')}</div>
     <div class="chart-inline">${chart('auc', '#95d5b2')}</div>
     <div class="chart-inline">${chart('escape', '#ef6a62')}</div>
     <div class="chart-inline">${chart('falseBlock', '#e0a458')}</div>
     <table class="table"><thead><tr><th>id</th><th>describe</th><th>mode</th><th>model</th><th>ADL</th><th>F1</th><th>AUC</th><th>escape</th><th>false-blk</th><th>dRecall</th></tr></thead><tbody>` +
     rows.map((r, i) => `<tr><td>${shortId(r.id)}</td><td>${esc(r.describe)}</td><td>${esc(r.mode)}</td><td>${esc(r.type)}</td>
       <td>${r.adl ? 'on' : 'off'}</td>
       <td>${fmtNum(r.f1)}</td><td>${fmtNum(r.auc)}</td>
       <td>${r.escape != null ? pct(r.escape) : '—'}</td>
       <td>${r.falseBlock != null ? pct(r.falseBlock) : '—'}</td>
       <td>${r.dRecall != null ? pct(r.dRecall) : '—'}</td></tr>`).join('') + '</tbody></table>';
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
      attrs: n.attrs, blocked: !!n.blocked, blocked_round: n.blocked_round,
      decision: n.decision || '', risk: n.risk || 0,
      r: n.blocked ? 6 : (n.label === 1 ? 7 : 3.6),
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
  const showB = $('#graphShowBlocked').checked;
  const colorRisk = $('#graphColorRisk').checked;
  const t = performance.now() / 1000;

  if (!GRAPH_NODES.length) {
    ctx.fillStyle = '#52b788';
    ctx.font = '13px Cascadia Code, monospace';
    ctx.textAlign = 'center';
    ctx.fillText('select a run and press “Load graph”', W / 2, H / 2);
    state.graphAnim = requestAnimationFrame(graphLoop);
    return;
  }

  const riskColor = (risk) => {
    const hue = Math.max(0, Math.min(1, risk)) * 130 - 10; // green (0) -> red (1)
    return `hsl(${hue}, 74%, 50%)`;
  };
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
    for (let j = i + 1; j < n; j++) {
      const b = GRAPH_NODES[j];
      let dx = a.x - b.x, dy = a.y - b.y;
      let d = Math.sqrt(dx * dx + dy * dy) || 0.001;
      const f = Math.min(2.2, repulsion / (d * d));
      a.vx += (dx / d) * f; a.vy += (dy / d) * f;
      b.vx -= (dx / d) * f; b.vy -= (dy / d) * f;
    }
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
    if (node.blocked && !showB) continue;
    if (node.label === 1 && !node.blocked && !showF) continue;
    if (node.label === 0 && !node.blocked && !showG) continue;
    const isFraud = node.label === 1;
    const pulse = isFraud ? 1 + Math.sin(t * 2.4 + node.i) * 0.22 : 1 + Math.sin(t * 1.4 + node.i) * 0.1;
    const r = node.r * pulse;
    let fill;
    if (node.blocked) {
      fill = '#5f6b66';
    } else if (colorRisk) {
      fill = riskColor(node.risk);
    } else if (isFraud) {
      fill = '#ef6a62';
    } else {
      fill = roundColor(node.round);
    }
    ctx.beginPath();
    ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
    if (node.blocked) {
      ctx.globalAlpha = 0.55;
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.globalAlpha = 1;
      // cross mark
      ctx.strokeStyle = 'rgba(0,0,0,.7)';
      ctx.lineWidth = 1.4;
      const s = r * 0.55;
      ctx.beginPath();
      ctx.moveTo(node.x - s, node.y - s); ctx.lineTo(node.x + s, node.y + s);
      ctx.moveTo(node.x + s, node.y - s); ctx.lineTo(node.x - s, node.y + s);
      ctx.stroke();
    } else if (isFraud) {
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
    const status = hit.blocked ? ` · BLOCKED r${hit.blocked_round}` : (hit.decision ? ` · ${hit.decision}` : '');
    tip.innerHTML = `<div class="gt-t">${hit.blocked ? 'BLOCKED' : (hit.label === 1 ? 'FRAUD' : 'GENUINE')} · acct ${hit.id} · r${roundTxt}${status}</div>` +
      `risk ${fmtNum(hit.risk, 3)}<br>` +
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
    buildAdlForm();
    status.className = 'status online';
    status.querySelector('.status-text').textContent = 'online · ' + (schema.defaults.rounds) + ' rounds default · ADL ' + (schema.defaults.adl_enabled ? 'on' : 'off');
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
    intelligent: '<b>intelligent</b> — the generator learns from the fraud that survived the whole defense and synthesises new strategies; it never copies a miss.',
    replay: '<b>replay (baseline)</b> — next round\'s fraud is an exact clone of the survivors, mirroring the original framework behaviour.',
  });
  $('#modeHint').innerHTML = '<b>intelligent</b> — the generator learns from the fraud that survived the whole defense and synthesises new strategies; it never copies a miss.';
  wireSeg('#typeSeg', null, null);
  wireSeg('#policySeg', '#policyHint', {
    adaptive: '<b>adaptive</b> — thresholds move every round: stricter when fraud escapes, more lenient when genuine users are blocked.',
    fixed: '<b>fixed</b> — thresholds stay at the configured values (baseline for comparing the adaptive policy).',
  });
  $('#policyHint').innerHTML = '<b>adaptive</b> — thresholds move every round: stricter when fraud escapes, more lenient when genuine users are blocked.';

  $('#launchBtn').addEventListener('click', launch);
  $('#liveRefreshBtn').addEventListener('click', () => renderLive());
  $('#defenseRefreshBtn').addEventListener('click', () => renderDefenseTab());
  $('#liveRunSelect').addEventListener('change', (e) => { if (e.target.value) selectLiveRun(e.target.value); });
  $('#defenseRunSelect').addEventListener('change', (e) => { if (e.target.value) selectDefenseRun(e.target.value); });
  $('#graphLoadBtn').addEventListener('click', loadGraph);
  $('#graphRunSelect').addEventListener('change', () => {});
  ['graphShowGenuine', 'graphShowFraud', 'graphShowBlocked', 'graphShowReferral', 'graphShowShared', 'graphColorRisk']
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
        renderDefenseTab();
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
