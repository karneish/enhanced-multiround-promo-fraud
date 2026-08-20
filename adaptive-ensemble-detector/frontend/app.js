/* =====================================================================
   Adaptive Ensemble Detector — dashboard logic
   ===================================================================== */
const API = '/api';
const MODEL_COLORS = {
  XGBoost: '#3b82f6',
  RandomForest: '#34d399',
  ExtraTrees: '#fbbf24',
  HistGradientBoosting: '#a78bfa',
  LogisticRegression: '#f472b6',
};
const MODEL_NAMES = Object.keys(MODEL_COLORS);

const state = {
  schema: null, datasets: [], sims: {}, selected: null, sse: null,
  graphData: null, graphAnim: null,
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}
function fmtNum(v, d = 3) {
  if (v === null || v === undefined || isNaN(v)) return '--';
  return Number(v).toFixed(d);
}
function shortId(id) { return String(id).slice(0, 8); }
function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function stateBadge(s) {
  const m = {running:'running',done:'done',error:'error',stopped:'stopped',pending:'pending'};
  return `<span class="badge badge-${m[s]||'pending'}">${s}</span>`;
}
function toast(msg, kind = 'info', ms = 3500) {
  const t = el('div', `toast ${kind}`, msg);
  $('#toasts').appendChild(t);
  setTimeout(() => { t.classList.add('out'); setTimeout(() => t.remove(), 450); }, ms);
}
function countUp(node, target, opts = {}) {
  const dec = opts.decimals != null ? opts.decimals : (Number.isInteger(target) ? 0 : 2);
  const dur = opts.dur || 700;
  if (!node) return;
  const start = performance.now();
  function tick(now) {
    const p = Math.min(1, (now - start) / dur);
    const e = 1 - Math.pow(1 - p, 3);
    node.textContent = dec ? (target * e).toFixed(dec) : Math.round(target * e).toLocaleString();
    if (p < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

async function apiGet(path) {
  const r = await fetch(API + path);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
async function apiPost(path, body) {
  const r = await fetch(API + path, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body || {}),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

/* ---- tabs ---- */
function moveIndicator() {
  const active = $('.tab.active'), ind = $('#tabIndicator');
  if (active && ind) { ind.style.width = active.offsetWidth + 'px'; ind.style.left = active.offsetLeft + 'px'; }
}
function switchTab(name) {
  $$('.tab').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
  $$('.panel').forEach(p => p.classList.toggle('active', p.id === 'tab-' + name));
  moveIndicator();
  if (name === 'graph') refreshGraphSelect();
}
$$('.tab').forEach(btn => btn.addEventListener('click', () => switchTab(btn.dataset.tab)));
window.addEventListener('resize', moveIndicator);

/* ---- SVG chart helpers ---- */
function lineChart(points, w, h, opts = {}) {
  if (!points || points.length < 1) return '<div class="empty">no data yet</div>';
  const pad = {l:36,r:12,t:14,b:24}, iw = w-pad.l-pad.r, ih = h-pad.t-pad.b;
  const xs = points.map(p=>p[0]), ys = points.map(p=>p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const yMin = Math.min(0,...ys), yMax = Math.max(0.05,...ys);
  const X = x => pad.l + (maxX===minX?0:(x-minX)/(maxX-minX))*iw;
  const Y = y => pad.t + (1-(y-yMin)/(yMax-yMin))*ih;
  const path = points.map((p,i)=>`${i?'L':'M'}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(' ');
  const grid = [0,0.25,0.5,0.75,1].map(v=>`<line x1="${pad.l}" y1="${Y(v).toFixed(1)}" x2="${w-pad.r}" y2="${Y(v).toFixed(1)}" class="grid-line"/>`).join('');
  const dots = points.map((p,i)=>`<circle cx="${X(p[0]).toFixed(1)}" cy="${Y(p[1]).toFixed(1)}" r="3" fill="${opts.color||'#3b82f6'}" style="animation-delay:${0.5+i*0.12}s" class="dot-anim"/>`).join('');
  const xlabels = points.map((p,i)=>`<text x="${X(p[0]).toFixed(1)}" y="${h-8}" class="axis" text-anchor="middle">r${p[0]}</text>`).join('');
  const last = points[points.length-1];
  const value = `<text x="${X(last[0]).toFixed(1)}" y="${(Y(last[1])-8).toFixed(1)}" class="val-label" text-anchor="middle">${fmtNum(last[1],opts.decimals||2)}</text>`;
  const gid = 'g'+Math.random().toString(36).slice(2,8);
  const area = opts.fill ? `<path d="${path} L${X(last[0]).toFixed(1)},${pad.t+ih} L${X(points[0][0]).toFixed(1)},${pad.t+ih} Z" fill="url(#${gid})" class="chart-area-gradient" opacity="0.25"/>` : '';
  return `<svg viewBox="0 0 ${w} ${h}" class="chart-svg"><defs><linearGradient id="${gid}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${opts.color||'#3b82f6'}" stop-opacity="0.9"/><stop offset="1" stop-color="${opts.color||'#3b82f6'}" stop-opacity="0"/></linearGradient></defs>${grid}${area}<path d="${path}" fill="none" stroke="${opts.color||'#3b82f6'}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" class="path-anim" style="filter:drop-shadow(0 0 6px ${opts.glow||opts.color||'#3b82f6'})"/>${dots}${xlabels}${value}</svg>`;
}

function stackedBarChart(roundData, modelNames, w, h) {
  if (!roundData || roundData.length === 0) return '<div class="empty">no data yet</div>';
  const pad = {l:36,r:12,t:14,b:24}, iw = w-pad.l-pad.r, ih = h-pad.t-pad.b;
  const nRounds = roundData.length;
  const barW = Math.max(6, Math.min(24, iw / nRounds - 4));
  const gap = 2;

  let bars = '';
  for (let r = 0; r < nRounds; r++) {
    const weights = roundData[r];
    let cumY = 0;
    for (const name of modelNames) {
      const val = weights[name] || 0;
      const bh = val * ih;
      const x = pad.l + r * (barW + gap + modelNames.length * 1);
      const y = pad.t + ih - cumY - bh;
      bars += `<rect x="${(pad.l + r * (iw/nRounds) + (iw/nRounds-barW*modelNames.length)/2 + modelNames.indexOf(name)*(barW)).toFixed(1)}" y="${y.toFixed(1)}" width="${barW}" height="${bh.toFixed(1)}" rx="2" fill="${MODEL_COLORS[name]||'#666'}" class="bar-anim" style="animation-delay:${r*0.06}s"><title>${name}: ${(val*100).toFixed(1)}%</title></rect>`;
      cumY += bh;
    }
    bars += `<text x="${(pad.l + r*(iw/nRounds) + iw/nRounds/2).toFixed(1)}" y="${h-8}" class="axis" text-anchor="middle">r${roundData[r]._round !== undefined ? roundData[r]._round : r}</text>`;
  }

  const legend = modelNames.map(n => `<rect x="${pad.l}" y="${pad.t + modelNames.indexOf(n)*14}" width="8" height="8" rx="2" fill="${MODEL_COLORS[n]}"/><text x="${pad.l+12}" y="${pad.t + modelNames.indexOf(n)*14 + 8}" class="axis" style="font-size:9px">${n.slice(0,6)}</text>`).join('');

  return `<svg viewBox="0 0 ${w} ${h}" class="chart-svg">${bars}</svg>`;
}

function barChart(items, w, h, opts = {}) {
  const sorted = Object.entries(items).sort((a,b)=>b[1]-a[1]).slice(0,12);
  if (!sorted.length) return '<div class="empty">no data yet</div>';
  const pad = {l:8,r:8,t:18,b:46}, iw = w-pad.l-pad.r, ih = h-pad.t-pad.b;
  const max = Math.max(...sorted.map(([,v])=>v), 1);
  const bw = Math.max(8, Math.min(30, (iw/sorted.length)-8));
  const bwTotal = bw + 8;
  const startX = pad.l + Math.max(0, (iw-bwTotal*sorted.length)/2);
  const bars = sorted.map(([k,v],i)=>{
    const bh = (v/max)*ih, x = startX + i*bwTotal, y = pad.t+ih-bh;
    const label = k.length>22?k.slice(0,20)+'...':k;
    const rotate = k.length>14?'transform="rotate(-24)"':'';
    return `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bw}" height="${bh.toFixed(1)}" rx="4" fill="${opts.color||'#3b82f6'}" class="bar-anim" style="animation-delay:${i*0.06}s"><title>${esc(k)}: ${v}</title></rect><text x="${(x+bw/2).toFixed(1)}" y="${(y-4).toFixed(1)}" class="val-label" text-anchor="middle">${v}</text><text x="${(x+bw/2).toFixed(1)}" y="${h-10}" class="axis" text-anchor="end" ${rotate}>${label}</text>`;
  }).join('');
  return `<svg viewBox="0 0 ${w} ${h}" class="chart-svg">${bars}</svg>`;
}

function radarChart(modelData, w, h) {
  if (!modelData || modelData.length === 0) return '<div class="empty">no data yet</div>';
  const cx = w/2, cy = h/2, R = Math.min(w,h)/2 - 30;
  const metrics = ['f1', 'recall', 'precision', 'auc'];
  const metricLabels = ['F1', 'Recall', 'Prec', 'AUC'];
  const nAxes = metrics.length;
  const angleStep = (2*Math.PI)/nAxes;

  let svg = `<svg viewBox="0 0 ${w} ${h}" class="chart-svg">`;
  for (let ring = 1; ring <= 4; ring++) {
    const r = (ring/4)*R;
    const pts = [];
    for (let i = 0; i < nAxes; i++) {
      const a = -Math.PI/2 + i*angleStep;
      pts.push(`${(cx+r*Math.cos(a)).toFixed(1)},${(cy+r*Math.sin(a)).toFixed(1)}`);
    }
    svg += `<polygon points="${pts.join(' ')}" fill="none" stroke="rgba(96,165,250,0.1)" stroke-width="0.5"/>`;
    svg += `<text x="${cx+3}" y="${cy-r-2}" class="axis" style="font-size:8px">${(ring*0.25).toFixed(2)}</text>`;
  }
  for (let i = 0; i < nAxes; i++) {
    const a = -Math.PI/2 + i*angleStep;
    svg += `<line x1="${cx}" y1="${cy}" x2="${(cx+R*Math.cos(a)).toFixed(1)}" y2="${(cy+R*Math.sin(a)).toFixed(1)}" stroke="rgba(96,165,250,0.15)" stroke-width="0.5"/>`;
    svg += `<text x="${(cx+(R+18)*Math.cos(a)).toFixed(1)}" y="${(cy+(R+18)*Math.sin(a)+3).toFixed(1)}" class="axis" text-anchor="middle" style="font-size:10px">${metricLabels[i]}</text>`;
  }
  for (const model of modelData) {
    const color = MODEL_COLORS[model.name] || '#666';
    const pts = [];
    for (let i = 0; i < nAxes; i++) {
      const a = -Math.PI/2 + i*angleStep;
      const v = Math.min(1, Math.max(0, model.values[i]));
      pts.push(`${(cx+R*v*Math.cos(a)).toFixed(1)},${(cy+R*v*Math.sin(a)).toFixed(1)}`);
    }
    svg += `<polygon points="${pts.join(' ')}" fill="${color}" fill-opacity="0.12" stroke="${color}" stroke-width="1.5"/>`;
    for (let i = 0; i < nAxes; i++) {
      const a = -Math.PI/2 + i*angleStep;
      const v = Math.min(1, Math.max(0, model.values[i]));
      svg += `<circle cx="${(cx+R*v*Math.cos(a)).toFixed(1)}" cy="${(cy+R*v*Math.sin(a)).toFixed(1)}" r="3" fill="${color}" class="dot-anim" style="animation-delay:${0.3+i*0.1}s"/>`;
    }
  }
  svg += '</svg>';
  return svg;
}

/* ---- live tab ---- */
function selectLiveRun(id) {
  state.selected = id;
  const sel = $('#liveRunSelect');
  if (sel) sel.value = id;
  const m = state.sims[id] && state.sims[id].meta;
  $('#liveRunLabel').textContent = m ? (m.describe || id) : '';
  openStream(id);
}

function openStream(id) {
  if (state.sse) state.sse.close();
  const sse = new EventSource(`${API}/stream/${id}`);
  state.sse = sse;
  sse.onmessage = e => {
    const ev = JSON.parse(e.data);
    state.sims[id] = state.sims[id] || {meta:{}, events:[], rounds:[], report:null};
    state.sims[id].events.push(ev);
    if (ev.type === 'round_result') {
      state.sims[id].rounds.push(ev);
      renderLive();
    } else if (ev.type === 'state' && ev.finished) {
      sse.close(); state.sse = null;
      loadReport(id).then(()=>{ renderLive(); loadHistory(); populateRunSelects(); });
      toast('Run finished: '+shortId(id), 'ok');
    }
    if (ev.type === 'log') renderLog(id);
  };
  sse.onerror = () => {};
}

function renderRoundTracker(sim) {
  const box = $('#roundTracker'); box.innerHTML = '';
  const total = (sim.report && sim.report.config && sim.report.config.rounds) ||
                (sim.meta && sim.meta.config && sim.meta.config.rounds) ||
                (sim.rounds.length + 1);
  box.appendChild(el('span', 'rt-label', 'rounds'));
  for (let r = 0; r < total; r++) {
    const rec = sim.rounds.find(x => x.round === r);
    const chip = el('div', 'round-chip');
    chip.appendChild(el('span', 'ball'));
    chip.appendChild(document.createTextNode('r'+r));
    if (rec) chip.classList.add('done');
    else if (r === sim.rounds.length) chip.classList.add('active');
    box.appendChild(chip);
  }
}

function renderMetrics(sim) {
  const r = sim.rounds[sim.rounds.length - 1];
  if (!r) return;
  const e = r.ensemble || {};
  const cards = [
    ['Round', `#${r.round}`, 'r', null],
    ['Nodes', r.num_nodes, 'n', null],
    ['Missed fraud', r.missed_fraud, r.missed_fraud > 0 ? 'bad' : 'good', null],
    ['Ensemble F1', e.f1, 'm', 3],
    ['Ensemble AUC', e.auc, 'a', 3],
    ['Recall', e.rec, 'm', 3],
    ['Precision', e.prec, 'p', 3],
    ['F1/AUC delta', e.f1-e.auc, 'a', 3],
  ];
  const wrap = $('#liveMetrics');
  wrap.innerHTML = '';
  cards.forEach(([k,v,c,dec], i) => {
    const card = el('div', `metric ${c}`);
    card.style.animationDelay = (i*0.05)+'s';
    card.appendChild(el('div', 'k', k));
    const vNode = el('div', 'v', '0');
    card.appendChild(vNode);
    wrap.appendChild(card);
    if (typeof v === 'number' && !isNaN(v)) countUp(vNode, v, {decimals:dec,dur:650});
    else vNode.textContent = v;
  });
}

function renderPerfChart(sim) {
  const rounds = sim.rounds;
  const w = 600, h = 180;
  const series = [
    ['F1', 'f1', '#3b82f6', true],
    ['AUC', 'auc', '#60a5fa', false],
    ['Recall', 'rec', '#34d399', false],
    ['Precision', 'prec', '#fbbf24', false],
  ];
  $('#perfLegend').innerHTML = series.map(([n,,c])=>`<span><i class="swatch" style="background:${c}"></i>${n}</span>`).join('');
  $('#livePerfChart').innerHTML = '<div class="stack">' + series.map(([name,key,color])=>
    `<div class="chart-inline"><div class="mini-label">${name}</div>${lineChart(rounds.map(r=>[r.round, r.ensemble?r.ensemble[key]:0]), w, h, {color, fill:true})}</div>`
  ).join('') + '</div>';
}

function renderWeightChart(sim) {
  const rounds = sim.rounds;
  if (!rounds.length) return;
  const w = 600, h = 180;
  const pad = {l:36,r:12,t:14,b:24}, iw = w-pad.l-pad.r, ih = h-pad.t-pad.b;
  const nR = rounds.length;
  let svg = `<svg viewBox="0 0 ${w} ${h}" class="chart-svg">`;

  const grid = [0,0.25,0.5,0.75,1].map(v=>{
    const y = pad.t + (1-v)*ih;
    return `<line x1="${pad.l}" y1="${y.toFixed(1)}" x2="${w-pad.r}" y2="${y.toFixed(1)}" class="grid-line"/>`;
  }).join('');
  svg += grid;

  for (const name of MODEL_NAMES) {
    const points = rounds.map((r,i)=>[i, r.weights?r.weights[name]||0:0]);
    const pts = points.map(p=>`${(pad.l+p[0]*(iw/Math.max(1,nR-1))).toFixed(1)},${(pad.t+(1-p[1])*ih).toFixed(1)}`);
    if (pts.length > 1) {
      svg += `<polyline points="${pts.join(' ')}" fill="none" stroke="${MODEL_COLORS[name]}" stroke-width="2" stroke-linecap="round" class="path-anim" style="filter:drop-shadow(0 0 4px ${MODEL_COLORS[name]})"/>`;
    }
    for (const p of points) {
      svg += `<circle cx="${(pad.l+p[0]*(iw/Math.max(1,nR-1))).toFixed(1)}" cy="${(pad.t+(1-p[1])*ih).toFixed(1)}" r="2.5" fill="${MODEL_COLORS[name]}" class="dot-anim" style="animation-delay:${0.5+p[0]*0.1}s"/>`;
    }
    const last = points[points.length-1];
    svg += `<text x="${(w-pad.r+4).toFixed(1)}" y="${(pad.t+(1-last[1])*ih+3).toFixed(1)}" class="axis" style="font-size:8px;fill:${MODEL_COLORS[name]}">${(last[1]*100).toFixed(0)}%</text>`;
  }
  for (let r = 0; r < nR; r++) {
    const x = pad.l + r*(iw/Math.max(1,nR-1));
    svg += `<text x="${x.toFixed(1)}" y="${h-8}" class="axis" text-anchor="middle">r${rounds[r].round}</text>`;
  }
  svg += '</svg>';
  $('#liveWeightChart').innerHTML = svg;
}

function renderIndivChart(sim) {
  const rounds = sim.rounds;
  if (!rounds.length) return;
  const w = 600, h = 180;
  const pad = {l:36,r:12,t:14,b:24}, iw = w-pad.l-pad.r, ih = h-pad.t-pad.b;
  const nR = rounds.length;
  let svg = `<svg viewBox="0 0 ${w} ${h}" class="chart-svg">`;
  svg += [0,0.25,0.5,0.75,1].map(v=>`<line x1="${pad.l}" y1="${(pad.t+(1-v)*ih).toFixed(1)}" x2="${w-pad.r}" y2="${(pad.t+(1-v)*ih).toFixed(1)}" class="grid-line"/>`).join('');

  for (const name of MODEL_NAMES) {
    const points = rounds.map((r,i)=>{
      const f1s = r.individual_f1 && r.individual_f1[name];
      return [i, f1s && f1s.length ? f1s[f1s.length-1] : 0];
    });
    const pts = points.map(p=>`${(pad.l+p[0]*(iw/Math.max(1,nR-1))).toFixed(1)},${(pad.t+(1-p[1])*ih).toFixed(1)}`);
    if (pts.length > 1) {
      svg += `<polyline points="${pts.join(' ')}" fill="none" stroke="${MODEL_COLORS[name]}" stroke-width="2" stroke-linecap="round" stroke-dasharray="6,3" class="path-anim" style="filter:drop-shadow(0 0 3px ${MODEL_COLORS[name]})"/>`;
    }
    for (const p of points) {
      svg += `<circle cx="${(pad.l+p[0]*(iw/Math.max(1,nR-1))).toFixed(1)}" cy="${(pad.t+(1-p[1])*ih).toFixed(1)}" r="3" fill="${MODEL_COLORS[name]}" class="dot-anim"/>`;
    }
  }
  for (let r = 0; r < nR; r++) {
    const x = pad.l + r*(iw/Math.max(1,nR-1));
    svg += `<text x="${x.toFixed(1)}" y="${h-8}" class="axis" text-anchor="middle">r${rounds[r].round}</text>`;
  }
  svg += '</svg>';
  $('#liveIndivChart').innerHTML = svg;
  $('#indivLegend').innerHTML = MODEL_NAMES.map(n=>`<span><i class="swatch" style="background:${MODEL_COLORS[n]}"></i>${n.slice(0,6)}</span>`).join('');
}

function renderAdsScores(sim) {
  const r = sim.rounds[sim.rounds.length - 1];
  if (!r) return;
  const scores = r.scores || {};
  const weights = r.weights || {};
  const modelData = MODEL_NAMES.filter(n => scores[n] !== undefined).map(n => ({
    name: n,
    values: [r.per_model&&r.per_model[n]?r.per_model[n].f1:0, r.per_model&&r.per_model[n]?r.per_model[n].recall:0, r.per_model&&r.per_model[n]?r.per_model[n].precision:0, r.per_model&&r.per_model[n]?r.per_model[n].auc:0],
  }));
  const w = 350, h = 280;
  $('#liveAdsScores').innerHTML = radarChart(modelData, w, h) + '<div style="padding:8px 0">' + MODEL_NAMES.filter(n=>scores[n]!==undefined).map(n =>
    `<div style="display:flex;align-items:center;gap:8px;padding:3px 0;font-size:11px"><span style="width:8px;height:8px;border-radius:50%;background:${MODEL_COLORS[n]};display:inline-block"></span><span style="color:var(--text-hi);min-width:90px">${n}</span><span style="color:var(--muted)">score:</span> <span style="color:var(--amber);font-family:var(--mono)">${fmtNum(scores[n])}</span><span style="color:var(--muted)">weight:</span> <span style="color:var(--cyan);font-family:var(--mono)">${fmtNum(weights[n],2)}</span></div>`
  ).join('') + '</div>';
}

function renderStrategies(sim) {
  const r = sim.rounds[sim.rounds.length - 1];
  if (!r) return;
  const counts = r.fraud_strategies || {};
  const w = 560, h = 240;
  $('#liveStrategies').innerHTML = barChart(counts, w, h, {color:'#3b82f6'});
}

function renderLog(id) {
  const sim = state.sims[id]; if (!sim) return;
  const box = $('#liveLog');
  const logs = sim.events.filter(e => e.type === 'log').slice(-220);
  if (!logs.length) { box.innerHTML = '<div class="empty">waiting for logs...</div>'; return; }
  box.innerHTML = logs.map(e => {
    const cls = /\[error\]/.test(e.text) ? 'err' : /\[warn\]/.test(e.text) ? 'warn' : /\[done\]/.test(e.text) ? 'ok' : '';
    return `<div class="log-line ${cls}"><span class="t">${fmtNum(e.t,1)}s</span><span class="tx">${esc(e.text)}</span></div>`;
  }).join('');
  if ($('#liveAutoScroll').checked) box.scrollTop = box.scrollHeight;
  $('#liveLogCount').textContent = `... ${logs.length} lines`;
}

function renderLive() {
  const sim = state.sims[state.selected];
  if (!sim || !sim.rounds.length) return;
  renderRoundTracker(sim);
  renderMetrics(sim);
  renderPerfChart(sim);
  renderWeightChart(sim);
  renderIndivChart(sim);
  renderAdsScores(sim);
  renderStrategies(sim);
  renderLog(state.selected);
  updateArchWeights(sim.rounds[sim.rounds.length-1]);
  updateModelTable(sim.rounds[sim.rounds.length-1]);
}

function updateArchWeights(r) {
  if (!r) return;
  const w = r.weights || {};
  MODEL_NAMES.forEach(n => {
    const node = $(`#arch-w-${n}`);
    if (node) node.textContent = w[n] !== undefined ? `w=${fmtNum(w[n],3)}` : '';
  });
}

function updateModelTable(r) {
  if (!r) return;
  const tbody = $('#modelTable tbody');
  const perModel = r.per_model || {};
  const weights = r.weights || {};
  const scores = r.scores || {};
  const f1Hist = r.individual_f1 || {};
  tbody.innerHTML = MODEL_NAMES.map(name => {
    const pm = perModel[name] || {};
    const w = weights[name] || 0;
    const s = scores[name] || 0;
    const hist = f1Hist[name] || [];
    const trend = hist.length >= 2 ? (hist[hist.length-1] > hist[hist.length-2] ? 'up' : hist[hist.length-1] < hist[hist.length-2] ? 'down' : 'flat') : '--';
    const trendColor = trend === 'up' ? 'var(--emerald)' : trend === 'down' ? 'var(--red)' : 'var(--muted)';
    return `<tr>
      <td><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${MODEL_COLORS[name]};margin-right:6px"></span>${name}</td>
      <td>${fmtNum(pm.f1)}</td><td>${fmtNum(pm.recall)}</td><td>${fmtNum(pm.precision)}</td><td>${fmtNum(pm.auc)}</td>
      <td style="color:var(--amber);font-family:var(--mono)">${fmtNum(s)}</td>
      <td style="color:var(--cyan);font-family:var(--mono)">${fmtNum(w,3)}</td>
      <td style="color:${trendColor};font-weight:700">${trend === 'up' ? '&#9650;' : trend === 'down' ? '&#9660;' : '--'}</td>
    </tr>`;
  }).join('');
}

/* ---- reports ---- */
async function loadReport(id, force) {
  if (state.sims[id] && state.sims[id].report && !force) return state.sims[id].report;
  const rep = await apiGet(`/report/${id}`);
  state.sims[id].report = rep;
  state.sims[id].rounds = [];
  for (const ev of rep.rounds || []) state.sims[id].rounds.push(ev);
  return rep;
}

/* ---- history ---- */
async function loadHistory() {
  const h = await apiGet('/history');
  const tbody = $('#historyTable tbody');
  tbody.innerHTML = h.map(m => {
    const s = state.sims[m.id];
    return `<tr>
      <td>${shortId(m.id)}</td><td>${esc(m.describe||'--')}</td><td>${m.rounds??'--'}</td>
      <td>${stateBadge(m.state)}</td><td>${new Date(m.started_at*1000).toLocaleTimeString()}</td>
      <td><button class="btn mini" data-live="${m.id}">open</button></td>
    </tr>`;
  }).join('') || '<tr><td colspan="6" class="empty">no runs yet</td></tr>';

  const runsBody = $('#runsTable tbody');
  runsBody.innerHTML = h.map(m => {
    const s = state.sims[m.id];
    const done = s && s.report ? s.report.rounds.length : (m.state==='done'?m.rounds:'...');
    return `<tr><td>${shortId(m.id)}</td><td>${esc(m.describe||'--')}</td><td>${m.rounds??'--'}</td><td>${done}</td><td>${stateBadge(m.state)}</td>
      <td><button class="btn mini" data-live="${m.id}">open</button></td></tr>`;
  }).join('') || '<tr><td colspan="6" class="empty">no runs yet</td></tr>';

  tbody.querySelectorAll('[data-live]').forEach(b => b.addEventListener('click', ()=>goLive(b.dataset.live)));
  runsBody.querySelectorAll('[data-live]').forEach(b => b.addEventListener('click', ()=>goLive(b.dataset.live)));
  renderCompare();
}

function goLive(id) { selectLiveRun(id); switchTab('live'); renderLive(); }

function renderCompare() {
  const rows = [];
  for (const id of Object.keys(state.sims)) {
    const sim = state.sims[id];
    if (!sim.report) continue;
    const rounds = sim.rounds;
    if (!rounds.length) continue;
    const last = rounds[rounds.length - 1].ensemble;
    rows.push({id, describe: sim.report.config.describe||id, f1: last.f1, auc: last.auc});
  }
  if (!rows.length) { $('#compareChart').innerHTML = '<div class="empty">no finished runs yet</div>'; return; }
  const w = 620, h = 190;
  const f1 = rows.map((r,i)=>[i+1,r.f1]);
  const auc = rows.map((r,i)=>[i+1,r.auc]);
  $('#compareChart').innerHTML =
    `<div class="legend"><span><i class="swatch" style="background:#3b82f6"></i>F1</span><span><i class="swatch" style="background:#60a5fa"></i>AUC</span></div>
     <div class="chart-inline">${lineChart(f1, w, h, {color:'#3b82f6'})}</div>
     <div class="chart-inline">${lineChart(auc, w, h, {color:'#60a5fa'})}</div>`;
}

/* ---- run selects ---- */
async function populateRunSelects() {
  const ids = Object.keys(state.sims);
  for (const sel of ['#liveRunSelect', '#graphRunSelect']) {
    const node = $(sel); if (!node) continue;
    const cur = node.value;
    const keep = node.querySelector('option[value=""]');
    node.innerHTML = '';
    if (keep) node.appendChild(keep);
    for (const id of ids) {
      const m = state.sims[id].meta || {};
      const o = el('option');
      o.value = id;
      o.textContent = `${shortId(id)} ... ${(m.describe||'--').slice(0,30)}`;
      node.appendChild(o);
    }
    if (cur && ids.includes(cur)) node.value = cur;
  }
}

/* ---- launch ---- */
const CONFIG_FIELDS = [
  ['rounds','Rounds','number','simulation rounds'],
  ['base_accounts','Base accounts','number','genuine in round 0'],
  ['initial_fraud','Initial fraud','number','fraud in round 0'],
  ['genuine_per_round','Genuine / round','number','new genuine each round'],
  ['fraud_per_round','Fraud / round','number','new fraud each round'],
  ['supervised_ratio','Supervised ratio','number','fraction labelled'],
  ['budget_pos','Review budget +','number','fraud revealed/round'],
  ['budget_neg','Review budget -','number','genuine revealed/round'],
  ['seed','Seed','number','random seed'],
];

function buildForm() {
  const d = state.schema.defaults;
  const wrap = $('#configForm');
  wrap.innerHTML = '';
  for (const [key,label,type,hint] of CONFIG_FIELDS) {
    const cell = el('div','field');
    const lab = el('label','',label);
    lab.title = hint||'';
    const input = el('input');
    input.type = type; input.className = 'text-input'; input.step = 'any';
    input.value = d[key]; input.dataset.key = key;
    cell.appendChild(lab); cell.appendChild(input);
    wrap.appendChild(cell);
  }
}

function readConfig() {
  const cfg = {describe: $('#cfgDescribe').value||'Untitled experiment'};
  $$('#configForm input').forEach(i => {
    const key = i.dataset.key;
    const def = state.schema.defaults[key];
    cfg[key] = typeof def === 'number' ? Number(i.value) : i.value;
  });
  return cfg;
}

function applyPreset(cfg) {
  $$('#configForm input').forEach(i => { if (cfg[i.dataset.key] !== undefined) i.value = cfg[i.dataset.key]; });
  if (cfg.describe) $('#cfgDescribe').value = cfg.describe;
}

function buildEnsembleOverview() {
  const wrap = $('#ensembleOverview');
  const models = [
    ['XGBoost', 'gradient boosted trees, warm-started across rounds', 'default'],
    ['Random Forest', '300 trees, balanced class weights, bagging', 'diverse'],
    ['Extra Trees', '300 randomized trees, extra splits', 'diverse'],
    ['HistGradientBoosting', '300 iterations, histogram-based boosting', 'boosting'],
    ['LogisticRegression', 'lbfgs solver, balanced weights, linear baseline', 'baseline'],
  ];
  wrap.innerHTML = models.map(([name,desc,type])=>
    `<div class="eo-model"><span class="eo-name" style="color:${MODEL_COLORS[name]}">${name}</span><span class="eo-type">${desc}</span><span class="eo-badge">${type}</span></div>`
  ).join('');
}

function buildAboutModels() {
  const wrap = $('#aboutModelGrid');
  if (!wrap) return;
  const models = [
    ['XGBoost', 'Gradient boosted decision trees. Warm-started across rounds for continual learning. Uses scale_pos_weight for class balancing.'],
    ['Random Forest', '300-tree ensemble with balanced class weights. Provides diverse tree-based predictions through bagging and random feature subsets.'],
    ['Extra Trees', 'Extremely randomized trees. Similar to RF but with randomized split thresholds, adding further diversity to the ensemble.'],
    ['HistGradientBoosting', 'Histogram-based gradient boosting (scikit-learn). Fast alternative to XGBoost with different splitting strategy.'],
    ['LogisticRegression', 'Linear classifier with lbfgs solver. Provides a fundamentally different (non-tree) baseline that is robust and interpretable.'],
  ];
  wrap.innerHTML = models.map(([name,desc])=>
    `<div class="model-card"><h4 style="color:${MODEL_COLORS[name]}">${name}</h4><p>${desc}</p></div>`
  ).join('');
}

async function launch() {
  $('#launchError').textContent = '';
  const btn = $('#launchBtn');
  btn.disabled = true; btn.classList.add('loading');
  try {
    const cfg = readConfig();
    const res = await apiPost('/run', cfg);
    state.sims[res.id] = {meta:res, events:[], rounds:[], report:null};
    toast('Launched: '+shortId(res.id), 'ok');
    await loadHistory(); await populateRunSelects();
    selectLiveRun(res.id); switchTab('live');
  } catch(err) {
    $('#launchError').textContent = 'Failed: '+err.message;
    toast('Launch failed: '+err.message, 'err');
  } finally { btn.disabled = false; btn.classList.remove('loading'); }
}

/* ---- graph ---- */
function refreshGraphSelect() { populateRunSelects(); }

async function loadGraph() {
  toast('Graph explorer requires per-run node/edge data (not available for this backend)', 'info');
}

/* ---- init ---- */
async function init() {
  const status = $('#serverStatus');
  status.className = 'status checking';
  status.querySelector('.status-text').textContent = 'checking...';
  try {
    const [schema, datasets, history] = await Promise.all([apiGet('/schema'), apiGet('/datasets'), apiGet('/history')]);
    state.schema = schema; state.datasets = datasets;
    buildForm(); buildEnsembleOverview(); buildAboutModels();
    status.className = 'status online';
    status.querySelector('.status-text').textContent = 'online';
    history.forEach(m => { if (!state.sims[m.id]) state.sims[m.id] = {meta:m,events:[],rounds:[],report:null}; });
    await populateRunSelects();
    const presetsWrap = $('#presets');
    datasets.forEach(ds => {
      const b = el('button','preset-btn',ds.label);
      b.addEventListener('click', ()=>{ applyPreset(ds.config); toast('Preset: '+ds.label, 'info'); });
      presetsWrap.appendChild(b);
    });
    loadHistory();
    moveIndicator();
  } catch(err) {
    status.className = 'status offline';
    status.querySelector('.status-text').textContent = 'offline ('+err.message+')';
    toast('Backend unreachable: '+err.message, 'err');
  }

  $('#launchBtn').addEventListener('click', launch);
  $('#liveRefreshBtn').addEventListener('click', ()=>renderLive());
  $('#liveRunSelect').addEventListener('change', e=>{ if(e.target.value) selectLiveRun(e.target.value); });

  setInterval(async()=>{
    try {
      const h = await apiGet('/history');
      const key = JSON.stringify(h);
      if (key !== state._lastHistory) {
        state._lastHistory = key;
        h.forEach(m=>{ if(!state.sims[m.id]) state.sims[m.id]={meta:m,events:[],rounds:[],report:null}; });
        await populateRunSelects(); loadHistory();
      }
    } catch {}
  }, 4000);
}

document.addEventListener('DOMContentLoaded', init);
