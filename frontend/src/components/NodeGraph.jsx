import { useMemo, useRef, useState, useEffect } from 'react';

const W = 1200;
const H = 640;

const DEC_COLOR = {
  allow: '#2ea043',
  review: '#d29922',
  block: '#ff5c57',
};
const UNKNOWN = '#3c4659';

function buildLayout(nodes, edges, positions, iterations) {
  const pos = positions;
  for (const n of nodes) {
    if (!pos.has(n.id)) {
      pos.set(n.id, {
        x: W * (0.15 + Math.random() * 0.7),
        y: H * (0.15 + Math.random() * 0.7),
        vx: 0, vy: 0,
      });
    }
  }
  const links = edges
    .map((e) => ({ a: pos.get(e.a), b: pos.get(e.b), w: e.weight }))
    .filter((l) => l.a && l.b);

  const cell = 150;
  for (let it = 0; it < iterations; it++) {
    const grid = new Map();
    for (const n of nodes) {
      const p = pos.get(n.id);
      const key = `${Math.floor(p.x / cell)},${Math.floor(p.y / cell)}`;
      const arr = grid.get(key) || [];
      arr.push(p);
      grid.set(key, arr);
    }

    for (const n of nodes) {
      const p = pos.get(n.id);
      const cx = Math.floor(p.x / cell);
      const cy = Math.floor(p.y / cell);
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          const others = grid.get(`${cx + dx},${cy + dy}`) || [];
          for (const o of others) {
            if (o === p) continue;
            const rx = p.x - o.x;
            const ry = p.y - o.y;
            const d2 = rx * rx + ry * ry;
            if (d2 > cell * cell * 3 || d2 < 1e-4) continue;
            const f = 5200 / d2;
            const inv = 1 / Math.sqrt(d2);
            p.vx += rx * inv * f;
            p.vy += ry * inv * f;
          }
        }
      }
    }

    for (const l of links) {
      const dx = l.b.x - l.a.x;
      const dy = l.b.y - l.a.y;
      const d = Math.sqrt(dx * dx + dy * dy) + 0.001;
      const target = 65 + (1 - Math.min(1, Math.log10(1 + l.w) / 3)) * 85;
      const force = (d - target) * 0.02 * (0.4 + 0.6 * Math.min(1, l.w / 8));
      const fx = (dx / d) * force;
      const fy = (dy / d) * force;
      l.a.vx += fx; l.a.vy += fy;
      l.b.vx -= fx; l.b.vy -= fy;
    }

    for (const n of nodes) {
      const p = pos.get(n.id);
      p.vx += (W / 2 - p.x) * 0.003;
      p.vy += (H / 2 - p.y) * 0.005;
      const sp = Math.hypot(p.vx, p.vy);
      if (sp > 40) { p.vx = (p.vx / sp) * 40; p.vy = (p.vy / sp) * 40; }
      p.x += p.vx * 0.85;
      p.y += p.vy * 0.85;
      p.vx *= 0.55;
      p.vy *= 0.55;
      p.x = Math.max(6, Math.min(W - 6, p.x));
      p.y = Math.max(6, Math.min(H - 6, p.y));
    }
  }
  return pos;
}

function edgePath(a, b, w) {
  const mx = (a.x + b.x) / 2;
  const my = (a.y + b.y) / 2;
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len = Math.hypot(dx, dy) || 1;
  const ox = -dy / len;
  const oy = dx / len;
  const off = Math.min(46, 10 + Math.log10(1 + w) * 9);
  return `M${a.x.toFixed(1)} ${a.y.toFixed(1)} Q${(mx + ox * off).toFixed(1)} ${(my + oy * off).toFixed(1)} ${b.x.toFixed(1)} ${b.y.toFixed(1)}`;
}

export default function NodeGraph({ graph, running }) {
  const positionsRef = useRef(new Map());
  const [layout, setLayout] = useState(null);
  const [selected, setSelected] = useState(null);
  const [tip, setTip] = useState(null);

  const { nodeMap, edges } = useMemo(() => {
    const nmap = new Map((graph?.nodes || []).map((n) => [n.id, n]));
    return { nodeMap: nmap, edges: graph?.edges || [] };
  }, [graph]);

  useEffect(() => {
    if (!graph || !graph.nodes.length) { setLayout(null); return; }
    const had = positionsRef.current.size > 0;
    const pos = buildLayout(graph.nodes, graph.edges, positionsRef.current, had ? 26 : 130);
    setLayout({ pos });
  }, [graph]);

  const nodes = [...nodeMap.values()];

  const byDegree = useMemo(
    () => [...nodes].sort((a, b) => (b.degree || 0) - (a.degree || 0)).slice(0, 8),
    [nodes],
  );

  const fmt = (x) => (x == null ? '—' : Number(x).toFixed(4));
  const fmtTS = (x) => (x == null || isNaN(x) ? '—' : new Date(x * 1000).toLocaleTimeString());
  const short = (id) => {
    const s = String(id ?? '');
    return s.length > 26 ? `${s.slice(0, 12)}…${s.slice(-12)}` : s;
  };

  let maxActivity = 1;
  for (const n of nodes) maxActivity = Math.max(maxActivity, n.n_events || 0);

  return (
    <section className="panel">
      <div className="panel-head">
        <h2 className="panel-title">Live node graph</h2>
        <div className="graph-stats">
          <span className="pill tiny"><b>{graph?.nodes?.length ?? 0}</b> nodes</span>
          <span className="pill tiny"><b>{graph?.edges?.length ?? 0}</b> edges</span>
          <span className="pill tiny">rev <b>{graph?.revision ?? 0}</b></span>
          {graph?.capped && <span className="pill tiny amber">capped (top {graph.nodes.length})</span>}
          <span className={`pill tiny ${running ? 'green' : 'gray'}`}>
            <span className="dot" /> {running ? 'live' : 'idle'}
          </span>
        </div>
      </div>

      {nodes.length === 0 ? (
        <div className="empty-state">
          Nothing seen yet — {running ? 'waiting for incoming events…' : 'start a session to begin mapping the network.'}
        </div>
      ) : (
        <>
          <div className="graph-legend">
            <span className="legend-item"><span className="lwdotted" style={{ color: UNKNOWN }} /> account</span>
            <span className="legend-item"><span className="lwdot" style={{ color: UNKNOWN }} /> counterparty</span>
            <span className="legend-item"><span className="lwdot" style={{ color: DEC_COLOR.block }} /> block</span>
            <span className="legend-item"><span className="lwdot" style={{ color: DEC_COLOR.review }} /> review</span>
            <span className="legend-item"><span className="lwdot" style={{ color: DEC_COLOR.allow }} /> allow</span>
            <span className="legend-item"><span className="lwdot" style={{ color: UNKNOWN }} /> unseen</span>
            <span className="graph-tip" style={{ position: 'static', transform: 'none', display: 'inline-flex' }}>hover/click + zoom hint</span>
          </div>

          {layout && (
            <div className="graph-wrap">
              <svg className="graph-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet"
                onMouseLeave={() => { setSelected(null); setTip(null); }}>
                <defs>
                  <pattern id="gdot" width="26" height="26" patternUnits="userSpaceOnUse">
                    <circle cx="1.2" cy="1.2" r="1.2" fill="rgba(122,146,214,0.10)" />
                  </pattern>
                  <filter id="glow" x="-60%" y="-60%" width="220%" height="220%">
                    <feGaussianBlur stdDeviation="3.2" result="b" />
                    <feMerge>
                      <feMergeNode in="b" />
                      <feMergeNode in="SourceGraphic" />
                    </feMerge>
                  </filter>
                </defs>
                <rect x="0" y="0" width={W} height={H} fill="url(#gdot)" rx="10" />

                <g>
                  {edges.map((e) => {
                    const a = layout.pos.get(e.a);
                    const b = layout.pos.get(e.b);
                    if (!a || !b) return null;
                    const d = edgePath(a, b, e.weight);
                    const hot = e.weight >= 3;
                    return (
                      <g key={`${e.a}|${e.b}`}>
                        <path d={d} className="g-edge" strokeWidth={Math.min(3.4, 0.8 + Math.log10(1 + e.weight) * 1.5)}
                          opacity={Math.min(0.6, 0.09 + 0.09 * Math.log10(1 + e.weight))} />
                        {hot && <path d={d} className="g-edge flow" strokeWidth={1.5} />}
                        <path d={d} className="g-edge-hit"
                          onMouseEnter={() => setSelected(e.a === selected ? null : e.a)}
                          onClick={() => setSelected(e.a)} />
                      </g>
                    );
                  })}
                </g>

                <g>
                  {nodes.map((n) => {
                    const p = layout.pos.get(n.id);
                    if (!p) return null;
                    const c = DEC_COLOR[n.last_decision] || UNKNOWN;
                    const s = (n.n_events || 0) / maxActivity;
                    const r = 3.5 + 13.5 * Math.sqrt(s);
                    const isAccount = n.kind === 'account';
                    const blocky = n.last_decision === 'block';
                    return (
                      <g key={n.id}
                        onMouseMove={(ev) => {
                          const g = ev.currentTarget.ownerSVGElement.getBoundingClientRect();
                          const sx = W / g.width;
                          const sy = H / g.height;
                          setTip({
                            x: (ev.clientX - g.left) * sx,
                            y: (ev.clientY - g.top) * sy,
                            label: short(n.id),
                            decision: n.last_decision || 'unseen',
                          });
                        }}
                        onMouseEnter={() => setSelected(n.id)}
                        onMouseLeave={() => setTip(null)}>
                        {isAccount && running && (
                          <circle className="g-pulse" cx={p.x} cy={p.y} r={r * 1.7} fill="none" stroke={c} strokeWidth="1.4" />
                        )}
                        <circle className={`g-node ${selected === n.id ? 'sel' : ''}`}
                          cx={p.x} cy={p.y} r={r}
                          fill={isAccount ? c : 'transparent'}
                          stroke={c} strokeWidth={isAccount ? (blocky ? 2.4 : 1.6) : 2}
                          strokeOpacity={isAccount ? 0.95 : 1}
                          style={blocky && isAccount ? { filter: 'url(#glow)' } : undefined} />
                      </g>
                    );
                  })}
                </g>
              </svg>

              {tip && (
                <div className="graph-tip" style={{ left: tip.x, top: tip.y }}>
                  <span style={{ color: DEC_COLOR[tip.decision] || UNKNOWN }}>●</span> {tip.label} · {tip.decision}
                </div>
              )}

              <div className="graph-detail">
                {selected && nodeMap.has(selected) ? (() => {
                  const n = nodeMap.get(selected);
                  return (
                    <>
                      <div className="g-d-head">
                        <b>{n.kind === 'account' ? 'Account' : 'Counterparty'}</b>
                        <span className={`pill tiny ${n.last_decision || 'gray'}`}>
                          {n.last_decision || 'unseen'}
                        </span>
                      </div>
                      <div className="g-d-id">{short(n.id)}</div>
                      <table className="g-d-table">
                        <tbody>
                          <tr><td>risk score</td><td>{fmt(n.last_prob)}</td></tr>
                          <tr><td>events</td><td>{n.n_events ?? 0}</td></tr>
                          <tr><td>links</td><td>{n.degree ?? 0}</td></tr>
                          <tr><td>known clean / fraud</td><td>{n.n_known ?? 0} / {n.n_fraud ?? 0}</td></tr>
                          <tr><td>sources</td><td>{(n.sources || []).join(', ') || '—'}</td></tr>
                          <tr><td>last seen</td><td>{fmtTS(n.last_ts)}</td></tr>
                        </tbody>
                      </table>
                    </>
                  );
                })() : (
                  <div className="g-d-placeholder">
                    <b>Most-connected nodes</b>
                    {byDegree.map((n) => (
                      <button key={n.id} className="g-degree" onClick={() => setSelected(n.id)}>
                        <span className="lwdot" style={{ background: DEC_COLOR[n.last_decision] || UNKNOWN }} />
                        <code>{short(n.id)}</code>
                        <b>{n.degree ?? 0}</b>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </>
      )}
    </section>
  );
}