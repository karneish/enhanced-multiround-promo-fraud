import { useMemo } from 'react';
import { C } from '../theme';

const W = 860;
const H = 220;
const P = { l: 36, r: 16, t: 14, b: 22 };
const IW = W - P.l - P.r;
const IH = H - P.t - P.b;

function yOf(p) { return P.t + IH * (1 - p); }

function smoothPath(pts) {
  if (pts.length < 2) return '';
  let d = `M ${pts[0].x.toFixed(2)} ${pts[0].y.toFixed(2)}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i - 1] || pts[i];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[i + 2] || p2;
    const c1x = p1.x + (p2.x - p0.x) / 6;
    const c1y = p1.y + (p2.y - p0.y) / 6;
    const c2x = p2.x - (p3.x - p1.x) / 6;
    const c2y = p2.y - (p3.y - p1.y) / 6;
    d += ` C ${c1x.toFixed(2)} ${c1y.toFixed(2)}, ${c2x.toFixed(2)} ${c2y.toFixed(2)}, ${p2.x.toFixed(2)} ${p2.y.toFixed(2)}`;
  }
  return d;
}

export default function TrendChart({ data, review = 0.3, block = 0.7 }) {
  const scalePts = useMemo(() => {
    if (!data.length) return [];
    const n = data.length;
    const xs = (i) => P.l + (n === 1 ? 0 : (i * IW) / (n - 1));
    return data.map((d, i) => ({ x: xs(i), y: yOf(Math.max(0, Math.min(1, d.prob))) }));
  }, [data]);

  if (!data.length) {
    return (
      <div className="chart-frame" style={{ height: H + 26 }}>
        <div className="chart-empty">waiting for scored events to draw the fraud-probability trend…</div>
      </div>
    );
  }

  const line = smoothPath(scalePts);
  const area = `${line} L ${scalePts[scalePts.length - 1].x.toFixed(2)} ${P.t + IH} L ${scalePts[0].x.toFixed(2)} ${P.t + IH} Z`;
  const last = data[data.length - 1];
  const lastPt = scalePts[scalePts.length - 1];

  return (
    <div className="chart-frame">
      <svg className="chart-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet">
        <defs>
          <linearGradient id="trendFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#35d6ff" stopOpacity="0.34" />
            <stop offset="100%" stopColor="#35d6ff" stopOpacity="0.02" />
          </linearGradient>
          <linearGradient id="trendStroke2" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#35d6ff" />
            <stop offset="100%" stopColor="#7d5bff" />
          </linearGradient>
        </defs>

        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <line key={v} className="grid-line" x1={P.l} x2={W - P.r} y1={yOf(v)} y2={yOf(v)} />
        ))}
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <text key={`t${v}`} className="axis-txt" x={P.l - 6} y={yOf(v) + 3} textAnchor="end">
            {v.toFixed(2)}
          </text>
        ))}
        <text className="axis-txt" x={P.l} y={H - 6}>fraud probability → latest</text>

        <rect x={P.l} y={yOf(block)} width={IW} height={P.t + IH - yOf(block)} fill={C.red} opacity="0.05" />
        <rect x={P.l} y={yOf(review)} width={IW} height={yOf(block) - yOf(review)} fill={C.amber} opacity="0.05" />

        <line className="thr-line" stroke={C.amber} x1={P.l} x2={W - P.r} y1={yOf(review)} y2={yOf(review)} />
        <line className="thr-line" stroke={C.red} x1={P.l} x2={W - P.r} y1={yOf(block)} y2={yOf(block)} />
        <text className="axis-txt" fill={C.amber} x={W - P.r} y={yOf(review) - 5} textAnchor="end">review {review}</text>
        <text className="axis-txt" fill={C.red} x={W - P.r} y={yOf(block) - 5} textAnchor="end">block {block}</text>

        <path d={area} fill="url(#trendFill)" />
        <path d={line} fill="none" stroke="url(#trendStroke2)" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />

        {scalePts.map((p, i) => {
          const d = data[i];
          return (
            <circle
              key={i}
              className={`trend-pt${d.decision === 'block' ? ' hot' : ''}`}
              cx={p.x}
              cy={p.y}
              r={3}
              fill={d.decision === 'allow' ? C.green : d.decision === 'review' ? C.amber : C.red}
              stroke="rgba(255,255,255,0.35)"
              strokeWidth="0.7"
            />
          );
        })}

        {last && (
          <g>
            <circle className="trend-pt hot" cx={lastPt.x} cy={lastPt.y} r={3} fill={last.decision === 'allow' ? C.green : last.decision === 'review' ? C.amber : C.red} />
          </g>
        )}
      </svg>
      <div className="chart-legend">
        <span style={{ color: C.green }}>◉ allow</span>
        <span style={{ color: C.amber }}>◉ review</span>
        <span style={{ color: C.red }}>◉ block</span>
        <span style={{ color: C.muted }}>last {data.length} events · source = colour</span>
      </div>
    </div>
  );
}