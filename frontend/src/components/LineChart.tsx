import React from 'react';

interface Point { x: number; y: number; label?: string; }
interface Props { points: Point[]; color?: string; w?: number; h?: number; yMin?: number; yMax?: number; decimals?: number; fill?: boolean; }
export default function LineChart({ points, color = '#3fb950', w = 560, h = 170, yMin, yMax, decimals = 3, fill = false }: Props) {
  if (!points.length) return <div className="empty">No data</div>;
  const pad = { l: 34, r: 12, t: 12, b: 22 };
  const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
  const xMin = Math.min(...points.map((p) => p.x)), xMax = Math.max(...points.map((p) => p.x));
  const dataYMin = yMin ?? Math.min(0, ...points.map((p) => p.y));
  const dataYMax = yMax ?? Math.max(0.01, ...points.map((p) => p.y));
  const X = (x: number) => pad.l + (xMax === xMin ? 0 : (x - xMin) / (xMax - xMin)) * iw;
  const Y = (y: number) => pad.t + (1 - (y - dataYMin) / (dataYMax - dataYMin)) * ih;
  const path = points.map((p, i) => `${i ? 'L' : 'M'}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join(' ');
  const gid = 'g' + Math.random().toString(36).slice(2, 8);
  const last = points[points.length - 1];
  const grid = [0, 0.25, 0.5, 0.75, 1].map((v) => {
    const y = dataYMin + v * (dataYMax - dataYMin);
    return <line key={v} x1={pad.l} y1={Y(y)} x2={w - pad.r} y2={Y(y)} className="grid-line" />;
  });
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="chart-svg">
      {fill && <defs><linearGradient id={gid} x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor={color} stopOpacity={0.6} /><stop offset="1" stopColor={color} stopOpacity={0} /></linearGradient></defs>}
      {grid}
      {fill && <path d={`${path} L${X(last.x)},${pad.t + ih} L${X(points[0].x)},${pad.t + ih} Z`} fill={`url(#${gid})`} className="chart-area-gradient" />}
      <path d={path} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="path-anim" style={{ filter: `drop-shadow(0 0 6px ${color})` }} />
      {points.map((p, i) => <circle key={i} cx={X(p.x)} cy={Y(p.y)} r="3.5" fill={color} stroke="#0c150f" strokeWidth="1" className="dot-anim" style={{ animationDelay: `${0.4 + i * 0.08}s` }} />)}
      <text x={X(last.x)} y={Y(last.y) - 8} className="val-label" textAnchor="middle">{last.y.toFixed(decimals)}</text>
      {points.map((p, i) => <text key={i} x={X(p.x)} y={h - 6} className="axis" textAnchor="middle">{p.label ?? `r${p.x}`}</text>)}
    </svg>
  );
}
