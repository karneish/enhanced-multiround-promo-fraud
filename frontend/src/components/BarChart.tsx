import React from 'react';

interface Props { items: Record<string, number>; color?: string; w?: number; h?: number; maxItems?: number; }
export default function BarChart({ items, color = '#3fb950', w = 500, h = 200, maxItems = 10 }: Props) {
  const sorted = Object.entries(items).sort((a, b) => b[1] - a[1]).slice(0, maxItems);
  if (!sorted.length) return <div className="empty">No data</div>;
  const pad = { l: 8, r: 8, t: 16, b: 42 };
  const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
  const max = Math.max(...sorted.map(([, v]) => v), 1);
  const bw = Math.max(8, Math.min(28, iw / sorted.length - 6));
  const bwT = bw + 6;
  const startX = pad.l + Math.max(0, (iw - bwT * sorted.length) / 2);
  const gid = 'bar-grad-' + Math.random().toString(36).slice(2, 8);
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="chart-svg">
      <defs>
        <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor={color} stopOpacity={1} />
          <stop offset="1" stopColor={color} stopOpacity={0.5} />
        </linearGradient>
      </defs>
      {sorted.map(([k, v], i) => {
        const bh = (v / max) * ih;
        const x = startX + i * bwT;
        const y = pad.t + ih - bh;
        const label = k.length > 16 ? k.slice(0, 14) + '..' : k;
        return (
          <React.Fragment key={k}>
            <rect x={x} y={y} width={bw} height={bh} rx="4" fill={`url(#${gid})`} className="bar-anim" style={{ animationDelay: `${i * 0.06}s` }}><title>{k}: {v}</title></rect>
            <text x={x + bw / 2} y={y - 4} className="val-label" textAnchor="middle">{v}</text>
            <text x={x + bw / 2} y={h - 8} className="axis" textAnchor="end" transform={`rotate(-20, ${x + bw / 2}, ${h - 8})`}>{label}</text>
          </React.Fragment>
        );
      })}
    </svg>
  );
}
