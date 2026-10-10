import { useMemo } from 'react';

const R = 64;
const TH = 15;
const ORDER = [
  { key: 'allow', color: '#2ea043', label: 'Allow' },
  { key: 'review', color: '#d29922', label: 'Review' },
  { key: 'block', color: '#ff5c57', label: 'Block' },
];

export default function DecisionDonut({ allow = 0, review = 0, block = 0 }) {
  const total = allow + review + block;
  const segs = useMemo(() => {
    let cum = 0;
    return ORDER.map((o) => {
      const pct = total ? (({ allow, review, block })[o.key] / total) * 100 : 0;
      const s = { ...o, pct, offset: cum };
      cum += pct;
      return s;
    });
  }, [allow, review, block, total]);

  return (
    <svg width="210" height="210" viewBox="0 0 210 210" style={{ display: 'block', margin: '0 auto' }}>
      <circle cx="105" cy="105" r={R} fill="none" stroke="rgba(122,146,214,0.12)" strokeWidth={TH} />
      <g transform={`rotate(-90 105 105)`}>
        {segs.map((s) =>
          s.pct > 0 ? (
            <circle
              key={s.key}
              className="donut-seg"
              cx="105"
              cy="105"
              r={R}
              fill="none"
              stroke={s.color}
              strokeWidth={TH}
              strokeLinecap="round"
              pathLength={100}
              strokeDasharray={`${s.pct} ${100 - s.pct}`}
              strokeDashoffset={-s.offset}
              style={{ filter: `drop-shadow(0 0 6px ${s.color}66)` }}
            />
          ) : null
        )}
      </g>
      <text className="donut-center-val" x="105" y="103" textAnchor="middle">{total}</text>
      <text className="donut-center-lbl" x="105" y="122" textAnchor="middle">DECISIONS</text>
    </svg>
  );
}