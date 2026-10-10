import { useMemo } from 'react';

export default function Sparkline({ values = [], color = '#35d6ff' }) {
  const { areaD, lineD } = useMemo(() => {
    if (!values.length) return { areaD: '', lineD: '' };
    const W = 200;
    const H = 34;
    const max = Math.max(1, ...values);
    const n = values.length;
    let d = '';
    values.forEach((v, i) => {
      const x = (i * W) / (n - 1);
      const y = H - (v / max) * (H - 3) - 1;
      d += `${i === 0 ? 'M' : 'L'}${x.toFixed(1)} ${y.toFixed(1)} `;
    });
    const tail = values.length ? d.split(' ').pop() : '';
    const areaD = values.length ? `${d}L${W} ${H} L0 ${H} Z` : '';
    return { areaD, lineD: d };
  }, [values]);

  if (!lineD) return <svg className="src-spark" viewBox="0 0 200 34" preserveAspectRatio="none"><text x="100" y="20" textAnchor="middle" fill="#8a94ab" fontSize="10" fontFamily="var(--mono)">no data yet</text></svg>;

  return (
    <svg className="src-spark" viewBox="0 0 200 34" preserveAspectRatio="none">
      <defs>
        <linearGradient id={`sp${color.replace('#', '')}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.4" />
          <stop offset="100%" stopColor={color} stopOpacity="0.02" />
        </linearGradient>
      </defs>
      <path d={areaD} fill={`url(#sp${color.replace('#', '')})`} />
      <path d={lineD} fill="none" stroke={color} strokeWidth="1.8" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}