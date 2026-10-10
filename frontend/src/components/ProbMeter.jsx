export default function ProbMeter({ prob, decision }) {
  const p = prob == null ? 0 : Math.max(0, Math.min(prob, 1));
  const pct = Math.round(p * 100);
  const cls = decision === 'block' ? 'danger' : decision === 'review' ? 'warn' : 'ok';
  const color = decision === 'block' ? '#ff9b97' : decision === 'review' ? '#ffcc6b' : '#7ee2a8';
  return (
    <div className="prob-cell">
      <div className="meter">
        <div className={`meter-fill ${cls}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="prob-val" style={{ color }}>{pct}%</span>
    </div>
  );
}