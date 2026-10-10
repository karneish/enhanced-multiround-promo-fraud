import DecisionDonut from '../charts/DecisionDonut.jsx';

const ORDER = [
  ['block', 'Block', 'red'],
  ['review', 'Review', 'amber'],
  ['allow', 'Allow', 'green'],
];

export default function DecisionMix({ decisions, config }) {
  const d = decisions || {};
  const total = (d.allow || 0) + (d.review || 0) + (d.block || 0);
  const pct = (v) => (total ? ((v / total) * 100).toFixed(1) : '0.0');
  const th = config?.decision_thresholds || {};

  return (
    <section className="panel">
      <div className="panel-head">
        <h2 className="panel-title">Decision mix</h2>
        <span className="hint">current session</span>
      </div>
      {total === 0 ? (
        <div className="empty-state">No decisions yet — start a stream</div>
      ) : (
        <div className="mix-donut-wrap">
          <DecisionDonut allow={d.allow || 0} review={d.review || 0} block={d.block || 0} />
        </div>
      )}
      <div className="mix-bar">
        {ORDER.map(([key, label, cls]) => (
          <div
            key={key}
            className={`mix-seg ${cls}`}
            style={{ width: `${pct(d[key] || 0)}%` }}
            title={`${label}: ${d[key] || 0} (${pct(d[key] || 0)}%)`}
          />
        ))}
      </div>
      <div className="mix-legend">
        {ORDER.map(([key, label, cls]) => (
          <div key={key} className="legend-row">
            <span className={`swatch ${cls}`} />
            <span className="legend-label">{label}</span>
            <b>{(d[key] || 0).toLocaleString()}</b>
            <span className="legend-pct">{pct(d[key] || 0)}%</span>
          </div>
        ))}
        <div className="legend-row total">
          <span className="legend-label">Total</span>
          <b>{total.toLocaleString()}</b>
          <span className="legend-pct">100%</span>
        </div>
      </div>
      <div className="thresholds">
        <span className="hint">bands</span>
        <code>review ≥ {th.review ?? '—'}</code>
        <code>block ≥ {th.block ?? '—'}</code>
      </div>
    </section>
  );
}