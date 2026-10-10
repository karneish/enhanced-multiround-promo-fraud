const MODEL_META = {
  XGBoost: { color: '#e0a15f', desc: 'GBDT · warm-started booster' },
  HistGradientBoosting: { color: '#4a90d9', desc: 'Histogram GBDT · grows max_iter' },
  ExtraTrees: { color: '#6bbf59', desc: 'Randomized bagging · refit' },
};

const SOURCES = ['replay', 'chain', 'webhook'];

const F1_COLOR = (v) => {
  if (v == null) return '#3c4659';
  if (v >= 0.5) return '#2ea043';
  if (v >= 0.3) return '#d29922';
  return '#ff5c57';
};

function WeightBar({ weights }) {
  const entries = Object.entries(weights || {});
  if (!entries.length) return null;
  return (
    <div className="weight-bar">
      {entries.map(([name, w]) => (
        <div
          key={name}
          className="weight-seg"
          style={{ width: `${(w * 100).toFixed(1)}%`, background: MODEL_META[name]?.color || '#888' }}
          title={`${name}: ${(w * 100).toFixed(1)}%`}
        />
      ))}
    </div>
  );
}

function ModelF1Rows({ ens, fitted }) {
  return (
    <div className="m-f1s">
      {Object.entries(MODEL_META).map(([name, meta]) => {
        const m = ens?.models?.[name];
        const f1 = m?.f1;
        const pct = f1 == null ? 0 : Math.max(0, Math.min(f1, 1)) * 100;
        return (
          <div key={name} className="m-f1">
            <span className="lbl">
              <span className="name-dot" style={{ background: meta.color }} />
              {name}
            </span>
            <div className="bar">
              <div className="fill" style={{ width: `${pct}%`, background: F1_COLOR(f1) }} />
            </div>
            <span className="val">{f1 == null ? (fitted ? 'waiting' : 'cold') : f1.toFixed(3)}</span>
          </div>
        );
      })}
    </div>
  );
}

function HParams({ config }) {
  if (!config) return null;
  const items = [
    ['min_fit_labels', config.min_fit_labels],
    ['retrain_interval_events', config.retrain_interval_events],
    ['xgb_rounds', config.xgb_rounds],
    ['xgb_extra_rounds', config.xgb_extra_rounds],
    ['hist_max_iter', config.hist_max_iter],
    ['et_n_estimators', config.et_n_estimators],
    ['buffer_max_labels', config.buffer_max_labels],
  ].filter(([, v]) => v != null);
  return (
    <div className="hparams">
      {items.map(([k, v]) => (
        <span key={k} className="hparam">
          <b>{k}</b>={v}
        </span>
      ))}
    </div>
  );
}

function EnsembleCard({ ens }) {
  const weights = ens?.weights || {};
  const entries = Object.entries(weights);
  const fitted = Boolean(ens?.fitted);
  const nLabeled = ens?.n_labeled ?? ens?.labels ?? 0;
  const buffered = ens?.buffered ?? 0;
  return (
    <div className="ensemble-card">
      <div className="ens-head">
        <div className="ens-labels">
          <span className={`src-chip chip-src-${ens?.source || ''}`}>{ens?.source || '—'}</span>
          <span className={`pill tiny ${fitted ? 'green' : 'amber'}`}>
            {fitted ? 'fitted' : 'cold start'}
          </span>
          <span className="ens-labels">{nLabeled} labeled · {buffered} buffered</span>
        </div>
        {entries.length > 0 && <span className="ens-badge">adaptive weights</span>}
      </div>

      {entries.length > 0 ? (
        <>
          <WeightBar weights={weights} />
          <div className="ens-weights">
            {entries.map(([name, w]) => {
              const meta = MODEL_META[name] || { color: '#888', desc: '' };
              return (
                <div key={name} className="ens-weight">
                  <span className="name-dot" style={{ background: meta.color }} />
                  <span className="w-name">{name}</span>
                  <b style={{ color: '#7ee2a8' }}>{(w * 100).toFixed(1)}%</b>
                  <small className="muted">{meta.desc}</small>
                </div>
              );
            })}
          </div>
        </>
      ) : (
        <div className="empty-state" style={{ padding: '14px', marginTop: '8px' }}>
          {fitted
            ? 'Not enough labels for weights yet.'
            : 'No training data yet — models fit once labels arrive via replay or labeled webhook events.'}
        </div>
      )}

      <ModelF1Rows ens={ens} fitted={fitted} />

      <table className="mini">
        <thead>
          <tr><th>model</th><th>precision</th><th>recall</th><th>F1</th><th>status</th></tr>
        </thead>
        <tbody>
          {Object.entries(MODEL_META).map(([name, meta]) => {
            const m = ens?.models?.[name];
            const fittedRow = fitted && m;
            return (
              <tr key={name}>
                <td>
                  <span className="name-dot" style={{ background: meta.color }} />
                  {name}
                </td>
                <td className="num">{m?.precision ?? '—'}</td>
                <td className="num">{m?.recall ?? '—'}</td>
                <td className="num"><b>{m?.f1 ?? '—'}</b></td>
                <td>{fittedRow ? 'trained' : fitted ? 'waiting' : 'cold'}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function ModelPanel({ status, config }) {
  const ensembles = status?.ensembles || {};
  const sourceList = Object.keys(ensembles).length ? Object.keys(ensembles) : SOURCES;
  const minFit = config?.min_fit_labels ?? 120;

  return (
    <section className="panel">
      <div className="panel-head">
        <div>
          <h2 className="panel-title">Adaptive model ensemble</h2>
          <HParams config={config} />
        </div>
        <span className="hint">
          3 models · weights ∝ trailing-holdout F1 · warm-start every {config?.retrain_interval_events ?? 500} labels
        </span>
      </div>

      <div className="model-hero">
        {Object.entries(MODEL_META).map(([name, meta]) => (
          <div key={name} className="model-hero-item">
            <div className="name"><span className="name-dot" style={{ background: meta.color }} />{name}</div>
            <div className="subtle" style={{ fontSize: '10.5px', color: 'var(--muted)', marginTop: '3px' }}>{meta.desc}</div>
          </div>
        ))}
      </div>
      <div className="model-hero-note">
        Fallback before first fit: clamped prior between 10–50%. First fit needs <code>{minFit}</code> labeled events.
      </div>

      <div className="ensembles">
        {sourceList.map((n) => (
          <EnsembleCard key={n} ens={ensembles[n]} />
        ))}
      </div>
      {sourceList.some((n) => !ensembles[n]) && (
        <div className="empty-state">
          Sources with no events yet ({sourceList.filter((n) => !ensembles[n]).join(', ')}) show
          their planned ensemble — they fit as soon as labeled events arrive.
        </div>
      )}
    </section>
  );
}