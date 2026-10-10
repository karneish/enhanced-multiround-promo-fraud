const MODEL_COLORS = {
  XGBoost: '#b87333',
  HistGradientBoosting: '#4a90d9',
  ExtraTrees: '#6bbf59',
};

export default function StatsPanel({ status, schema }) {
  if (!status) return null;
  const cfg = schema?.config || {};

  const counters = status.counters || {};
  const decisions = status.decisions || {};
  const sources = status.source_states || {};
  const ensembles = status.ensembles || {};
  const chain = status.chain || {};
  const budget = status.review_budget || {};

  const total = Object.values(counters).reduce((a, b) => a + b, 0);
  const scammers = status.scam_labels || {};

  const sum = (o) => Object.values(o).reduce((a, b) => a + (b || 0), 0);

  return (
    <section className="stats card">
      <h2>State</h2>
      <div className="stat-grid">
        <div className="stat">
          <b>{total}</b> events
          <small>{status.running ? 'streaming' : 'idle'} · mode {status.mode}</small>
        </div>
        <div className="stat">
          <b>{sum(decisions)}</b> decisions
          <small>allow {decisions.allow ?? 0} · review {decisions.review ?? 0} · block {decisions.block ?? 0}</small>
        </div>
        <div className="stat">
          <b>{status.replay_finished ? 'done' : sources.replay?.finished ? 'done' : '…'}</b> replay
          <small>{sources.replay?.events ?? 0} events · {sources.replay?.labeled ?? 0} labeled</small>
        </div>
        <div className="stat">
          <b>{sources.chain?.events ?? 0}</b> chain events
          <small>
            block {chain.last_block ?? '—'}
            {chain.last_error ? ` · err: ${chain.last_error.slice(0, 40)}` : ''}
          </small>
        </div>
        <div className="stat">
          <b>{budget.pos ?? 0} / {budget.neg ?? 0}</b> review budget
          <small>{status.review_spent ?? 0} spent</small>
        </div>
        <div className="stat">
          <b>{scammers.count ?? 0}</b> scam addrs
          <small>{scammers.available ? 'label bridge active' : 'list not loaded'}</small>
        </div>
      </div>

      {Object.keys(ensembles).length > 0 && (
        <div className="ensembles">
          {Object.entries(ensembles).map(([source, ens]) => (
            <div key={source} className="ensemble">
              <h3>{source} · {ens.fitted ? 'fitted' : 'cold start'}</h3>
              <div className="weight-row">
                {(cfg.decision_thresholds) && (
                  <span className="muted">
                    bands: review≥{cfg.decision_thresholds.review} · block≥{cfg.decision_thresholds.block}
                  </span>
                )}
                {(ens.weights && Object.keys(ens.weights).length) ? (
                  Object.entries(ens.weights).map(([name, w]) => (
                    <span key={name} className="weight">
                      <i style={{ background: MODEL_COLORS[name] || '#888' }} />
                      {name} {(w * 100).toFixed(0)}%
                    </span>
                  ))
                ) : <span className="muted">waiting for {cfg.min_fit_labels ?? 120} labels</span>}
              </div>
              {(ens.models && Object.keys(ens.models).filter((m) => ens.models[m]).length > 0) && (
                <table className="mini">
                  <thead>
                    <tr><th>model</th><th>precision</th><th>recall</th><th>f1</th></tr>
                  </thead>
                  <tbody>
                    {Object.entries(ens.models).filter(([, m]) => m).map(([name, m]) => (
                      <tr key={name}>
                        <td>{name}</td>
                        <td>{m.precision ?? '—'}</td>
                        <td>{m.recall ?? '—'}</td>
                        <td>{m.f1 ?? '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              <small className="muted">{ens.n_labeled ?? 0} labeled events kept in the rolling window</small>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}