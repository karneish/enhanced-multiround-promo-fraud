import Sparkline from '../charts/Sparkline.jsx';

const SRC_ACCENT = { replay: '#e0a15f', chain: '#4a90d9', webhook: '#a58ef0' };

function fmtTS(ts) {
  if (ts == null) return '—';
  return new Date(ts * 1000).toLocaleTimeString();
}

function SourceCard({ id, icon, name, state, note, history }) {
  const events = state?.events ?? 0;
  const labeled = state?.labeled ?? 0;
  const finished = Boolean(state?.finished);
  const error = Boolean(state?.error);
  const accent = SRC_ACCENT[id] || '#5b8cff';

  return (
    <div className={`source-card${error ? ' error' : ''}${finished ? ' finished' : ''}`}>
      <div className="src-bar" style={{ background: accent, '--src': accent, boxShadow: `0 0 14px ${accent}` }} />
      <div className="source-head">
        <div className="src-left">
          <span className="src-ico" style={{ background: accent, color: '#0b1024' }}>{icon}</span>
          <div>
            <span className="src-name">{name}</span>
            <div className="badge-row">
              <span className={`pill tiny ${error ? 'red' : finished ? 'gray' : 'green'}`}>
                {error ? 'error' : finished ? 'finished' : 'active'}
              </span>
              {state?.sample && <span className="pill tiny amber">sample (no real CSV)</span>}
            </div>
          </div>
        </div>
        {events > 0 ? (
          <span className="src-state live" style={{ color: accent }}>● live</span>
        ) : (
          <span className="src-state">listening…</span>
        )}
      </div>
      <div className="source-stats">
        <div className="mini-stat"><span className="v">{events.toLocaleString()}</span><span className="k">events</span></div>
        <div className="mini-stat"><span className="v">{labeled.toLocaleString()}</span><span className="k">labeled</span></div>
        {state?.last_ts != null && (
          <div className="mini-stat"><span className="v" style={{ fontSize: '12px' }}>{fmtTS(state.last_ts)}</span><span className="k">last event</span></div>
        )}
      </div>
      <Sparkline values={(history || []).slice(-80)} color={accent} />
      <div className="source-note">{note}</div>
      {error && <div className="src-err">{state.error}</div>}
    </div>
  );
}

export default function SourceCards({ status, config, history }) {
  const st = status?.source_states || {};
  const chain = status?.chain || {};
  const scammers = status?.scam_labels || {};
  const ieeeF = config?.ieee || {};
  const chainF = config?.chain || {};
  const errCount = Object.values(status?.errors || {}).reduce((a, b) => a + b, 0);

  return (
    <section className="panel">
      <div className="panel-head">
        <h2 className="panel-title">Data sources</h2>
        <span className="hint">stream health per source</span>
      </div>
      <div className="sources">
        <SourceCard
          id="replay"
          icon="RS"
          name="IEEE-CIS replay"
          state={st.replay}
          note={st.replay?.sample
            ? `deterministic sample fallback · ${config?.stream_slice ?? '—'} slice · real labels`
            : `${ieeeF.replay_skip_first ?? 0} skipped · ${config?.stream_slice ?? '—'} slice · real labels`}
          history={history?.replay}
        />
        <SourceCard
          id="chain"
          icon="ETH"
          name="Ethereum live"
          state={st.chain}
          note={`block ${chain.last_block ?? '—'} · ${chain.active_rpc_url || chainF.rpc_url || 'rpc'} · ${chainF.poll_interval_seconds ?? '—'}s poll`}
          history={history?.chain}
        />
      </div>
      <div className="bridge">
        <span>Scam-address label bridge:</span>
        <b className={scammers.available ? 'ok' : 'warn'}>
          {scammers.available ? `${scammers.count} addresses loaded` : 'no list — review loop only'}
        </b>
        {errCount > 0 && (
          <span className="pill tiny red">errors: {errCount}</span>
        )}
      </div>
    </section>
  );
}