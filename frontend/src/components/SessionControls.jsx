const MODES = [
  ['replay-only', 'IEEE replay', 'Timed replay of real labeled IEEE-CIS transactions'],
  ['chain-only', 'Ethereum live', 'Genuinely live Ethereum transactions (keyless gateway)'],
  ['handoff', 'Handoff', 'Train/evaluate on replay, then switch to the live chain stream'],
  ['concurrent', 'Concurrent', 'Replay + chain together, source-tagged'],
];

function ReadyBadge({ label, ready }) {
  return (
    <span className={`ready ${ready ? 'ok' : 'warn'}`}>
      {ready ? '● ready' : '○ missing'}
      <span className="ready-label">{label}</span>
    </span>
  );
}

export default function SessionControls({ controls, setControls, running, onStart, onStop, dataReady, schema }) {
  const set = (k) => (e) => setControls((c) => ({ ...c, [k]: e.target.value }));
  const num = (k) => (e) => setControls((c) => ({ ...c, [k]: e.target.value }));

  const ieeeReady = !!dataReady?.ieee_files?.train_transaction || !!dataReady?.ieee_files?.final_fraud;
  const scamReady = !!dataReady?.scam_list;

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Session controls</h2>
        <span className="hint">Configure and start/stop a streaming session</span>
      </div>

      <div className="mode-grid">
        {MODES.map(([value, label, desc]) => (
          <button
            key={value}
            type="button"
            className={`mode-btn ${controls.mode === value ? 'active' : ''}`}
            disabled={running}
            onClick={() => setControls((c) => ({ ...c, mode: value }))}
          >
            <b>{label}</b>
            <small>{desc}</small>
          </button>
        ))}
      </div>

      <div className="ctrl-grid">
        <label className="field">
          <span>Replay rate (events/sec)</span>
          <input type="number" min="1" max="5000" value={controls.events_per_sec}
                 disabled={running} onChange={num('events_per_sec')} />
        </label>
        <label className="field">
          <span>Replay slice (rows)</span>
          <input type="number" min="1000" step="1000" value={controls.stream_slice}
                 disabled={running} onChange={num('stream_slice')} />
        </label>
        <label className="field">
          <span>Review threshold (≥)</span>
          <input type="number" min="0" max="1" step="0.01" value={controls.review_threshold}
                 disabled={running} onChange={num('review_threshold')} />
        </label>
        <label className="field">
          <span>Block threshold (≥)</span>
          <input type="number" min="0" max="1" step="0.01" value={controls.block_threshold}
                 disabled={running} onChange={num('block_threshold')} />
        </label>
        <label className="field">
          <span>Review budget · positive</span>
          <input type="number" min="0" value={controls.budget_pos}
                 disabled={running} onChange={num('budget_pos')} />
        </label>
        <label className="field">
          <span>Review budget · negative</span>
          <input type="number" min="0" value={controls.budget_neg}
                 disabled={running} onChange={num('budget_neg')} />
        </label>
      </div>

      <div className="data-ready">
        <ReadyBadge label="IEEE-CIS dataset (data/)" ready={ieeeReady} />
        <ReadyBadge label="Sample replay fallback" ready={!!dataReady?.sample_fallback} />
        <ReadyBadge label="Scam-address list (chain)" ready={scamReady} />
        {!ieeeReady && schema && (
          <span className="hint">
            {dataReady?.sample_fallback
              ? <>No real CSVs — using a deterministic labeled sample; real files in <code>data/</code> override it.</>
              : <>Run <code>scripts\\fetch_ieee.py</code> or drop CSVs into <code>data/</code></>}
          </span>
        )}
      </div>

      <div className="actions">
        {!running ? (
          <button className="primary big" onClick={onStart}>▶ Start live stream</button>
        ) : (
          <button className="danger big" onClick={onStop}>■ Stop stream</button>
        )}
      </div>
    </section>
  );
}