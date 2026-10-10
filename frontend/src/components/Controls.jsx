export default function Controls({ mode, setMode, eps, setEps, running, busy, onStart, onStop }) {
  const modes = [
    ['replay-only', 'IEEE replay'],
    ['chain-only', 'Ethereum live'],
    ['handoff', 'Handoff (replay → chain)'],
    ['concurrent', 'Concurrent (both)'],
  ];

  return (
    <section className="controls card">
      <label className="field">
        <span>Live mode</span>
        <select value={mode} onChange={(e) => setMode(e.target.value)} disabled={running}>
          {modes.map(([value, label]) => (
            <option key={value} value={value}>{label}</option>
          ))}
        </select>
      </label>
      <label className="field">
        <span>Replay rate (events/sec)</span>
        <input
          type="number"
          min="1"
          max="1000"
          value={eps}
          disabled={running}
          onChange={(e) => setEps(e.target.value)}
        />
      </label>
      <div className="actions">
        {!running ? (
          <button className="primary" disabled={busy} onClick={onStart}>
            {busy ? 'Starting…' : 'Start live stream'}
          </button>
        ) : (
          <button className="danger" disabled={busy} onClick={onStop}>
            {busy ? 'Stopping…' : 'Stop'}
          </button>
        )}
      </div>
    </section>
  );
}