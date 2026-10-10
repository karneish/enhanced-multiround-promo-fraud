import { useState } from 'react';
import { ingestEvent } from '../api.js';

const PRESETS = [
  { label: 'Known fraud', value: { "account_id": "fraud-acct-001", "amount": 812.4, "label": 1, "counterparties": ["dev-42"] } },
  { label: 'Genuine', value: { "account_id": "good-acct-002", "amount": 24.5, "label": 0, "counterparties": ["dev-7"] } },
  { label: 'Unlabeled', value: { "account_id": "new-acct-003", "amount": 105.9, "counterparties": ["dev-99"] } },
];

export default function EventIntake({ running }) {
  const [account, setAccount] = useState('test-acct-001');
  const [amount, setAmount] = useState('100');
  const [label, setLabel] = useState('');
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState('');

  const send = async (payload) => {
    setBusy(true); setErr(''); setResult(null);
    try {
      const r = await ingestEvent(payload);
      setResult(r);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const payload = {
      account_id: account,
      amount: Number(amount) || 0,
    };
    if (label === '0') payload.label = 0;
    else if (label === '1') payload.label = 1;
    payload.counterparties = [];
    send(payload);
  };

  return (
    <section className="panel">
      <div className="panel-head"><h2>Manual event intake</h2><span className="hint">POST /api/events/ingest · works in every mode</span></div>
      <p className="muted" style={{ marginTop: 0 }}>
        Simulate a producer posting a canonical event. Fill in a payload and it is scored immediately.
      </p>

      <div className="preset-row">
        <span className="hint">Quick presets:</span>
        {PRESETS.map((p) => (
          <button key={p.label} className="ghost tiny" onClick={() => send(p.value)}>{p.label}</button>
        ))}
      </div>

      <form className="intake-form" onSubmit={handleSubmit}>
        <label className="field">
          <span>Account id</span>
          <input value={account} onChange={(e) => setAccount(e.target.value)} />
        </label>
        <label className="field">
          <span>Amount</span>
          <input type="number" value={amount} onChange={(e) => setAmount(e.target.value)} />
        </label>
        <label className="field">
          <span>Label (optional)</span>
          <select value={label} onChange={(e) => setLabel(e.target.value)}>
            <option value="">unknown</option>
            <option value="1">1 — fraud</option>
            <option value="0">0 — genuine</option>
          </select>
        </label>
        <button className="primary" disabled={busy || !running}>{busy ? 'Ingesting…' : 'Ingest event'}</button>
      </form>

      {err && <div className="banner error">{err}</div>}
      {result && (
        <div className="ingest-result">
          <span className={`chip dec-${result.decision}`}>{result.decision}</span>
          <span>probability <b>{result.prob}</b></span>
          <span>label {result.label ?? 'unknown'}</span>
          <code className="muted">{result.event_id}</code>
        </div>
      )}
    </section>
  );
}