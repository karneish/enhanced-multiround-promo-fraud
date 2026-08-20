import React, { useState, useEffect, useCallback, useRef } from 'react';
import { adlApi } from '../api';
import { useSSE } from '../hooks/useSSE';
import MetricCard from '../components/MetricCard';
import LineChart from '../components/LineChart';
import LogPanel from '../components/LogPanel';
import RoundTracker from '../components/RoundTracker';
import DataTable from '../components/DataTable';

type Tab = 'live' | 'defense' | 'launch' | 'history' | 'about';

const ADL_FIELDS: [string, string][] = [
  ['rounds', 'Rounds'], ['base_accounts', 'Base Accounts'], ['initial_fraud', 'Initial Fraud'],
  ['genuine_per_round', 'Genuine / Round'], ['fraud_per_round', 'Fraud / Round'],
  ['seed', 'Seed'], ['supervised_ratio', 'Supervised Ratio'], ['budget_pos', 'Budget +'],
  ['budget_neg', 'Budget -'], ['gan_epochs', 'GAN Epochs'],
];

const ADL_CONFIG: [string, string][] = [
  ['t1', 'T1 (Review Threshold)'], ['t2', 'T2 (Block Threshold)'],
  ['threshold_alpha', 'Alpha (adapt rate)'], ['review_catch_rate', 'Review Catch Rate'],
  ['w_pf', 'w*Pf'], ['w_centrality', 'w*Centrality'], ['w_ring', 'w*Ring'],
  ['w_velocity', 'w*Velocity'], ['w_trust', 'w*Trust'],
];

export default function AdlPage() {
  const [tab, setTab] = useState<Tab>('live');
  const [schema, setSchema] = useState<any>(null);
  const [datasets, setDatasets] = useState<any[]>([]);
  const [history, setHistory] = useState<any[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [rounds, setRounds] = useState<any[]>([]);
  const [config, setConfig] = useState<Record<string, any>>({});
  const [describe, setDescribe] = useState('');
  const [adlEnabled, setAdlEnabled] = useState(true);
  const [policy, setPolicy] = useState('adaptive');
  const [launching, setLaunching] = useState(false);
  const logRef = useRef<{ t: number; text: string }[]>([]);
  const [logLines, setLogLines] = useState<{ t: number; text: string }[]>([]);
  const sseUrl = selected ? adlApi.streamUrl(selected) : null;
  const { events, connected } = useSSE(sseUrl);
  const lastIdxRef = useRef(0);

  useEffect(() => { adlApi.schema().then(setSchema).catch(() => {}); adlApi.datasets().then(setDatasets).catch(() => {}); adlApi.history().then(setHistory).catch(() => {}); }, []);
  useEffect(() => { if (schema) setConfig(schema.defaults || {}); }, [schema]);

  useEffect(() => {
    const start = lastIdxRef.current;
    if (start >= events.length) return;
    lastIdxRef.current = events.length;
    for (let i = start; i < events.length; i++) {
      const ev = events[i];
      if (ev.type === 'round_result') setRounds((p) => [...p, ev]);
      else if (ev.type === 'log') { logRef.current = [...logRef.current.slice(-200), { t: ev.t ?? 0, text: ev.text ?? '' }]; setLogLines([...logRef.current]); }
      else if (ev.type === 'state' && ev.finished) { adlApi.history().then(setHistory).catch(() => {}); }
    }
  }, [events, selected]);

  const launch = useCallback(async () => {
    setLaunching(true);
    try {
      const body = { ...config, describe: describe || 'ADL experiment', adl_enabled: adlEnabled, threshold_policy: policy };
      const res = await adlApi.launch(body);
      setSelected(res.id); setRounds([]); logRef.current = []; setLogLines([]); setTab('live'); lastIdxRef.current = 0;
    } catch (e: any) { alert('Launch failed: ' + e.message); }
    finally { setLaunching(false); }
  }, [config, describe, adlEnabled, policy]);

  const lastRound = rounds[rounds.length - 1];
  const defense = lastRound?.defense;

  return (
    <div>
      <div className="page-header">
        <h2>Adaptive Defensive Layer</h2>
        <p>Risk-based Allow / Review / Block decisions with adaptive thresholds</p>
      </div>
      <div className="tab-bar">
        {(['live', 'defense', 'launch', 'history', 'about'] as Tab[]).map((t) => (
          <button key={t} className={`tab-btn ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
            {t === 'live' ? '\u25CF' : t === 'defense' ? '\u2666' : t === 'launch' ? '\u25B2' : t === 'history' ? '\u25BC' : '\u2139'} {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {tab === 'live' && (
        <div className="tab-panel">
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
            <select className="input" style={{ width: 300 }} value={selected ?? ''} onChange={(e) => { setSelected(e.target.value || null); setRounds([]); logRef.current = []; setLogLines([]); lastIdxRef.current = 0; }}>
              <option value="">Select a run...</option>
              {history.map((h) => <option key={h.id} value={h.id}>{h.id.slice(0, 8)} ... {(h.describe || '').slice(0, 25)}</option>)}
            </select>
            <span className={`sse-pill ${connected ? 'on' : ''}`}><span className="dot" />{connected ? 'Streaming' : 'Off'}</span>
          </div>
          {lastRound ? (
            <>
              <RoundTracker total={config.rounds ?? 8} current={lastRound.round ?? 0} />
              <div className="metric-grid">
                <MetricCard label="Round" value={`#${lastRound.round}`} cls="blue" />
                <MetricCard label="Nodes" value={lastRound.num_nodes ?? '--'} cls="blue" />
                <MetricCard label="Missed" value={lastRound.missed ?? '--'} cls={lastRound.missed > 0 ? 'red' : 'green'} />
                <MetricCard label="F1" value={(lastRound.metrics?.f1 ?? 0).toFixed(3)} cls="green" />
                <MetricCard label="AUC" value={(lastRound.metrics?.auc ?? 0).toFixed(3)} cls="purple" />
                <MetricCard label="Recall" value={(lastRound.metrics?.rec ?? 0).toFixed(3)} cls="green" />
                <MetricCard label="Precision" value={(lastRound.metrics?.prec ?? 0).toFixed(3)} cls="amber" />
                {defense && <>
                  <MetricCard label="Blocked" value={defense.n_block ?? '--'} cls="red" />
                  <MetricCard label="Escape Rate" value={((defense.escape_rate ?? 0) * 100).toFixed(1) + '%'} cls={defense.escape_rate > 0.05 ? 'red' : 'green'} />
                  <MetricCard label="False Block" value={((defense.false_block_rate ?? 0) * 100).toFixed(1) + '%'} cls={defense.false_block_rate > 0.05 ? 'red' : 'green'} />
                  <MetricCard label="Defense Recall" value={(defense.defense_recall ?? 0).toFixed(3)} cls="green" />
                  <MetricCard label="Avg Risk" value={(defense.avg_risk ?? 0).toFixed(3)} cls="amber" />
                </>}
              </div>
              <div className="row">
                <div className="grow card">
                  <div className="card-head"><h3>Detector Performance</h3></div>
                  <div className="stack">
                    <div className="chart-inline"><span className="mini-label">F1</span><LineChart points={rounds.map((r) => ({ x: r.round, y: r.metrics?.f1 ?? 0 }))} color="#3fb950" fill /></div>
                    <div className="chart-inline"><span className="mini-label">AUC</span><LineChart points={rounds.map((r) => ({ x: r.round, y: r.metrics?.auc ?? 0 }))} color="#60a5fa" /></div>
                    <div className="chart-inline"><span className="mini-label">Recall</span><LineChart points={rounds.map((r) => ({ x: r.round, y: r.metrics?.rec ?? 0 }))} color="#a78bfa" /></div>
                    <div className="chart-inline"><span className="mini-label">Prec</span><LineChart points={rounds.map((r) => ({ x: r.round, y: r.metrics?.prec ?? 0 }))} color="#f0b429" /></div>
                  </div>
                </div>
                <div className="grow card">
                  <div className="card-head"><h3>Defense Metrics</h3></div>
                  {defense ? (
                    <div className="stack">
                      <div className="chart-inline"><span className="mini-label">Escape</span><LineChart points={rounds.map((r) => ({ x: r.round, y: r.defense?.escape_rate ?? 0 }))} color="#ef4444" yMin={0} yMax={1} decimals={3} /></div>
                      <div className="chart-inline"><span className="mini-label">False Blk</span><LineChart points={rounds.map((r) => ({ x: r.round, y: r.defense?.false_block_rate ?? 0 }))} color="#f0b429" yMin={0} yMax={1} decimals={3} /></div>
                      <div className="chart-inline"><span className="mini-label">dRecall</span><LineChart points={rounds.map((r) => ({ x: r.round, y: r.defense?.defense_recall ?? 0 }))} color="#3fb950" yMin={0} yMax={1} decimals={3} /></div>
                      <div className="chart-inline"><span className="mini-label">dPrec</span><LineChart points={rounds.map((r) => ({ x: r.round, y: r.defense?.defense_precision ?? 0 }))} color="#60a5fa" yMin={0} yMax={1} decimals={3} /></div>
                    </div>
                  ) : <div className="empty">No defense data (ADL disabled?)</div>}
                </div>
              </div>
              <div className="card">
                <div className="card-head"><h3>Adaptive Thresholds</h3></div>
                <div className="stack">
                  <div className="chart-inline"><span className="mini-label">T1</span><LineChart points={rounds.map((r) => ({ x: r.round, y: r.defense?.t1 ?? 0 }))} color="#f0b429" yMin={0} yMax={1} decimals={3} /></div>
                  <div className="chart-inline"><span className="mini-label">T2</span><LineChart points={rounds.map((r) => ({ x: r.round, y: r.defense?.t2 ?? 0 }))} color="#ef4444" yMin={0} yMax={1} decimals={3} /></div>
                </div>
              </div>
            </>
          ) : <div className="empty" style={{ padding: 48 }}>Select a run to see live results.</div>}
          <div className="card"><div className="card-head"><h3>Event Log</h3></div><LogPanel lines={logLines} /></div>
        </div>
      )}

      {tab === 'defense' && (
        <div className="tab-panel">
          {defense && lastRound ? (
            <>
              <div className="metric-grid">
                <MetricCard label="Policy" value={config.threshold_policy ?? 'adaptive'} cls="blue" />
                <MetricCard label="T1" value={(defense.t1 ?? 0).toFixed(3)} cls="amber" />
                <MetricCard label="T2" value={(defense.t2 ?? 0).toFixed(3)} cls="red" />
                <MetricCard label="Review Catch" value={((defense.review_catch_rate ?? 0) * 100).toFixed(0) + '%'} cls="blue" />
                <MetricCard label="Defense Recall" value={(defense.defense_recall ?? 0).toFixed(3)} cls="green" />
                <MetricCard label="Defense Prec" value={(defense.defense_precision ?? 0).toFixed(3)} cls="green" />
                <MetricCard label="Escape Rate" value={((defense.escape_rate ?? 0) * 100).toFixed(1) + '%'} cls="red" />
                <MetricCard label="False Block" value={((defense.false_block_rate ?? 0) * 100).toFixed(1) + '%'} cls="amber" />
              </div>
              <div className="row">
                <div className="grow card">
                  <div className="card-head"><h3>Risk Score Distribution</h3></div>
                  <div className="stack">
                    {defense.risk_components && Object.entries(defense.risk_components).map(([k, v]) => (
                      <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 12 }}>
                        <span style={{ width: 90, color: '#9bb8a3', fontWeight: 500 }}>{k}</span>
                        <div className="risk-bar-track">
                          <div className="risk-bar-fill" style={{ width: `${Math.min(100, (v as number) * 100)}%` }} />
                        </div>
                        <span style={{ fontFamily: "'JetBrains Mono', monospace", color: '#c8d8c8', width: 48, textAlign: 'right', fontSize: 11 }}>{(v as number).toFixed(3)}</span>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="grow card">
                  <div className="card-head"><h3>Decisions Per Round</h3></div>
                  <div className="stack">
                    <div className="chart-inline"><span className="mini-label">Allow</span><LineChart points={rounds.map((r) => ({ x: r.round, y: r.defense?.allow_count ?? 0 }))} color="#3fb950" /></div>
                    <div className="chart-inline"><span className="mini-label">Review</span><LineChart points={rounds.map((r) => ({ x: r.round, y: r.defense?.review_count ?? 0 }))} color="#f0b429" /></div>
                    <div className="chart-inline"><span className="mini-label">Block</span><LineChart points={rounds.map((r) => ({ x: r.round, y: r.defense?.block_count ?? 0 }))} color="#ef4444" /></div>
                  </div>
                </div>
              </div>
            </>
          ) : <div className="empty" style={{ padding: 48 }}>No defense data. Select a run with ADL enabled.</div>}
        </div>
      )}

      {tab === 'launch' && (
        <div className="tab-panel">
          <div className="card">
            <div className="card-head"><h3>Presets</h3></div>
            <div className="presets">{datasets.map((ds: any) => <button key={ds.key} className="preset-btn" onClick={() => { setConfig((c) => ({ ...c, ...ds.config })); setDescribe(ds.config.describe || ds.label); if (ds.config.adl_enabled !== undefined) setAdlEnabled(ds.config.adl_enabled); if (ds.config.threshold_policy) setPolicy(ds.config.threshold_policy); }}>{ds.label}</button>)}</div>
          </div>
          <div className="card">
            <div className="card-head"><h3>ADL Configuration</h3></div>
            <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 14 }}>
              <label className="inline-check">
                <input type="checkbox" checked={adlEnabled} onChange={(e) => setAdlEnabled(e.target.checked)} /> ADL Enabled
              </label>
              <div style={{ display: 'flex', gap: 6 }}>
                {['adaptive', 'fixed'].map((p) => <button key={p} className={`btn mini ${policy === p ? 'primary' : ''}`} onClick={() => setPolicy(p)}>{p}</button>)}
              </div>
            </div>
            <div className="form-grid">
              {ADL_CONFIG.map(([key, label]) => (
                <div className="field" key={key}><label>{label}</label>
                  <input className="input" type="number" step="any" value={config[key] ?? ''} onChange={(e) => setConfig((c) => ({ ...c, [key]: Number(e.target.value) }))} />
                </div>
              ))}
            </div>
          </div>
          <div className="card">
            <div className="card-head"><h3>Simulation Config</h3></div>
            <div className="field" style={{ marginBottom: 14 }}>
              <label>Describe</label>
              <input className="input" style={{ width: '100%' }} value={describe} onChange={(e) => setDescribe(e.target.value)} placeholder="Name this experiment..." />
            </div>
            <div className="form-grid">
              {ADL_FIELDS.map(([key, label]) => (
                <div className="field" key={key}><label>{label}</label>
                  <input className="input" type="number" step="any" value={config[key] ?? ''} onChange={(e) => setConfig((c) => ({ ...c, [key]: Number(e.target.value) }))} />
                </div>
              ))}
            </div>
          </div>
          <button className="btn primary" onClick={launch} disabled={launching}>{launching ? 'Launching...' : '\u25B6 Launch'}</button>
          <div className="card" style={{ marginTop: 16 }}>
            <div className="card-head"><h3>Recent Runs</h3></div>
            <DataTable columns={['ID', 'Describe', 'ADL', 'State', 'Actions']} rows={history.map((h) => ({ ID: h.id?.slice(0, 8), Describe: (h.describe || '').slice(0, 25), ADL: h.adl_enabled ? (h.threshold_policy || 'on') : 'off', State: h.state, Actions: '' }))}
              renderCell={(col, row, i) => { if (col === 'State') return <span className={`badge badge-${row.State}`}>{row.State}</span>; if (col === 'Actions') return <button className="btn mini" onClick={() => { setSelected(history[i].id); setTab('live'); }}>Open</button>; return row[col]; }} />
          </div>
        </div>
      )}

      {tab === 'history' && (
        <div className="tab-panel">
          <div className="card">
            <div className="card-head"><h3>Run History</h3></div>
            <DataTable columns={['ID', 'Describe', 'ADL', 'State', 'Started', 'Actions']} rows={history.map((h) => ({ ID: h.id?.slice(0, 8), Describe: (h.describe || '').slice(0, 25), ADL: h.adl_enabled ? (h.threshold_policy || 'on') : 'off', State: h.state, Started: h.started_at ? new Date(h.started_at * 1000).toLocaleTimeString() : '--', Actions: '' }))}
              renderCell={(col, row, i) => { if (col === 'State') return <span className={`badge badge-${row.State}`}>{row.State}</span>; if (col === 'Actions') return <button className="btn mini" onClick={() => { setSelected(history[i].id); setTab('live'); }}>Open</button>; return row[col]; }} />
          </div>
        </div>
      )}

      {tab === 'about' && (
        <div className="tab-panel">
          <div className="card">
            <h3 style={{ marginBottom: 12, color: '#e4ede6' }}>Adaptive Defensive Layer (ADL)</h3>
            <p style={{ fontSize: 13, lineHeight: 1.8, color: '#c8d8c8' }}>
              A <strong style={{ color: '#6ec47e' }}>risk-based decision layer</strong> that sits between the fraud detector and the marketplace.
              Each account receives a risk score computed from 5 components: prediction confidence (Pf),
              graph centrality (C), referral ring membership (S), transaction velocity (V), and trust history (A).
              Based on two thresholds (T1 for review, T2 for block), accounts are classified as Allow / Review / Block.
              Thresholds adapt each round using an exponential moving average to balance escape rate vs false-block rate.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
