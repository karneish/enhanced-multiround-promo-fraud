import React, { useState, useEffect, useCallback, useRef } from 'react';
import { genApi } from '../api';
import { useSSE } from '../hooks/useSSE';
import MetricCard from '../components/MetricCard';
import LineChart from '../components/LineChart';
import BarChart from '../components/BarChart';
import LogPanel from '../components/LogPanel';
import RoundTracker from '../components/RoundTracker';
import DataTable from '../components/DataTable';

type Tab = 'live' | 'launch' | 'history' | 'about';

const GEN_FIELDS: [string, string][] = [
  ['rounds', 'Rounds'], ['base_accounts', 'Base Accounts'], ['initial_fraud', 'Initial Fraud'],
  ['genuine_per_round', 'Genuine / Round'], ['fraud_per_round', 'Fraud / Round'],
  ['seed', 'Seed'], ['supervised_ratio', 'Supervised Ratio'], ['budget_pos', 'Review Budget +'],
  ['budget_neg', 'Review Budget -'], ['gan_epochs', 'GAN Epochs'],
  ['gan_noise_dim', 'Noise Dim'], ['gan_hidden', 'Hidden Size'],
  ['diversity', 'Diversity'], ['conn_coef', 'Conn Coef'], ['ring_ratio', 'Ring Ratio'],
  ['profile_window', 'Profile Window'],
];

const STRATEGY_COLORS: Record<string, string> = {
  fake_identity: '#3fb950', referral_farming: '#f0b429', device_spray: '#d4a017', vpn_hop: '#60a5fa', quiet_sampler: '#a78bfa',
};

export default function GeneratorPage() {
  const [tab, setTab] = useState<Tab>('live');
  const [schema, setSchema] = useState<any>(null);
  const [datasets, setDatasets] = useState<any[]>([]);
  const [history, setHistory] = useState<any[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [rounds, setRounds] = useState<any[]>([]);
  const [config, setConfig] = useState<Record<string, any>>({});
  const [describe, setDescribe] = useState('');
  const [genMode, setGenMode] = useState('intelligent');
  const [genType, setGenType] = useState('GAN');
  const [launching, setLaunching] = useState(false);
  const logRef = useRef<{ t: number; text: string }[]>([]);
  const [logLines, setLogLines] = useState<{ t: number; text: string }[]>([]);
  const sseUrl = selected ? genApi.streamUrl(selected) : null;
  const { events, connected } = useSSE(sseUrl);
  const lastIdxRef = useRef(0);

  useEffect(() => { genApi.schema().then(setSchema).catch(() => {}); genApi.datasets().then(setDatasets).catch(() => {}); genApi.history().then(setHistory).catch(() => {}); }, []);
  useEffect(() => { if (schema) setConfig(schema.defaults || {}); }, [schema]);

  useEffect(() => {
    const start = lastIdxRef.current;
    if (start >= events.length) return;
    lastIdxRef.current = events.length;
    for (let i = start; i < events.length; i++) {
      const ev = events[i];
      if (ev.type === 'round_result') setRounds((p) => [...p, ev]);
      else if (ev.type === 'log') { logRef.current = [...logRef.current.slice(-200), { t: ev.t ?? 0, text: ev.text ?? '' }]; setLogLines([...logRef.current]); }
      else if (ev.type === 'state' && ev.finished) { genApi.history().then(setHistory).catch(() => {}); }
    }
  }, [events, selected]);

  const launch = useCallback(async () => {
    setLaunching(true);
    try {
      const body = { ...config, describe: describe || 'Generator experiment', generator_mode: genMode, gen_type: genType };
      const res = await genApi.launch(body);
      setSelected(res.id); setRounds([]); logRef.current = []; setLogLines([]); setTab('live'); lastIdxRef.current = 0;
    } catch (e: any) { alert('Launch failed: ' + e.message); }
    finally { setLaunching(false); }
  }, [config, describe, genMode, genType]);

  const lastRound = rounds[rounds.length - 1];

  return (
    <div>
      <div className="page-header">
        <h2>Intelligent Fraud Generator</h2>
        <p>Adaptive fraud simulation with GAN/probabilistic generation and replay baseline</p>
      </div>
      <div className="tab-bar">
        {(['live', 'launch', 'history', 'about'] as Tab[]).map((t) => (
          <button key={t} className={`tab-btn ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
            {t === 'live' ? '\u25CF' : t === 'launch' ? '\u25B2' : t === 'history' ? '\u25BC' : '\u2139'} {t.charAt(0).toUpperCase() + t.slice(1)}
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
                <MetricCard label="Missed Fraud" value={lastRound.missed ?? '--'} cls={lastRound.missed > 0 ? 'red' : 'green'} />
                <MetricCard label="Macro-F1" value={(lastRound.metrics?.f1 ?? 0).toFixed(3)} cls="green" />
                <MetricCard label="AUC" value={(lastRound.metrics?.auc ?? 0).toFixed(3)} cls="purple" />
                <MetricCard label="Recall" value={(lastRound.metrics?.rec ?? 0).toFixed(3)} cls="green" />
                <MetricCard label="Precision" value={(lastRound.metrics?.prec ?? 0).toFixed(3)} cls="amber" />
                <MetricCard label="Feat Diversity" value={(lastRound.gen?.gen_feat_div ?? 0).toFixed(3)} cls="amber" />
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
                  <div className="card-head"><h3>Generator Diagnostics</h3></div>
                  <div className="stack">
                    <div className="chart-inline"><span className="mini-label">Diversity</span><LineChart points={rounds.map((r) => ({ x: r.round, y: r.gen?.gen_feat_div ?? 0 }))} color="#a78bfa" /></div>
                    <div className="chart-inline"><span className="mini-label">Shift</span><LineChart points={rounds.map((r) => ({ x: r.round, y: r.gen?.gen_feat_shift ?? 0 }))} color="#f0b429" /></div>
                    <div className="chart-inline"><span className="mini-label">Ring Ratio</span><LineChart points={rounds.map((r) => ({ x: r.round, y: r.gen?.gen_ring_ratio ?? 0 }))} color="#d4a017" yMin={0} yMax={1} decimals={2} /></div>
                  </div>
                </div>
              </div>
              {lastRound.gen?.gen_strategies && (
                <div className="card">
                  <div className="card-head"><h3>Evolved Strategies</h3></div>
                  <BarChart items={lastRound.gen.gen_strategies} color="#3fb950" />
                </div>
              )}
            </>
          ) : <div className="empty" style={{ padding: 48 }}>Select a run to see live results.</div>}
          <div className="card"><div className="card-head"><h3>Event Log</h3></div><LogPanel lines={logLines} /></div>
        </div>
      )}

      {tab === 'launch' && (
        <div className="tab-panel">
          <div className="card">
            <div className="card-head"><h3>Presets</h3></div>
            <div className="presets">{datasets.map((ds: any) => <button key={ds.key} className="preset-btn" onClick={() => { setConfig((c) => ({ ...c, ...ds.config })); setDescribe(ds.config.describe || ds.label); }}>{ds.label}</button>)}</div>
          </div>
          <div className="card">
            <div className="card-head"><h3>Generator Mode</h3></div>
            <div style={{ display: 'flex', gap: 8, marginBottom: 14 }}>
              {['intelligent', 'replay'].map((m) => <button key={m} className={`btn ${genMode === m ? 'primary' : ''}`} onClick={() => setGenMode(m)}>{m}</button>)}
              <span style={{ width: 10 }} />
              {['GAN', 'PROB'].map((t) => <button key={t} className={`btn ${genType === t ? 'primary' : ''}`} onClick={() => setGenType(t)}>{t}</button>)}
            </div>
          </div>
          <div className="card">
            <div className="card-head"><h3>Configuration</h3></div>
            <div className="field" style={{ marginBottom: 14 }}>
              <label>Describe</label>
              <input className="input" style={{ width: '100%' }} value={describe} onChange={(e) => setDescribe(e.target.value)} placeholder="Name this experiment..." />
            </div>
            <div className="form-grid">
              {GEN_FIELDS.map(([key, label]) => (
                <div className="field" key={key}><label>{label}</label>
                  <input className="input" type="number" step="any" value={config[key] ?? ''} onChange={(e) => setConfig((c) => ({ ...c, [key]: Number(e.target.value) }))} />
                </div>
              ))}
            </div>
          </div>
          <button className="btn primary" onClick={launch} disabled={launching}>{launching ? 'Launching...' : '\u25B6 Launch'}</button>
          <div className="card" style={{ marginTop: 16 }}>
            <div className="card-head"><h3>Recent Runs</h3></div>
            <DataTable columns={['ID', 'Describe', 'Mode', 'State', 'Actions']} rows={history.map((h) => ({ ID: h.id?.slice(0, 8), Describe: (h.describe || '').slice(0, 25), Mode: h.generator_mode || '--', State: h.state, Actions: '' }))}
              renderCell={(col, row, i) => { if (col === 'State') return <span className={`badge badge-${row.State}`}>{row.State}</span>; if (col === 'Actions') return <button className="btn mini" onClick={() => { setSelected(history[i].id); setTab('live'); }}>Open</button>; return row[col]; }} />
          </div>
        </div>
      )}

      {tab === 'history' && (
        <div className="tab-panel">
          <div className="card">
            <div className="card-head"><h3>Run History</h3></div>
            <DataTable columns={['ID', 'Describe', 'Mode', 'State', 'Started', 'Actions']} rows={history.map((h) => ({ ID: h.id?.slice(0, 8), Describe: (h.describe || '').slice(0, 25), Mode: h.generator_mode || '--', State: h.state, Started: h.started_at ? new Date(h.started_at * 1000).toLocaleTimeString() : '--', Actions: '' }))}
              renderCell={(col, row, i) => { if (col === 'State') return <span className={`badge badge-${row.State}`}>{row.State}</span>; if (col === 'Actions') return <button className="btn mini" onClick={() => { setSelected(history[i].id); setTab('live'); }}>Open</button>; return row[col]; }} />
          </div>
        </div>
      )}

      {tab === 'about' && (
        <div className="tab-panel">
          <div className="card">
            <h3 style={{ marginBottom: 12, color: '#e4ede6' }}>Intelligent Fraud Generator</h3>
            <p style={{ fontSize: 13, lineHeight: 1.8, color: '#c8d8c8' }}>
              An adaptive fraud simulation engine that uses either a <strong style={{ color: '#6ec47e' }}>GAN</strong> or <strong style={{ color: '#6ec47e' }}>probabilistic resampling</strong> approach
              to generate evolved fraud strategies each round. The generator learns from missed (false-negative) fraud accounts
              and adapts its feature profiles, referral patterns, device reuse, and IP behavior to evade the detector.
              A <strong style={{ color: '#6ec47e' }}>replay baseline</strong> mode is also available that simply replays original fraud templates.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
