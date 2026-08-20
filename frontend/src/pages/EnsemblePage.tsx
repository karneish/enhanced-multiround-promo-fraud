import React, { useState, useEffect, useCallback, useRef } from 'react';
import { ensApi } from '../api';
import { useSSE } from '../hooks/useSSE';
import MetricCard from '../components/MetricCard';
import LineChart from '../components/LineChart';
import LogPanel from '../components/LogPanel';
import RoundTracker from '../components/RoundTracker';
import DataTable from '../components/DataTable';

const MODEL_COLORS: Record<string, string> = {
  XGBoost: '#60a5fa', RandomForest: '#3fb950', ExtraTrees: '#f0b429',
  HistGradientBoosting: '#a78bfa', LogisticRegression: '#f472b6',
};
const MODEL_NAMES = Object.keys(MODEL_COLORS);

type Tab = 'live' | 'launch' | 'models' | 'history' | 'about';

const CONFIG_FIELDS: [string, string, string][] = [
  ['rounds', 'Rounds', 'number'], ['base_accounts', 'Base Accounts', 'number'],
  ['initial_fraud', 'Initial Fraud', 'number'], ['genuine_per_round', 'Genuine / Round', 'number'],
  ['fraud_per_round', 'Fraud / Round', 'number'], ['supervised_ratio', 'Supervised Ratio', 'number'],
  ['budget_pos', 'Review Budget +', 'number'], ['budget_neg', 'Review Budget -', 'number'],
  ['seed', 'Seed', 'number'],
];

export default function EnsemblePage() {
  const [tab, setTab] = useState<Tab>('live');
  const [schema, setSchema] = useState<any>(null);
  const [datasets, setDatasets] = useState<any[]>([]);
  const [history, setHistory] = useState<any[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [report, setReport] = useState<any>(null);
  const [rounds, setRounds] = useState<any[]>([]);
  const [config, setConfig] = useState<Record<string, any>>({});
  const [describe, setDescribe] = useState('');
  const [launching, setLaunching] = useState(false);
  const logRef = useRef<{ t: number; text: string }[]>([]);
  const [logLines, setLogLines] = useState<{ t: number; text: string }[]>([]);
  const sseUrl = selected ? ensApi.streamUrl(selected) : null;
  const { events, connected } = useSSE(sseUrl);
  const lastIdxRef = useRef(0);

  useEffect(() => {
    ensApi.schema().then(setSchema).catch(() => {});
    ensApi.datasets().then(setDatasets).catch(() => {});
    ensApi.history().then(setHistory).catch(() => {});
  }, []);

  useEffect(() => {
    if (!schema) return;
    setConfig(schema.defaults || {});
  }, [schema]);

  useEffect(() => {
    const start = lastIdxRef.current;
    if (start >= events.length) return;
    lastIdxRef.current = events.length;
    for (let i = start; i < events.length; i++) {
      const ev = events[i];
      if (ev.type === 'round_result') {
        setRounds((prev) => [...prev, ev]);
      } else if (ev.type === 'log') {
        logRef.current = [...logRef.current.slice(-200), { t: ev.t ?? 0, text: ev.text ?? '' }];
        setLogLines([...logRef.current]);
      } else if (ev.type === 'state' && ev.finished) {
        if (selected) ensApi.report(selected).then(setReport).catch(() => {});
        ensApi.history().then(setHistory).catch(() => {});
      }
    }
  }, [events, selected]);

  const launch = useCallback(async () => {
    setLaunching(true);
    try {
      const body = { ...config, describe: describe || 'Ensemble experiment' };
      const res = await ensApi.launch(body);
      setSelected(res.id);
      setRounds([]); logRef.current = []; setLogLines([]); setReport(null); lastIdxRef.current = 0;
      setTab('live');
    } catch (e: any) { alert('Launch failed: ' + e.message); }
    finally { setLaunching(false); }
  }, [config, describe]);

  const lastRound = rounds[rounds.length - 1];

  let poolTP = 0, poolFP = 0, poolTN = 0, poolFN = 0;
  let overallAUC = 0, overallRec = 0;
  if (rounds.length > 0) {
    for (const r of rounds) {
      poolTP += r.ensemble?.tp ?? 0;
      poolFP += r.ensemble?.fp ?? 0;
      poolTN += r.ensemble?.tn ?? 0;
      poolFN += r.ensemble?.fn ?? 0;
      overallAUC += r.ensemble?.auc ?? 0;
      overallRec += r.ensemble?.rec ?? 0;
    }
    overallAUC /= rounds.length;
    overallRec /= rounds.length;
  }
  const total = poolTP + poolFP + poolTN + poolFN;
  const pooledAccuracy = total > 0 ? (poolTP + poolTN) / total : 0;
  const pooledPrecision = (poolTP + poolFP) > 0 ? poolTP / (poolTP + poolFP) : 0;
  const pooledRecall = (poolTP + poolFN) > 0 ? poolTP / (poolTP + poolFN) : 0;
  const pooledF1 = (pooledPrecision + pooledRecall) > 0
    ? 2 * (pooledPrecision * pooledRecall) / (pooledPrecision + pooledRecall)
    : 0;
  const pooledF1Neg = (poolTN + poolFP) > 0 && (poolTN + poolFN) > 0
    ? 2 * (poolTN / (poolTN + poolFP) * (poolTN / (poolTN + poolFN))) /
      ((poolTN / (poolTN + poolFP)) + (poolTN / (poolTN + poolFN)))
    : 0;
  const pooledMacroF1 = rounds.length > 0 ? (pooledF1 + pooledF1Neg) / 2 : 0;

  const avgModelF1: Record<string, number> = {};
  const modelWins: Record<string, number> = {};
  MODEL_NAMES.forEach((n) => { avgModelF1[n] = 0; modelWins[n] = 0; });
  const roundLeaders: { round: number; model: string; f1: number; ensembleF1: number }[] = [];
  for (const r of rounds) {
    let bestName = '';
    let bestF1 = -1;
    for (const name of MODEL_NAMES) {
      const f1 = r.per_model?.[name]?.f1 ?? 0;
      avgModelF1[name] += f1;
      if (f1 > bestF1) { bestF1 = f1; bestName = name; }
    }
    if (bestName) modelWins[bestName]++;
    roundLeaders.push({ round: r.round, model: bestName, f1: bestF1, ensembleF1: r.ensemble?.f1 ?? 0 });
  }
  if (rounds.length > 0) {
    MODEL_NAMES.forEach((n) => { avgModelF1[n] /= rounds.length; });
  }
  const bestModelOverall = Object.entries(avgModelF1).sort((a, b) => b[1] - a[1])[0];

  return (
    <div>
      <div className="page-header">
        <h2>Adaptive Ensemble Detector</h2>
        <p>5-model ensemble with Adaptive Detector Score (ADS) weighting</p>
      </div>
      <div className="tab-bar">
        {(['live', 'launch', 'models', 'history', 'about'] as Tab[]).map((t) => (
          <button key={t} className={`tab-btn ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
            {t === 'live' ? '\u25CF' : t === 'launch' ? '\u25B2' : t === 'models' ? '\u25A0' : t === 'history' ? '\u25BC' : '\u2139'} {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {tab === 'live' && (
        <div className="tab-panel">
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
            <select className="input" style={{ width: 280 }} value={selected ?? ''} onChange={(e) => {
              const id = e.target.value || null;
              setSelected(id); setRounds([]); logRef.current = []; setLogLines([]); setReport(null); lastIdxRef.current = 0;
            }}>
              <option value="">Select a run...</option>
              {history.map((h) => <option key={h.id} value={h.id}>{h.id.slice(0, 8)} ... {(h.describe || '').slice(0, 25)}</option>)}
            </select>
            <span className={`sse-pill ${connected ? 'on' : ''}`}><span className="dot" />{connected ? 'Streaming' : 'Disconnected'}</span>
          </div>
          {lastRound && (
            <>
              <RoundTracker total={config.rounds ?? 5} current={lastRound.round ?? 0} />
              <div className="metric-grid">
                <MetricCard label="Round" value={`#${lastRound.round}`} cls="blue" />
                <MetricCard label="Nodes" value={lastRound.num_nodes ?? '--'} cls="blue" />
                <MetricCard label="Missed Fraud" value={lastRound.missed_fraud ?? '--'} cls={lastRound.missed_fraud > 0 ? 'red' : 'green'} />
                <MetricCard label="Ensemble F1" value={(lastRound.ensemble?.f1 ?? 0).toFixed(3)} cls="purple" />
                <MetricCard label="Ensemble AUC" value={(lastRound.ensemble?.auc ?? 0).toFixed(3)} cls="purple" />
                <MetricCard label="Recall" value={(lastRound.ensemble?.rec ?? 0).toFixed(3)} cls="green" />
                <MetricCard label="Precision" value={(lastRound.ensemble?.prec ?? 0).toFixed(3)} cls="amber" />
                <MetricCard label="Supervised" value={lastRound.supervised_count ?? '--'} cls="blue" />
              </div>
              <div className="metric-grid wide">
                <MetricCard label="Final Accuracy" value={pooledAccuracy.toFixed(4)} cls="green" sub={`${poolTP + poolTN} / ${total} correct`} />
                <MetricCard label="Final F1 (Macro)" value={pooledMacroF1.toFixed(4)} cls="purple" sub="pooled confusion matrix" />
                <MetricCard label="Final Precision" value={pooledPrecision.toFixed(4)} cls="amber" sub={`TP: ${poolTP}  FP: ${poolFP}`} />
                <MetricCard label="Final Recall" value={pooledRecall.toFixed(4)} cls="blue" sub={`TP: ${poolTP}  FN: ${poolFN}`} />
                <MetricCard label="Avg AUC" value={overallAUC.toFixed(4)} cls="purple" sub={`${rounds.length} rounds`} />
                <MetricCard label="Best Model (Avg)" value={bestModelOverall ? bestModelOverall[0] : '--'} cls="amber" sub={bestModelOverall ? `avg F1: ${bestModelOverall[1].toFixed(4)}` : ''} />
              </div>
              <div className="metric-grid wide">
                <MetricCard label="Top Round F1" value={rounds.length > 0 ? Math.max(...rounds.map((r) => r.ensemble?.f1 ?? 0)).toFixed(4) : '--'} cls="green" />
                <MetricCard label="Win Counts" value={Object.entries(modelWins).filter(([,v]) => v > 0).map(([k, v]) => `${k.slice(0, 4)}:${v}`).join(' ') || '--'} cls="blue" />
                <MetricCard label="Confusion Matrix" value={`TP:${poolTP} FP:${poolFP} TN:${poolTN} FN:${poolFN}`} cls="blue" sub="aggregated across all rounds" />
              </div>
              <div className="row">
                <div className="grow card">
                  <div className="card-head"><h3>Ensemble Performance</h3></div>
                  <div className="stack">
                    <div className="chart-inline"><span className="mini-label">F1</span><LineChart points={rounds.map((r) => ({ x: r.round, y: r.ensemble?.f1 ?? 0 }))} color="#60a5fa" fill /></div>
                    <div className="chart-inline"><span className="mini-label">AUC</span><LineChart points={rounds.map((r) => ({ x: r.round, y: r.ensemble?.auc ?? 0 }))} color="#a78bfa" /></div>
                    <div className="chart-inline"><span className="mini-label">Recall</span><LineChart points={rounds.map((r) => ({ x: r.round, y: r.ensemble?.rec ?? 0 }))} color="#3fb950" /></div>
                    <div className="chart-inline"><span className="mini-label">Prec</span><LineChart points={rounds.map((r) => ({ x: r.round, y: r.ensemble?.prec ?? 0 }))} color="#f0b429" /></div>
                  </div>
                </div>
                <div className="grow card">
                  <div className="card-head"><h3>Model Weights Over Rounds</h3></div>
                  <div className="stack">
                    {MODEL_NAMES.map((name) => (
                      <div className="chart-inline" key={name}>
                        <span className="mini-label" style={{ color: MODEL_COLORS[name] }}>{name.slice(0, 6)}</span>
                        <LineChart points={rounds.map((r) => ({ x: r.round, y: r.weights?.[name] ?? 0 }))} color={MODEL_COLORS[name]} yMin={0} yMax={1} decimals={2} />
                      </div>
                    ))}
                  </div>
                </div>
                <div className="grow card">
                  <div className="card-head"><h3>Per-Model F1 Over Rounds</h3></div>
                  <div className="stack">
                    {MODEL_NAMES.map((name) => (
                      <div className="chart-inline" key={name}>
                        <span className="mini-label" style={{ color: MODEL_COLORS[name] }}>{name.slice(0, 6)}</span>
                        <LineChart points={rounds.map((r) => ({ x: r.round, y: r.per_model?.[name]?.f1 ?? 0 }))} color={MODEL_COLORS[name]} yMin={0} yMax={1} decimals={3} />
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              <div className="card">
                <div className="card-head"><h3>Model Comparison</h3></div>
                <DataTable
                  columns={['Rank', 'Model', 'F1', 'Recall', 'Precision', 'AUC', 'ADS Score', 'Weight', 'Wins']}
                  rows={(() => {
                    const ranked = MODEL_NAMES.map((name) => ({
                      name,
                      f1: lastRound.per_model?.[name]?.f1 ?? 0,
                      recall: lastRound.per_model?.[name]?.recall ?? 0,
                      precision: lastRound.per_model?.[name]?.precision ?? 0,
                      auc: lastRound.per_model?.[name]?.auc ?? 0,
                      ads: lastRound.scores?.[name] ?? 0,
                      weight: lastRound.weights?.[name] ?? 0,
                      avgF1: avgModelF1[name],
                      wins: modelWins[name],
                    })).sort((a, b) => b.f1 - a.f1);
                    return ranked.map((r, i) => ({
                      Rank: `#${i + 1}`,
                      Model: r.name,
                      F1: r.f1.toFixed(4),
                      Recall: r.recall.toFixed(4),
                      Precision: r.precision.toFixed(4),
                      AUC: r.auc.toFixed(4),
                      'ADS Score': r.ads.toFixed(4),
                      Weight: r.weight.toFixed(4),
                      Wins: `${r.wins}/${rounds.length}`,
                      _isTop: i === 0,
                      _avgF1: r.avgF1,
                    }));
                  })()}
                  renderCell={(col, row: any) => {
                    const v = row[col];
                    if (col === 'Rank') {
                      const color = row._isTop ? '#f0b429' : '#6b8a6b';
                      return <span style={{ color, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace" }}>{row._isTop ? '\u2605 ' : ''}{v}</span>;
                    }
                    if (col === 'Model') return <span style={{ color: MODEL_COLORS[v] || '#c8d8c8', fontWeight: row._isTop ? 800 : 600 }}>{v}{row._isTop ? ' \u2605' : ''}</span>;
                    if (col === 'F1') {
                      const isBest = row._isTop;
                      return <span style={{ color: isBest ? '#f0b429' : '#c8d8c8', fontWeight: isBest ? 800 : 400, fontFamily: "'JetBrains Mono', monospace" }}>{v}</span>;
                    }
                    if (col === 'Wins') return <span style={{ color: row._isTop ? '#3fb950' : '#6b8a6b', fontFamily: "'JetBrains Mono', monospace" }}>{v}</span>;
                    if (col === 'ADS Score') return <span style={{ color: '#f0b429', fontFamily: "'JetBrains Mono', monospace" }}>{v}</span>;
                    if (col === 'Weight') return <span style={{ color: '#60a5fa', fontFamily: "'JetBrains Mono', monospace" }}>{v}</span>;
                    return v;
                  }}
                />
              </div>
              {roundLeaders.length > 0 && (
                <div className="card">
                  <div className="card-head"><h3>Round Leaders</h3></div>
                  <DataTable
                    columns={['Round', 'Ensemble F1', 'Best Model', 'Best Model F1', 'Delta']}
                    rows={roundLeaders.map((rl) => ({
                      Round: `R${rl.round}`,
                      'Ensemble F1': rl.ensembleF1.toFixed(4),
                      'Best Model': rl.model,
                      'Best Model F1': rl.f1.toFixed(4),
                      Delta: (rl.f1 - rl.ensembleF1).toFixed(4),
                    }))}
                    renderCell={(col, row: any) => {
                      const v = row[col];
                      if (col === 'Round') return <span style={{ color: '#60a5fa', fontWeight: 700, fontFamily: "'JetBrains Mono', monospace" }}>{v}</span>;
                      if (col === 'Best Model') {
                        const name = roundLeaders.find((rl) => `R${rl.round}` === row.Round)?.model;
                        return <span style={{ color: MODEL_COLORS[name || ''] || '#c8d8c8', fontWeight: 700 }}>{v} <span style={{ color: '#f0b429' }}>{'\u2605'}</span></span>;
                      }
                      if (col === 'Best Model F1') return <span style={{ color: '#3fb950', fontWeight: 700, fontFamily: "'JetBrains Mono', monospace" }}>{v}</span>;
                      if (col === 'Delta') {
                        const num = parseFloat(v);
                        const color = num > 0 ? '#3fb950' : num < 0 ? '#f472b6' : '#6b8a6b';
                        return <span style={{ color, fontFamily: "'JetBrains Mono', monospace" }}>{num > 0 ? '+' : ''}{v}</span>;
                      }
                      return v;
                    }}
                  />
                </div>
              )}
            </>
          )}
          {!lastRound && <div className="empty" style={{ padding: 48 }}>Select a run or launch a new experiment to see live results.</div>}
          <div className="card"><div className="card-head"><h3>Event Log</h3></div><LogPanel lines={logLines} /></div>
        </div>
      )}

      {tab === 'launch' && (
        <div className="tab-panel">
          <div className="card">
            <div className="card-head"><h3>Presets</h3></div>
            <div className="presets">
              {datasets.map((ds: any) => (
                <button key={ds.label} className="preset-btn" onClick={() => { setConfig((c) => ({ ...c, ...ds.config })); setDescribe(ds.config.describe || ds.label); }}>{ds.label}</button>
              ))}
            </div>
          </div>
          <div className="card">
            <div className="card-head"><h3>Configuration</h3></div>
            <div className="field" style={{ marginBottom: 14 }}>
              <label>Describe</label>
              <input className="input" style={{ width: '100%' }} value={describe} onChange={(e) => setDescribe(e.target.value)} placeholder="Name this experiment..." />
            </div>
            <div className="form-grid">
              {CONFIG_FIELDS.map(([key, label, type]) => (
                <div className="field" key={key}>
                  <label>{label}</label>
                  <input className="input" type={type} step="any" value={config[key] ?? ''} onChange={(e) => setConfig((c) => ({ ...c, [key]: Number(e.target.value) }))} />
                </div>
              ))}
            </div>
          </div>
          <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
            <button className="btn primary" onClick={launch} disabled={launching}>{launching ? 'Launching...' : '\u25B6 Launch Experiment'}</button>
          </div>
          <div className="card" style={{ marginTop: 16 }}>
            <div className="card-head"><h3>Recent Runs</h3></div>
            <DataTable
              columns={['ID', 'Describe', 'Rounds', 'State', 'Actions']}
              rows={history.map((h) => ({ ID: h.id?.slice(0, 8), Describe: (h.describe || '').slice(0, 30), Rounds: h.rounds ?? '--', State: h.state, Actions: '' }))}
              renderCell={(col, row, i) => {
                if (col === 'State') return <span className={`badge badge-${row.State}`}>{row.State}</span>;
                if (col === 'Actions') return <button className="btn mini" onClick={() => { setSelected(history[i].id); setTab('live'); }}>Open</button>;
                return row[col];
              }}
            />
          </div>
        </div>
      )}

      {tab === 'models' && (
        <div className="tab-panel">
          <div className="card">
            <div className="card-head"><h3>Ensemble Architecture</h3></div>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10, padding: '20px 0' }}>
              <div className="arch-node" style={{ background: 'rgba(96,165,250,0.08)', border: '1px solid rgba(96,165,250,0.2)' }}>
                <div style={{ fontWeight: 700, color: '#60a5fa', fontSize: 14 }}>TPNE Embedder</div>
                <div style={{ fontSize: 10, color: '#9bb8a3', marginTop: 2 }}>17-D feature vector</div>
              </div>
              <div className="arch-arrow">{'\u2193'}</div>
              <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', justifyContent: 'center' }}>
                {MODEL_NAMES.map((name) => (
                  <div key={name} className="arch-node" style={{ background: `${MODEL_COLORS[name]}08`, border: `1px solid ${MODEL_COLORS[name]}30`, minWidth: 120 }}>
                    <div style={{ fontWeight: 700, color: MODEL_COLORS[name], fontSize: 12 }}>{name}</div>
                    {lastRound && <div style={{ fontSize: 10, color: '#f0b429', fontFamily: "'JetBrains Mono', monospace", marginTop: 3 }}>w={lastRound.weights?.[name]?.toFixed(3) ?? '--'}</div>}
                  </div>
                ))}
              </div>
              <div className="arch-arrow">{'\u2193'}</div>
              <div className="arch-node" style={{ background: 'rgba(167,139,250,0.08)', border: '1px solid rgba(167,139,250,0.2)' }}>
                <div style={{ fontWeight: 700, color: '#a78bfa', fontSize: 14 }}>Adaptive Detector Score</div>
                <div style={{ fontSize: 10, color: '#9bb8a3', marginTop: 2 }}>F1 + Recall + Stability + Historical</div>
              </div>
              <div className="arch-arrow">{'\u2193'}</div>
              <div className="arch-node" style={{ background: 'rgba(240,180,41,0.08)', border: '1px solid rgba(240,180,41,0.2)' }}>
                <div style={{ fontWeight: 700, color: '#f0b429', fontSize: 14 }}>Weighted Ensemble</div>
                <div style={{ fontSize: 10, color: '#9bb8a3', marginTop: 2 }}>Dynamic model reweighting per round</div>
              </div>
              <div className="arch-arrow">{'\u2193'}</div>
              <div className="arch-node" style={{ background: 'rgba(63,185,80,0.08)', border: '1px solid rgba(63,185,80,0.2)' }}>
                <div style={{ fontWeight: 700, color: '#3fb950', fontSize: 14 }}>Fraud Probability</div>
              </div>
            </div>
          </div>
          {lastRound && (
            <div className="card">
              <div className="card-head"><h3>Per-Model Metrics (Latest Round)</h3></div>
              <DataTable
                columns={['Model', 'F1', 'Avg F1', 'Wins', 'Recall', 'Precision', 'AUC', 'ADS Score', 'Weight']}
                rows={MODEL_NAMES.map((name) => ({
                  Model: name, F1: (lastRound.per_model?.[name]?.f1 ?? 0).toFixed(4),
                  'Avg F1': avgModelF1[name].toFixed(4),
                  Wins: `${modelWins[name]}/${rounds.length}`,
                  Recall: (lastRound.per_model?.[name]?.recall ?? 0).toFixed(4),
                  Precision: (lastRound.per_model?.[name]?.precision ?? 0).toFixed(4),
                  AUC: (lastRound.per_model?.[name]?.auc ?? 0).toFixed(4),
                  'ADS Score': (lastRound.scores?.[name] ?? 0).toFixed(4),
                  Weight: (lastRound.weights?.[name] ?? 0).toFixed(4),
                }))}
                renderCell={(col, row) => {
                  if (col === 'Model') return <span style={{ color: MODEL_COLORS[row[col]] || '#c8d8c8', fontWeight: 600 }}>{row[col]}</span>;
                  if (col === 'ADS Score') return <span style={{ color: '#f0b429', fontFamily: "'JetBrains Mono', monospace" }}>{row[col]}</span>;
                  if (col === 'Weight') return <span style={{ color: '#60a5fa', fontFamily: "'JetBrains Mono', monospace" }}>{row[col]}</span>;
                  if (col === 'Avg F1') return <span style={{ color: '#3fb950', fontWeight: 700, fontFamily: "'JetBrains Mono', monospace" }}>{row[col]}</span>;
                  if (col === 'Wins') return <span style={{ color: row[col].startsWith('0/') ? '#6b8a6b' : '#f0b429', fontFamily: "'JetBrains Mono', monospace" }}>{row[col]}</span>;
                  return row[col];
                }}
              />
            </div>
          )}
        </div>
      )}

      {tab === 'history' && (
        <div className="tab-panel">
          <div className="card">
            <div className="card-head"><h3>Run History</h3></div>
            <DataTable
              columns={['ID', 'Describe', 'Rounds', 'State', 'Actions']}
              rows={history.map((h) => ({ ID: h.id?.slice(0, 8), Describe: (h.describe || '').slice(0, 30), Rounds: h.rounds ?? '--', State: h.state, Actions: '' }))}
              renderCell={(col, row, i) => {
                if (col === 'State') return <span className={`badge badge-${row.State}`}>{row.State}</span>;
                if (col === 'Actions') return <button className="btn mini" onClick={() => { setSelected(history[i].id); setTab('live'); }}>Open</button>;
                return row[col];
              }}
            />
          </div>
        </div>
      )}

      {tab === 'about' && (
        <div className="tab-panel">
          <div className="card">
            <h3 style={{ marginBottom: 12, color: '#e4ede6' }}>Adaptive Multi-Model Ensemble</h3>
            <p style={{ fontSize: 13, lineHeight: 1.8, color: '#c8d8c8', marginBottom: 16 }}>
              This module implements a <strong style={{ color: '#6ec47e' }}>5-model adaptive ensemble</strong> for fraud detection.
              Each model (XGBoost, Random Forest, Extra Trees, HistGradientBoosting, Logistic Regression)
              contributes predictions weighted by an <strong style={{ color: '#6ec47e' }}>Adaptive Detector Score (ADS)</strong> that
              considers F1, recall, stability, and historical performance. Weights are re-computed every round,
              allowing the ensemble to automatically favor the best-performing models as fraud strategies evolve.
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 12 }}>
              {MODEL_NAMES.map((name) => (
                <div key={name} style={{ padding: 16, borderRadius: 12, background: 'rgba(12,21,15,0.6)', border: '1px solid rgba(63,185,80,0.1)' }}>
                  <div style={{ fontWeight: 700, color: MODEL_COLORS[name], fontSize: 14, marginBottom: 6 }}>{name}</div>
                  <div style={{ fontSize: 11, color: '#9bb8a3', lineHeight: 1.6 }}>
                    {name === 'XGBoost' && 'Gradient boosted trees with warm-start'}
                    {name === 'RandomForest' && '300-tree ensemble with balanced weights'}
                    {name === 'ExtraTrees' && 'Randomized split thresholds for diversity'}
                    {name === 'HistGradientBoosting' && 'Histogram-based fast gradient boosting'}
                    {name === 'LogisticRegression' && 'Linear baseline with lbfgs solver'}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
