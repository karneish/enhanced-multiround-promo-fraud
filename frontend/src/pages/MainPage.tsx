import React, { useState, useEffect, useCallback, useRef } from 'react';
import { mainApi } from '../api';
import { useSSE } from '../hooks/useSSE';
import MetricCard from '../components/MetricCard';
import LogPanel from '../components/LogPanel';
import DataTable from '../components/DataTable';

type Tab = 'live' | 'launch' | 'results' | 'graph' | 'about';

export default function MainPage() {
  const [tab, setTab] = useState<Tab>('live');
  const [schema, setSchema] = useState<any>(null);
  const [runs, setRuns] = useState<any[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [experiments, setExperiments] = useState<any[]>([]);
  const [expDetail, setExpDetail] = useState<any>(null);
  const [datasets, setDatasets] = useState<any[]>([]);
  const [config, setConfig] = useState<Record<string, any>>({});
  const [desc, setDesc] = useState('');
  const [launching, setLaunching] = useState(false);
  const [graphDset, setGraphDset] = useState('');
  const [graphData, setGraphData] = useState<any>(null);
  const [graphN, setGraphN] = useState(180);
  const logRef = useRef<{ t: number; text: string }[]>([]);
  const [logLines, setLogLines] = useState<{ t: number; text: string }[]>([]);
  const [metrics, setMetrics] = useState<any[]>([]);
  const [runState, setRunState] = useState('idle');

  const sseUrl = selected ? mainApi.streamUrl(selected) : null;
  const { events, connected } = useSSE(sseUrl);
  const lastIdxRef = useRef(0);

  useEffect(() => {
    mainApi.schema().then((s) => { setSchema(s); setDatasets(s.datasets || []); }).catch(() => {});
    mainApi.experiments().then(setExperiments).catch(() => {});
    mainApi.runs().then(setRuns).catch(() => {});
  }, []);

  useEffect(() => {
    if (!schema) return;
    const d = schema.defaults || {};
    setConfig({
      TRIAL_NUM: 1, FAILURE_LIMIT: 2, EXPERIMENT_DESC: 'Dashboard experiment',
      LIST_DSET: [d.datasets?.[0] || ''], LIST_TRAIN_DSET: [],
      EXP_DICT: d.main || {}, MODEL_CONFIG: d.model || {}, TRAIN_CONFIG: d.train || {},
      STRAT_CONFIG: d.strat || {}, ADVER_CONFIG: d.adver || {},
    });
  }, [schema]);

  const lineCounter = useRef(0);
  useEffect(() => {
    const start = lastIdxRef.current;
    if (start >= events.length) return;
    lastIdxRef.current = events.length;
    for (let i = start; i < events.length; i++) {
      const ev = events[i];
      if (ev.type === 'log') {
        lineCounter.current += 1;
        logRef.current = [...logRef.current.slice(-300), { t: lineCounter.current, text: ev.line ?? '' }];
        setLogLines([...logRef.current]);
        const m = ev.line?.match?.(/Best Val: REC ([0-9.]+) PRE ([0-9.]+) MF1 ([0-9.]+) AUC ([0-9.]+)/);
        if (m) setMetrics((p) => [...p, { rec: +m[1], prec: +m[2], f1: +m[3], auc: +m[4] }]);
      } else if (ev.type === 'state') {
        setRunState(ev.state || 'done');
        mainApi.runs().then(setRuns).catch(() => {});
      }
    }
  }, [events, selected]);

  const launch = useCallback(async () => {
    setLaunching(true); setMetrics([]); logRef.current = []; setLogLines([]); setRunState('running'); lastIdxRef.current = 0;
    try {
      const res = await mainApi.launch({ ...config, EXPERIMENT_DESC: desc || 'Dashboard experiment' });
      setSelected(res.run_id); setTab('live');
    } catch (e: any) { alert('Launch failed: ' + e.message); setRunState('error'); }
    finally { setLaunching(false); }
  }, [config, desc]);

  const loadGraph = useCallback(async () => {
    if (!graphDset) return;
    try { setGraphData(await mainApi.graph(graphDset, graphN)); } catch (e: any) { alert(e.message); }
  }, [graphDset, graphN]);

  const loadExperiment = useCallback(async (cname: string, ts: string) => {
    try {
      const meta = await mainApi.expCsv(cname, ts, '');
      setExpDetail({ cname, ts, meta });
    } catch { setExpDetail({ cname, ts }); }
  }, []);

  return (
    <div>
      <div className="page-header">
        <h2>TPNE-XGB Research Framework</h2>
        <p>Multi-round adversarial fraud detection with DGL, PyTorch, and XGBoost</p>
      </div>
      <div className="tab-bar">
        {(['live', 'launch', 'results', 'graph', 'about'] as Tab[]).map((t) => (
          <button key={t} className={`tab-btn ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
            {t === 'live' ? '\u25CF' : t === 'launch' ? '\u25B2' : t === 'results' ? '\u25BC' : t === 'graph' ? '\u21BB' : '\u2139'} {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {tab === 'live' && (
        <div className="tab-panel">
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
            <select className="input" style={{ width: 280 }} value={selected ?? ''} onChange={(e) => { setSelected(e.target.value || null); logRef.current = []; setLogLines([]); setMetrics([]); setRunState('idle'); lastIdxRef.current = 0; }}>
              <option value="">Select a run...</option>
              {runs.map((r) => <option key={r.id} value={r.id}>{r.id.slice(0, 10)} ... {(r.desc || '').slice(0, 25)}</option>)}
            </select>
            <span className={`sse-pill ${connected ? 'on' : ''}`}><span className="dot" />{connected ? 'Streaming' : 'Off'}</span>
            <span className={`badge badge-${runState}`}>{runState}</span>
          </div>
          <div className="metric-grid">
            <MetricCard label="Status" value={runState} cls={runState === 'running' ? 'blue' : runState === 'done' ? 'green' : 'red'} />
            <MetricCard label="Metrics" value={metrics.length} cls="blue" />
            {metrics.length > 0 && <>
              <MetricCard label="Latest F1" value={metrics[metrics.length - 1].f1.toFixed(3)} cls="green" />
              <MetricCard label="Latest AUC" value={metrics[metrics.length - 1].auc.toFixed(3)} cls="purple" />
              <MetricCard label="Latest Recall" value={metrics[metrics.length - 1].rec.toFixed(3)} cls="green" />
              <MetricCard label="Latest Prec" value={metrics[metrics.length - 1].prec.toFixed(3)} cls="amber" />
            </>}
          </div>
          <div className="card">
            <div className="card-head"><h3>Experiment Log</h3></div>
            <LogPanel lines={logLines} />
          </div>
        </div>
      )}

      {tab === 'launch' && (
        <div className="tab-panel">
          <div className="card">
            <div className="card-head"><h3>Quick Start</h3></div>
            <div className="presets">
              {schema?.models?.map((m: string) => <button key={m} className="preset-btn" onClick={() => setConfig((c) => ({ ...c, EXP_DICT: { ...c.EXP_DICT, model_name: m } }))}>{m}</button>)}
            </div>
          </div>
          <div className="card">
            <div className="card-head"><h3>Configuration</h3></div>
            <div className="field" style={{ marginBottom: 14 }}>
              <label>Description</label>
              <input className="input" style={{ width: '100%' }} value={desc} onChange={(e) => setDesc(e.target.value)} placeholder="Name this experiment..." />
            </div>
            <div className="form-grid">
              <div className="field"><label>Trials</label><input className="input" type="number" value={config.TRIAL_NUM ?? 1} onChange={(e) => setConfig((c) => ({ ...c, TRIAL_NUM: +e.target.value }))} /></div>
              <div className="field"><label>Failure Limit</label><input className="input" type="number" value={config.FAILURE_LIMIT ?? 2} onChange={(e) => setConfig((c) => ({ ...c, FAILURE_LIMIT: +e.target.value }))} /></div>
              <div className="field"><label>Rounds</label><input className="input" type="number" value={config.EXP_DICT?.rounds ?? 5} onChange={(e) => setConfig((c) => ({ ...c, EXP_DICT: { ...c.EXP_DICT, rounds: +e.target.value } }))} /></div>
              <div className="field"><label>Model</label>
                <select className="input" value={config.EXP_DICT?.model_name ?? ''} onChange={(e) => setConfig((c) => ({ ...c, EXP_DICT: { ...c.EXP_DICT, model_name: e.target.value } }))}>
                  <option value="">--</option>
                  {schema?.models?.map((m: string) => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>
              <div className="field"><label>Adversary Mod</label>
                <select className="input" value={config.EXP_DICT?.adver_mod ?? ''} onChange={(e) => setConfig((c) => ({ ...c, EXP_DICT: { ...c.EXP_DICT, adver_mod: e.target.value } }))}>
                  {schema?.adver_mod?.map((m: string) => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>
            </div>
            <div className="field" style={{ marginTop: 8 }}>
              <label>Datasets</label>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {datasets.map((d: any) => {
                  const name = d.name || d;
                  const sel = config.LIST_DSET?.includes(name);
                  return <button key={name} className={`btn mini ${sel ? 'primary' : ''}`} onClick={() => {
                    setConfig((c) => {
                      const ds = c.LIST_DSET || [];
                      return { ...c, LIST_DSET: sel ? ds.filter((x: string) => x !== name) : [...ds, name] };
                    });
                  }}>{name}</button>;
                })}
              </div>
            </div>
          </div>
          <button className="btn primary" onClick={launch} disabled={launching}>{launching ? 'Launching...' : '\u25B6 Launch Experiment'}</button>
          <div className="card" style={{ marginTop: 16 }}>
            <div className="card-head"><h3>Session Runs</h3></div>
            <DataTable columns={['ID', 'State', 'Description', 'Actions']} rows={runs.map((r) => ({ ID: r.id?.slice(0, 10), State: r.state, Description: (r.desc || '').slice(0, 30), Actions: '' }))}
              renderCell={(col, row, i) => { if (col === 'State') return <span className={`badge badge-${row.State}`}>{row.State}</span>; if (col === 'Actions') return <button className="btn mini" onClick={() => { setSelected(runs[i].id); setTab('live'); }}>Open</button>; return row[col]; }} />
          </div>
        </div>
      )}

      {tab === 'results' && (
        <div className="tab-panel">
          <div className="card">
            <div className="card-head"><h3>Completed Experiments</h3></div>
            {experiments.length === 0 && <div className="empty">No experiments found in result/ directory.</div>}
            {experiments.map((e, i) => (
              <div key={i} style={{ padding: '10px 0', borderBottom: '1px solid rgba(63,185,80,0.06)', cursor: 'pointer', fontSize: 13, transition: 'background 0.15s' }} onClick={() => loadExperiment(e.cname, e.ts)}>
                <span style={{ color: '#e4ede6', fontWeight: 600 }}>{e.desc || e.cname}</span>
                <span className="muted" style={{ marginLeft: 8 }}>{e.cname}/{e.ts}</span>
                {e.has_combined && <span className="badge badge-done" style={{ marginLeft: 8 }}>CSV</span>}
              </div>
            ))}
          </div>
        </div>
      )}

      {tab === 'graph' && (
        <div className="tab-panel">
          <div className="card">
            <div className="card-head"><h3>Graph Explorer</h3></div>
            <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 16 }}>
              <select className="input" style={{ width: 240 }} value={graphDset} onChange={(e) => setGraphDset(e.target.value)}>
                <option value="">Select dataset...</option>
                {datasets.map((d: any) => <option key={d.name || d} value={d.name || d}>{d.name || d}</option>)}
              </select>
              <div className="field" style={{ margin: 0, width: 80 }}>
                <input className="input" type="number" min={30} max={600} value={graphN} onChange={(e) => setGraphN(+e.target.value)} />
              </div>
              <button className="btn" onClick={loadGraph}>Load</button>
            </div>
            {graphData && (
              <div>
                <div style={{ fontSize: 12, color: '#9bb8a3', marginBottom: 10 }}>
                  {graphData.name} - {graphData.num_nodes_total} nodes total, showing {graphData.sampled_nodes?.length ?? 0}
                </div>
                <svg viewBox="-1.1 -1.1 2.2 2.2" style={{ width: '100%', maxWidth: 600, background: 'rgba(4,8,6,0.8)', borderRadius: 12, border: '1px solid rgba(63,185,80,0.1)' }}>
                  {graphData.edges?.slice(0, 2000).map(([u, v]: [number, number], i: number) => {
                    const nu = graphData.sampled_nodes?.find((n: any) => n.id === u);
                    const nv = graphData.sampled_nodes?.find((n: any) => n.id === v);
                    if (!nu || !nv) return null;
                    return <line key={i} x1={nu.x} y1={nu.y} x2={nv.x} y2={nv.y} stroke="rgba(63,185,80,0.12)" strokeWidth="0.3" />;
                  })}
                  {graphData.sampled_nodes?.map((n: any, i: number) => (
                    <circle key={i} cx={n.x} cy={n.y} r={n.label === 1 ? 0.025 : 0.012} fill={n.label === 1 ? '#ef4444' : '#3fb950'} opacity={0.85} />
                  ))}
                </svg>
              </div>
            )}
          </div>
        </div>
      )}

      {tab === 'about' && (
        <div className="tab-panel">
          <div className="card">
            <h3 style={{ marginBottom: 12, color: '#e4ede6' }}>TPNE-XGB Research Framework</h3>
            <p style={{ fontSize: 13, lineHeight: 1.8, color: '#c8d8c8' }}>
              A multi-round adversarial fraud detection framework built on <strong style={{ color: '#6ec47e' }}>DGL (Deep Graph Library)</strong>,
              <strong style={{ color: '#6ec47e' }}> PyTorch</strong>, and <strong style={{ color: '#6ec47e' }}> XGBoost</strong>. Each round the adversary evolves fraud strategies
              (via REPLAY, PERTURB, MIXING, or INTELLIGENT modes) while the detector retrains on the updated graph.
              Supports temporal TPNE embeddings, multiple GNN architectures (GCN, GAT, GIN, GraphSAGE, BWGNN, etc.),
              graph augmentation strategies, and comprehensive evaluation with CSV result exports.
            </p>
          </div>

          <div className="card">
            <div className="card-head"><h3>Experiment Results — Macro-F1 on tolokers_bid</h3></div>
            <div className="metric-grid" style={{ marginBottom: 16 }}>
              <div className="metric" style={{ borderLeft: '3px solid #ef4444' }}>
                <div className="k">Paper Baseline (XGB-SP)</div>
                <div className="v" style={{ color: '#ef4444' }}>34.9%</div>
                <div className="s">Single XGBoost reported in paper</div>
              </div>
              <div className="metric" style={{ borderLeft: '3px solid #f59e0b' }}>
                <div className="k">Our XGB-SP Baseline (4 trials)</div>
                <div className="v" style={{ color: '#f59e0b' }}>68.2%</div>
                <div className="s">R0 val_best MF1 avg over 4 trials</div>
              </div>
              <div className="metric" style={{ borderLeft: '3px solid #3b82f6' }}>
                <div className="k">6-Model ADAPTIVE + DFEAT</div>
                <div className="v" style={{ color: '#3b82f6' }}>69.9%</div>
                <div className="s">R1 val_best MF1 (5 epochs GNN)</div>
              </div>
              <div className="metric" style={{ borderLeft: '3px solid #10b981' }}>
                <div className="k">6-Model Entire Graph R1</div>
                <div className="v" style={{ color: '#10b981' }}>62.8%</div>
                <div className="s">Full-graph MF1 with 128 GAN nodes</div>
              </div>
            </div>

            <h4 style={{ color: '#e4ede6', marginBottom: 8 }}>Baseline Comparison (avg over 4 trials, 2 rounds)</h4>
            <DataTable
              columns={['Config', 'R0 MF1', 'R0 AUC', 'R1 MF1', 'R1 AUC', 'Adversary']}
              rows={[
                { Config: 'XGB-SP (config_cpu)', 'R0 MF1': '68.2%', 'R0 AUC': '81.2%', 'R1 MF1': '65.1%', 'R1 AUC': '76.5%', Adversary: 'REPLAY' },
                { Config: 'XGB-SP (config_intelligent)', 'R0 MF1': '67.0%', 'R0 AUC': '80.0%', 'R1 MF1': '65.7%', 'R1 AUC': '74.5%', Adversary: 'INTELLIGENT' },
                { Config: '6-Model ADAPTIVE + DFEAT (min)', 'R0 MF1': '62.6%', 'R0 AUC': '76.0%', 'R1 MF1': '69.9%', 'R1 AUC': '81.1%', Adversary: 'INTELLIGENT' },
              ]}
            />

            <h4 style={{ color: '#e4ede6', marginTop: 16, marginBottom: 8 }}>Individual Model F1 (Round 1, 6-Model Ensemble)</h4>
            <DataTable
              columns={['Model', 'MF1', 'Recall', 'ADS Weight', 'Role']}
              rows={[
                { Model: 'LightGBM', MF1: '69.4%', Recall: '62.2%', 'ADS Weight': '16.4%', Role: 'Leaf-wise boosting' },
                { Model: 'HistGradientBoosting', MF1: '68.7%', Recall: '53.1%', 'ADS Weight': '17.9%', Role: 'Histogram-based' },
                { Model: 'XGBoost', MF1: '66.7%', Recall: '47.7%', 'ADS Weight': '15.6%', Role: 'Level-wise boosting' },
                { Model: 'ExtraTrees', MF1: '64.9%', Recall: '36.6%', 'ADS Weight': '17.2%', Role: 'Random splitting' },
                { Model: 'RandomForest', MF1: '64.2%', Recall: '33.2%', 'ADS Weight': '15.9%', Role: 'Bootstrap agg.' },
                { Model: 'LogisticRegression', MF1: '52.0%', Recall: '96.3%', 'ADS Weight': '16.4%', Role: 'Linear baseline' },
              ]}
            />

            <h4 style={{ color: '#e4ede6', marginTop: 16, marginBottom: 8 }}>Key Innovations</h4>
            <ul style={{ fontSize: 13, lineHeight: 1.8, color: '#c8d8c8', paddingLeft: 20 }}>
              <li><strong style={{ color: '#6ec47e' }}>6-Model Ensemble:</strong> XGBoost + RandomForest + ExtraTrees + HistGradientBoosting + LogisticRegression + LightGBM</li>
              <li><strong style={{ color: '#6ec47e' }}>Adaptive Detector Score (ADS):</strong> Dynamically reweights models based on 0.25xF1 + 0.25xRecall + 0.25xStability + 0.25xHistorical</li>
              <li><strong style={{ color: '#6ec47e' }}>DFEAT Add-on:</strong> Combines FeatureDistThreshold (DBSCAN) + DegreeActivityThreshold (hub detection) via OR aggregation</li>
              <li><strong style={{ color: '#6ec47e' }}>TemporalMixedEmbedder:</strong> Self-supervised GNN with attention-gated temporal features (192D embeddings)</li>
              <li><strong style={{ color: '#6ec47e' }}>GAN Adversary:</strong> 128 intelligent fraud variants per round, evolves from missed detections</li>
            </ul>
            <p style={{ fontSize: 12, lineHeight: 1.6, color: '#7a9a7a', marginTop: 12, fontStyle: 'italic' }}>
              Note: All F1 scores are macro-F1 (sklearn average='macro') with threshold search (0.05-0.95).
              The 6-Model ADAPTIVE experiment above used only 5 GNN epochs; full 20-epoch runs pending.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
