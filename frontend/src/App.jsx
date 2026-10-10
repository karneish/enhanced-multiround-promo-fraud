import { useState, useEffect, useCallback, useRef } from 'react';
import { getSchema, getStatus, getGraph, startLive, stopLive } from './api.js';
import { useCountUp } from './hooks.js';
import StatusBar from './components/StatusBar.jsx';
import SessionControls from './components/SessionControls.jsx';
import SourceCards from './components/SourceCards.jsx';
import ModelPanel from './components/ModelPanel.jsx';
import ReviewQueuePanel from './components/ReviewQueuePanel.jsx';
import DecisionMix from './components/DecisionMix.jsx';
import LiveFeed from './components/LiveFeed.jsx';
import NodeGraph from './components/NodeGraph.jsx';
import EventLog from './components/EventLog.jsx';
import EventIntake from './components/EventIntake.jsx';
import ConfigInspector from './components/ConfigInspector.jsx';
import TrendChart from './charts/TrendChart.jsx';

const LOG_TYPES = new Set([
  'event', 'refit', 'refit_error', 'source_status', 'source_error', 'process_error',
]);

const SRC_KEYS = { ieee: 'replay', chain: 'chain', webhook: 'webhook' };

function HeroTile({ label, value, sub, cls, width }) {
  const v = useCountUp(value || 0, { duration: 600 });
  return (
    <div className={`hero-tile ${cls || ''}`} style={{ '--w': width || '40%' }}>
      <div className="hero-label">{label}</div>
      <div className="hero-val">{v.toLocaleString()}</div>
      <div className="hero-sub">{sub}</div>
    </div>
  );
}

export default function App() {
  const [schema, setSchema] = useState(null);
  const [status, setStatus] = useState(null);
  const [events, setEvents] = useState([]);
  const [log, setLog] = useState([]);
  const [graph, setGraph] = useState(null);
  const [feedKey, setFeedKey] = useState(0);
  const [sseState, setSseState] = useState('disconnected');
  const [error, setError] = useState('');
  const [logBusy, setLogBusy] = useState(false);
  const [trend, setTrend] = useState([]);
  const [history, setHistory] = useState({ replay: [], chain: [], webhook: [] });

  const [controls, setControls] = useState({
    mode: 'concurrent',
    events_per_sec: 20,
    stream_slice: 100000,
    review_threshold: 0.30,
    block_threshold: 0.70,
    budget_pos: 10,
    budget_neg: 200,
  });

  const running = Boolean(status?.running);
  const esRef = useRef(null);
  const seenRef = useRef(new Set());
  const trendRef = useRef([]);
  const srcCountRef = useRef({ replay: 0, chain: 0, webhook: 0 });
  const histRef = useRef({ replay: [], chain: [], webhook: [] });

  const pushUnique = useCallback((setter, payload, cap) => {
    const key = typeof payload === 'object' ? JSON.stringify(payload) : String(payload);
    if (seenRef.current.has(key)) return;
    if (seenRef.current.size > 800) {
      const arr = [...seenRef.current];
      seenRef.current = new Set(arr.slice(-400));
    }
    seenRef.current.add(key);
    setter((prev) => [{ ...payload, _at: Date.now() }, ...prev].slice(0, cap));
  }, []);

  useEffect(() => {
    getSchema()
      .then((s) => {
        setSchema(s);
        const c = s?.config;
        if (c) {
          const dth = c.decision_thresholds || {};
          const bd = c.budget_pos != null ? { budget_pos: c.budget_pos, budget_neg: c.budget_neg } : {};
          setControls((prev) => ({
            ...prev,
            mode: c.live_mode ?? prev.mode,
            events_per_sec: c.events_per_sec ?? prev.events_per_sec,
            review_threshold: dth.review ?? prev.review_threshold,
            block_threshold: dth.block ?? prev.block_threshold,
            ...bd,
          }));
        }
      })
      .catch((e) => setError(String(e)));
    getStatus().then(setStatus).catch((e) => setError(String(e)));
    getGraph().then(setGraph).catch(() => {});
  }, []);

  // (Re)connect SSE whenever streaming starts
  useEffect(() => {
    if (!running) {
      setSseState('disconnected');
      if (esRef.current) { esRef.current.close(); esRef.current = null; }
      return;
    }
    const es = new EventSource('/api/stream');
    esRef.current = es;
    es.onopen = () => setSseState('open');
    es.onerror = () => setSseState('disconnected');
    es.addEventListener('status', (e) => {
      try { setStatus(JSON.parse(e.data)); } catch { /* ignore */ }
    });
    es.addEventListener('message', (e) => {
      try {
        const msg = JSON.parse(e.data);
        if (!msg || !msg.type) return;
        if (msg.type === 'event' && msg.event_id) {
          trendRef.current.push({ prob: msg.prob, decision: msg.decision, source: msg.source, ts: msg.ts });
          if (trendRef.current.length > 420) trendRef.current = trendRef.current.slice(-350);
          pushUnique(setEvents, msg, 120);
          const k = SRC_KEYS[msg.source];
          if (k) srcCountRef.current[k] += 1;
        }
        if (LOG_TYPES.has(msg.type)) {
          pushUnique(setLog, msg, 300);
        }
      } catch { /* ignore */ }
    });
    return () => { es.close(); esRef.current = null; };
  }, [running]);

  // Throttled chart snapshots (trend + per-source history)
  useEffect(() => {
    if (!running) {
      setTrend(trendRef.current.slice(-180));
      return;
    }
    const id = setInterval(() => {
      setTrend(trendRef.current.slice(-180));
      const snap = { replay: [], chain: [], webhook: [] };
      for (const k of ['replay', 'chain', 'webhook']) {
        histRef.current[k].push(srcCountRef.current[k]);
        if (histRef.current[k].length > 120) histRef.current[k] = histRef.current[k].slice(-120);
        snap[k] = histRef.current[k];
      }
      setHistory(snap);
    }, 400);
    return () => clearInterval(id);
  }, [running]);

  // Poll the node graph while streaming
  useEffect(() => {
    if (!running) return;
    let dead = false;
    const tick = async () => {
      try {
        const g = await getGraph();
        if (!dead) setGraph(g);
      } catch { /* ignore */ }
    };
    tick();
    const id = setInterval(tick, 2500);
    return () => { dead = true; clearInterval(id); };
  }, [running]);

  const handleStart = useCallback(async () => {
    setError('');
    const body = {
      live_mode: controls.mode,
      events_per_sec: Number(controls.events_per_sec),
      stream_slice: Number(controls.stream_slice),
      decision_thresholds: {
        review: Number(controls.review_threshold),
        block: Number(controls.block_threshold),
      },
      budget_pos: Number(controls.budget_pos),
      budget_neg: Number(controls.budget_neg),
    };
    try {
      const res = await startLive(body);
      setStatus(res);
      setEvents([]);
      seenRef.current = new Set();
      trendRef.current = [];
      histRef.current = { replay: [], chain: [], webhook: [] };
      srcCountRef.current = { replay: 0, chain: 0, webhook: 0 };
      setTrend([]);
      setHistory({ replay: [], chain: [], webhook: [] });
      setLog((prev) => [{ _at: Date.now(), type: 'source_status', source: 'sim', status: `started (${controls.mode})` }, ...prev]);
    } catch (e) {
      setError(String(e));
    }
  }, [controls]);

  const handleStop = useCallback(async () => {
    setError('');
    try {
      const res = await stopLive();
      setStatus(res);
      setLog((prev) => [{ _at: Date.now(), type: 'source_status', source: 'sim', status: 'stopped' }, ...prev]);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  const pushIngestLog = useCallback((rec) => {
    setLog((prev) => [{ _at: Date.now(), type: 'event', ...rec }, ...prev].slice(0, 300));
  }, []);

  const clearFeed = () => { setEvents([]); setLog([]); setFeedKey((k) => k + 1); };

  const counters = status?.counters || {};
  const decisions = status?.decisions || {};
  const totalEvents = Object.values(counters).reduce((a, b) => a + (b || 0), 0);
  const labeled = Object.values(status?.source_states || {}).reduce((a, s) => a + (s?.labeled || 0), 0);
  const blocked = decisions.block || 0;
  const inReview = decisions.review || 0;
  const th = schema?.config?.decision_thresholds || {};

  return (
    <div>
      <div className="backdrop" />
      <div className="orb orb-a" />
      <div className="orb orb-b" />
      <div className="orb orb-c" />

      <div className="app-shell">
        <StatusBar running={running} mode={status?.mode} sse={sseState} uptime={status?.uptime_seconds} />
        <div className="gap" />

        <div className="hero-grid">
          <HeroTile label="Events processed" value={totalEvents} width="62%" sub="across all live sources · this session" />
          <HeroTile label="Blocked" value={blocked} cls="bad" width="70%" sub="risk ≥ block threshold" />
          <HeroTile label="In review" value={inReview} cls="warn" width="58%" sub={`≥ ${th.review ?? '—'} · < ${th.block ?? '—'}`} />
          <HeroTile label="Labeled" value={labeled} cls="ok" width="38%" sub="ground truth available for weights" />
        </div>
        <div className="gap" />

        <SessionControls
          controls={controls}
          setControls={setControls}
          running={running}
          onStart={handleStart}
          onStop={handleStop}
          dataReady={schema?.data}
          schema={schema}
        />
        <div className="gap" />

        <div className="row">
          <div className="col-3">
            <section className="panel">
              <div className="panel-head">
                <h2 className="panel-title">Fraud-probability trend</h2>
                <span className="hint">rolling scored events · decision bands shaded</span>
              </div>
              <TrendChart data={trend} review={th.review} block={th.block} />
            </section>
          </div>
          <div className="col-2">
            <SourceCards status={status} config={schema?.config} history={history} />
          </div>
        </div>
        <div className="gap" />

        {error && <div className="banner error">⚠ {error}</div>}

        <ModelPanel status={status} config={schema?.config} />
        <div className="gap" />

        <ReviewQueuePanel status={status} revision={feedKey} />
        <div className="gap" />

        <NodeGraph graph={graph} running={running} pollMs={2500} />
        <div className="gap" />

        <div className="grid-cols-2">
          <div>
            <div className="log-header">
              <h2 className="panel-title">Live feed</h2>
              <div className="log-actions">
                <button className="ghost tiny" onClick={clearFeed}>Clear feed + log</button>
              </div>
            </div>
            <LiveFeed events={events} feedKey={feedKey} running={running} />
          </div>
          <div>
            <div className="log-header">
              <h2 className="panel-title">Event log</h2>
              <div className="log-actions">
                <button className="ghost tiny" onClick={() => setLog([])}>Clear log</button>
              </div>
            </div>
            <EventLog log={log} key={feedKey} />
          </div>
        </div>
        <div className="gap" />

        <div className="grid-cols-3">
          <EventIntake running={running} onResult={pushIngestLog} />
          <DecisionMix decisions={status?.decisions} config={schema?.config} />
          <ConfigInspector schema={schema} />
        </div>
      </div>
    </div>
  );
}