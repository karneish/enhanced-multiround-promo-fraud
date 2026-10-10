import { useCountUp } from '../hooks.js';

export default function StatusBar({ running, mode, sse, uptime }) {
  const fmt = (s) => {
    if (s == null) return '—';
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = Math.floor(s % 60);
    const p = (n) => String(n).padStart(2, '0');
    return h > 0 ? `${h}:${p(m)}:${p(sec)}` : `${m}:${p(sec)}`;
  };
  const up = useCountUp(Math.floor(uptime || 0), { duration: 500 });

  return (
    <header className="statusbar">
      <div className="brand">
        <span className="logo" />
        <div className="brand-text">
          <span className="brand-title">Real-Time Adaptive Fraud Detector</span>
          <span className="brand-sub">XGBoost · HistGradientBoosting · ExtraTrees + adaptive weighting</span>
        </div>
      </div>
      <div className="status-group">
        <div className={`pill ${running ? 'green' : 'gray'}`}>
          <span className="dot" /> {running ? 'Streaming' : 'Idle'}
        </div>
        <div className="pill">mode: <b>{mode || '—'}</b></div>
        <div className="pill">uptime: <b>{fmt(up)}</b></div>
        <div className={`pill ${sse === 'open' ? 'green' : 'amber'}`}>
          SSE {sse}
        </div>
      </div>
    </header>
  );
}