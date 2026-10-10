function Row({ k, v }) {
  let val;
  if (typeof v === 'boolean') val = v ? 'true' : 'false';
  else if (typeof v === 'object' && v !== null) val = JSON.stringify(v);
  else val = String(v ?? '—');
  return (
    <div className="cfg-row">
      <code>{k}</code>
      <span>{val}</span>
    </div>
  );
}

function Section({ title, obj }) {
  if (!obj || typeof obj !== 'object') return null;
  const entries = Object.entries(obj);
  return (
    <div className="cfg-section">
      <div className="cfg-title">{title}</div>
      {entries.map(([k, v]) => (
        <Row key={k} k={k} v={v} />
      ))}
    </div>
  );
}

export default function ConfigInspector({ schema }) {
  const cfg = schema?.config;
  if (!cfg) return <section className="panel"><div className="empty-state">Config unavailable</div></section>;

  const { host, port, ...top } = cfg;

  return (
    <section className="panel">
      <div className="panel-head"><h2>Configuration</h2><span className="hint">backend/config.py + config.local.json</span></div>
      <div className="cfg-scroll">
        <Section title="Server" obj={{ host, port }} />
        <Section title="Session" obj={top} />
        <Section title="ieee (replay)" obj={cfg.ieee} />
        <Section title="chain (ethereum)" obj={cfg.chain} />
        <Section title="event schema" obj={schema?.event_schema} />
        <Section title="data availability" obj={schema?.data} />
      </div>
    </section>
  );
}