import { useMemo, useState } from 'react';

const TYPE_CLASS = {
  event: 'blue',
  refit: 'amber',
  source_status: 'gray',
  refit_error: 'red',
  process_error: 'red',
  source_error: 'red',
};

const TYPE_LABEL = {
  event: 'event',
  refit: 'refit',
  source_status: 'status',
  refit_error: 'refit err',
  process_error: 'process err',
  source_error: 'source err',
};

function summarize(msg) {
  switch (msg.type) {
    case 'event': {
      return `${msg.source} · ${msg.account_id || ''} · ${msg.decision}` +
        (msg.prob != null ? ` (${(msg.prob * 100).toFixed(1)}%)` : '') +
        (msg.known_fraud ? ' · known fraud' : '');
    }
    case 'refit': {
      const models = Object.entries(msg.models || {});
      return models.length
        ? `${msg.source}: ${models.map(([name, m]) => `${name}=${(m?.f1 ?? 0).toFixed(2)}`).join(' · ')}`
        : `${msg.source}: refit complete`;
    }
    case 'source_status':
    case 'source_error':
    case 'process_error':
    case 'refit_error':
      return `${msg.source ?? msg.origin ?? ''}: ${msg.status ?? msg.error ?? ''}`;
    default:
      return JSON.stringify(msg);
  }
}

export default function EventLog({ log }) {
  const [filter, setFilter] = useState('all');
  const [search, setSearch] = useState('');
  const [open, setOpen] = useState(null);

  const counts = useMemo(() => {
    const c = {};
    for (const m of log || []) if (m.type) c[m.type] = (c[m.type] || 0) + 1;
    return c;
  }, [log]);

  const filtered = useMemo(() => {
    let rows = log || [];
    if (filter !== 'all') rows = rows.filter((m) => m.type === filter);
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      rows = rows.filter((m) => JSON.stringify(m).toLowerCase().includes(q));
    }
    return rows.slice(0, 250);
  }, [log, filter, search]);

  const time = (t) =>
    t ? new Date(t).toLocaleTimeString('en-US', { hour12: false }) : '';

  return (
    <div className="event-log">
      <div className="feed-toolbar">
        <select className="field-sel" value={filter} onChange={(e) => setFilter(e.target.value)}>
          <option value="all">all ({log.length})</option>
          {Object.entries(counts).map(([t, n]) => (
            <option key={t} value={t}>{t} ({n})</option>
          ))}
        </select>
        <input
          className="search"
          placeholder="search log…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <span className={`feed-ind ${log.length ? '' : 'muted'}`}>
          {filtered.length} shown / {log.length} total
        </span>
      </div>

      <div className="log-wrap">
        {filtered.length === 0 ? (
          <div className="empty-state">No log entries yet.</div>
        ) : (
          filtered.map((m, i) => {
            const key = `${m._at}-${i}`;
            const cls = TYPE_CLASS[m.type] || 'gray';
            const detail = { ...m };
            delete detail._at;
            return (
              <div key={key} className={`log-row log-${m.type || 'other'}`}>
                <div className="log-main">
                  <code className="log-time">{time(m._at)}</code>
                  <span className={`chip log-type ${cls}`}>{TYPE_LABEL[m.type] || m.type}</span>
                  <span className="log-src">{m.source || ''}</span>
                  <span className="log-summary">{summarize(m)}</span>
                  <button className="ghost tiny log-toggle" onClick={() => setOpen(open === key ? null : key)}>
                    {open === key ? 'hide' : 'detail'}
                  </button>
                </div>
                {open === key && (
                  <pre className="log-detail">{JSON.stringify(detail, null, 2)}</pre>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}