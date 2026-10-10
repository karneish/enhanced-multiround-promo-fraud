import { useMemo, useState } from 'react';
import ProbMeter from './ProbMeter.jsx';

const SRC_LABEL = { ieee: 'IEEE', chain: 'ETH', webhook: 'WEB' };
const DECISION_CLASS = { allow: 'allow', review: 'review', block: 'block' };

function fmtTS(ts) {
  if (ts == null) return '—';
  return new Date(ts * 1000).toLocaleTimeString();
}

export default function LiveFeed({ events, running, feedKey }) {
  const [filterSrc, setFilterSrc] = useState('all');
  const [filterDec, setFilterDec] = useState('all');
  const [search, setSearch] = useState('');

  const filtered = useMemo(() => {
    return events.filter((e) => {
      if (filterSrc !== 'all' && e.source !== filterSrc) return false;
      if (filterDec !== 'all' && e.decision !== filterDec) return false;
      if (search.trim() && !(e.account_id || '').toLowerCase().includes(search.trim().toLowerCase())) return false;
      return true;
    });
  }, [events, filterSrc, filterDec, search]);

  const srcCount = useMemo(() => {
    const c = { ieee: 0, chain: 0, webhook: 0 };
    events.forEach((e) => { if (c[e.source] != null) c[e.source]++; });
    return c;
  }, [events]);

  return (
    <section className="panel">
      <div className="panel-head">
        <h2 className="panel-title">Live feed</h2>
        <span className="hint">every scored event, newest first · {events.length} buffered</span>
      </div>

      <div className="feed-toolbar">
        <select className="field-sel" value={filterSrc} onChange={(e) => setFilterSrc(e.target.value)}>
          <option value="all">All sources ({Object.values(srcCount).reduce((a, b) => a + b, 0)})</option>
          <option value="ieee">IEEE ({srcCount.ieee})</option>
          <option value="chain">Ethereum ({srcCount.chain})</option>
          <option value="webhook">Webhook ({srcCount.webhook})</option>
        </select>
        <select className="field-sel" value={filterDec} onChange={(e) => setFilterDec(e.target.value)}>
          <option value="all">All decisions</option>
          <option value="allow">Allow</option>
          <option value="review">Review</option>
          <option value="block">Block</option>
        </select>
        <input
          className="search"
          placeholder="Search account…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <span className={`feed-ind ${running ? 'live' : ''}`}>{running ? '● live' : 'paused'}</span>
      </div>

      <div className="feed-wrap">
        <table className="feed">
          <thead>
            <tr>
              <th>time</th>
              <th>src</th>
              <th>account</th>
              <th>amount</th>
              <th>fraud prob</th>
              <th>decision</th>
              <th>label</th>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 && (
              <tr key="empty"><td colSpan="7" className="empty">No matching events yet</td></tr>
            )}
            {filtered.map((e) => (
              <tr key={`${feedKey}-${e.event_id}`} className={`row-${DECISION_CLASS[e.decision] || 'allow'}`}>
                <td>{fmtTS(e.ts)}</td>
                <td><span className={`src chip-src-${e.source}`}>{SRC_LABEL[e.source] || e.source}</span></td>
                <td className="acct">{e.account_id}</td>
                <td>{e.amount != null ? e.amount.toFixed(4) : '—'}</td>
                <td><ProbMeter prob={e.prob} decision={e.decision} /></td>
                <td>
                  <span className={`chip dec-${e.decision}`}>{e.decision}</span>
                  {e.known_fraud && <span className="chip known">known</span>}
                  {e.reviewed && <span className="chip reviewed">reviewed</span>}
                </td>
                <td>{e.label == null ? <span className="muted">?</span> : e.label === 1 ? <b className="red">1</b> : '0'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}