import { useCallback, useEffect, useRef, useState } from 'react';
import { getReviewQueue, submitReview, submitReviewBatch } from '../api.js';

/**
 * Human / LLM review queue.
 *
 * This panel is the *only* way an unlabeled stream (chain, webhook) can learn:
 * the pipeline queues candidates it could not triage, and a reviewer supplies
 * the ground-truth label.  Nothing here auto-labels - a button click or a
 * POST /api/review is a deliberate human (or LLM) judgement, and the label goes
 * straight into that source's ensemble for the next warm-start refit.
 */
const POLL_MS = 3000;

function age(ts) {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.round(s / 60)}m`;
  return `${(s / 3600).toFixed(1)}h`;
}

function probClass(p) {
  if (p >= 0.7) return 'block';
  if (p >= 0.3) return 'review';
  return 'allow';
}

function ReviewRow({ item, selected, onToggle, onLabel, busy }) {
  const cps = item?.extra?.counterparties || [];
  return (
    <div className={`review-row${selected ? ' selected' : ''}`}>
      <input
        type="checkbox"
        className="review-pick"
        checked={selected}
        disabled={busy}
        onChange={() => onToggle(item.review_id)}
        aria-label={`select ${item.review_id}`}
      />
      <div className="review-main">
        <div className="review-title">
          <span className={`src-chip chip-src-${item.source}`}>{item.source}</span>
          <code className="review-acct">{item.account_id}</code>
          <span className={`chip dec-${probClass(item.prob)}`}>p={item.prob}</span>
        </div>
        <div className="review-meta">
          <span>amount <b>{Number(item.amount).toFixed(4)}</b></span>
          {cps.length > 0 && <span title={cps.join(', ')}>{cps.length} counterparty{cps.length > 1 ? 'ies' : ''}</span>}
          <span>queued {age(item.queued_at)} ago</span>
          <code className="muted">{item.review_id}</code>
        </div>
      </div>
      <div className="review-actions">
        <button className="danger tiny" disabled={busy} onClick={() => onLabel(item.review_id, 1)}>
          Fraud
        </button>
        <button className="ok tiny" disabled={busy} onClick={() => onLabel(item.review_id, 0)}>
          Genuine
        </button>
      </div>
    </div>
  );
}

export default function ReviewQueuePanel({ status, revision }) {
  const [data, setData] = useState(null);
  const [reviewer, setReviewer] = useState('human');
  const [note, setNote] = useState('');
  const [busy, setBusy] = useState(false);
  const [sel, setSel] = useState(() => new Set());
  const [msg, setMsg] = useState(null);
  const [err, setErr] = useState('');
  const topRef = useRef(null);

  const load = useCallback(async () => {
    try {
      const r = await getReviewQueue(25);
      setData(r);
    } catch (e) {
      setErr(String(e));
    }
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(load, POLL_MS);
    return () => clearInterval(t);
  }, [load]);

  // re-pull whenever the session reports new review activity
  useEffect(() => {
    if (status?.review?.queued != null) load();
  }, [status?.review?.queued, status?.generation, revision, load]);

  const label = useCallback(
    async (reviewId, value) => {
      setBusy(true);
      setErr('');
      setMsg(null);
      try {
        const r = await submitReview(reviewId, value, reviewer, note || undefined);
        const n = r?.ensemble?.n_labeled;
        const refit = r?.refit?.queued ? ' · refit queued' : '';
        setMsg(`labeled ${reviewId} as ${value} · source now has ${n} labels${refit}`);
        setNote('');
        load();
      } catch (e) {
        setErr(String(e));
        load();
      } finally {
        setBusy(false);
      }
    },
    [note, reviewer, load],
  );

  // ---------------------------------------------------------- batch triage
  // A reviewer's pass over many rows is one deliberate judgement per row, so
  // batch is still explicit: pick the rows, then choose the label once.  There
  // is no default and nothing is ever labeled without this click.
  const toggle = useCallback((id) => {
    setSel((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const labelBatch = useCallback(
    async (value) => {
      const ids = [...sel];
      if (ids.length === 0) return;
      setBusy(true);
      setErr('');
      setMsg(null);
      try {
        const r = await submitReviewBatch(
          ids.map((id) => ({ review_id: id, label: value })),
          reviewer,
        );
        const done = r?.labeled?.length ?? 0;
        const errors = r?.errors || [];
        setMsg(
          `batch: ${done}/${ids.length} labeled as ${value}` +
            (errors.length ? ` · ${errors.length} rejected` : ''),
        );
        if (errors.length) {
          setErr(
            errors
              .map((e) => `${e.review_id || '?'}: ${e.error || 'rejected'}`)
              .join(' · '),
          );
        }
        setSel(new Set());
        load();
      } catch (e) {
        setErr(String(e));
      } finally {
        setBusy(false);
      }
    },
    [sel, reviewer, load],
  );


  // keyboard triage: the top candidate is the one being looked at
  useEffect(() => {
    const onKey = (e) => {
      if (e.target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) return;
      if (e.key !== '1' && e.key !== '0') return;
      const first = data?.pending?.[0];
      if (!first || busy) return;
      e.preventDefault();
      label(first.review_id, Number(e.key));
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [data, busy, label]);

  const stats = data?.stats || {};
  const budget = data?.budget || {};
  const pending = data?.pending || [];
  const allShown = pending.length > 0 && pending.every((p) => sel.has(p.review_id));
  const remaining = (budget.pos || 0) + (budget.neg || 0);
  const held = stats.held || 0;
  const total = stats.queued || 0;
  const resolvedPct = total ? ((stats.resolved || 0) / total) * 100 : 0;

  return (
    <section className="panel">
      <div className="panel-head">
        <div>
          <h2 className="panel-title">Human / LLM review queue</h2>
          <span className="hint">GET /api/review/queue · POST /api/review · POST /api/review/batch · the only source of new labels for chain &amp; webhook</span>
        </div>
        <span className={`pill tiny ${stats.pending ? 'amber' : 'gray'}`}>
          {stats.pending || 0} pending
        </span>
      </div>

      <div className="review-budget">
        <div className="review-budget-item">
          <span className="k">review budget left</span>
          <span className="v">{remaining}</span>
          <span className="muted">high-value {budget.pos ?? 0} · rest {budget.neg ?? 0}</span>
        </div>
        <div className="review-budget-item">
          <span className="k">labeled</span>
          <span className="v">{stats.resolved || 0}</span>
          <span className="muted">{stats.by_label?.[1] || 0} fraud · {stats.by_label?.[0] || 0} genuine</span>
        </div>
        <div className="review-budget-item">
          <span className="k">held / expired</span>
          <span className="v">{held}</span>
          <span className="muted">{stats.expired || 0} expired from a {stats.queue_max}-deep queue</span>
        </div>
      </div>

      <div className="bar" style={{ marginTop: 10 }}>
        <div className="fill" style={{ width: `${resolvedPct.toFixed(1)}%`, background: 'var(--green)' }} />
      </div>

      <div className="intake-form" style={{ marginTop: 14 }}>
        <label className="field">
          <span>Reviewer</span>
          <input value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="human or llm name" />
        </label>
        <label className="field" style={{ flex: 2 }}>
          <span>Note (optional, applied to the next label)</span>
          <input value={note} onChange={(e) => setNote(e.target.value)} placeholder="why this label" />
        </label>
        <button className="ghost tiny" onClick={load}>Refresh</button>
      </div>

      {err && <div className="banner error">{err}</div>}
      {msg && <div className="banner info">{msg}</div>}

      {pending.length > 0 && (
        <div className="review-batch">
          <label className="review-batch-pick">
            <input
              type="checkbox"
              checked={allShown}
              disabled={busy}
              onChange={() => setSel(allShown ? new Set() : new Set(pending.map((p) => p.review_id)))}
            />
            <span className="muted">select all shown</span>
          </label>
          <span className="muted">{sel.size} selected</span>
          <span className="spacer" />
          <button className="danger tiny" disabled={busy || sel.size === 0} onClick={() => labelBatch(1)}>
            Batch fraud
          </button>
          <button className="ok tiny" disabled={busy || sel.size === 0} onClick={() => labelBatch(0)}>
            Batch genuine
          </button>
        </div>
      )}

      {pending.length === 0 ? (
        <div className="empty-state">
          {stats.queued
            ? 'Queue is clear — every candidate has been labeled.'
            : 'No candidates yet. The pipeline queues an unlabeled event when no model can triage it, or when the review budget is still available.'}
        </div>
      ) : (
        <div className="review-list" ref={topRef}>
          {pending.map((item) => (
            <ReviewRow
              key={item.review_id}
              item={item}
              selected={sel.has(item.review_id)}
              onToggle={toggle}
              onLabel={label}
              busy={busy}
            />
          ))}
        </div>
      )}

      <p className="muted" style={{ fontSize: 11.5, marginBottom: 0 }}>
        Press <b>1</b> for fraud / <b>0</b> for genuine on the top candidate. Batch buttons apply the chosen
        label to the ticked rows only. Labels train that source's ensemble on the next warm-start refit;
        nothing is labeled automatically.
      </p>
    </section>
  );
}
