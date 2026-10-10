const json = (res) => {
  if (!res.ok) return res.json().then((j) => Promise.reject(j.error || res.statusText));
  return res.json();
};

export const getSchema = () => fetch('/api/schema').then(json);
export const getStatus = () => fetch('/api/live/status').then(json);
export const getGraph = () => fetch('/api/graph').then(json);

export const startLive = (body) =>
  fetch('/api/live/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(json);

export const stopLive = () =>
  fetch('/api/live/stop', { method: 'POST' }).then(json);

export const ingestEvent = (payload) =>
  fetch('/api/events/ingest', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }).then(json);

// --- human / LLM review loop -------------------------------------------
// A review candidate is an *unlabeled* event the model could not triage on
// its own.  The label returned here is the only thing that can teach an
// unlabeled stream (chain / webhook) - nothing is auto-labeled.

export const getReviewQueue = (limit = 50, source) =>
  fetch(
    `/api/review/queue?limit=${limit}` + (source ? `&source=${encodeURIComponent(source)}` : ''),
  ).then(json);

export const submitReview = (reviewId, label, reviewer, note) =>
  fetch('/api/review', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ review_id: reviewId, label, reviewer, note }),
  }).then(json);

export const submitReviewBatch = (items, reviewer) =>
  fetch('/api/review/batch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ items, reviewer }),
  }).then(json);
