/* =====================================================================
   Unified API client — each sub-app gets a prefix-rewritten proxy path
   ===================================================================== */

async function apiFetch<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

/* ---- sub-app API namespaces ---- */

export const mainApi = {
  health:   () => apiFetch<any>('/api/main/health'),
  schema:   () => apiFetch<any>('/api/main/schema'),
  datasets: () => apiFetch<any[]>('/api/main/datasets'),
  graph:    (dset: string, n = 180) => apiFetch<any>(`/api/main/datasets/${dset}/graph?n=${n}`),
  experiments: () => apiFetch<any[]>('/api/main/experiments'),
  expCsv:   (cname: string, ts: string, csv: string) => apiFetch<any>(`/api/main/experiments/${cname}/${ts}/${csv}`),
  launch:   (cfg: any) => apiPost<{run_id: string}>('/api/main/run', cfg),
  runStatus:(id: string) => apiFetch<any>(`/api/main/run/${id}`),
  stopRun:  (id: string) => apiPost<any>(`/api/main/run/${id}/stop`, {}),
  runs:     () => apiFetch<any[]>('/api/main/runs'),
  streamUrl:(id: string) => `/api/main/run/${id}/stream`,
};

export const genApi = {
  health:   () => apiFetch<any>('/api/gen/health'),
  schema:   () => apiFetch<any>('/api/gen/schema'),
  datasets: () => apiFetch<any[]>('/api/gen/datasets'),
  launch:   (cfg: any) => apiPost<{id: string; state: string; config: any}>('/api/gen/run', cfg),
  runStatus:(id: string) => apiFetch<any>(`/api/gen/run/${id}`),
  report:   (id: string) => apiFetch<any>(`/api/gen/report/${id}`),
  graph:    (id: string) => apiFetch<any>(`/api/gen/graph/${id}`),
  history:  () => apiFetch<any[]>('/api/gen/history'),
  streamUrl:(id: string) => `/api/gen/stream/${id}`,
};

export const adlApi = {
  health:   () => apiFetch<any>('/api/adl/health'),
  schema:   () => apiFetch<any>('/api/adl/schema'),
  datasets: () => apiFetch<any[]>('/api/adl/datasets'),
  launch:   (cfg: any) => apiPost<{id: string; state: string; config: any}>('/api/adl/run', cfg),
  runStatus:(id: string) => apiFetch<any>(`/api/adl/run/${id}`),
  report:   (id: string) => apiFetch<any>(`/api/adl/report/${id}`),
  graph:    (id: string) => apiFetch<any>(`/api/adl/graph/${id}`),
  history:  () => apiFetch<any[]>('/api/adl/history'),
  streamUrl:(id: string) => `/api/adl/stream/${id}`,
};

export const ensApi = {
  health:   () => apiFetch<any>('/api/ens/health'),
  schema:   () => apiFetch<any>('/api/ens/schema'),
  datasets: () => apiFetch<any[]>('/api/ens/datasets'),
  launch:   (cfg: any) => apiPost<{id: string; describe: string; config: any; state: string}>('/api/ens/run', cfg),
  runStatus:(id: string) => apiFetch<any>(`/api/ens/run/${id}`),
  report:   (id: string) => apiFetch<any>(`/api/ens/report/${id}`),
  history:  () => apiFetch<any[]>('/api/ens/history'),
  streamUrl:(id: string) => `/api/ens/stream/${id}`,
};
