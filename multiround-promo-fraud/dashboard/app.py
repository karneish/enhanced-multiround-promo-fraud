import sys, os, json, re, time, threading, subprocess, datetime, glob, io, random
import networkx as nx

from flask import Flask, jsonify, request, Response, render_template, abort, send_from_directory

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
SCRIPT_DIR = os.path.join(PROJECT_ROOT, 'scripts')
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')
RESULT_DIR = os.path.join(PROJECT_ROOT, 'result')

sys.path.append(SRC_DIR)

app = Flask(__name__)

try:
    import dgl
    import torch
    from utils import utils_const as UC
except Exception as e:
    dgl = torch = UC = None
    _IMPORT_ERROR = str(e)

RUNS = {}
RUNS_LOCK = threading.Lock()
GRAPH_CACHE = {}
DSET_CACHE = {}


def now_str():
    return datetime.datetime.now().strftime("%H:%M:%S")


# --------------------------------------------------------------------------
# Schema / knowledge of available options
# --------------------------------------------------------------------------
def build_schema():
    models = list(UC.MODEL_DICT.keys()) if UC else ['GCN', 'GCNII', 'GraphSAGE', 'GIN', 'GAT', 'BWGNN', 'GHRN', 'XGB', 'XGB-SP']
    augments = list(UC.AUGMENT_DICT.keys()) if UC else ['NONE', 'RANDOM', 'REAGE']
    adver_choose = list(UC.ADVER_CHOOSE_DICT.keys()) if UC else ['RANDOM', 'GREEDY', 'OGREEDY']
    adver_mod = list(UC.ADVER_MOD_DICT.keys()) if UC else ['REPLAY', 'PERTURB-ABS', 'PERTURB-REL', 'MIXING', 'INTELLIGENT']
    addons = list(UC.ADDON_DICT.keys()) if UC else ['NONE', 'FTHR', 'AFTHR', 'DEGREE', 'DFEAT', 'DAFEAT']

    defaults = {}
    if UC:
        for name, cfg in [('main', UC.DEFAULT_MAIN_CONFIG), ('train', UC.DEFAULT_TRAIN_CONFIG),
                          ('model', UC.DEFAULT_MODEL_CONFIG), ('strat', UC.DEFAULT_STRAT_CONFIG),
                          ('adver', UC.DEFAULT_ADVER_CONFIG)]:
            defaults[name] = {k: str(v) if not isinstance(v, (int, float, bool, list, str)) else v
                              for k, v in cfg.items()}
    return {
        'datasets': list_datasets(force=True),
        'models': models,
        'augments': augments,
        'adver_choose': adver_choose,
        'adver_mod': adver_mod,
        'addons': addons,
        'defaults': defaults,
        'import_error': _IMPORT_ERROR if UC is None else None,
    }


# --------------------------------------------------------------------------
# Dataset helpers
# --------------------------------------------------------------------------
def list_datasets(force=False):
    if DSET_CACHE and not force:
        return DSET_CACHE['names']
    names = sorted([d for d in os.listdir(DATASET_DIR)
                    if os.path.isfile(os.path.join(DATASET_DIR, d)) and not d.startswith('.')])
    DSET_CACHE['names'] = names
    return names


def dataset_stats(dset):
    path = os.path.join(DATASET_DIR, dset)
    if not os.path.isfile(path):
        return None
    if dset in DSET_CACHE and 'stats' in DSET_CACHE[dset]:
        return DSET_CACHE[dset]['stats']
    if dgl is None:
        return {'name': dset, 'error': _IMPORT_ERROR}
    graphs, _ = dgl.load_graphs(path)
    g = graphs[0]
    labels = g.ndata['label']
    if labels.dim() > 1:
        labels = labels.argmax(1)
    lc = labels.unique(return_counts=True)
    counts = {int(k): int(v) for k, v in zip(lc[0].tolist(), lc[1].tolist())}
    stats = {
        'name': dset,
        'num_nodes': int(g.num_nodes()),
        'num_edges': int(g.num_edges()),
        'label_counts': counts,
        'num_labels': len(counts),
        'feature_dim': int(g.ndata['feature'].shape[1]),
        'is_homogeneous': g.is_homogeneous,
    }
    DSET_CACHE.setdefault(dset, {})['stats'] = stats
    return stats


def sample_graph(dset, n_nodes=180):
    key = (dset, n_nodes)
    if key in GRAPH_CACHE:
        return GRAPH_CACHE[key]
    path = os.path.join(DATASET_DIR, dset)
    graphs, _ = dgl.load_graphs(path)
    g = graphs[0]
    n = g.num_nodes()

    seeds = []
    if n > 0:
        degrees = (g.in_degrees() + g.out_degrees())
        top = torch.topk(degrees, k=min(5, n), largest=True).indices.tolist()
        seeds.extend(top)
    seeds = list(set(seeds))
    while len(seeds) < min(8, n):
        seeds.append(random.randint(0, n - 1))
    seeds = list(set(seeds))

    visited = set(seeds)
    frontier = list(seeds)
    while len(visited) < n_nodes and frontier:
        nxt = []
        for u in frontier:
            nbrs = g.successors(u).tolist()
            random.shuffle(nbrs)
            for v in nbrs:
                if v not in visited and len(visited) < n_nodes:
                    visited.add(v)
                    nxt.append(v)
                if len(visited) >= n_nodes:
                    break
            if len(visited) >= n_nodes:
                break
        frontier = nxt

    nodes = sorted(visited)
    idx = {u: i for i, u in enumerate(nodes)}
    labels = g.ndata['label']
    if labels.dim() > 1:
        labels = labels.argmax(1)
    degs = (g.in_degrees() + g.out_degrees())

    edges = []
    for u in nodes:
        for v in g.successors(u).tolist():
            if v in idx:
                edges.append([idx[u], idx[v]])

    sg = nx.Graph()
    sg.add_nodes_from(range(len(nodes)))
    sg.add_edges_from(edges)
    try:
        pos = nx.spring_layout(sg, seed=42, k=0.9, iterations=80)
    except Exception:
        pos = nx.circular_layout(sg)
    xmin, xmax = min(p[0] for p in pos.values()), max(p[0] for p in pos.values())
    ymin, ymax = min(p[1] for p in pos.values()), max(p[1] for p in pos.values())
    xr = max(xmax - xmin, 1e-6); yr = max(ymax - ymin, 1e-6)

    node_data = [{
        'id': u,
        'label': int(labels[u].item()),
        'degree': int(degs[u].item()),
        'x': (pos[i][0] - xmin) / xr * 2 - 1,
        'y': (pos[i][1] - ymin) / yr * 2 - 1,
    } for i, u in enumerate(nodes)]

    result = {
        'name': dset,
        'num_nodes_total': int(n),
        'num_edges_total': int(g.num_edges()),
        'sampled_nodes': node_data,
        'edges': edges[:4000],
    }
    GRAPH_CACHE[key] = result
    return result


# --------------------------------------------------------------------------
# Experiment run management
# --------------------------------------------------------------------------
def parse_progress_line(run, line):
    m = re.search(r'TRIAL NUMBER (\d+)', line)
    if m:
        run['parsed']['trial'] = int(m.group(1))
        run['parsed']['events'].append({'t': time.time(), 'type': 'trial', 'trial': int(m.group(1))})
    m = re.search(r'Starting round (\d+)\.\.\.', line)
    if m:
        run['parsed']['round'] = int(m.group(1))
        run['parsed']['events'].append({'t': time.time(), 'type': 'round_start', 'round': int(m.group(1))})
    m = re.search(r'Best Val: REC ([0-9.]+) PRE ([0-9.]+) MF1 ([0-9.]+) AUC ([0-9.]+) TP (\d+) FP (\d+) TN (\d+) FN (\d+)', line)
    if m:
        rec, pre, f1, auc = (float(m.group(i)) for i in range(1, 5))
        ev = {'t': time.time(), 'type': 'round_metric', 'round': run['parsed']['round'],
              'metric': {'rec': rec, 'prec': pre, 'f1': f1, 'auc': auc, 'tp': int(m.group(5)), 'fp': int(m.group(6)), 'tn': int(m.group(7)), 'fn': int(m.group(8))}}
        run['parsed']['events'].append(ev)
        run['parsed']['rounds'].append(ev)
    m = re.search(r'(Dataset - [^:]+|Seeds - [^:]+): REC ([0-9.]+) PRE ([0-9.]+) MF1 ([0-9.]+) AUC ([0-9.]+) TP (\d+) FP (\d+) TN (\d+) FN (\d+)', line)
    if m:
        run['parsed']['events'].append({
            't': time.time(), 'type': 'eval', 'scope': m.group(1).strip(),
            'round': run['parsed']['round'], 'metric': {
                'rec': float(m.group(2)), 'prec': float(m.group(3)),
                'f1': float(m.group(4)), 'auc': float(m.group(5)),
                'tp': int(m.group(6)), 'fp': int(m.group(7)), 'tn': int(m.group(8)), 'fn': int(m.group(9))}})
    m = re.search(r'Experiment ended, experienced (\d+) failures', line)
    if m:
        run['parsed']['failures'] = int(m.group(1))
    m = re.search(r'Elapsed experiment time ([0-9.]+)s', line)
    if m:
        run['parsed']['elapsed'] = float(m.group(1))
    m = re.search(r'\(best ([0-9.]+)\)', line)
    if m:
        run['parsed']['best_f1'] = float(m.group(1))


def reader_thread(run):
    proc = run['proc']
    try:
        for line in iter(proc.stdout.readline, ''):
            line = line.rstrip('\n')
            with RUNS_LOCK:
                run['lines'].append(line)
                try:
                    parse_progress_line(run, line)
                except Exception:
                    pass
                try:
                    run['cond'].notify_all()
                except Exception:
                    pass
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        rc = proc.wait()
        with RUNS_LOCK:
            run['returncode'] = rc
            run['state'] = 'done' if rc == 0 else 'failed'
            run['finished_at'] = time.time()
            try:
                run['cond'].notify_all()
            except Exception:
                pass


def start_run(config_payload):
    run_id = datetime.datetime.now().strftime("%y%m%d%H%M%S%f")
    cname = f"run_{run_id}"
    cfg_path = os.path.join(SCRIPT_DIR, f"{cname}.json")
    with open(cfg_path, 'w', encoding='utf-8') as f:
        json.dump(config_payload, f, indent=4)

    run = {
        'id': run_id,
        'cname': cname,
        'config': config_payload,
        'state': 'starting',
        'lines': [],
        'parsed': {'trial': 0, 'round': 0, 'rounds': [], 'events': [], 'failures': None, 'elapsed': None},
        'started_at': time.time(),
        'finished_at': None,
        'returncode': None,
        'proc': None,
        'cond': threading.Condition(RUNS_LOCK),
    }

    kwargs = {}
    if sys.platform == 'win32':
        kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW

    # Spawn the experiment worker directly with the base interpreter.
    # On Windows a venv's python.exe is a redirector that spawns the base
    # interpreter as a child and detaches from the captured pipe, which cuts
    # off the live output and makes the reader hang. Bypassing the redirector
    # (running the base interpreter with __PYVENV_LAUNCHER__ set) keeps the
    # worker as the direct child so the pipe stays open for the whole run.
    env = None
    python = sys.executable
    if sys.prefix != sys.base_prefix:
        base_python = os.path.join(sys.base_exec_prefix, 'python.exe')
        if os.path.isfile(base_python):
            python = base_python
            env = os.environ.copy()
            env['__PYVENV_LAUNCHER__'] = sys.executable

    proc = subprocess.Popen(
        [python, 'main.py', '-c', cname],
        cwd=SCRIPT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env,
        **kwargs,
    )
    run['proc'] = proc
    with RUNS_LOCK:
        run['state'] = 'running'
        RUNS[run_id] = run
    threading.Thread(target=reader_thread, args=(run,), daemon=True).start()
    return run


def stop_run(run_id):
    with RUNS_LOCK:
        run = RUNS.get(run_id)
    if not run or run['proc'] is None:
        return False
    try:
        run['proc'].terminate()
    except Exception:
        pass
    if sys.platform == 'win32':
        try:
            subprocess.run(['taskkill', '/PID', str(run['proc'].pid), '/T', '/F'],
                           capture_output=True, timeout=10)
        except Exception:
            pass
    with RUNS_LOCK:
        if run['state'] == 'running':
            run['state'] = 'stopped'
        run['cond'].notify_all()
    return True


def find_result_csv(run):
    folder = os.path.join(RESULT_DIR, run['cname'])
    if not os.path.isdir(folder):
        return None
    subs = sorted(os.listdir(folder))
    if not subs:
        return None
    latest = os.path.join(folder, subs[-1])
    combined = os.path.join(latest, 'combined_result.csv')
    if os.path.isfile(combined):
        return combined
    csvs = glob.glob(os.path.join(latest, '*.csv'))
    return csvs[0] if csvs else None


def run_status(run_id):
    with RUNS_LOCK:
        run = RUNS.get(run_id)
        if not run:
            return {'error': 'not found'}
        snapshot = {
            'id': run['id'],
            'state': run['state'],
            'started_at': run['started_at'],
            'finished_at': run['finished_at'],
            'returncode': run['returncode'],
            'parsed': run['parsed'],
            'line_count': len(run['lines']),
            'last_lines': run['lines'][-80:],
        }
    if run['state'] in ('done', 'failed', 'stopped'):
        csv = find_result_csv(run)
        snapshot['result_csv'] = csv
    return snapshot


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/health')
def health():
    return jsonify({'ok': True, 'import_error': _IMPORT_ERROR if UC is None else None,
                    'torch': torch.__version__ if torch else None,
                    'dgl': dgl.__version__ if dgl else None})


@app.route('/api/schema')
def schema():
    return jsonify(build_schema())


@app.route('/api/datasets')
def datasets():
    out = []
    for d in list_datasets(force=True):
        out.append(dataset_stats(d) or {'name': d, 'error': 'load failed'})
    return jsonify(out)


@app.route('/api/datasets/<dset>/stats')
def dset_stats(dset):
    s = dataset_stats(dset)
    return jsonify(s or {'error': 'not found'})


@app.route('/api/datasets/<dset>/graph')
def dset_graph(dset):
    try:
        n_nodes = int(request.args.get('n', 180))
        n_nodes = max(30, min(n_nodes, 600))
    except Exception:
        n_nodes = 180
    g = sample_graph(dset, n_nodes)
    return jsonify(g or {'error': 'not found'})


@app.route('/api/experiments')
def experiments():
    if not os.path.isdir(RESULT_DIR):
        return jsonify([])
    runs = []
    for cname in sorted(os.listdir(RESULT_DIR)):
        cdir = os.path.join(RESULT_DIR, cname)
        if not os.path.isdir(cdir):
            continue
        for ts in sorted(os.listdir(cdir)):
            tdir = os.path.join(cdir, ts)
            if not os.path.isdir(tdir):
                continue
            meta_path = os.path.join(tdir, 'meta.txt')
            desc = ''
            if os.path.isfile(meta_path):
                with open(meta_path, encoding='utf-8', errors='replace') as f:
                    desc = f.readline().strip()
            csvs = sorted([f for f in os.listdir(tdir) if f.endswith('.csv')])
            n_csvs = sum(1 for c in csvs if c != 'combined_result.csv')
            has_combined = 'combined_result.csv' in csvs
            runs.append({
                'cname': cname, 'ts': ts, 'desc': desc,
                'csvs': csvs, 'n_csvs': n_csvs, 'has_combined': has_combined,
            })
    return jsonify(runs)


@app.route('/api/experiments/<cname>/<ts>/meta')
def experiment_meta(cname, ts):
    p = os.path.join(RESULT_DIR, cname, ts, 'meta.txt')
    if not os.path.isfile(p):
        abort(404)
    with open(p, encoding='utf-8', errors='replace') as f:
        return jsonify({'meta': f.read()})


@app.route('/api/experiments/<cname>/<ts>/<csvfile>')
def experiment_csv(cname, ts, csvfile):
    p = os.path.join(RESULT_DIR, cname, ts, csvfile)
    if not os.path.isfile(p) or not csvfile.endswith('.csv'):
        abort(404)
    df = pd_read_csv(p)
    return jsonify({'columns': df['columns'], 'rows': df['rows']})


def pd_read_csv(p):
    import pandas as pd
    df = pd.read_csv(p)
    return {'columns': list(df.columns), 'rows': df.to_dict(orient='records')}


@app.route('/api/run', methods=['POST'])
def launch():
    payload = request.get_json(force=True)
    required = ['TRIAL_NUM', 'FAILURE_LIMIT', 'EXPERIMENT_DESC', 'LIST_DSET', 'LIST_TRAIN_DSET', 'EXP_DICT']
    for k in required:
        if k not in payload:
            return jsonify({'error': f'missing key {k}'}), 400
    run = start_run(payload)
    return jsonify({'run_id': run['id']})


@app.route('/api/run/<run_id>')
def run_status_api(run_id):
    return jsonify(run_status(run_id))


@app.route('/api/run/<run_id>/stop', methods=['POST'])
def run_stop(run_id):
    ok = stop_run(run_id)
    return jsonify({'ok': ok})


@app.route('/api/run/<run_id>/stream')
def run_stream(run_id):
    with RUNS_LOCK:
        run = RUNS.get(run_id)
        if not run:
            return jsonify({'error': 'not found'}), 404
        start_idx = len(run['lines'])

    def gen():
        idx = start_idx
        while True:
            with RUNS_LOCK:
                run_local = RUNS.get(run_id)
                if not run_local:
                    yield "event: error\ndata: {}\n\n"
                    return
                new_lines = run_local['lines'][idx:]
                idx = len(run_local['lines'])
                state = run_local['state']
                parsed = run_local['parsed']
                for line in new_lines:
                    yield f"data: {json.dumps({'type': 'log', 'line': line})}\n\n"
                if state != 'running':
                    yield f"event: state\ndata: {json.dumps({'state': state, 'parsed': parsed, 'returncode': run_local['returncode']})}\n\n"
                    return
                yield ": keepalive\n\n"
            time.sleep(0.2)

    return Response(gen(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/runs')
def runs_list():
    with RUNS_LOCK:
        out = [{'id': r['id'], 'state': r['state'], 'desc': r['config'].get('EXPERIMENT_DESC', ''),
                'started_at': r['started_at'], 'cname': r['cname']} for r in RUNS.values()]
    return jsonify(sorted(out, key=lambda x: -x['started_at']))


@app.route('/static/<path:path>')
def static_files(path):
    return send_from_directory(os.path.join(os.path.dirname(__file__), 'static'), path)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    print(f"TPNE-XGB Dashboard running at http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=False, threaded=True)
