"""Flask HTTP API for the Adaptive Defensive Layer (ADL) platform.

Endpoints
---------
GET  /api/health          - liveness + version
GET  /api/schema          - configuration schema + feature schema + ADL fields
GET  /api/datasets        - ready-made experiment presets
POST /api/run             - launch a new simulation (config in the body)
GET  /api/stream/<id>     - Server-Sent-Events feed of a running simulation
GET  /api/run/<id>        - snapshot status of one simulation
GET  /api/report/<id>     - full report (metrics, generator stats, defence)
GET  /api/history         - metadata of every run started this session
GET  /api/graph/<id>      - node/edge dump for the graph explorer (incl. ADL)
"""

import json
import os
import threading
import time
import uuid

from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS

from .simulation import DEFAULTS, Simulation
from .adl import (DEFAULT_T1, DEFAULT_T2, DEFAULT_ALPHA,
                  DEFAULT_REVIEW_CATCH_RATE, DEFAULT_WEIGHTS,
                  RISK_COMPONENT_NAMES)
from .world import (FEATURE_NAMES, INTRINSIC_NAMES, GRAPH_NAMES,
                    INITIAL_STRATEGIES)

app = Flask(__name__, static_folder=None)
CORS(app)

SIMS = {}               # sim_id -> Simulation
HISTORY = []            # recent run metadata
HISTORY_LOCK = threading.Lock()

DECISIONS = ['allow', 'review', 'block']


def _json_default(obj):
    if isinstance(obj, (int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_default(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_default(v) for v in obj]
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _run_summary(sim):
    snap = sim.snapshot()
    return {
        'id': sim.id,
        'state': sim.state,
        'describe': sim.config.get('describe', ''),
        'generator_mode': sim.config.get('generator_mode'),
        'gen_type': sim.config.get('gen_type'),
        'adl_enabled': sim.config.get('adl_enabled'),
        'threshold_policy': sim.config.get('threshold_policy'),
        'rounds': sim.config.get('rounds'),
        'rounds_done': snap['rounds_done'],
        'num_nodes': snap['num_nodes'],
        'active_nodes': snap['active_nodes'],
        'blocked_nodes': snap['blocked_nodes'],
        'num_fraud': snap['num_fraud'],
        'started_at': sim.started_at,
        'finished_at': sim.finished_at,
    }


def _iter_events(sim, after_index):
    """Yield SSE-formatted events newer than ``after_index``."""
    while sim.state not in ('done', 'error', 'stopped'):
        with sim.lock:
            pending = list(sim.events[after_index:])
            after_index = len(sim.events)
        for ev in pending:
            yield f'data: {json.dumps(ev, default=_json_default)}\n\n'
        time.sleep(0.15)
    # drain the tail
    with sim.lock:
        pending = list(sim.events[after_index:])
    for ev in pending:
        yield f'data: {json.dumps(ev, default=_json_default)}\n\n'
    final = {'type': 'state', 'state': sim.state,
             'error': sim.error, 'finished': True}
    yield f'data: {json.dumps(final, default=_json_default)}\n\n'


# ---------------------------------------------------------------------------
# schema / datasets
# ---------------------------------------------------------------------------
@app.get('/api/health')
def health():
    return jsonify({'ok': True, 'service': 'adaptive-defensive-layer',
                    'time': time.time()})


@app.get('/api/schema')
def schema():
    return jsonify({
        'defaults': DEFAULTS,
        'feature_names': FEATURE_NAMES,
        'intrinsic_names': INTRINSIC_NAMES,
        'graph_names': GRAPH_NAMES,
        'decisions': DECISIONS,
        'risk_components': RISK_COMPONENT_NAMES,
        'adl_defaults': {
            'policy': 'adaptive',
            't1': DEFAULT_T1,
            't2': DEFAULT_T2,
            'alpha': DEFAULT_ALPHA,
            'review_catch_rate': DEFAULT_REVIEW_CATCH_RATE,
            'weights': DEFAULT_WEIGHTS,
        },
        'strategies': {k: {'device_spray': v['device_spray'],
                           'ip_reuse': v['ip_reuse'],
                           'ring_affinity': v['ring_affinity']}
                       for k, v in INITIAL_STRATEGIES.items()},
    })


@app.get('/api/datasets')
def datasets():
    return jsonify([
        {'key': 'quick', 'label': 'Quick demo (3 rounds, ADL adaptive)',
         'config': {'rounds': 3, 'base_accounts': 300, 'initial_fraud': 40,
                    'genuine_per_round': 35, 'fraud_per_round': 20,
                    'gan_epochs': 100, 'supervised_ratio': 0.25,
                    'budget_pos': 6, 'budget_neg': 15,
                    'adl_enabled': True, 'threshold_policy': 'adaptive',
                    'describe': 'Quick ADL demo'}},
        {'key': 'quick_fixed', 'label': 'Quick demo (fixed thresholds)',
         'config': {'rounds': 3, 'base_accounts': 300, 'initial_fraud': 40,
                    'genuine_per_round': 35, 'fraud_per_round': 20,
                    'gan_epochs': 100, 'supervised_ratio': 0.25,
                    'budget_pos': 6, 'budget_neg': 15,
                    'adl_enabled': True, 'threshold_policy': 'fixed',
                    'describe': 'Quick ADL fixed-threshold baseline'}},
        {'key': 'no_adl', 'label': 'Standard (8 rounds, no ADL baseline)',
         'config': {'rounds': 8, 'base_accounts': 500, 'initial_fraud': 60,
                    'genuine_per_round': 45, 'fraud_per_round': 30,
                    'gan_epochs': 120, 'supervised_ratio': 0.25,
                    'budget_pos': 6, 'budget_neg': 15,
                    'adl_enabled': False,
                    'describe': 'Standard without defense (baseline)'}},
        {'key': 'default', 'label': 'Standard (8 rounds, ADL adaptive)',
         'config': {'rounds': 8, 'base_accounts': 500, 'initial_fraud': 60,
                    'genuine_per_round': 45, 'fraud_per_round': 30,
                    'gan_epochs': 120, 'supervised_ratio': 0.25,
                    'budget_pos': 6, 'budget_neg': 15,
                    'adl_enabled': True, 'threshold_policy': 'adaptive',
                    'describe': 'Standard ADL experiment'}},
        {'key': 'deep', 'label': 'Long evolution (10 rounds, ADL adaptive)',
         'config': {'rounds': 10, 'base_accounts': 800, 'initial_fraud': 90,
                    'genuine_per_round': 60, 'fraud_per_round': 40,
                    'gan_epochs': 150, 'supervised_ratio': 0.22,
                    'budget_pos': 8, 'budget_neg': 20,
                    'adl_enabled': True, 'threshold_policy': 'adaptive',
                    'describe': 'Long evolution with ADL'}},
    ])


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------
@app.post('/api/run')
def start_run():
    body = request.get_json(silent=True) or {}
    sim_id = str(uuid.uuid4())[:8]
    sim = Simulation(sim_id, body)
    SIMS[sim_id] = sim
    with HISTORY_LOCK:
        HISTORY.append({'id': sim_id, 'describe': sim.config.get('describe', ''),
                        'state': 'running', 'started_at': sim.started_at})
    sim.start()
    return jsonify({'id': sim_id, 'state': sim.state,
                    'config': sim.config})


@app.get('/api/run/<sim_id>')
def run_status(sim_id):
    sim = SIMS.get(sim_id)
    if sim is None:
        return jsonify({'error': 'run not found'}), 404
    return jsonify(_run_summary(sim))


@app.get('/api/report/<sim_id>')
def report(sim_id):
    sim = SIMS.get(sim_id)
    if sim is None:
        return jsonify({'error': 'run not found'}), 404
    return jsonify(sim.report())


@app.get('/api/graph/<sim_id>')
def graph(sim_id):
    sim = SIMS.get(sim_id)
    if sim is None or sim.world is None:
        return jsonify({'error': 'run not found or not started'}), 404
    world = sim.world
    nodes = []
    for a in world.accounts:
        nodes.append({
            'id': int(a.idx),
            'label': int(a.label),
            'round': int(a.creation_round),
            'base': a.base_strategy or '',
            'strategy': a.strategy or '',
            'device': int(a.device_id),
            'ip': int(a.ip_id),
            'blocked': bool(a.blocked),
            'blocked_round': int(a.blocked_round),
            'decision': a.decision or '',
            'reviewed': bool(a.reviewed),
            'risk': round(float(a.risk), 4),
            'attrs': {k: round(float(v), 3) for k, v in a.attrs.items()},
        })
    edges = {
        'referral': [[int(x) for x in e] for e in world.referral],
        'device': [[int(x) for x in e] for e in world.device_edges],
        'ip': [[int(x) for x in e] for e in world.ip_edges],
    }
    return jsonify({'nodes': nodes, 'edges': edges})


@app.get('/api/history')
def history():
    out = []
    with HISTORY_LOCK:
        for h in HISTORY[-50:]:
            sim = SIMS.get(h['id'])
            row = dict(h)
            if sim is not None:
                row['state'] = sim.state
                row['describe'] = sim.config.get('describe', h.get('describe', ''))
                row['rounds'] = sim.config.get('rounds')
                row['generator_mode'] = sim.config.get('generator_mode')
                row['gen_type'] = sim.config.get('gen_type')
                row['adl_enabled'] = sim.config.get('adl_enabled')
                row['threshold_policy'] = sim.config.get('threshold_policy')
            out.append(row)
    return jsonify(out)


@app.get('/api/stream/<sim_id>')
def stream(sim_id):
    sim = SIMS.get(sim_id)
    if sim is None:
        return jsonify({'error': 'run not found'}), 404
    return Response(
        stream_with_context(_iter_events(sim, 0)),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache',
                 'X-Accel-Buffering': 'no'},
    )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '5050'))
    print(f'Adaptive Defensive Layer dashboard -> http://127.0.0.1:{port}')
    app.run(host='127.0.0.1', port=port, threaded=True, debug=False)
