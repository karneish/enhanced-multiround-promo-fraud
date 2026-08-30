"""Flask REST + SSE API for Adaptive Ensemble Detector."""
import json
import time
import threading

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

from .simulation import Simulation, DEFAULTS

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

SIMS = {}
HISTORY = []
_lock = threading.Lock()


def _add_history(meta):
    with _lock:
        for i, m in enumerate(HISTORY):
            if m['id'] == meta['id']:
                HISTORY[i] = meta
                return
        HISTORY.append(meta)


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'service': 'adaptive-ensemble-detector'})


@app.route('/api/schema')
def schema():
    return jsonify({
        'defaults': DEFAULTS,
        'model_names': ['XGBoost', 'HistGradientBoosting', 'ExtraTrees'],
        'ads_components': ['f1', 'recall', 'stability', 'historical'],
        'feature_names': [
            'age', 'email_disposable', 'phone_verified', 'device_fresh',
            'ip_proxy', 'loc_entropy', 'login_night', 'amount_mean',
            'amount_std', 'txn_count', 'txn_freq',
            'referral_count', 'degree', 'clustering', 'shared_device', 'shared_ip', 'fraud_neighbor_ratio',
        ],
        'fraud_strategies': list({
            'fake_identity', 'referral_farming', 'device_spray', 'vpn_hop', 'quiet_sampler', 'evolved'
        }),
    })


@app.route('/api/datasets')
def datasets():
    presets = [
        {
            'label': 'Quick (3 rounds)',
            'config': {
                'rounds': 3, 'base_accounts': 300, 'initial_fraud': 40,
                'genuine_per_round': 30, 'fraud_per_round': 20,
                'supervised_ratio': 0.25, 'budget_pos': 5, 'budget_neg': 10,
                'seed': 42, 'describe': 'Quick adaptive ensemble test',
            },
        },
        {
            'label': 'Standard (5 rounds)',
            'config': {
                'rounds': 5, 'base_accounts': 500, 'initial_fraud': 60,
                'genuine_per_round': 45, 'fraud_per_round': 30,
                'supervised_ratio': 0.25, 'budget_pos': 6, 'budget_neg': 15,
                'seed': 42, 'describe': 'Standard adaptive ensemble experiment',
            },
        },
        {
            'label': 'Long (8 rounds)',
            'config': {
                'rounds': 8, 'base_accounts': 700, 'initial_fraud': 80,
                'genuine_per_round': 50, 'fraud_per_round': 35,
                'supervised_ratio': 0.30, 'budget_pos': 8, 'budget_neg': 20,
                'seed': 42, 'describe': 'Long adaptive ensemble study',
            },
        },
        {
            'label': 'Extended (10 rounds)',
            'config': {
                'rounds': 10, 'base_accounts': 900, 'initial_fraud': 100,
                'genuine_per_round': 60, 'fraud_per_round': 40,
                'supervised_ratio': 0.30, 'budget_pos': 10, 'budget_neg': 25,
                'seed': 42, 'describe': 'Extended adaptive ensemble experiment',
            },
        },
        {
            'label': 'Deep (12 rounds)',
            'config': {
                'rounds': 12, 'base_accounts': 1100, 'initial_fraud': 120,
                'genuine_per_round': 70, 'fraud_per_round': 50,
                'supervised_ratio': 0.35, 'budget_pos': 12, 'budget_neg': 30,
                'seed': 42, 'describe': 'Deep adaptive ensemble study',
            },
        },
    ]
    return jsonify(presets)


@app.route('/api/run', methods=['POST'])
def run_simulation():
    cfg = request.get_json(force=True, silent=True) or {}
    sim = Simulation(cfg)
    SIMS[sim.id] = sim
    meta = {
        'id': sim.id,
        'describe': sim.cfg.get('describe', 'Untitled'),
        'config': sim.cfg,
        'state': 'pending',
        'rounds': 0,
        'started_at': sim.started_at,
    }
    _add_history(meta)
    sim.start()
    meta['state'] = 'running'
    _add_history(meta)
    return jsonify(meta)


@app.route('/api/stream/<sim_id>')
def stream(sim_id):
    sim = SIMS.get(sim_id)
    if not sim:
        return jsonify({'error': 'not found'}), 404

    def generate():
        idx = 0
        while True:
            if idx < len(sim.events):
                ev = sim.events[idx]
                yield f"data: {json.dumps(ev)}\n\n"
                idx += 1
            elif sim.state in ('done', 'error', 'stopped'):
                remaining = sim.events[idx:]
                for ev in remaining:
                    yield f"data: {json.dumps(ev)}\n\n"
                if not any(ev.get('type') == 'state' for ev in remaining):
                    yield f"data: {json.dumps({'type': 'state', 'state': sim.state, 'finished': True})}\n\n"
                break
            else:
                time.sleep(0.15)

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/run/<sim_id>')
def get_run(sim_id):
    sim = SIMS.get(sim_id)
    if not sim:
        return jsonify({'error': 'not found'}), 404
    return jsonify({
        'id': sim.id,
        'describe': sim.cfg.get('describe', ''),
        'state': sim.state,
        'rounds_done': len(sim.round_results),
        'config': sim.cfg,
    })


@app.route('/api/report/<sim_id>')
def get_report(sim_id):
    sim = SIMS.get(sim_id)
    if not sim:
        return jsonify({'error': 'not found'}), 404
    return jsonify({
        'id': sim.id,
        'config': sim.cfg,
        'state': sim.state,
        'rounds': sim.round_results,
        'total_time': sim.report.get('total_time', 0) if sim.report else 0,
    })


@app.route('/api/history')
def get_history():
    with _lock:
        h = list(HISTORY)
    h.sort(key=lambda x: x.get('started_at', 0), reverse=True)
    for item in h:
        sim = SIMS.get(item['id'])
        if sim:
            item['state'] = sim.state
            item['rounds'] = len(sim.round_results)
    return jsonify(h)
