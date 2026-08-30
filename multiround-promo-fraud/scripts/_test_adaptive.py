"""
Smoke test for the Adaptive Multi-Model Detector.

Tests:
1. ADS scoring mechanism
2. Import and model registration
3. Full adaptive experiment (2 rounds)
4. Single-model baseline (e.g. RF-only via AdaptiveDetector)
5. Save/load cycle

Usage:
    python _test_adaptive.py
"""
import sys
import os
import gc

sys.path.append('../src')

import torch
import dgl
import numpy as np

from utils.utils_const import (
    DEFAULT_MAIN_CONFIG, DEFAULT_TRAIN_CONFIG,
    DEFAULT_ADVER_CONFIG, DEFAULT_MODEL_CONFIG, DEFAULT_STRAT_CONFIG
)
from experiment.supervised_multi import MultiroundExperiment


def create_test_graph():
    num_nodes = 200
    num_edges = 600
    feat_dim = 32
    num_pos = 20
    num_neg = 180

    features = torch.randn(num_nodes, feat_dim)
    labels = torch.zeros(num_nodes, dtype=torch.long)
    labels[:num_pos] = 1

    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    graph = dgl.graph((src, dst), num_nodes=num_nodes)
    graph = dgl.add_self_loop(graph)
    graph.ndata['feature'] = features
    graph.ndata['label'] = labels

    return graph


def make_configs(round_num=2, model_name='ADAPTIVE', adaptive_model_list=None, adaptive_components=None):
    main_config = DEFAULT_MAIN_CONFIG.copy()
    train_config = DEFAULT_TRAIN_CONFIG.copy()
    model_config = DEFAULT_MODEL_CONFIG.copy()
    strat_config = DEFAULT_STRAT_CONFIG.copy()
    adver_config = DEFAULT_ADVER_CONFIG.copy()

    main_config['device'] = 'cpu'
    main_config['exp_type'] = 'ADVER'
    main_config['task_type'] = 'NODE'
    main_config['round_num'] = round_num
    main_config['round_new_pos'] = 5
    main_config['round_new_neg'] = 20
    main_config['round_budget_pos'] = 0
    main_config['round_budget_neg'] = 0

    model_config['model_name'] = model_name
    model_config['embed_type'] = 'temporal'
    model_config['h_feats'] = 32
    model_config['num_layers'] = 1
    model_config['round_window'] = 3
    model_config['num_epoch'] = 3
    model_config['num_round_epoch'] = 3
    model_config['early_stopping'] = 3
    model_config['mlp_feats'] = 32

    if adaptive_model_list is not None:
        model_config['adaptive_model_list'] = adaptive_model_list
    if adaptive_components is not None:
        model_config['adaptive_components'] = adaptive_components

    train_config['num_epoch'] = 3
    train_config['num_round_epoch'] = 3
    train_config['early_stopping'] = 3
    train_config['round_reset_model'] = False

    adver_config['adver_choose_name'] = 'GREEDY'
    adver_config['adver_mod_name'] = 'INTELLIGENT'
    adver_config['adver_gen_type'] = 'GAN'
    adver_config['adver_gen_epochs'] = 10
    adver_config['adver_gen_feat_coef'] = 1.0
    adver_config['adver_gen_conn_coef'] = 0.5
    adver_config['adver_gen_ring_ratio'] = 0.5

    return main_config, train_config, model_config, strat_config, adver_config


def run_experiment(graph, main_config, model_config, strat_config, adver_config, train_config):
    exp = MultiroundExperiment(
        graph,
        main_config=main_config,
        model_config=model_config,
        strat_config=strat_config,
        adver_config=adver_config,
        train_config=train_config
    )

    for round_num in range(main_config['round_num']):
        success = exp.one_round_node(round_num)
        if not success:
            return None
    return exp


def test_ads():
    print("="*50)
    print("TEST 1: AdaptiveDetectorScore")
    print("="*50)

    from models.proposed_supervised.adaptive_detector import AdaptiveDetectorScore

    ads = AdaptiveDetectorScore(
        model_names=['A', 'B', 'C'],
        history_window=5,
        components=['f1', 'recall', 'stability', 'historical'],
    )

    ads.update('A', f1=0.90, recall=0.85)
    ads.update('B', f1=0.80, recall=0.90)
    ads.update('C', f1=0.70, recall=0.70)

    weights = ads.compute_weights()
    scores = ads.compute_all_scores()

    print(f"  Scores: {scores}")
    print(f"  Weights: {weights}")
    assert abs(sum(weights.values()) - 1.0) < 1e-6, "Weights should sum to 1"
    print("  PASS: Weights sum to 1.0")

    ads.update('A', f1=0.85, recall=0.80)
    ads.update('B', f1=0.88, recall=0.92)
    ads.update('C', f1=0.72, recall=0.71)

    weights2 = ads.compute_weights()
    scores2 = ads.compute_all_scores()
    print(f"\n  After round 2:")
    print(f"  Scores: {scores2}")
    print(f"  Weights: {weights2}")
    assert abs(sum(weights2.values()) - 1.0) < 1e-6
    print("  PASS: Weights still sum to 1.0 after 2 rounds")

    print("\n  Test empty components (equal average):")
    ads_eq = AdaptiveDetectorScore(model_names=['A', 'B'], components=[])
    ads_eq.update('A', f1=0.9, recall=0.8)
    ads_eq.update('B', f1=0.7, recall=0.9)
    weights_eq = ads_eq.compute_weights()
    print(f"  Equal weights: {weights_eq}")
    assert weights_eq['A'] == 0.5 and weights_eq['B'] == 0.5
    print("  PASS: Empty components gives equal weights")

    print("  ALL ADS TESTS PASSED\n")


def test_import():
    print("="*50)
    print("TEST 2: Import and Model Registration")
    print("="*50)

    from utils.utils_const import MODEL_DICT
    assert 'ADAPTIVE' in MODEL_DICT, "ADAPTIVE not found in MODEL_DICT"
    print(f"  MODEL_DICT keys: {list(MODEL_DICT.keys())}")
    print("  PASS: ADAPTIVE registered in MODEL_DICT\n")


def test_full_adaptive():
    print("="*50)
    print("TEST 3: Full Adaptive Experiment (2 rounds)")
    print("="*50)

    graph = create_test_graph()
    print(f"  Created graph: {graph.num_nodes()} nodes, {graph.num_edges()} edges")
    print(f"  Labels: {dict(zip(*torch.unique(graph.ndata['label'], return_counts=True)))}")

    main_config, train_config, model_config, strat_config, adver_config = make_configs(
        round_num=2,
        model_name='ADAPTIVE',
        adaptive_model_list=['XGBoost', 'RandomForest', 'ExtraTrees', 'HistGradientBoosting', 'LogisticRegression'],
        adaptive_components=['f1', 'recall', 'stability', 'historical'],
    )

    exp = run_experiment(graph, main_config, model_config, strat_config, adver_config, train_config)
    assert exp is not None, "Experiment failed"
    assert type(exp.model).__name__ == 'AdaptiveDetector'
    print("  PASS: Correct model type instantiated")

    model = exp.model
    print(f"  Weight history rounds: {len(model.round_weight_history)}")
    print(f"  Score history rounds: {len(model.round_score_history)}")
    for name in model.model_names:
        f1_hist = model.round_individual_f1[name]
        print(f"    {name} F1 history: {[f'{x:.4f}' for x in f1_hist]}")

    if len(model.round_weight_history) > 1:
        w0 = model.round_weight_history[0]
        w1 = model.round_weight_history[1]
        weights_shifted = any(abs(w0[n] - w1[n]) > 1e-4 for n in model.model_names)
        print(f"  Weights shifted between rounds: {weights_shifted}")

    print("\n  Testing save/load...")
    save_path = '../checkpoint/_test_adaptive_save'
    model.save_model(save_path)
    exp.clean_temp_files()

    del exp, graph
    gc.collect()

    graph2 = create_test_graph()
    exp2 = run_experiment(graph2, main_config, model_config, strat_config, adver_config, train_config)
    assert exp2 is not None
    exp2.model.load_model(save_path)
    print(f"  Loaded model, XGBoost fitted: {exp2.model.classifier_fitted.get('XGBoost', False)}")
    print(f"  Loaded model, RF fitted: {exp2.model.classifier_fitted.get('RandomForest', False)}")

    for suffix in ['_xgb.json', '_sklearn.pt', '_ads_state.json']:
        path = f"{save_path}{suffix}"
        if os.path.exists(path):
            os.remove(path)
            print(f"  Cleaned up {path}")

    del exp2, graph2
    gc.collect()

    print("  ALL FULL ADAPTIVE TESTS PASSED\n")


def test_single_model_baseline():
    print("="*50)
    print("TEST 4: Single-Model Baseline (RF via AdaptiveDetector)")
    print("="*50)

    graph = create_test_graph()
    print(f"  Created graph: {graph.num_nodes()} nodes, {graph.num_edges()} edges")

    main_config, train_config, model_config, strat_config, adver_config = make_configs(
        round_num=2,
        model_name='ADAPTIVE',
        adaptive_model_list=['RandomForest'],
        adaptive_components=[],
    )

    exp = run_experiment(graph, main_config, model_config, strat_config, adver_config, train_config)
    assert exp is not None, "Single-model experiment failed"

    model = exp.model
    assert model.model_names == ['RandomForest']
    print(f"  Model names: {model.model_names}")

    if model.round_weight_history:
        w = model.round_weight_history[-1]
        print(f"  Final weights: {w}")
        assert abs(w['RandomForest'] - 1.0) < 1e-6, "Single model should have weight 1.0"

    if model.round_individual_f1['RandomForest']:
        f1s = model.round_individual_f1['RandomForest']
        print(f"  RF F1 history: {[f'{x:.4f}' for x in f1s]}")

    exp.clean_temp_files()
    del exp, graph
    gc.collect()

    print("  ALL SINGLE-MODEL BASELINE TESTS PASSED\n")


if __name__ == '__main__':
    test_ads()
    test_import()
    test_full_adaptive()
    test_single_model_baseline()

    print("="*50)
    print("ALL SMOKE TESTS PASSED!")
    print("="*50)
