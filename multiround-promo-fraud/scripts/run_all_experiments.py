"""
Full Experiment Runner for Adaptive Multi-Model Detection Layer Research

Runs all comparison experiments:
  1. Baseline: TPNE + XGBoost (XGB-SP, base paper)
  2. Individual models: TPNE + RF/ET/HGB/LR (via AdaptiveDetector with single model)
  3. Proposed: TPNE + Adaptive Multi-Model Detector (full ADS)
  4. Ablation variants of the adaptive detector

Usage:
    python run_all_experiments.py -c config_adaptive
    python run_all_experiments.py -c config_adaptive --skip-baseline
    python run_all_experiments.py -c config_adaptive --only-adaptive
    python run_all_experiments.py -c config_adaptive --variants baseline_xgb individual_rf adaptive_full
"""

import argparse
import sys
import os
import copy
import gc
import json
import itertools
import datetime
import warnings

import torch
import dgl
import pandas as pd
import numpy as np

from time import time

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

sys.path.append('../src')

from experiment.supervised_multi import MultiroundExperiment
from utils.utils_const import (
    DEFAULT_MAIN_CONFIG, DEFAULT_TRAIN_CONFIG,
    DEFAULT_ADVER_CONFIG, DEFAULT_MODEL_CONFIG,
    DEFAULT_STRAT_CONFIG, LOSS_DICT, BACKBONE_DICT
)


SINGLE_MODEL_BASELINES = {
    'XGBoost': ['XGBoost'],
    'RandomForest': ['RandomForest'],
    'ExtraTrees': ['ExtraTrees'],
    'HistGradientBoosting': ['HistGradientBoosting'],
    'LogisticRegression': ['LogisticRegression'],
}


EXPERIMENT_VARIANTS = {
    # === BASELINE ===
    'baseline_xgb': {
        'model_name': 'XGB-SP',
        'description': 'Baseline: TPNE + XGBoost (base paper classifier)',
        'category': 'baseline',
    },

    # === INDIVIDUAL MODELS (using AdaptiveDetector with single model) ===
    'individual_xgb': {
        'model_name': 'ADAPTIVE',
        'adaptive_model_list': ['XGBoost'],
        'adaptive_components': [],
        'description': 'Single: TPNE + XGBoost (via AdaptiveDetector)',
        'category': 'individual',
    },
    'individual_rf': {
        'model_name': 'ADAPTIVE',
        'adaptive_model_list': ['RandomForest'],
        'adaptive_components': [],
        'description': 'Single: TPNE + Random Forest',
        'category': 'individual',
    },
    'individual_et': {
        'model_name': 'ADAPTIVE',
        'adaptive_model_list': ['ExtraTrees'],
        'adaptive_components': [],
        'description': 'Single: TPNE + Extra Trees',
        'category': 'individual',
    },
    'individual_hgb': {
        'model_name': 'ADAPTIVE',
        'adaptive_model_list': ['HistGradientBoosting'],
        'adaptive_components': [],
        'description': 'Single: TPNE + HistGradientBoosting',
        'category': 'individual',
    },
    'individual_lr': {
        'model_name': 'ADAPTIVE',
        'adaptive_model_list': ['LogisticRegression'],
        'adaptive_components': [],
        'description': 'Single: TPNE + Logistic Regression',
        'category': 'individual',
    },
    'individual_lgb': {
        'model_name': 'ADAPTIVE',
        'adaptive_model_list': ['LightGBM'],
        'adaptive_components': [],
        'description': 'Single: TPNE + LightGBM (via AdaptiveDetector)',
        'category': 'individual',
    },

    # === PROPOSED ===
    'adaptive_full': {
        'model_name': 'ADAPTIVE',
        'description': 'Proposed: TPNE + Adaptive Multi-Model Detector (full ADS)',
        'category': 'proposed',
    },
    'adaptive_6model': {
        'model_name': 'ADAPTIVE',
        'adaptive_model_list': ['XGBoost', 'RandomForest', 'ExtraTrees', 'HistGradientBoosting', 'LogisticRegression', 'LightGBM'],
        'description': 'Enhanced: 6-Model Ensemble (XGB+RF+ET+HGB+LR+LGB)',
        'category': 'proposed',
    },
    'adaptive_6model_dfeat': {
        'model_name': 'ADAPTIVE',
        'adaptive_model_list': ['XGBoost', 'RandomForest', 'ExtraTrees', 'HistGradientBoosting', 'LogisticRegression', 'LightGBM'],
        'addon_name': 'DFEAT',
        'addon_perc': 0.05,
        'addon_round_window': 3,
        'addon_internal_agg': 'OR',
        'description': 'Enhanced: 6-Model + DFEAT Addon (FTHR+DEGREE)',
        'category': 'proposed',
    },
    'adaptive_6model_dfeat_mixed': {
        'model_name': 'ADAPTIVE',
        'adaptive_model_list': ['XGBoost', 'RandomForest', 'ExtraTrees', 'HistGradientBoosting', 'LogisticRegression', 'LightGBM'],
        'embed_type': 'mixed',
        'addon_name': 'DFEAT',
        'addon_perc': 0.05,
        'addon_round_window': 3,
        'addon_internal_agg': 'OR',
        'description': 'Enhanced: Mixed Embedder + 6-Model + DFEAT',
        'category': 'proposed',
    },

    # === ABLATIONS ===
    'adaptive_no_stability': {
        'model_name': 'ADAPTIVE',
        'adaptive_components': ['f1', 'recall', 'historical'],
        'description': 'Ablation: Adaptive without stability component',
        'category': 'ablation',
    },
    'adaptive_no_historical': {
        'model_name': 'ADAPTIVE',
        'adaptive_components': ['f1', 'recall', 'stability'],
        'description': 'Ablation: Adaptive without historical component',
        'category': 'ablation',
    },
    'adaptive_f1_only': {
        'model_name': 'ADAPTIVE',
        'adaptive_components': ['f1'],
        'description': 'Ablation: Adaptive with F1-only scoring',
        'category': 'ablation',
    },
    'adaptive_equal_avg': {
        'model_name': 'ADAPTIVE',
        'adaptive_components': [],
        'adaptive_weights': {},
        'description': 'Ablation: Simple equal-weight average ensemble',
        'category': 'ablation',
    },
}


def load_graph(dataset_path):
    dataset, _ = dgl.load_graphs(dataset_path)
    graph = dataset[0].long()
    if len(graph.ndata['label'].shape) > 1:
        graph.ndata['label'] = graph.ndata['label'].argmax(1)
        graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
    graph.ndata['feature'] = graph.ndata['feature'].float()
    return graph


def run_single_experiment(
    variant_name, variant_config, dataset_path, train_dataset_path,
    main_config, train_config, model_config, strat_config, adver_config,
    trial_num=1, failure_limit=2
):
    print(f'\n{"="*60}')
    print(f'EXPERIMENT: {variant_name}')
    print(f'DESCRIPTION: {variant_config["description"]}')
    print(f'{"="*60}')

    model_config_run = copy.deepcopy(model_config)
    main_config_run = copy.deepcopy(main_config)
    train_config_run = copy.deepcopy(train_config)
    strat_config_run = copy.deepcopy(strat_config)
    adver_config_run = copy.deepcopy(adver_config)

    model_config_run['model_name'] = variant_config['model_name']

    for key in ['adaptive_components', 'adaptive_weights', 'adaptive_model_list',
                'adaptive_history_window', 'adaptive_ema_alpha', 'adaptive_recall_importance']:
        if key in variant_config:
            model_config_run[key] = variant_config[key]

    for key in ['num_epoch', 'num_round_epoch', 'early_stopping']:
        model_config_run[key] = train_config_run[key]

    graph = load_graph(dataset_path)
    pos = (graph.ndata['label'] == 1).sum().item()
    neg = (graph.ndata['label'] == 0).sum().item()
    main_config_run['round_new_pos'] = int(0.05 * pos)
    main_config_run['round_new_neg'] = int(0.05 * neg)
    main_config_run['round_budget_pos'] = 0
    main_config_run['round_budget_neg'] = 0
    del graph

    if train_dataset_path and train_dataset_path != 'NONE':
        train_graph = load_graph(train_dataset_path)
    else:
        train_graph = None

    dfs = []
    trial_counter, failure_counter = 0, 0
    start = time()

    while trial_counter < trial_num:
        print(f'\n  Trial {trial_counter + 1}/{trial_num}')

        graph = load_graph(dataset_path)

        if train_dataset_path and train_dataset_path != 'NONE':
            train_graph = load_graph(train_dataset_path)
        else:
            train_graph = None

        try:
            exp = MultiroundExperiment(
                graph, train_graph=train_graph,
                main_config=main_config_run, model_config=model_config_run,
                strat_config=strat_config_run, adver_config=adver_config_run,
                train_config=train_config_run
            )

            round_counter = 0
            round_flag = True
            while (round_counter < main_config_run['round_num']) and round_flag:
                print(f'    Round {round_counter}...')
                round_flag = exp.one_round_node(round_counter)
                round_counter += 1

            if round_flag:
                eval_df = pd.DataFrame(
                    sum([r['log_single_eval'] for r in exp.rounds], []),
                    columns=['round', 'eval_type', 'time', 'rec', 'prec', 'f1', 'auc', 'tp', 'fp', 'tn', 'fn']
                )
                trainlog_df = pd.DataFrame([r['log_round'] for r in exp.rounds])
                log_df = pd.merge(left=eval_df, right=trainlog_df, on='round', how='outer')
                log_df['variant'] = variant_name
                log_df['trial'] = trial_counter
                log_df['description'] = variant_config['description']
                log_df['category'] = variant_config['category']
                dfs.append(log_df)
                trial_counter += 1
            else:
                failure_counter += 1
                print(f'    Round failed, retrying...')

            exp.clean_temp_files()

        except Exception as e:
            import traceback
            print(f'    Error: {e}')
            traceback.print_exc()
            failure_counter += 1

        if failure_counter > failure_limit:
            print(f'  Too many failures ({failure_counter}), skipping variant.')
            break

        del graph, train_graph
        gc.collect()
        torch.cuda.empty_cache()

    elapsed = time() - start
    print(f'\n  Completed {variant_name}: {trial_counter} trials, {failure_counter} failures, {elapsed:.1f}s')

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description='Run all adaptive detector experiments')
    parser.add_argument('-c', '--config', type=str, default='config_adaptive',
                        help='Base config file name (without .json)')
    parser.add_argument('--dset', type=str, default=None,
                        help='Override dataset name')
    parser.add_argument('--rounds', type=int, default=None,
                        help='Override number of rounds')
    parser.add_argument('--trials', type=int, default=1,
                        help='Number of trials per variant')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use')
    parser.add_argument('--skip-baseline', action='store_true',
                        help='Skip baseline experiments (XGB-SP)')
    parser.add_argument('--skip-individual', action='store_true',
                        help='Skip individual model experiments')
    parser.add_argument('--skip-ablation', action='store_true',
                        help='Skip ablation experiments')
    parser.add_argument('--only-adaptive', action='store_true',
                        help='Run only the full adaptive detector')
    parser.add_argument('--only-core', action='store_true',
                        help='Run baseline + proposed only (no individuals, no ablations)')
    parser.add_argument('--variants', nargs='+', default=None,
                        help='Run specific variant names only')
    args = parser.parse_args()

    config_path = f'{args.config}.json'
    with open(config_path) as f:
        config_file = json.loads(f.read())

    main_config = copy.deepcopy(DEFAULT_MAIN_CONFIG)
    train_config = copy.deepcopy(DEFAULT_TRAIN_CONFIG)
    model_config = copy.deepcopy(DEFAULT_MODEL_CONFIG)
    strat_config = copy.deepcopy(DEFAULT_STRAT_CONFIG)
    adver_config = copy.deepcopy(DEFAULT_ADVER_CONFIG)

    main_config['device'] = args.device
    main_config['exp_type'] = 'ADVER'
    main_config['task_type'] = 'NODE'

    dataset_name = args.dset or config_file['LIST_DSET'][0]
    train_dataset_name = 'NONE'
    if not args.dset and len(config_file.get('LIST_TRAIN_DSET', [])) > 0:
        train_dataset_name = config_file['LIST_TRAIN_DSET'][0]

    dataset_path = f'../dataset/{dataset_name}'
    train_dataset_path = None if train_dataset_name == 'NONE' else f'../dataset/{train_dataset_name}'

    round_num = args.rounds or config_file['EXP_DICT'].get('round_num', [3])[0]
    main_config['round_num'] = round_num

    for key in ['num_epoch', 'num_round_epoch', 'early_stopping']:
        if key in config_file['EXP_DICT']:
            val = config_file['EXP_DICT'][key]
            if isinstance(val, list):
                val = val[0]
            train_config[key] = val
            model_config[key] = val

    for key in ['embed_type', 'h_feats', 'num_layers', 'round_window',
                'loss_type', 'norm_name', 'temporal_agg', 'alpha', 'beta']:
        if key in config_file['EXP_DICT']:
            val = config_file['EXP_DICT'][key]
            if isinstance(val, list):
                val = val[0]
            model_config[key] = val

    if 'adver_choose_name' in config_file['EXP_DICT']:
        val = config_file['EXP_DICT']['adver_choose_name']
        if isinstance(val, list):
            val = val[0]
        adver_config['adver_choose_name'] = val

    if 'adver_mod_name' in config_file['EXP_DICT']:
        val = config_file['EXP_DICT']['adver_mod_name']
        if isinstance(val, list):
            val = val[0]
        adver_config['adver_mod_name'] = val

    for key in ['adver_gen_type', 'adver_gen_epochs', 'adver_gen_feat_coef',
                'adver_gen_conn_coef', 'adver_gen_ring_ratio']:
        if key in config_file['EXP_DICT']:
            val = config_file['EXP_DICT'][key]
            if isinstance(val, list):
                val = val[0]
            adver_config[key] = val

    for key in ['adaptive_history_window', 'adaptive_ema_alpha', 'adaptive_recall_importance']:
        if key in config_file['EXP_DICT']:
            val = config_file['EXP_DICT'][key]
            if isinstance(val, list):
                val = val[0]
            model_config[key] = val

    if 'adaptive_components' in config_file['EXP_DICT']:
        val = config_file['EXP_DICT']['adaptive_components']
        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], list):
            val = val[0]
        model_config['adaptive_components'] = val

    if 'adaptive_model_list' in config_file['EXP_DICT']:
        val = config_file['EXP_DICT']['adaptive_model_list']
        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], list):
            val = val[0]
        model_config['adaptive_model_list'] = val

    variants_to_run = {}
    if args.variants:
        for v in args.variants:
            if v in EXPERIMENT_VARIANTS:
                variants_to_run[v] = EXPERIMENT_VARIANTS[v]
    elif args.only_adaptive:
        variants_to_run = {k: v for k, v in EXPERIMENT_VARIANTS.items() if v['category'] == 'proposed'}
    elif args.only_core:
        variants_to_run = {k: v for k, v in EXPERIMENT_VARIANTS.items() if v['category'] in ['baseline', 'proposed']}
    else:
        for name, config in EXPERIMENT_VARIANTS.items():
            if args.skip_baseline and config['category'] == 'baseline':
                continue
            if args.skip_individual and config['category'] == 'individual':
                continue
            if args.skip_ablation and config['category'] == 'ablation':
                continue
            variants_to_run[name] = config

    print(f'\n{"="*60}')
    print(f'ADAPTIVE MULTI-MODEL DETECTION LAYER - EXPERIMENT SUITE')
    print(f'{"="*60}')
    print(f'Dataset: {dataset_name}')
    print(f'Rounds: {round_num}')
    print(f'Trials per variant: {args.trials}')
    print(f'Device: {args.device}')
    print(f'Variants to run ({len(variants_to_run)}): {list(variants_to_run.keys())}')
    print(f'{"="*60}\n')

    all_results = []
    overall_start = time()

    for variant_name, variant_config in variants_to_run.items():
        result_df = run_single_experiment(
            variant_name=variant_name,
            variant_config=variant_config,
            dataset_path=dataset_path,
            train_dataset_path=train_dataset_path,
            main_config=main_config,
            train_config=train_config,
            model_config=model_config,
            strat_config=strat_config,
            adver_config=adver_config,
            trial_num=args.trials,
            failure_limit=config_file.get('FAILURE_LIMIT', 2),
        )

        if not result_df.empty:
            all_results.append(result_df)

        gc.collect()
        torch.cuda.empty_cache()

    overall_elapsed = time() - overall_start

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)

        ts = datetime.datetime.now().strftime("%y%m%d%H%M%S")
        output_dir = f'../result/adaptive_experiment/{ts}'
        os.makedirs(output_dir, exist_ok=True)

        combined.to_csv(f'{output_dir}/all_results.csv', index=False)

        summary_rows = []
        for variant_name in variants_to_run.keys():
            vdf = combined[combined['variant'] == variant_name]
            if vdf.empty:
                continue

            overall_evals = vdf[vdf['eval_type'] == 'entire_graph']
            if overall_evals.empty:
                overall_evals = vdf

            for _, row in overall_evals.iterrows():
                summary_entry = {
                    'variant': variant_name,
                    'category': variants_to_run[variant_name]['category'],
                    'description': variants_to_run[variant_name]['description'],
                    'round': row.get('round', ''),
                    'f1': row.get('f1', 0),
                    'recall': row.get('rec', 0),
                    'precision': row.get('prec', 0),
                    'auc': row.get('auc', 0),
                }

                for col in row.index:
                    if col.startswith('weight_') or col.startswith('ads_score_') or col.startswith('individual_f1_') or col.startswith('individual_recall_'):
                        summary_entry[col] = row[col]

                summary_rows.append(summary_entry)

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(f'{output_dir}/summary.csv', index=False)

        print(f'\n{"="*60}')
        print(f'EXPERIMENT SUITE COMPLETE')
        print(f'{"="*60}')
        print(f'Total time: {overall_elapsed:.1f}s')
        print(f'Results saved to: {output_dir}')
        print(f'\nSummary:')

        if not summary_df.empty:
            display_cols = ['variant', 'round', 'f1', 'recall', 'precision', 'auc']
            print(summary_df[display_cols].to_string(index=False))

        print(f'\nTo analyze: python analyze_results.py -r {output_dir}')
        print(f'{"="*60}')
    else:
        print('\nNo results collected.')


if __name__ == '__main__':
    main()
