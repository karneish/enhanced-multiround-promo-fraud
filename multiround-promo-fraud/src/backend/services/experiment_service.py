import os
import sys
import copy
import gc
import json
import uuid
import time
import datetime
import traceback
import threading
import pandas as pd
import numpy as np

import torch
import dgl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiment.supervised_multi import MultiroundExperiment
from utils.utils_const import (
    DEFAULT_MAIN_CONFIG, DEFAULT_TRAIN_CONFIG,
    DEFAULT_ADVER_CONFIG, DEFAULT_MODEL_CONFIG,
    DEFAULT_STRAT_CONFIG
)
from backend.config import Config


class ExperimentStatus:
    QUEUED = 'queued'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'


class ExperimentRecord:
    def __init__(self, experiment_id, config):
        self.id = experiment_id
        self.config = config
        self.status = ExperimentStatus.QUEUED
        self.progress = {
            'current_round': 0,
            'total_rounds': config.get('round_num', 5),
            'phase': 'queued',
            'message': 'Experiment queued'
        }
        self.result = None
        self.error = None
        self.started_at = None
        self.completed_at = None
        self.output_dir = None

    def to_dict(self):
        return {
            'id': self.id,
            'config': self.config,
            'status': self.status,
            'progress': self.progress,
            'result': self.result,
            'error': self.error,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'output_dir': self.output_dir,
        }


class ExperimentService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.experiments = {}
        self.experiments_lock = threading.Lock()
        os.makedirs(Config.RESULT_DIR, exist_ok=True)
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    def list_experiments(self):
        with self.experiments_lock:
            return [r.to_dict() for r in self.experiments.values()]

    def get_experiment(self, experiment_id):
        with self.experiments_lock:
            record = self.experiments.get(experiment_id)
        if record is None:
            return None
        return record.to_dict()

    def get_experiment_status(self, experiment_id):
        with self.experiments_lock:
            record = self.experiments.get(experiment_id)
        if record is None:
            return None
        return {'id': record.id, 'status': record.status, 'progress': record.progress}

    def list_results(self):
        results = []
        if not os.path.exists(Config.RESULT_DIR):
            return results
        for dirpath, dirnames, filenames in os.walk(Config.RESULT_DIR):
            csv_files = [f for f in filenames if f.endswith('.csv')]
            if csv_files:
                rel_path = os.path.relpath(dirpath, Config.RESULT_DIR)
                results.append({
                    'path': rel_path,
                    'files': csv_files,
                })
        return results

    def get_result_csv(self, relative_path, filename):
        full_path = os.path.join(Config.RESULT_DIR, relative_path, filename)
        full_path = os.path.normpath(full_path)
        if not full_path.startswith(os.path.normpath(Config.RESULT_DIR)):
            return None
        if not os.path.exists(full_path):
            return None
        try:
            df = pd.read_csv(full_path)
            return df.to_dict(orient='records')
        except Exception:
            return None

    def get_result_plot(self, relative_path, filename):
        full_path = os.path.join(Config.RESULT_DIR, relative_path, filename)
        full_path = os.path.normpath(full_path)
        if not full_path.startswith(os.path.normpath(Config.RESULT_DIR)):
            return None
        if not os.path.exists(full_path):
            return None
        return full_path

    def list_datasets(self):
        datasets = []
        if not os.path.exists(Config.DATASET_DIR):
            return datasets
        for name in os.listdir(Config.DATASET_DIR):
            path = os.path.join(Config.DATASET_DIR, name)
            if os.path.isfile(path) and not name.startswith('.'):
                try:
                    graphs, _ = dgl.load_graphs(path)
                    g = graphs[0]
                    datasets.append({
                        'name': name,
                        'nodes': g.num_nodes(),
                        'edges': g.num_edges(),
                        'features': g.ndata['feature'].shape[1] if 'feature' in g.ndata else 0,
                        'labels': {
                            str(k.item()): v.item()
                            for k, v in zip(*torch.unique(g.ndata['label'], return_counts=True))
                        } if 'label' in g.ndata else {},
                    })
                except Exception:
                    datasets.append({'name': name, 'error': 'Failed to load'})
        return datasets

    def list_models(self):
        from utils.utils_const import MODEL_DICT
        return list(MODEL_DICT.keys())

    def list_available_configs(self):
        configs = []
        if not os.path.exists(Config.SCRIPT_DIR):
            return configs
        for name in os.listdir(Config.SCRIPT_DIR):
            if name.startswith('config_') and name.endswith('.json'):
                try:
                    path = os.path.join(Config.SCRIPT_DIR, name)
                    with open(path) as f:
                        cfg = json.load(f)
                    configs.append({
                        'filename': name,
                        'name': name.replace('config_', '').replace('.json', ''),
                        'description': cfg.get('EXPERIMENT_DESC', ''),
                        'dataset': cfg.get('LIST_DSET', []),
                        'rounds': cfg.get('EXP_DICT', {}).get('round_num', []),
                        'model': cfg.get('EXP_DICT', {}).get('model_name', []),
                    })
                except Exception:
                    pass
        return configs

    def start_experiment(self, experiment_config):
        experiment_id = str(uuid.uuid4())[:8]
        record = ExperimentRecord(experiment_id, experiment_config)

        with self.experiments_lock:
            self.experiments[experiment_id] = record

        thread = threading.Thread(
            target=self._run_experiment,
            args=(experiment_id,),
            daemon=True
        )
        thread.start()

        return experiment_id

    def _run_experiment(self, experiment_id):
        with self.experiments_lock:
            record = self.experiments.get(experiment_id)
        if record is None:
            return

        try:
            record.status = ExperimentStatus.RUNNING
            record.started_at = datetime.datetime.now().isoformat()
            record.progress['phase'] = 'initializing'
            record.progress['message'] = 'Loading dataset and initializing...'

            config = record.config
            dataset_name = config.get('dataset', 'tolokers_bid')
            round_num = config.get('round_num', 5)
            device = config.get('device', 'cpu')
            trials = config.get('trials', 1)

            dataset_path = os.path.join(Config.DATASET_DIR, dataset_name)
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f'Dataset not found: {dataset_path}')

            graph = self._load_graph(dataset_path)
            pos = (graph.ndata['label'] == 1).sum().item()
            neg = (graph.ndata['label'] == 0).sum().item()

            main_config = copy.deepcopy(DEFAULT_MAIN_CONFIG)
            train_config = copy.deepcopy(DEFAULT_TRAIN_CONFIG)
            model_config = copy.deepcopy(DEFAULT_MODEL_CONFIG)
            strat_config = copy.deepcopy(DEFAULT_STRAT_CONFIG)
            adver_config = copy.deepcopy(DEFAULT_ADVER_CONFIG)

            main_config['device'] = device
            main_config['exp_type'] = 'ADVER'
            main_config['task_type'] = 'NODE'
            main_config['round_num'] = round_num
            main_config['round_new_pos'] = config.get('round_new_pos', int(0.05 * pos))
            main_config['round_new_neg'] = config.get('round_new_neg', int(0.05 * neg))
            main_config['round_budget_pos'] = 0
            main_config['round_budget_neg'] = 0

            model_name = config.get('model_name', 'ADAPTIVE')
            model_config['model_name'] = model_name
            model_config['embed_type'] = config.get('embed_type', 'temporal')
            model_config['h_feats'] = config.get('h_feats', 64)
            model_config['num_layers'] = config.get('num_layers', 2)
            model_config['round_window'] = config.get('round_window', 7)
            model_config['mlp_feats'] = config.get('h_feats', 64)
            model_config['num_epoch'] = config.get('num_epoch', 20)
            model_config['num_round_epoch'] = config.get('num_round_epoch', 10)
            model_config['early_stopping'] = config.get('early_stopping', 10)
            model_config['loss_type'] = config.get('loss_type', 'ndist')
            model_config['norm_name'] = config.get('norm_name', 'layer')
            model_config['temporal_agg'] = config.get('temporal_agg', 'weight')
            model_config['alpha'] = config.get('alpha', 1)
            model_config['beta'] = config.get('beta', 1)

            if 'adaptive_model_list' in config:
                model_config['adaptive_model_list'] = config['adaptive_model_list']
            if 'adaptive_components' in config:
                model_config['adaptive_components'] = config['adaptive_components']

            train_config['num_epoch'] = model_config['num_epoch']
            train_config['num_round_epoch'] = model_config['num_round_epoch']
            train_config['early_stopping'] = model_config['early_stopping']
            train_config['round_reset_model'] = False

            adver_config['adver_choose_name'] = config.get('adver_choose_name', 'GREEDY')
            adver_config['adver_mod_name'] = config.get('adver_mod_name', 'INTELLIGENT')
            adver_config['adver_gen_type'] = config.get('adver_gen_type', 'GAN')
            adver_config['adver_gen_epochs'] = config.get('adver_gen_epochs', 50)
            adver_config['adver_gen_feat_coef'] = config.get('adver_gen_feat_coef', 1.0)
            adver_config['adver_gen_conn_coef'] = config.get('adver_gen_conn_coef', 0.5)
            adver_config['adver_gen_ring_ratio'] = config.get('adver_gen_ring_ratio', 0.5)

            ts = datetime.datetime.now().strftime("%y%m%d%H%M%S")
            output_dir = os.path.join(Config.RESULT_DIR, f'api_experiment/{ts}')
            os.makedirs(output_dir, exist_ok=True)
            record.output_dir = output_dir

            all_dfs = []
            trial_counter = 0
            failure_counter = 0

            while trial_counter < trials:
                record.progress['phase'] = 'training'
                record.progress['message'] = f'Trial {trial_counter + 1}/{trials}'

                graph = self._load_graph(dataset_path)

                try:
                    exp = MultiroundExperiment(
                        graph,
                        main_config=copy.deepcopy(main_config),
                        model_config=copy.deepcopy(model_config),
                        strat_config=copy.deepcopy(strat_config),
                        adver_config=copy.deepcopy(adver_config),
                        train_config=copy.deepcopy(train_config)
                    )

                    round_flag = True
                    for round_num_idx in range(round_num):
                        record.progress['current_round'] = round_num_idx
                        record.progress['message'] = f'Trial {trial_counter + 1} - Round {round_num_idx}/{round_num - 1}'
                        round_flag = exp.one_round_node(round_num_idx)
                        if not round_flag:
                            break

                    if round_flag:
                        eval_df = pd.DataFrame(
                            sum([r['log_single_eval'] for r in exp.rounds], []),
                            columns=['round', 'eval_type', 'time', 'rec', 'prec', 'f1', 'auc', 'tp', 'fp', 'tn', 'fn']
                        )
                        trainlog_df = pd.DataFrame([r['log_round'] for r in exp.rounds])
                        log_df = pd.merge(left=eval_df, right=trainlog_df, on='round', how='outer')
                        log_df['variant'] = model_name
                        log_df['trial'] = trial_counter
                        all_dfs.append(log_df)
                        trial_counter += 1
                    else:
                        failure_counter += 1

                    exp.clean_temp_files()

                except Exception as e:
                    failure_counter += 1
                    print(f'Trial {trial_counter} error: {e}')
                    traceback.print_exc()

                if failure_counter > 3:
                    raise RuntimeError(f'Too many failures ({failure_counter})')

                del graph
                gc.collect()
                torch.cuda.empty_cache()

            if all_dfs:
                combined = pd.concat(all_dfs, ignore_index=True)
                combined.to_csv(os.path.join(output_dir, 'all_results.csv'), index=False)

                overall_evals = combined[combined['eval_type'] == 'entire_graph']
                if overall_evals.empty:
                    overall_evals = combined

                result_data = []
                for _, row in overall_evals.iterrows():
                    entry = {
                        'round': row.get('round', ''),
                        'f1': float(row.get('f1', 0)),
                        'recall': float(row.get('rec', 0)),
                        'precision': float(row.get('prec', 0)),
                        'auc': float(row.get('auc', 0)),
                    }
                    for col in row.index:
                        if col.startswith('weight_') or col.startswith('individual_f1_'):
                            try:
                                entry[col] = float(row[col])
                            except (ValueError, TypeError):
                                entry[col] = str(row[col])
                    result_data.append(entry)

                record.result = result_data

            record.status = ExperimentStatus.COMPLETED
            record.completed_at = datetime.datetime.now().isoformat()
            record.progress['phase'] = 'completed'
            record.progress['message'] = 'Experiment completed successfully'
            record.progress['current_round'] = round_num

        except Exception as e:
            record.status = ExperimentStatus.FAILED
            record.error = str(e)
            record.completed_at = datetime.datetime.now().isoformat()
            record.progress['phase'] = 'failed'
            record.progress['message'] = f'Experiment failed: {str(e)}'
            traceback.print_exc()

        finally:
            gc.collect()
            torch.cuda.empty_cache()

    def _load_graph(self, path):
        dataset, _ = dgl.load_graphs(path)
        graph = dataset[0].long()
        if len(graph.ndata['label'].shape) > 1:
            graph.ndata['label'] = graph.ndata['label'].argmax(1)
            graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
        graph.ndata['feature'] = graph.ndata['feature'].float()
        return graph
