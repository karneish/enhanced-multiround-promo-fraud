from utils.utils_func import verPrint

import gc
import os
import json
import torch
import numpy as np
import time
import xgboost as xgb
import lightgbm as lgb

from torch.optim import Adam
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
from collections import deque

from models.base_model import BaseModel
from models.proposed_supervised.mixed import (
    TemporalEmbedder, VanillaEmbedder, TemporalMixedEmbedder
)

EPS = 1e-10


class AdaptiveDetectorScore:

    def __init__(
        self,
        model_names,
        history_window=5,
        weights=None,
        ema_alpha=0.7,
        recall_importance=1.0,
        components=None,
    ):
        self.model_names = list(model_names)
        self.history_window = history_window
        self.ema_alpha = ema_alpha
        self.recall_importance = recall_importance

        if components is None:
            self.components = ['f1', 'recall', 'stability', 'historical']
        else:
            self.components = list(components)

        if weights is None:
            if len(self.components) > 0:
                self.weights = {c: 1.0 / len(self.components) for c in self.components}
            else:
                self.weights = {}
        else:
            self.weights = dict(weights)

        self.history = {name: deque(maxlen=history_window) for name in self.model_names}
        self.ema_scores = {name: 0.0 for name in self.model_names}

    def reset(self):
        for name in self.model_names:
            self.history[name].clear()
            self.ema_scores[name] = 0.0

    def update(self, model_name, f1, recall):
        self.history[model_name].append({'f1': f1, 'recall': recall})

        if len(self.history[model_name]) == 1:
            self.ema_scores[model_name] = f1
        else:
            self.ema_scores[model_name] = (
                self.ema_alpha * f1 + (1 - self.ema_alpha) * self.ema_scores[model_name]
            )

    def _compute_f1_score(self, model_name):
        hist = self.history[model_name]
        if len(hist) == 0:
            return 0.0
        return hist[-1]['f1']

    def _compute_recall_score(self, model_name):
        hist = self.history[model_name]
        if len(hist) == 0:
            return 0.0
        return hist[-1]['recall'] * self.recall_importance

    def _compute_stability(self, model_name):
        hist = self.history[model_name]
        if len(hist) < 2:
            return 1.0
        f1_values = [h['f1'] for h in hist]
        mean_f1 = np.mean(f1_values)
        std_f1 = np.std(f1_values)
        stability = 1.0 - min(std_f1 / (mean_f1 + EPS), 1.0)
        return stability

    def _compute_historical(self, model_name):
        return self.ema_scores[model_name]

    def compute_score(self, model_name):
        component_values = {}

        if 'f1' in self.components:
            component_values['f1'] = self._compute_f1_score(model_name)
        if 'recall' in self.components:
            component_values['recall'] = self._compute_recall_score(model_name)
        if 'stability' in self.components:
            component_values['stability'] = self._compute_stability(model_name)
        if 'historical' in self.components:
            component_values['historical'] = self._compute_historical(model_name)

        if len(component_values) == 0:
            return 1.0

        total_weight = sum(self.weights.get(c, 0) for c in component_values)
        if total_weight == 0:
            total_weight = 1.0

        score = sum(
            self.weights.get(c, 0) * v for c, v in component_values.items()
        ) / total_weight

        return max(score, EPS)

    def compute_all_scores(self):
        return {name: self.compute_score(name) for name in self.model_names}

    def compute_weights(self):
        scores = self.compute_all_scores()
        total_score = sum(scores.values())
        if total_score == 0:
            n = len(self.model_names)
            return {name: 1.0 / n for name in self.model_names}
        return {name: s / total_score for name, s in scores.items()}

    def get_diagnostics(self):
        return {
            name: {
                'scores': self.compute_all_scores(),
                'weights': self.compute_weights(),
                'history_len': len(self.history[name]),
                'ema': self.ema_scores[name],
                'f1_history': [h['f1'] for h in self.history[name]],
                'recall_history': [h['recall'] for h in self.history[name]],
            }
            for name in self.model_names
        }


class AdaptiveDetector(BaseModel):

    def __init__(
        self,
        in_feats, h_feats, num_layers,
        embed_type='mixed',
        round_window=7,
        temporal_agg='sum_final',
        gamma=1, alpha=1, beta=1,
        loss_type='neigh', tloss_type='normal',
        loss_sample=False, loss_sample_ratio=0.1,
        dropout_rate=0.5, act_name='ReLU', norm_name='layer',
        num_epoch=300, num_round_epoch=150, early_stopping=25,
        device='cuda:0', verbose=0,
        temp_model_path='../checkpoint/working_model_file',

        adaptive_model_list=None,
        adaptive_weights=None,
        adaptive_components=None,
        adaptive_history_window=5,
        adaptive_ema_alpha=0.7,
        adaptive_recall_importance=1.0,

        boost_metric=None,
        training_type='round',
        **kwargs
    ):
        super().__init__()
        self.verbose = verbose
        self.device = device
        self.temp_model_path = temp_model_path

        self.in_feats = in_feats
        self.h_feats = h_feats
        self.num_layers = num_layers
        self.round_window = round_window

        self.eval_metric = boost_metric
        self.num_epoch = num_epoch
        self.num_round_epoch = num_round_epoch
        self.early_stopping = early_stopping

        self.training_type = training_type

        if adaptive_model_list is None:
            self.model_names = ['XGBoost', 'RandomForest', 'ExtraTrees', 'HistGradientBoosting', 'LogisticRegression']
        else:
            self.model_names = list(adaptive_model_list)

        self.classifiers = {}
        self.classifier_fitted = {}
        self._init_classifiers()

        self.ads = AdaptiveDetectorScore(
            model_names=self.model_names,
            history_window=adaptive_history_window,
            weights=adaptive_weights,
            ema_alpha=adaptive_ema_alpha,
            recall_importance=adaptive_recall_importance,
            components=adaptive_components,
        )

        self.round_weight_history = []
        self.round_score_history = []
        self.round_individual_f1 = {name: [] for name in self.model_names}
        self.round_individual_recall = {name: [] for name in self.model_names}

        if embed_type == 'temporal':
            self.embedder = TemporalEmbedder(
                in_feats, h_feats, num_layers,
                round_window=round_window, temporal_agg=temporal_agg,
                gamma=gamma, alpha=alpha, beta=beta,
                loss_type=loss_type, tloss_type=tloss_type,
                loss_sample=loss_sample, loss_sample_ratio=loss_sample_ratio,
                dropout_rate=dropout_rate, act_name=act_name, norm_name=norm_name,
                device=device, verbose=verbose
            )
        elif embed_type == 'vanilla':
            self.embedder = VanillaEmbedder(
                in_feats, h_feats, num_layers,
                loss_type=loss_type, tloss_type=tloss_type,
                loss_sample=loss_sample, loss_sample_ratio=loss_sample_ratio,
                dropout_rate=dropout_rate, act_name=act_name, norm_name=norm_name,
                device=device, verbose=verbose
            )
        elif embed_type == 'mixed':
            self.embedder = TemporalMixedEmbedder(
                in_feats, h_feats, num_layers,
                round_window=round_window, temporal_agg=temporal_agg,
                gamma=gamma, alpha=alpha, beta=beta,
                loss_type=loss_type, tloss_type=tloss_type,
                loss_sample=loss_sample, loss_sample_ratio=loss_sample_ratio,
                dropout_rate=dropout_rate, act_name=act_name, norm_name=norm_name,
                device=device, verbose=verbose
            )

    def _init_classifiers(self):
        if 'XGBoost' in self.model_names:
            self.classifiers['XGBoost'] = None
            self.classifier_fitted['XGBoost'] = False

        if 'RandomForest' in self.model_names:
            self.classifiers['RandomForest'] = RandomForestClassifier(
                n_estimators=300, max_depth=None, min_samples_split=5,
                class_weight='balanced', random_state=42, n_jobs=-1
            )
            self.classifier_fitted['RandomForest'] = False

        if 'ExtraTrees' in self.model_names:
            self.classifiers['ExtraTrees'] = ExtraTreesClassifier(
                n_estimators=300, max_depth=None, min_samples_split=5,
                class_weight='balanced', random_state=42, n_jobs=-1
            )
            self.classifier_fitted['ExtraTrees'] = False

        if 'HistGradientBoosting' in self.model_names:
            self.classifiers['HistGradientBoosting'] = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.1, max_depth=6,
                random_state=42
            )
            self.classifier_fitted['HistGradientBoosting'] = False

        if 'LogisticRegression' in self.model_names:
            self.classifiers['LogisticRegression'] = LogisticRegression(
                max_iter=5000, solver='lbfgs', class_weight='balanced',
                random_state=42, tol=1e-4, C=1.0
            )
            self.classifier_fitted['LogisticRegression'] = False

        if 'LightGBM' in self.model_names:
            self.classifiers['LightGBM'] = None
            self.classifier_fitted['LightGBM'] = False

    def __call__(self, graph, feats, **kwargs):
        agg_feats = self.embed_nodes(graph, feats)

        all_probs = {}
        for name in self.model_names:
            probs = self._predict_single(name, agg_feats)
            if probs is not None:
                all_probs[name] = probs

        if len(all_probs) == 0:
            fallback = torch.zeros(feats.shape[0], 2, device=self.device)
            fallback[:, 0] = 1.0
            return fallback, None, None

        weights = self.ads.compute_weights()
        ensemble_probs = torch.zeros(feats.shape[0], device=self.device)
        total_weight = 0.0

        for name, probs in all_probs.items():
            w = weights.get(name, 0.0)
            ensemble_probs += w * probs
            total_weight += w

        if total_weight > 0:
            ensemble_probs = ensemble_probs / total_weight

        result = torch.stack([1 - ensemble_probs, ensemble_probs], dim=1)
        return result, None, None

    def _predict_single(self, name, feats):
        feats_np = feats.detach().cpu().numpy()

        if not self.classifier_fitted.get(name, False):
            return None

        if name == 'XGBoost' and self.classifiers[name] is not None:
            dmatrix = xgb.DMatrix(feats_np)
            raw = self.classifiers[name].predict(dmatrix)
            return torch.tensor(raw, dtype=torch.float32, device=self.device)
        elif name == 'LightGBM' and self.classifiers.get(name) is not None:
            raw = self.classifiers[name].predict(feats_np)
            return torch.tensor(raw, dtype=torch.float32, device=self.device)
        elif self.classifiers.get(name) is not None:
            proba = self.classifiers[name].predict_proba(feats_np)
            if proba.shape[1] == 1:
                return torch.tensor(proba[:, 0], dtype=torch.float32, device=self.device)
            return torch.tensor(proba[:, 1], dtype=torch.float32, device=self.device)

        return None

    def embed_nodes(self, graph, feats):
        return self.embedder.embed_nodes(graph, feats)

    def train(self, graph, weight, round_num):
        if (self.training_type == 'round') or ((self.training_type == 'init') and (round_num == 0)):
            embedder_finish = False
            while not embedder_finish:
                embedder_finish = self.train_embedder(graph, weight, round_num)

        train_score, val_score = self.train_classifiers(graph, weight, round_num)
        return train_score, val_score

    def train_embedder(self, graph, weight, round_num):
        verPrint(self.verbose, 2, 'TRAINING NODE EMBEDDER (ADAPTIVE)')

        self.embedder = self.embedder.to(self.device)
        graph = graph.to(self.device)

        best_loss = None
        epoch_counter, stagnant_counter = 0, 0
        stop_training = False

        self.optimizer = Adam(self.embedder.parameters(), lr=0.01)
        self.embedder.dist_calculated = False

        features = graph.ndata['feature']

        while not stop_training:
            self.embedder.train()
            _, embedding_loss = self.embedder(graph, features, **{'epoch': epoch_counter, 'ce_weight': weight})

            self.optimizer.zero_grad()
            embedding_loss.backward()
            self.optimizer.step()

            self.embedder.eval()
            current_loss = embedding_loss.item()

            if (best_loss is None) or (current_loss < best_loss):
                while True:
                    try:
                        self.save_embedder(f'{self.temp_model_path}_adaptive_epoch')
                        break
                    except RuntimeError as e:
                        verPrint(self.verbose, 4, f"CHECKPOINTING ERROR {e}")

                best_loss = current_loss
                stagnant_counter = 0
            else:
                stagnant_counter += 1

            epoch_counter += 1
            verPrint(self.verbose, 2, f'Epoch {epoch_counter}, loss: {current_loss:.8f}-(best {best_loss:.8f})')

            stop_training = (
                (epoch_counter >= (self.num_epoch if round_num == 0 else self.num_round_epoch))
                or (stagnant_counter >= self.early_stopping)
            )

        self.load_embedder(f'{self.temp_model_path}_adaptive_epoch')
        verPrint(self.verbose, 2, '>> Reached final epoch. Loading best embedder model...')
        return True

    def train_classifiers(self, graph, weight, round_num):
        verPrint(self.verbose, 2, 'TRAINING ALL CLASSIFIERS (ADAPTIVE)')

        feats = self.embed_nodes(graph, graph.ndata['feature'].clone().detach())
        labels = graph.ndata['ps_label']

        train_X = feats[graph.ndata['ps_train_mask']].clone().detach().cpu().numpy()
        train_y = labels[graph.ndata['ps_train_mask']].clone().detach().cpu().numpy()
        val_X = feats[graph.ndata['val_mask']].clone().detach().cpu().numpy()
        val_y = labels[graph.ndata['val_mask']].clone().detach().cpu().numpy()

        sample_weights = compute_sample_weight('balanced', train_y)

        if np.unique(train_y).size < 2:
            verPrint(self.verbose, 2, '  >> WARNING: only one class in training data, skipping classifier training')
            return 0.0, 0.0

        best_val_f1 = 0.0

        for name in self.model_names:
            verPrint(self.verbose, 2, f'  >> Training {name}...')

            if name == 'XGBoost':
                self._train_xgboost(train_X, train_y, val_X, val_y, weight, round_num)
            elif name == 'LightGBM':
                self._train_lightgbm(train_X, train_y, val_X, val_y, weight, round_num)
            else:
                self._train_sklearn(name, train_X, train_y, val_X, val_y, sample_weights, round_num)

            val_preds = self._predict_single(name, torch.tensor(val_X, dtype=torch.float32))
            if val_preds is not None:
                val_preds_np = val_preds.cpu().numpy()
                val_preds_binary = (val_preds_np > 0.5).astype(int)

                round_f1 = f1_score(val_y, val_preds_binary, average='macro', zero_division=0)
                round_recall = recall_score(val_y, val_preds_binary, zero_division=0)

                self.ads.update(name, round_f1, round_recall)
                self.round_individual_f1[name].append(round_f1)
                self.round_individual_recall[name].append(round_recall)

                verPrint(self.verbose, 2,
                    f'    {name} val: F1={round_f1:.4f}, Recall={round_recall:.4f}')

                if round_f1 > best_val_f1:
                    best_val_f1 = round_f1

        weights = self.ads.compute_weights()
        scores = self.ads.compute_all_scores()
        self.round_weight_history.append(weights)
        self.round_score_history.append(scores)

        verPrint(self.verbose, 2, f'  >> ADS Weights: {weights}')
        verPrint(self.verbose, 2, f'  >> ADS Scores: {scores}')

        return best_val_f1, best_val_f1

    def _train_xgboost(self, train_X, train_y, val_X, val_y, weight, round_num):
        params = {
            "objective": "binary:logistic",
            "scale_pos_weight": weight,
            "tree_method": "hist",
            "max_depth": 6,
            "device": "cpu"
        }

        dtrain = xgb.DMatrix(train_X, train_y)
        dval = xgb.DMatrix(val_X, val_y)

        existing_model = self.classifiers['XGBoost'] if self.classifier_fitted.get('XGBoost', False) else None

        self.classifiers['XGBoost'] = xgb.train(
            params, dtrain,
            num_boost_round=500,
            early_stopping_rounds=100,
            evals=[(dtrain, 'Train'), (dval, 'Eval')],
            verbose_eval=False,
            xgb_model=existing_model
        )
        self.classifier_fitted['XGBoost'] = True

    def _train_lightgbm(self, train_X, train_y, val_X, val_y, weight, round_num):
        params = {
            "objective": "binary",
            "scale_pos_weight": weight,
            "max_depth": 6,
            "learning_rate": 0.1,
            "verbose": -1,
            "n_jobs": -1,
            "random_state": 42,
        }

        dtrain = lgb.Dataset(train_X, label=train_y)
        dval = lgb.Dataset(val_X, label=val_y, reference=dtrain)

        callbacks = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]

        existing_model = self.classifiers.get('LightGBM') if self.classifier_fitted.get('LightGBM', False) else None

        if existing_model is not None:
            self.classifiers['LightGBM'] = lgb.train(
                params, dtrain,
                num_boost_round=500,
                valid_sets=[dtrain, dval],
                callbacks=callbacks,
                init_model=existing_model
            )
        else:
            self.classifiers['LightGBM'] = lgb.train(
                params, dtrain,
                num_boost_round=500,
                valid_sets=[dtrain, dval],
                callbacks=callbacks
            )
        self.classifier_fitted['LightGBM'] = True

    def _train_sklearn(self, name, train_X, train_y, val_X, val_y, sample_weights, round_num):
        clf = self.classifiers[name]

        try:
            clf.fit(train_X, train_y, sample_weight=sample_weights)
        except (TypeError, ValueError):
            try:
                clf.fit(train_X, train_y)
            except Exception:
                verPrint(self.verbose, 2, f'  >> WARNING: {name} training failed, skipping')
                return

        self.classifier_fitted[name] = True

    def eval(self):
        return

    def set_graph(self, graph, round_num, device, **kwargs):
        self.graph = graph.to(device)
        self.preprocess_graph(round_num)

        self.graph.ndata['ps_label'] = self.graph.ndata['label'].clone()
        self.graph.ndata['ps_train_mask'] = self.graph.ndata['train_mask'].clone()

    def release_graph(self):
        if hasattr(self, 'graph'):
            del self.graph
            gc.collect()

    def augment_graph(self, augment_strat=None, round_num=0, **kwargs):
        if augment_strat is not None:
            augment_strat(self, round_num=round_num, **kwargs)
            self.preprocess_graph(round_num)

    def preprocess_graph(self, round_num, **kwargs):
        self.graph.ndata['age'] = round_num - self.graph.ndata['creation_round']

    def save_model(self, path):
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        if self.classifier_fitted.get('XGBoost', False) and self.classifiers.get('XGBoost') is not None:
            self.classifiers['XGBoost'].save_model(f'{path}_xgb.json')

        if self.classifier_fitted.get('LightGBM', False) and self.classifiers.get('LightGBM') is not None:
            self.classifiers['LightGBM'].save_model(f'{path}_lgb.txt')

        sklearn_state = {}
        for name in self.model_names:
            if name != 'XGBoost' and self.classifier_fitted.get(name, False):
                sklearn_state[name] = self.classifiers[name]

        if sklearn_state:
            torch.save(sklearn_state, f'{path}_sklearn.pt')

        ads_state = {
            'history': {name: list(self.ads.history[name]) for name in self.model_names},
            'ema_scores': dict(self.ads.ema_scores),
            'weights': self.round_weight_history,
            'scores': self.round_score_history,
            'individual_f1': {n: list(v) for n, v in self.round_individual_f1.items()},
            'individual_recall': {n: list(v) for n, v in self.round_individual_recall.items()},
        }
        with open(f'{path}_ads_state.json', 'w') as f:
            json.dump(ads_state, f, default=str)

    def load_model(self, path):
        xgb_path = f'{path}_xgb.json'
        if os.path.exists(xgb_path):
            self.classifiers['XGBoost'] = xgb.Booster()
            self.classifiers['XGBoost'].load_model(xgb_path)
            self.classifier_fitted['XGBoost'] = True

        lgb_path = f'{path}_lgb.txt'
        if os.path.exists(lgb_path):
            self.classifiers['LightGBM'] = lgb.Booster(model_file=lgb_path)
            self.classifier_fitted['LightGBM'] = True

        sklearn_path = f'{path}_sklearn.pt'
        if os.path.exists(sklearn_path):
            sklearn_state = torch.load(sklearn_path)
            for name, clf in sklearn_state.items():
                self.classifiers[name] = clf
                self.classifier_fitted[name] = True

        ads_path = f'{path}_ads_state.json'
        if os.path.exists(ads_path):
            with open(ads_path, 'r') as f:
                ads_state = json.load(f)
            for name in self.model_names:
                if name in ads_state.get('history', {}):
                    self.ads.history[name] = deque(
                        ads_state['history'][name],
                        maxlen=self.ads.history_window
                    )
                if name in ads_state.get('ema_scores', {}):
                    self.ads.ema_scores[name] = ads_state['ema_scores'][name]
            self.round_weight_history = ads_state.get('weights', [])
            self.round_score_history = ads_state.get('scores', [])

    def save_embedder(self, path):
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        torch.save({'model_state_dict': self.embedder.state_dict()}, f'{path}.pt')

    def load_embedder(self, path):
        checkpoint = torch.load(f'{path}.pt')
        self.embedder.load_state_dict(checkpoint['model_state_dict'])

    def postBackprop(self, **kwargs):
        return

    def get_latest_trainlog(self, **kwargs):
        base_log = self.embedder.get_latest_trainlog()

        if self.round_weight_history:
            latest_weights = self.round_weight_history[-1]
            for name in self.model_names:
                base_log[f'weight_{name}'] = latest_weights.get(name, 0.0)

        if self.round_score_history:
            latest_scores = self.round_score_history[-1]
            for name in self.model_names:
                base_log[f'ads_score_{name}'] = latest_scores.get(name, 0.0)

        for name in self.model_names:
            if self.round_individual_f1[name]:
                base_log[f'individual_f1_{name}'] = self.round_individual_f1[name][-1]
            if self.round_individual_recall[name]:
                base_log[f'individual_recall_{name}'] = self.round_individual_recall[name][-1]

        return base_log
