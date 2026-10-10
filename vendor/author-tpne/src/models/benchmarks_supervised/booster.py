import torch
import gc
import numpy as np

from torch import nn

import dgl.nn.pytorch.conv as dglnn
import dgl.utils as ut
import dgl.function as fn

import xgboost as xgb
EPS = 1e-10

#####################
### GIN BACKBONES ###
#####################

class GIN_noparam(nn.Module):
    def __init__(self, num_layers=2, agg='mean', init_eps=-1, **kwargs):
        super().__init__()
        self.gnn = dglnn.GINConv(None, activation=None, init_eps=init_eps, aggregator_type=agg)
        self.num_layers = num_layers

    def forward(self, graph):
        h = graph.ndata['feature']
        h_final = h.detach().clone()
        for i in range(self.num_layers):
            h = self.gnn(graph, h)
            h_final = torch.cat([h_final, h], -1)
        return h_final
        
class RoundGIN_noparam(nn.Module):
    def __init__(
            self, num_layers=2, round_window=3, 
            agg='sum', init_eps=-1, device='cpu', **kwargs):
        super().__init__()
        self.device = device

        self.gnn = dglnn.GINConv(None, activation=None, init_eps=init_eps,aggregator_type=agg)
        self.num_layers = num_layers
        self.round_window = round_window

    def forward(self, blocks):
        with blocks.local_scope():
            x = blocks.ndata['feature']
            h_feats = x.shape[1]

            # Expand to window size
            h_exp = x.repeat(self.round_window, 1)

            # Create mask
            age = blocks.ndata['age'].unsqueeze(1).expand(-1, h_exp.shape[1]).repeat(self.round_window, 1)
            age_range = torch.arange(self.round_window).unsqueeze(1).to(self.device)
            age_range = age_range.repeat_interleave(x.shape[0], dim=0)
            age_range = age_range.repeat_interleave(age.shape[1], dim=1)
            feat_mask = (age_range <= age)

            age_feats = blocks.ndata['age'].unsqueeze(1).expand(-1, h_feats).repeat(self.round_window, 1)
            age_range_feats = torch.arange(self.round_window).unsqueeze(1).to(self.device)
            age_range_feats = age_range_feats.repeat_interleave(x.shape[0], dim=0)
            age_range_feats = age_range_feats.repeat_interleave(age_feats.shape[1], dim=1)
            feat_mask_feats = (age_range_feats <= age_feats)

            # Calculate h and split
            h = h_exp * feat_mask

            h_windows = []
            for w in range(self.round_window):
                h_layer = h[x.shape[0]*w:x.shape[0]*(w+1),]
                feat_mask_layer =  feat_mask_feats[x.shape[0]*w:x.shape[0]*(w+1),]          
                h_windows.append(h_layer)

                for i in range(self.num_layers):                  
                    h_layer = self.gnn(blocks, h_layer) * feat_mask_layer
                    h_windows.append(h_layer)
            
            h_final = torch.cat(h_windows, dim=1)

            return h_final

class SplitRoundGIN_noparam(nn.Module):
    def __init__(
            self, num_layers=2, round_window=3, 
            agg='sum', temporal_agg='mean_final', norm_name=None, init_eps=-1, device='cpu', **kwargs):
        super().__init__()
        self.device = device

        self.gnn = dglnn.GINConv(None, activation=None, init_eps=init_eps, aggregator_type=agg)
        self.num_layers = num_layers
        self.norm_name = norm_name
        self.round_window = round_window
        self.temporal_agg = temporal_agg

    def _normalize(self, x):
        norm = None

        if self.norm_name == 'batch':
            norm = torch.nn.BatchNorm1d(x.shape[1], affine=False).to(self.device)
            res = norm(x)
        elif self.norm_name == 'layer':
            norm = torch.nn.LayerNorm(x.shape[1], elementwise_affine=False).to(self.device)
            res = norm(x)
        elif self.norm_name == 'none':
            res = x

        return res

    def forward(self, blocks):
        with blocks.local_scope():
            x = blocks.ndata['feature']
            h_feats = x.shape[1]

            # Expand to window size
            h_exp = x.repeat(self.round_window, 1)
            h_exp = self._normalize(h_exp)

            # Create mask
            age = blocks.ndata['age'].unsqueeze(1).expand(-1, h_exp.shape[1]).repeat(self.round_window, 1)
            age_range = torch.arange(self.round_window).unsqueeze(1).to(self.device)
            age_range = age_range.repeat_interleave(x.shape[0], dim=0)
            age_range = age_range.repeat_interleave(age.shape[1], dim=1)
            feat_mask = (age_range <= age)

            #age_feats = blocks.ndata['age'].unsqueeze(1).expand(-1, h_feats).repeat(self.round_window, 1)
            #age_range_feats = torch.arange(self.round_window).unsqueeze(1).to(self.device)
            #age_range_feats = age_range_feats.repeat_interleave(x.shape[0], dim=0)
            #age_range_feats = age_range_feats.repeat_interleave(age_feats.shape[1], dim=1)
            #feat_mask_feats = (age_range_feats <= age_feats)

            # Calculate h and split
            h = h_exp * feat_mask

            h_windows = []
            
            for w in range(self.round_window):
                h_neigh = []
                
                h_layer = h[x.shape[0]*w:x.shape[0]*(w+1),]
                #feat_mask_layer =  feat_mask_feats[x.shape[0]*w:x.shape[0]*(w+1),]          
                h_neigh.append(h_layer)

                for i in range(self.num_layers):                  
                    h_layer = self.gnn(blocks, h_layer) # * feat_mask_layer
                    h_layer = self._normalize(h_layer)
                    h_layer = torch.nn.functional.relu(h_layer)

                    h_neigh.append(h_layer)
            
                h_windows.append(torch.cat(h_neigh, dim=1))

            # Calculate h inherent and h temporal
            h_self = h_windows[0]

            if self.round_window > 1:
                if self.temporal_agg == 'mean_final':
                    h_temp = torch.stack([(h_windows[0] - h_windows[i]) for i in range(1, self.round_window)]).mean(dim=0)
                elif self.temporal_agg == 'sum_final':
                    h_temp = torch.stack([(h_windows[0] - h_windows[i]) for i in range(1, self.round_window)]).sum(dim=0)
                elif self.temporal_agg == 'mean_step':
                    h_temp = torch.stack([(h_windows[i-1] - h_windows[i]) for i in range(1, self.round_window)]).mean(dim=0)
                elif self.temporal_agg == 'sum_step':
                    h_temp = torch.stack([(h_windows[i-1] - h_windows[i]) for i in range(1, self.round_window)]).sum(dim=0)
            else:
                h_temp = torch.zeros_like(h_windows[0])
            
            h_final = torch.cat([h_self, h_temp], dim=1)

            return h_final

####################
### MAIN CLASSES ###
####################

class GraphBoost():
    ### BASIC METHODS ###
    def __init__(self, boost_agg_backbone, boost_predictor, boost_metric, device='cpu', num_epoch=500, early_stopping=100, **kwargs):
        self.eval_metric = boost_metric
        self.agg_backbone = boost_agg_backbone(device=device, **kwargs).to(device) if boost_agg_backbone != None else None
        self.predictor = boost_predictor(eval_metric=self.eval_metric)
        self.predictor_fitted = False
        self.device = device
        self.model = None

        self.num_epoch = num_epoch
        self.early_stopping = early_stopping

    def __call__(self, graph, feats, **kwargs):
        agg_feats = self.agg_backbone(graph) if self.agg_backbone != None else graph.ndata['feature']    
        probs = self.predictor.predict(xgb.DMatrix(agg_feats.cpu()))    
        return torch.Tensor([1 - probs, probs]).swapaxes(0, 1).to(self.device) , None, None

    def train(self, graph, weight, round_num):
        feats = self.embed_nodes(graph, None)
        labels = graph.ndata['ps_label']

        train_X = feats[graph.ndata['ps_train_mask']].cpu().numpy()
        train_y = labels[graph.ndata['ps_train_mask']].cpu().numpy()
        val_X = feats[graph.ndata['val_mask']].cpu().numpy()
        val_y = labels[graph.ndata['val_mask']].cpu().numpy()
        
        params = {"objective": "binary:logistic", "scale_pos_weight": weight, "tree_method": "hist", "max_depth": 6, "device": self.device}
        self.predictor = xgb.train(
            params, xgb.DMatrix(train_X, train_y), 
            num_boost_round=self.num_epoch, early_stopping_rounds=self.early_stopping,
            evals=[(xgb.DMatrix(train_X, train_y), 'Train'), (xgb.DMatrix(val_X, val_y), 'Eval')], verbose_eval=True, 
            xgb_model=self.predictor if self.predictor_fitted else None)
        self.predictor_fitted = True
        
        pred_train = self.predictor.predict(xgb.DMatrix(train_X))
        pred_val = self.predictor.predict(xgb.DMatrix(val_X))

        train_score = self.eval_metric(train_y.flatten(), pred_train.flatten())
        val_score = self.eval_metric(val_y.flatten(), pred_val.flatten())

        return train_score, val_score
    
    def embed_nodes(self, graph, x):
        return self.agg_backbone(graph) if self.agg_backbone != None else graph.ndata['feature']

    def eval(self):
        return
    
    ### SET/RELEASE GRAPH ###
    def set_graph(self, graph, round_num, device, **kwargs):
        self.graph = graph.to(device)
        self.preprocess_graph(round_num)

    def release_graph (self):
        if hasattr(self, 'graph'):
            del self.graph
            gc.collect()

    ### GRAPH PREPROCESSING
    def pseudolabel_graph(self, pseudo_strat=None, round_num=0, **kwargs):
        if (pseudo_strat != None) and (round_num > 0):
            label, train_mask = pseudo_strat(self, round_num=round_num, **kwargs)
        else:
            label = self.graph.ndata['label']
            train_mask = self.graph.ndata['train_mask']

        self.graph.ndata['ps_label'] = label
        self.graph.ndata['ps_train_mask'] = train_mask

        self.graph.ndata['val_mask'][train_mask] = False
        self.graph.ndata['test_mask'][train_mask] = False
    
    def augment_graph(self, augment_strat=None, round_num=0, **kwargs):
        if (augment_strat != None):
            augment_strat(self, round_num=round_num, **kwargs)
            self.preprocess_graph(round_num)


    def preprocess_graph(self, round_num, **kwargs):
        self.graph.ndata['age'] = round_num - self.graph.ndata['creation_round'] # For round aggregator
    
    ### CHECKPOINTING ###
    def save_model(self, path):
        if self.predictor_fitted:
            self.predictor.save_model(f'{path}.json')

    def load_model(self, path):
        if self.predictor_fitted:
            self.predictor.load_model(f'{path}.json')

    ### OTHER HOOKS ###
    def postBackprop(self, **kwargs):
        return
    
    def get_latest_trainlog(self, **kwargs):
        return {}