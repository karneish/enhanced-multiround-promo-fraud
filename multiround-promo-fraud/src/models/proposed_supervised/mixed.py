from utils.utils_func import verPrint

import gc
import os
import torch
import numpy
import time
import xgboost as xgb
import dgl.nn.pytorch.conv as dglnn
import dgl.function as fn
import dgl.sampling as samp

from torch import nn
from torch.nn import init
from torch.optim import Adam
from models.base_model import BaseModel
from models.benchmarks_supervised.simple import MLP

EPS = 1e-10
E_CONSTANT = 2.7182817459106445

TEMP_MODEL_SAVE_PATH = '../checkpoint/working_model_file'

#################################
### SELF-SUPERVISED EMBEDDERS ###
#################################

class BaseEmbedder(nn.Module):
    def __init__(self, device='cuda:0', verbose=0, **kwargs):
        super().__init__()
        self.verbose=verbose
        self.device=device
        
        self.latest_trainlog = {}

    def forward(self, blocks, x, **kwargs):
        pass
        
    def embed_nodes(self, blocks, x):
        pass

    def get_latest_trainlog(self, **kwargs):
        return self.latest_trainlog

    def _get_nonparametric_embedding(self, blocks):
        pass

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

class VanillaEmbedder(BaseEmbedder):
    def __init__(
        self, in_feats, h_feats, num_layers,
        layer_agg = 'mean', temporal_agg='mean_final', 
        loss_type='neigh', loss_sample=False, loss_sample_ratio=0.1, loss_sample_k=10,
        dropout_rate=0.5, act_name='ReLU', norm_name='batch',
        device='cuda:0', verbose=0, **kwargs):

        super().__init__()
        self.verbose=verbose
        self.device=device
        
        # General params
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.num_layers = num_layers

        # Saved normalization term
        self.dist_calculated = False
        self.current_dists = None

        # Setting for loss
        self.loss_type = loss_type
        self.loss_sample = loss_sample
        self.loss_sample_ratio = loss_sample_ratio
        self.loss_sample_k = loss_sample_k

        # Other modules
        self.act = getattr(nn, act_name)()
        self.norm_name = norm_name
        self.layer_agg = layer_agg
        self.temporal_agg = temporal_agg
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.gin = dglnn.GINConv(None, activation=None, init_eps=-1, aggregator_type=self.layer_agg).to(self.device)
        
        # Layers
        self.init_weight = nn.Parameter(torch.Tensor(self.in_feats, self.h_feats)).to(self.device)
        init.xavier_uniform_(self.init_weight)

        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(dglnn.GraphConv(self.h_feats, self.h_feats, activation=None, norm='none'))
        
        for i in range(num_layers-1):
            self.gnn_layers.append(dglnn.GraphConv(self.h_feats, self.h_feats, activation=None, norm='none'))

    def forward(self, blocks, x, **kwargs):
        with blocks.local_scope():
            blocks.to(self.device)

            start = time.time()
            h_final = self.embed_nodes(blocks, x) # Embed
            verPrint(self.verbose, 5, f'Embedding: {time.time() - start:.8f}s')

            # Reconstruction Loss
            if self.loss_type == 'ndist':
                r_loss = self._neighborhood_dist_loss(blocks, h_final, loss_sample=self.loss_sample, loss_sample_ratio=self.loss_sample_ratio)
            elif self.loss_type == 'ndot':
                r_loss = self._neighborhood_dot_loss(blocks, h_final, loss_sample=self.loss_sample, loss_sample_ratio=self.loss_sample_ratio)

            final_loss = r_loss
            verPrint(self.verbose, 4, f"Detailed Loss: recon {r_loss.item():.8f} | total {final_loss:.8f}")

             # Store to log
            self.latest_trainlog = {
                'recon_loss': r_loss.item(),
                'final_loss': final_loss.item()
            }
    
            return h_final, final_loss
        
    def embed_nodes(self, blocks, x):
        h_layer = torch.matmul(x, self.init_weight)
        h_layer = self._normalize(h_layer)
        h_layer = self.act(h_layer)

        for i, layer in enumerate(self.gnn_layers):
            if i > 0:
                h_layer = self.dropout(h_layer)              
            h_layer = layer(blocks, h_layer)
            h_layer = self._normalize(h_layer)
            h_layer = self.act(h_layer)
                                
        return h_layer
      
    def _neighborhood_dist_loss(self, blocks, h_current, loss_sample=False, loss_sample_ratio=0.1):
        # Calculate non-parametric average neighborhood distance        
        if self.dist_calculated == False:

            h_nonpar_final = self._get_nonparametric_embedding(blocks)

            start = time.time()
            if loss_sample:
                self.sampled_nodes = torch.randperm(blocks.num_nodes())[:int(blocks.num_nodes() * loss_sample_ratio)].to(self.device)
                sg = samp.sample_neighbors(blocks, self.sampled_nodes, self.loss_sample_k, replace=False)

                self.sampled_edges = sg.edata['_ID'].detach().clone()
                del sg
                gc.collect()
            else:
                self.sampled_nodes = torch.arange(blocks.num_nodes()).to(self.device)
                self.sampled_edges = '__ALL__'

            blocks.ndata['h_nonpar'] = h_nonpar_final
            blocks.apply_edges(lambda e: {'d_nonpar': torch.nn.functional.pairwise_distance(e.src['h_nonpar'], e.dst['h_nonpar'])}, edges=self.sampled_edges)
            blocks.pull(self.sampled_nodes, fn.copy_e('d_nonpar', 'm'), fn.mean('m', f'mean_dist_nonpar'))
            self.current_dists[f'mean_dist_nonpar'] = blocks.ndata[f'mean_dist_nonpar'].detach().clone()
            
            self.dist_calculated = True
            verPrint(self.verbose, 5, f'Nonparametric distance calc {time.time() - start:.8f}s')
    
        start = time.time()
        # Calculate current average neighborhood distance              
        blocks.ndata['h_par'] = h_current     
        blocks.apply_edges(lambda e: {'d_par': torch.nn.functional.pairwise_distance(e.src['h_par'], e.dst['h_par'])}, edges=self.sampled_edges)
        blocks.pull(self.sampled_nodes, fn.copy_e('d_par', 'm'), fn.mean('m', f'mean_dist_par'))
            
        verPrint(self.verbose, 5, f'Parametric distance calc {time.time() - start:.8f}s')
        start = time.time()

        # Local distance term
        preservation_loss = torch.nn.functional.mse_loss(torch.log(blocks.ndata[f'mean_dist_par'][self.sampled_nodes]), torch.log(self.current_dists[f'mean_dist_nonpar'][self.sampled_nodes]), reduction='mean')
        verPrint(self.verbose, 5, f'Preservation loss calc {time.time() - start:.8f}s')

        return preservation_loss
    
    def _neighborhood_dot_loss(self, blocks, h_current, loss_sample=False, loss_sample_ratio=0.1):
        # Calculate non-parametric average neighborhood distance        
        if self.dist_calculated == False:

            h_nonpar_final = self._get_nonparametric_embedding(blocks)

            start = time.time()
            if loss_sample:
                self.sampled_nodes = torch.randperm(blocks.num_nodes())[:int(blocks.num_nodes() * loss_sample_ratio)].to(self.device)
                sg = samp.sample_neighbors(blocks, self.sampled_nodes, self.loss_sample_k, replace=False)

                self.sampled_edges = sg.edata['_ID'].detach().clone()
                del sg
                gc.collect()
            else:
                self.sampled_nodes = torch.arange(blocks.num_nodes()).to(self.device)
                self.sampled_edges = '__ALL__'

            blocks.ndata['h_nonpar'] = h_nonpar_final
            blocks.apply_edges(lambda e: {'d_nonpar': (((e.src['h_nonpar'] * e.dst['h_nonpar']).sum(dim=1)) / (torch.norm(e.src['h_nonpar'], p=2, dim=1) * torch.norm(e.dst['h_nonpar'], p=2, dim=1)))}, edges=self.sampled_edges)
            blocks.pull(self.sampled_nodes, fn.copy_e('d_nonpar', 'm'), fn.mean('m', f'mean_dist_nonpar'))
            self.current_dists[f'mean_dist_nonpar'] = blocks.ndata[f'mean_dist_nonpar'].detach().clone()
            
            self.dist_calculated = True
            verPrint(self.verbose, 5, f'Nonparametric distance calc {time.time() - start:.8f}s')
    
        start = time.time()
        # Calculate current average neighborhood distance              
        blocks.ndata['h_par'] = h_current     
        blocks.apply_edges(lambda e: {'d_par': (((e.src['h_par'] * e.dst['h_par']).sum(dim=1)) / ((torch.norm(e.src['h_par'], p=2, dim=1) * torch.norm(e.dst['h_par'], p=2, dim=1) + EPS)))}, edges=self.sampled_edges)
        blocks.pull(self.sampled_nodes, fn.copy_e('d_par', 'm'), fn.mean('m', f'mean_dist_par'))
            
        verPrint(self.verbose, 5, f'Parametric distance calc {time.time() - start:.8f}s')
        start = time.time()

        # Local distance term
        preservation_loss = torch.nn.functional.mse_loss(torch.log(blocks.ndata[f'mean_dist_par'][self.sampled_nodes]), torch.log(self.current_dists[f'mean_dist_nonpar'][self.sampled_nodes]), reduction='mean')
        verPrint(self.verbose, 5, f'Preservation loss calc {time.time() - start:.8f}s')

        return preservation_loss

    def _get_nonparametric_embedding(self, blocks):
        start = time.time()
        self.current_dists = {}
        
        h_nonpar = blocks.ndata['feature']
        h_nonpar = self._normalize(h_nonpar)
        h_nonpar = self.act(h_nonpar) if self.act != None else h_nonpar

        for i in range(self.num_layers):            
            h_nonpar = self.gin(blocks, h_nonpar)
            h_nonpar = self._normalize(h_nonpar)
            h_nonpar = self.act(h_nonpar) if self.act != None else h_nonpar

        verPrint(self.verbose, 5, f'Nonparametric agg {time.time() - start:.8f}s')

        return h_nonpar

class TemporalEmbedder(BaseEmbedder):
    def __init__(
        self, in_feats, h_feats, num_layers, round_window, 
        layer_agg = 'mean', temporal_agg='mean_final', 
        loss_type='neigh', tloss_type='normal',
        alpha=1, beta=1, 
        loss_sample=False, loss_sample_ratio=0.1, loss_sample_k=10,
        dropout_rate=0.5, act_name='ReLU', norm_name='batch',
        device='cuda:0', verbose=0, **kwargs):

        super().__init__()
        self.verbose=verbose
        self.device=device
        
        # General params
        self.in_feats = in_feats
        self.round_window = round_window
        self.num_layers = num_layers
        self.h_feats = h_feats
        self.alpha = alpha
        self.beta = beta

        # Saved normalization term
        self.dist_calculated = False
        self.current_dists = None

        # Setting for loss
        self.loss_type = loss_type
        self.tloss_type = tloss_type
        self.loss_sample=loss_sample
        self.loss_sample_ratio=loss_sample_ratio
        self.loss_sample_k=loss_sample_k

        # Other modules
        self.act = getattr(nn, act_name)()
        self.norm_name = norm_name
        self.layer_agg = layer_agg
        self.temporal_agg = temporal_agg
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.gin = dglnn.GINConv(None, activation=None, init_eps=-1, aggregator_type=self.layer_agg).to(self.device)
        
        # Layers
        self.init_weight = nn.Parameter(torch.Tensor(self.in_feats, self.h_feats)).to(self.device)
        self.temporal_weight = nn.Linear(self.h_feats * (self.num_layers + 1) * self. round_window, self.h_feats * (self.num_layers + 1), device=self.device)
        init.xavier_uniform_(self.init_weight)

        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(dglnn.GraphConv(self.h_feats, self.h_feats, activation=None, norm='none'))
        for i in range(num_layers-1):
            self.gnn_layers.append(dglnn.GraphConv(self.h_feats, self.h_feats, activation=None, norm='none'))

    def forward(self, blocks, x, **kwargs):
        with blocks.local_scope():
            blocks.to(self.device)

            start = time.time()
            h_final = self.embed_nodes(blocks, x) # Embed
            verPrint(self.verbose, 5, f'Embedding: {time.time() - start:.8f}s')
            
            # Split
            h_current = h_final[:,:(self.h_feats * (self.num_layers + 1))]
            h_temp = h_final[:,-(self.h_feats * (self.num_layers + 1)):]

            # Reconstruction Loss
            if self.loss_type == 'ndist':
                r_loss = self._neighborhood_dist_loss(blocks, h_current, loss_sample=self.loss_sample, loss_sample_ratio=self.loss_sample_ratio)
            elif self.loss_type == 'ndot':
                r_loss = self._neighborhood_dot_loss(blocks, h_current, loss_sample=self.loss_sample, loss_sample_ratio=self.loss_sample_ratio)

            # Disentanglement Loss
            start = time.time()
            d_loss = self._correlation_loss(h_current, h_temp, loss_sample=self.loss_sample, loss_sample_ratio=self.loss_sample_ratio)
            verPrint(self.verbose, 5, f'Correlation loss calc {time.time() - start:.8f}s')

            # Temporal Maximization Loss
            start = time.time()
            t_loss = self._temporal_loss_standard(blocks, h_temp)
            verPrint(self.verbose, 5, f'Temporal loss calc {time.time() - start:.8f}s')

            # Final loss
            norm_recon = 1 / (1 + self.alpha + self.beta)
            norm_disent = self.alpha / (1 + self.alpha + self.beta)
            norm_temporal = self.beta / (1 + self.alpha + self.beta)

            final_loss = (norm_recon * r_loss) + (norm_disent* d_loss) + (norm_temporal * t_loss)
            verPrint(self.verbose, 4, f"Detailed Loss: recon {r_loss.item():.8f}-{norm_recon * r_loss.item():.8f} | disen {d_loss.item():.8f}-{norm_disent * d_loss.item():.8f} | temporal {t_loss.item():.8f}-{norm_temporal * t_loss.item():.8f} | total {final_loss:.8f}")

             # Store to log
            self.latest_trainlog = {
                'recon_loss': r_loss.item(),
                'weighted_recon_loss': norm_recon * r_loss.item(),
                'disen_loss': d_loss.item(),
                'weighted_disen_loss': norm_disent * d_loss.item(),
                'temp_loss': t_loss.item(),
                'weighted_temp_loss': norm_temporal * t_loss.item(),\
                'final_loss': final_loss.item()
            }
    
            return h_final, final_loss
        
    def embed_nodes(self, blocks, x):
        # Expand to window size
        h_exp = x.repeat(self.round_window, 1).to(self.device)        

        # Create mask
        age = blocks.ndata['age'].unsqueeze(1).expand(-1, h_exp.shape[1]).repeat(self.round_window, 1).to(self.device)
        age_range = torch.arange(self.round_window).unsqueeze(1).to(self.device)
        age_range = age_range.repeat_interleave(x.shape[0], dim=0)
        age_range = age_range.repeat_interleave(age.shape[1], dim=1)
        feat_mask = (age_range <= age)

        # Calculate h and split
        h = h_exp * feat_mask
        h_windows = []
        
        # Get parametrized embedding for all hop and all round
        for w in range(self.round_window):
            h_neigh = []

            h_layer = torch.matmul(h[x.shape[0]*w:x.shape[0]*(w+1),], self.init_weight)
            h_layer = self._normalize(h_layer)
            h_layer = self.act(h_layer)
            
            h_neigh.append(h_layer)

            for i, layer in enumerate(self.gnn_layers):
                if i > 0:
                    h_layer = self.dropout(h_layer)              
                h_layer = layer(blocks, h_layer)
                h_layer = self._normalize(h_layer)
                h_layer = self.act(h_layer)

                h_neigh.append(h_layer)

            h_windows.append(torch.cat(h_neigh, dim=1))
        
        # Get temporal embedding
        if self.temporal_agg == 'mean_final':
            h_temp = torch.stack([(h_windows[0] - h_windows[i]) for i in range(1, self.round_window)]).mean(dim=0)
        elif self.temporal_agg == 'sum_final':
            h_temp = torch.stack([(h_windows[0] - h_windows[i]) for i in range(1, self.round_window)]).sum(dim=0)
        elif self.temporal_agg == 'mean_step':
            h_temp = torch.stack([(h_windows[i-1] - h_windows[i]) for i in range(1, self.round_window)]).mean(dim=0)
        elif self.temporal_agg == 'sum_step':
            h_temp = torch.stack([(h_windows[i-1] - h_windows[i]) for i in range(1, self.round_window)]).sum(dim=0)
        elif self.temporal_agg == 'weight':
            h_temp = self.temporal_weight(torch.cat(h_windows, dim=1))
                                
        # Final concatenation
        h_current = h_windows[0]
        h_final = torch.cat([h_current, h_temp], dim=1)
        return h_final
    
    def _temporal_loss_standard(self, blocks, h_temp):
        temporal_norm = torch.linalg.vector_norm(h_temp, ord=2, dim=1).mean()
        t_loss = 1 / torch.log(temporal_norm + E_CONSTANT)
        return t_loss
          
    def _neighborhood_dist_loss(self, blocks, h_current, loss_sample=False, loss_sample_ratio=0.1):
        # Calculate non-parametric average neighborhood distance        
        if self.dist_calculated == False:

            h_nonpar_final = self._get_nonparametric_embedding(blocks)
            feat_dim_nonpar = int(h_nonpar_final.shape[1]/(self.num_layers+1))

            start = time.time()
            if loss_sample:
                self.sampled_nodes = torch.randperm(blocks.num_nodes())[:int(blocks.num_nodes() * loss_sample_ratio)].to(self.device)
                sg = samp.sample_neighbors(blocks, self.sampled_nodes, self.loss_sample_k, replace=False)

                self.sampled_edges = sg.edata['_ID'].detach().clone()
                del sg
                gc.collect()
            else:
                self.sampled_nodes = torch.arange(blocks.num_nodes()).to(self.device)
                self.sampled_edges = '__ALL__'

            for i in range(self.num_layers + 1):
                blocks.ndata['h_nonpar'] = h_nonpar_final[:,feat_dim_nonpar*i:feat_dim_nonpar*(i+1)]
                blocks.apply_edges(lambda e: {'d_nonpar': torch.nn.functional.pairwise_distance(e.src['h_nonpar'], e.dst['h_nonpar'])}, edges=self.sampled_edges)
                blocks.pull(self.sampled_nodes, fn.copy_e('d_nonpar', 'm'), fn.mean('m', f'mean_dist_nonpar_{i}'))

                self.current_dists[f'mean_dist_nonpar_{i}'] = blocks.ndata[f'mean_dist_nonpar_{i}'].detach().clone()
            
            self.dist_calculated = True
            verPrint(self.verbose, 5, f'Nonparametric distance calc {time.time() - start:.8f}s')
    
        start = time.time()
        # Calculate current average neighborhood distance
        feat_dim = int(h_current.shape[1]/(self.num_layers+1))
                
        for i in range(self.num_layers + 1):
            blocks.ndata['h_par'] = h_current[:,feat_dim*i:feat_dim*(i+1)]     
            blocks.apply_edges(lambda e: {'d_par': torch.nn.functional.pairwise_distance(e.src['h_par'], e.dst['h_par'])}, edges=self.sampled_edges)
            blocks.pull(self.sampled_nodes, fn.copy_e('d_par', 'm'), fn.mean('m', f'mean_dist_par_{i}'))
            
        verPrint(self.verbose, 5, f'Parametric distance calc {time.time() - start:.8f}s')
        start = time.time()

        # Local distance term
        preservation_loss = None
        for i in range(self.num_layers + 1):
            layer_loss = torch.nn.functional.mse_loss(torch.log(blocks.ndata[f'mean_dist_par_{i}'][self.sampled_nodes]), torch.log(self.current_dists[f'mean_dist_nonpar_{i}'][self.sampled_nodes]), reduction='mean')
            preservation_loss = layer_loss if preservation_loss == None else (preservation_loss + layer_loss)

        preservation_loss = preservation_loss / (self.num_layers + 1)
        verPrint(self.verbose, 5, f'Preservation loss calc {time.time() - start:.8f}s')

        return preservation_loss
    
    def _neighborhood_dot_loss(self, blocks, h_current, loss_sample=False, loss_sample_ratio=0.1):
        # Calculate non-parametric average neighborhood distance        
        if self.dist_calculated == False:

            h_nonpar_final = self._get_nonparametric_embedding(blocks)
            feat_dim_nonpar = int(h_nonpar_final.shape[1]/(self.num_layers+1))

            start = time.time()
            if loss_sample:
                self.sampled_nodes = torch.randperm(blocks.num_nodes())[:int(blocks.num_nodes() * loss_sample_ratio)].to(self.device)
                sg = samp.sample_neighbors(blocks, self.sampled_nodes, self.loss_sample_k, replace=False)

                self.sampled_edges = sg.edata['_ID'].detach().clone()
                del sg
                gc.collect()
            else:
                self.sampled_nodes = torch.arange(blocks.num_nodes()).to(self.device)
                self.sampled_edges = '__ALL__'

            for i in range(self.num_layers + 1):
                blocks.ndata['h_nonpar'] = h_nonpar_final[:,feat_dim_nonpar*i:feat_dim_nonpar*(i+1)]
                blocks.apply_edges(lambda e: {'d_nonpar': (((e.src['h_nonpar'] * e.dst['h_nonpar']).sum(dim=1)) / (torch.norm(e.src['h_nonpar'], p=2, dim=1) * torch.norm(e.dst['h_nonpar'], p=2, dim=1)))}, edges=self.sampled_edges)
                blocks.pull(self.sampled_nodes, fn.copy_e('d_nonpar', 'm'), fn.mean('m', f'mean_dist_nonpar_{i}'))

                self.current_dists[f'mean_dist_nonpar_{i}'] = blocks.ndata[f'mean_dist_nonpar_{i}'].detach().clone()
            
            self.dist_calculated = True
            verPrint(self.verbose, 5, f'Nonparametric distance calc {time.time() - start:.8f}s')
    
        start = time.time()
        # Calculate current average neighborhood distance
        feat_dim = int(h_current.shape[1]/(self.num_layers+1))
                
        for i in range(self.num_layers + 1):
            blocks.ndata['h_par'] = h_current[:,feat_dim*i:feat_dim*(i+1)]
            
            blocks.apply_edges(lambda e: {'d_par': (((e.src['h_par'] * e.dst['h_par']).sum(dim=1)) / ((torch.norm(e.src['h_par'], p=2, dim=1) * torch.norm(e.dst['h_par'], p=2, dim=1) + EPS)))}, edges=self.sampled_edges)
            blocks.pull(self.sampled_nodes, fn.copy_e('d_par', 'm'), fn.mean('m', f'mean_dist_par_{i}'))
            
        verPrint(self.verbose, 5, f'Parametric distance calc {time.time() - start:.8f}s')
        start = time.time()

        # Local distance term
        preservation_loss = None
        for i in range(self.num_layers + 1):
            layer_loss = torch.nn.functional.mse_loss(blocks.ndata[f'mean_dist_par_{i}'][self.sampled_nodes], self.current_dists[f'mean_dist_nonpar_{i}'][self.sampled_nodes], reduction='mean')
            preservation_loss = layer_loss if preservation_loss == None else (preservation_loss + layer_loss)

        preservation_loss = preservation_loss / (self.num_layers + 1)
        verPrint(self.verbose, 5, f'Preservation loss calc {time.time() - start:.8f}s')

        return preservation_loss

    def _correlation_loss(self, h_current, h_temp, loss_sample=False, loss_sample_ratio=0.1):                    
        return 1 - (((h_current * h_temp).sum(dim=1)) / (EPS + (torch.norm(h_current, p=2, dim=1) * torch.norm(h_temp, p=2, dim=1)))).mean()
    
    def _get_nonparametric_embedding(self, blocks):
        start = time.time()
        self.current_dists = {}
        
        h_nonpar = blocks.ndata['feature']
        h_nonpar = self._normalize(h_nonpar)
        h_nonpar = self.act(h_nonpar) if self.act != None else h_nonpar

        h_nonpar_final = h_nonpar.detach().clone()
        for i in range(self.num_layers):            
            h_nonpar = self.gin(blocks, h_nonpar)
            h_nonpar = self._normalize(h_nonpar)
            h_nonpar = self.act(h_nonpar) if self.act != None else h_nonpar

            h_nonpar_final = torch.cat([h_nonpar_final, h_nonpar], -1)

        verPrint(self.verbose, 5, f'Nonparametric agg {time.time() - start:.8f}s')

        return h_nonpar_final

class TemporalMixedEmbedder(BaseEmbedder):
    def __init__(
        self, in_feats, h_feats, num_layers, round_window, 
        layer_agg = 'mean', temporal_agg='mean_final', 
        loss_type='neigh', tloss_type='normal',
        alpha=1, beta=1, 
        loss_sample=False, loss_sample_ratio=0.1, loss_sample_k=10,
        dropout_rate=0.5, act_name='ReLU', norm_name='batch',
        device='cuda:0', verbose=0, **kwargs):

        super().__init__()
        self.verbose=verbose
        self.device=device
        
        # General params
        self.in_feats = in_feats
        self.round_window = round_window
        self.num_layers = num_layers
        self.h_feats = h_feats
        self.alpha = alpha
        self.beta = beta

        # Saved normalization term
        self.dist_calculated = False
        self.current_dists = None
        self.latest_h_nontemp = None
        self.latest_h_temp = None

        # Setting for loss
        self.loss_type = loss_type
        self.tloss_type = tloss_type
        self.loss_sample=loss_sample
        self.loss_sample_ratio=loss_sample_ratio
        self.loss_sample_k=loss_sample_k

        # Other modules
        self.act = getattr(nn, act_name)()
        self.norm_name = norm_name
        self.layer_agg = layer_agg
        self.temporal_agg = temporal_agg
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.gin = dglnn.GINConv(None, activation=None, init_eps=-1, aggregator_type=self.layer_agg).to(self.device)
        
        # Layers
        self.init_weight = nn.Parameter(torch.Tensor(self.in_feats, self.h_feats)).to(self.device)
        init.xavier_uniform_(self.init_weight)

        self.temporal_weight = nn.Linear(self.h_feats * (self.num_layers + 1) * self. round_window, self.h_feats * (self.num_layers + 1), device=self.device)
        self.temporal_att = MLP(self.h_feats * (self.num_layers + 1) * 2, h_feats = self.h_feats, num_classes=1, num_layers=self.num_layers, device=self.device)

        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(dglnn.GraphConv(self.h_feats, self.h_feats, activation=None, norm='none'))
        for i in range(num_layers-1):
            self.gnn_layers.append(dglnn.GraphConv(self.h_feats, self.h_feats, activation=None, norm='none'))

    def forward(self, blocks, x, **kwargs):
        with blocks.local_scope():
            blocks.to(self.device)

            start = time.time()
            h_final, h_current, h_temp, a_temp = self.embed_nodes(blocks, x, return_components=True) # Embed
            verPrint(self.verbose, 5, f'Embedding: {time.time() - start:.8f}s')

            # Save target if first epoch
            if self.dist_calculated == False:
                self.a_target = a_temp.detach().clone()

            # Reconstruction Loss
            if self.loss_type == 'ndist':
                r_loss = self._neighborhood_dist_loss(blocks, h_current, loss_sample=self.loss_sample, loss_sample_ratio=self.loss_sample_ratio)
            elif self.loss_type == 'ndot':
                r_loss = self._neighborhood_dot_loss(blocks, h_current, loss_sample=self.loss_sample, loss_sample_ratio=self.loss_sample_ratio)

            # Disentanglement Loss
            start = time.time()
            d_loss = self._correlation_loss(h_current, h_temp, loss_sample=self.loss_sample, loss_sample_ratio=self.loss_sample_ratio)
            verPrint(self.verbose, 5, f'Correlation loss calc {time.time() - start:.8f}s')

            # Temporal Maximization Loss
            start = time.time()
            t_loss = self._temporal_loss_standard(blocks, h_temp)
            verPrint(self.verbose, 5, f'Temporal loss calc {time.time() - start:.8f}s')
    
            # Attention Loss
            start = time.time()
            a_loss = self._attention_loss(blocks, a_temp)
            verPrint(self.verbose, 5, f'Attention loss calc {time.time() - start:.8f}s')

            # Final loss
            norm_recon = 1 / (1 + self.alpha + self.beta + 1)
            norm_disent = self.alpha / (1 + self.alpha + self.beta + 1)
            norm_temporal = self.beta / (1 + self.alpha + self.beta + 1)
            norm_atten = 1 / (1 + self.alpha + self.beta + 1)

            final_loss = (norm_recon * r_loss) + (norm_disent * d_loss) + (norm_temporal * t_loss)
            final_loss = final_loss if a_loss == None else final_loss + (norm_atten * a_loss)

            verPrint(self.verbose, 4, f"Detailed Loss: recon {r_loss.item():.8f}-{norm_recon * r_loss.item():.8f} | disen {d_loss.item():.8f}-{norm_disent * d_loss.item():.8f} | temporal {t_loss.item():.8f}-{norm_temporal * t_loss.item():.8f} | attention {0 if a_loss == None else a_loss.item():.8f}-{0 if a_loss == None else norm_atten * a_loss.item():.8f} | total {final_loss:.8f}")

            # Store to log
            self.latest_trainlog = {
                'recon_loss': r_loss.item(),
                'weighted_recon_loss': norm_recon * r_loss.item(),
                'disen_loss': d_loss.item(),
                'weighted_disen_loss': norm_disent * d_loss.item(),
                'temp_loss': t_loss.item(),
                'weighted_temp_loss': norm_temporal * t_loss.item(),
                'atten_loss': 0 if a_loss == None else a_loss.item(),
                'weighted_atten_loss': 0 if a_loss == None else norm_atten * a_loss.item(),
                'final_loss': final_loss.item()
            }

            return h_final, final_loss
        
    def embed_nodes(self, blocks, x, return_components=False):
        # Expand to window size
        h_exp = x.repeat(self.round_window, 1).to(self.device)        

        # Create mask
        age = blocks.ndata['age'].unsqueeze(1).expand(-1, h_exp.shape[1]).repeat(self.round_window, 1).to(self.device)
        age_range = torch.arange(self.round_window).unsqueeze(1).to(self.device)
        age_range = age_range.repeat_interleave(x.shape[0], dim=0)
        age_range = age_range.repeat_interleave(age.shape[1], dim=1)
        feat_mask = (age_range <= age)

        # Calculate h and split
        h = h_exp * feat_mask
        h_windows = []
        
        # Get parametrized embedding for all hop and all round
        for w in range(self.round_window):
            h_neigh = []

            h_layer = torch.matmul(h[x.shape[0]*w:x.shape[0]*(w+1),], self.init_weight)
            h_layer = self._normalize(h_layer)
            h_layer = self.act(h_layer)
            
            h_neigh.append(h_layer)

            for i, layer in enumerate(self.gnn_layers):
                if i > 0:
                    h_layer = self.dropout(h_layer)              
                h_layer = layer(blocks, h_layer)
                h_layer = self._normalize(h_layer)
                h_layer = self.act(h_layer)

                h_neigh.append(h_layer)

            h_windows.append(torch.cat(h_neigh, dim=1))
        
        # Get temporal embedding
        if self.temporal_agg == 'mean_final':
            h_temp = torch.stack([(h_windows[0] - h_windows[i]) for i in range(1, self.round_window)]).mean(dim=0)
        elif self.temporal_agg == 'sum_final':
            h_temp = torch.stack([(h_windows[0] - h_windows[i]) for i in range(1, self.round_window)]).sum(dim=0)
        elif self.temporal_agg == 'mean_step':
            h_temp = torch.stack([(h_windows[i-1] - h_windows[i]) for i in range(1, self.round_window)]).mean(dim=0)
        elif self.temporal_agg == 'sum_step':
            h_temp = torch.stack([(h_windows[i-1] - h_windows[i]) for i in range(1, self.round_window)]).sum(dim=0)
        elif self.temporal_agg == 'weight':
            h_temp = self.temporal_weight(torch.cat(h_windows, dim=1))
                                
        # Final concatenation
        h_current = h_windows[0]
        a_temp = self.temporal_att(torch.cat([h_current, h_temp], dim=1), is_graph=False)
        a_temp = getattr(nn, 'Sigmoid')()(a_temp)
        h_final = ((1- a_temp) * h_current) + (a_temp * h_temp)

        if not return_components:
            return h_final
        else:
            return h_final, h_current, h_temp, a_temp
    
    def _temporal_loss_standard(self, blocks, h_temp):
        temporal_norm = torch.linalg.vector_norm(h_temp, ord=2, dim=1).mean()
        t_loss = 1 / torch.log(temporal_norm + E_CONSTANT)
        return t_loss
        
    def _neighborhood_dist_loss(self, blocks, h_current, loss_sample=False, loss_sample_ratio=0.1):
        # Calculate non-parametric average neighborhood distance        
        if self.dist_calculated == False:

            h_nonpar_final = self._get_nonparametric_embedding(blocks)
            feat_dim_nonpar = int(h_nonpar_final.shape[1]/(self.num_layers+1))

            start = time.time()
            if loss_sample:
                self.sampled_nodes = torch.randperm(blocks.num_nodes())[:int(blocks.num_nodes() * loss_sample_ratio)].to(self.device)
                sg = samp.sample_neighbors(blocks, self.sampled_nodes, self.loss_sample_k, replace=False)

                self.sampled_edges = sg.edata['_ID'].detach().clone()
                del sg
                gc.collect()
            else:
                self.sampled_nodes = torch.arange(blocks.num_nodes()).to(self.device)
                self.sampled_edges = '__ALL__'

            for i in range(self.num_layers + 1):
                blocks.ndata['h_nonpar'] = h_nonpar_final[:,feat_dim_nonpar*i:feat_dim_nonpar*(i+1)]
                blocks.apply_edges(lambda e: {'d_nonpar': torch.nn.functional.pairwise_distance(e.src['h_nonpar'], e.dst['h_nonpar'])}, edges=self.sampled_edges)
                blocks.pull(self.sampled_nodes, fn.copy_e('d_nonpar', 'm'), fn.mean('m', f'mean_dist_nonpar_{i}'))

                self.current_dists[f'mean_dist_nonpar_{i}'] = blocks.ndata[f'mean_dist_nonpar_{i}'].detach().clone()
            
            self.dist_calculated = True
            verPrint(self.verbose, 5, f'Nonparametric distance calc {time.time() - start:.8f}s')
    
        start = time.time()
        # Calculate current average neighborhood distance
        feat_dim = int(h_current.shape[1]/(self.num_layers+1))
                
        for i in range(self.num_layers + 1):
            blocks.ndata['h_par'] = h_current[:,feat_dim*i:feat_dim*(i+1)]     
            blocks.apply_edges(lambda e: {'d_par': torch.nn.functional.pairwise_distance(e.src['h_par'], e.dst['h_par'])}, edges=self.sampled_edges)
            blocks.pull(self.sampled_nodes, fn.copy_e('d_par', 'm'), fn.mean('m', f'mean_dist_par_{i}'))
            
        verPrint(self.verbose, 5, f'Parametric distance calc {time.time() - start:.8f}s')
        start = time.time()

        # Local distance term
        preservation_loss = None
        for i in range(self.num_layers + 1):
            layer_loss = torch.nn.functional.mse_loss(torch.log(blocks.ndata[f'mean_dist_par_{i}'][self.sampled_nodes]), torch.log(self.current_dists[f'mean_dist_nonpar_{i}'][self.sampled_nodes]), reduction='mean')
            preservation_loss = layer_loss if preservation_loss == None else (preservation_loss + layer_loss)

        preservation_loss = preservation_loss / (self.num_layers + 1)
        verPrint(self.verbose, 5, f'Preservation loss calc {time.time() - start:.8f}s')

        return preservation_loss
    
    def _neighborhood_dot_loss(self, blocks, h_current, loss_sample=False, loss_sample_ratio=0.1):
        # Calculate non-parametric average neighborhood distance        
        if self.dist_calculated == False:

            h_nonpar_final = self._get_nonparametric_embedding(blocks)
            feat_dim_nonpar = int(h_nonpar_final.shape[1]/(self.num_layers+1))

            start = time.time()
            if loss_sample:
                self.sampled_nodes = torch.randperm(blocks.num_nodes())[:int(blocks.num_nodes() * loss_sample_ratio)].to(self.device)
                sg = samp.sample_neighbors(blocks, self.sampled_nodes, self.loss_sample_k, replace=False)

                self.sampled_edges = sg.edata['_ID'].detach().clone()
                del sg
                gc.collect()
            else:
                self.sampled_nodes = torch.arange(blocks.num_nodes()).to(self.device)
                self.sampled_edges = '__ALL__'

            for i in range(self.num_layers + 1):
                blocks.ndata['h_nonpar'] = h_nonpar_final[:,feat_dim_nonpar*i:feat_dim_nonpar*(i+1)]
                blocks.apply_edges(lambda e: {'d_nonpar': (((e.src['h_nonpar'] * e.dst['h_nonpar']).sum(dim=1)) / (torch.norm(e.src['h_nonpar'], p=2, dim=1) * torch.norm(e.dst['h_nonpar'], p=2, dim=1)))}, edges=self.sampled_edges)
                blocks.pull(self.sampled_nodes, fn.copy_e('d_nonpar', 'm'), fn.mean('m', f'mean_dist_nonpar_{i}'))

                self.current_dists[f'mean_dist_nonpar_{i}'] = blocks.ndata[f'mean_dist_nonpar_{i}'].detach().clone()
            
            self.dist_calculated = True
            verPrint(self.verbose, 5, f'Nonparametric distance calc {time.time() - start:.8f}s')
    
        start = time.time()
        # Calculate current average neighborhood distance
        feat_dim = int(h_current.shape[1]/(self.num_layers+1))
                
        for i in range(self.num_layers + 1):
            blocks.ndata['h_par'] = h_current[:,feat_dim*i:feat_dim*(i+1)]
            
            blocks.apply_edges(lambda e: {'d_par': (((e.src['h_par'] * e.dst['h_par']).sum(dim=1)) / ((torch.norm(e.src['h_par'], p=2, dim=1) * torch.norm(e.dst['h_par'], p=2, dim=1) + EPS)))}, edges=self.sampled_edges)
            blocks.pull(self.sampled_nodes, fn.copy_e('d_par', 'm'), fn.mean('m', f'mean_dist_par_{i}'))
            
        verPrint(self.verbose, 5, f'Parametric distance calc {time.time() - start:.8f}s')
        start = time.time()

        # Local distance term
        preservation_loss = None
        for i in range(self.num_layers + 1):
            layer_loss = torch.nn.functional.mse_loss(blocks.ndata[f'mean_dist_par_{i}'][self.sampled_nodes], self.current_dists[f'mean_dist_nonpar_{i}'][self.sampled_nodes], reduction='mean')
            preservation_loss = layer_loss if preservation_loss == None else (preservation_loss + layer_loss)

        preservation_loss = preservation_loss / (self.num_layers + 1)
        verPrint(self.verbose, 5, f'Preservation loss calc {time.time() - start:.8f}s')

        return preservation_loss

    def _correlation_loss(self, h_current, h_temp, loss_sample=False, loss_sample_ratio=0.1):                    
        return 1 - (((h_current * h_temp).sum(dim=1)) / (EPS + (torch.norm(h_current, p=2, dim=1) * torch.norm(h_temp, p=2, dim=1)))).mean()
    
    def _attention_loss(self, blocks, a_temp):
        if blocks.ndata['predicted'].all():
            return None
        else:
            ceil_age = torch.minimum(blocks.ndata['age'].detach().clone(), torch.full(blocks.ndata['age'].shape, self.round_window).to(self.device))
            rate = ceil_age / self.round_window
            rate[blocks.ndata['predicted']] = self.a_target.flatten()[blocks.ndata['predicted']]
            a_loss = torch.nn.functional.mse_loss(torch.log(a_temp + EPS), torch.log(rate + EPS).unsqueeze(-1), reduction='mean')
            return a_loss

    def _get_nonparametric_embedding(self, blocks):
        start = time.time()
        self.current_dists = {}
        
        h_nonpar = blocks.ndata['feature']
        h_nonpar = self._normalize(h_nonpar)
        h_nonpar = self.act(h_nonpar) if self.act != None else h_nonpar

        h_nonpar_final = h_nonpar.detach().clone()
        for i in range(self.num_layers):            
            h_nonpar = self.gin(blocks, h_nonpar)
            h_nonpar = self._normalize(h_nonpar)
            h_nonpar = self.act(h_nonpar) if self.act != None else h_nonpar

            h_nonpar_final = torch.cat([h_nonpar_final, h_nonpar], -1)

        verPrint(self.verbose, 5, f'Nonparametric agg {time.time() - start:.8f}s')

        return h_nonpar_final


########################
### MAIN MODEL CLASS ###
########################

class EmbedBoost(BaseModel):

    ### BASIC METHODS ###
    def __init__(
        self, 
        boost_predictor, boost_metric,
        in_feats, h_feats, num_layers, embed_type='mixed',
        round_window=7, temporal_agg='sum_final', training_type='round',
        gamma=1, alpha=1, beta=1, 
        loss_type='neigh', tloss_type='normal', loss_sample=False, loss_sample_ratio=0.1,
        dropout_rate=0.5, act_name='ReLU', norm_name='layer',
        num_epoch=300, num_round_epoch=150, early_stopping=25, 
        device='cuda:0', verbose=0, temp_model_path='../checkpoint/working_model_file', **kwargs):

        super().__init__()
        self.verbose=verbose
        self.device=device
        self.temp_model_path = temp_model_path
        self.model = None
        
        self.in_feats= in_feats
        self.h_feats = h_feats
        self.num_layers = num_layers
        self.round_window=round_window
 
        self.eval_metric = boost_metric
        self.num_epoch = num_epoch
        self.num_round_epoch = num_round_epoch
        self.early_stopping = early_stopping

        # GNN Modules
        self.training_type = training_type
        if embed_type == 'temporal':
            self.embedder = TemporalEmbedder(
                in_feats, h_feats, num_layers,
                round_window=round_window, temporal_agg=temporal_agg, 
                gamma=gamma, alpha=alpha, beta=beta, loss_type=loss_type, tloss_type=tloss_type, loss_sample=loss_sample, loss_sample_ratio=loss_sample_ratio,
                dropout_rate=dropout_rate, act_name=act_name, norm_name=norm_name,
                device=device, verbose=verbose 
            )
        elif embed_type == 'vanilla':
            self.embedder = VanillaEmbedder(
                in_feats, h_feats, num_layers, 
                loss_type=loss_type, tloss_type=tloss_type, loss_sample=loss_sample, loss_sample_ratio=loss_sample_ratio,
                dropout_rate=dropout_rate, act_name=act_name, norm_name=norm_name,
                device=device, verbose=verbose 
            )
        elif embed_type == 'mixed':
            self.embedder = TemporalMixedEmbedder(
                in_feats, h_feats, num_layers,
                round_window=round_window, temporal_agg=temporal_agg, 
                gamma=gamma, alpha=alpha, beta=beta, loss_type=loss_type, tloss_type=tloss_type, loss_sample=loss_sample, loss_sample_ratio=loss_sample_ratio,
                dropout_rate=dropout_rate, act_name=act_name, norm_name=norm_name,
                device=device, verbose=verbose 
            )

        # XGB predictor Module
        self.predictor = boost_predictor(eval_metric=self.eval_metric)
        self.predictor_fitted = False

    def __call__(self, graph, feats, **kwargs):
        # Generate node embedding
        agg_feats = self.embed_nodes(graph, feats)
        
        # Predict
        probs = self.predictor.predict(xgb.DMatrix(agg_feats.clone().detach().cpu()))    
        return torch.Tensor(numpy.array([1 - probs, probs])).swapaxes(0, 1).to(self.device) , None, None

    def embed_nodes(self, graph, feats):
        return self.embedder.embed_nodes(graph, feats)

    ### TRAINING ###
    def train(self, graph, weight, round_num):
        
        if (self.training_type == 'round') or ((self.training_type == 'init') and (round_num == 0)):
            embedder_finish = False
            while not embedder_finish:
                embedder_finish = self.train_embedder(graph, weight, round_num)
        
        train_score, val_score = self.train_predictor(graph, weight, round_num)
        return train_score, val_score
    
    def train_embedder(self, graph, weight, round_num, stuck_stopping=50, stuck_thres=3):
        verPrint(self.verbose, 2, f'TRAINING NODE EMBEDDER')

        # Load everything to GPU
        self.embedder = self.embedder.to(self.device)
        graph = graph.to(self.device)

        # Initialize evaluation vars
        best_loss = None

        # Initialize epoch flags
        epoch_counter, stagnant_counter = 0, 0
        stop_training = False

        # Initialize training module
        self.optimizer = Adam(self.embedder.parameters(), lr=0.01)
        self.embedder.dist_calculated = False
        
        # Get required tensors
        features = graph.ndata['feature']
        
        # TRAIN
        while not stop_training:
            # TRAIN STEP     
            self.embedder.train()

            _, embedding_loss = self.embedder(graph, features, **{'epoch': epoch_counter, 'ce_weight': weight})

            start = time.time()
            self.optimizer.zero_grad()
            embedding_loss.backward()
            self.optimizer.step()     
            verPrint(self.verbose, 5, f'Backward step {time.time() - start:.8f}s')

            # EVALUATE
            self.embedder.eval()
            current_loss = embedding_loss.item()

            # Check F1 improvement
            if (best_loss == None) or (current_loss < best_loss):
                # Model checkpointing
                while True:
                    try:
                        self.save_embedder(f'{self.temp_model_path}_hybrid_epoch')
                        break
                    except RuntimeError as e:
                            verPrint(self.verbose, 4, f"CHECKPOINTING ERROR {e}")

                best_loss = current_loss
                stagnant_counter = 0
            else:
                stagnant_counter = stagnant_counter + 1

            # Update epoch counter
            epoch_counter = epoch_counter + 1

            # Print and log
            verPrint(self.verbose, 2, f'Epoch {epoch_counter}, loss: {current_loss:.8f}-(best {best_loss:.8f})\n')
            verPrint(self.verbose, 5, f'\n')

            # Check epoch exit condition
            stop_training = (epoch_counter >= (self.num_epoch if round_num == 0 else self.num_round_epoch)) or (stagnant_counter >= self.early_stopping)

        # Unload from GPU
        # self.embedder = self.embedder.to('cpu')
        # graph = graph.to('cpu')

        # Load best model
        self.load_embedder(f'{self.temp_model_path}_hybrid_epoch')
        verPrint(self.verbose, 2, '>> Reached final epoch. Loading best model...')

        return True
    
    def train_predictor(self, graph, weight, round_num):
        verPrint(self.verbose, 2, f'TRAINING XGB PREDICTOR')

        feats = self.embed_nodes(graph, graph.ndata['feature'].clone().detach())
        labels = graph.ndata['ps_label']

        train_X = feats[graph.ndata['ps_train_mask']].clone().detach().cpu().numpy()
        train_y = labels[graph.ndata['ps_train_mask']].clone().detach().cpu().numpy()
        val_X = feats[graph.ndata['val_mask']].clone().detach().cpu().numpy()
        val_y = labels[graph.ndata['val_mask']].clone().detach().cpu().numpy()
        
        # Train
        params = {"objective": "binary:logistic", "scale_pos_weight": weight, "tree_method": "hist", "max_depth": 6, "device": 'cpu'}        
        self.predictor = xgb.train(
            params, xgb.DMatrix(train_X, train_y), 
            num_boost_round=500, early_stopping_rounds=100,
            evals=[(xgb.DMatrix(train_X, train_y), 'Train'), (xgb.DMatrix(val_X, val_y), 'Eval')], verbose_eval=False, 
            xgb_model=self.predictor if self.predictor_fitted else None)
        self.predictor_fitted = True
        
        # Predict
        train_score = self.eval_metric(train_y.flatten(), self.predictor.predict(xgb.DMatrix(train_X)).flatten())
        val_score = self.eval_metric(val_y.flatten(), self.predictor.predict(xgb.DMatrix(val_X)).flatten())

        return train_score, val_score

    ### EVAL ###
    def eval(self):
        return
    
    ### SET/RELEASE GRAPH ###
    def set_graph(self, graph, round_num, device, **kwargs):
        self.graph = graph.to(device)
        self.preprocess_graph(round_num)

        # Initialize ps_label/ptrainmask
        self.graph.ndata['ps_label'] = self.graph.ndata['label'].clone()
        self.graph.ndata['ps_train_mask'] = self.graph.ndata['train_mask'].clone()
        

    def release_graph (self):
        if hasattr(self, 'graph'):
            del self.graph
            gc.collect()

    def augment_graph(self, augment_strat=None, round_num=0, **kwargs):
        if (augment_strat != None):
            augment_strat(self, round_num=round_num, **kwargs)
            self.preprocess_graph(round_num)

    def preprocess_graph(self, round_num, **kwargs):
        self.graph.ndata['age'] = round_num - self.graph.ndata['creation_round'] # For round aggregator
    
    ### CHECKPOINTING ###
    def save_model(self, path):
        if self.predictor_fitted:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.predictor.save_model(f'{path}.json')

    def load_model(self, path):
        if self.predictor_fitted:
            self.predictor.load_model(f'{path}.json')

    def save_embedder(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'model_state_dict': self.embedder.state_dict()}, f'{path}.pt')

    def load_embedder(self, path):
        checkpoint = torch.load(f'{path}.pt')
        self.embedder.load_state_dict(checkpoint['model_state_dict'])

    ### OTHER HOOKS ###
    def postBackprop(self, **kwargs):
        return
    
    def get_latest_trainlog(self, **kwargs):
        return self.embedder.get_latest_trainlog()