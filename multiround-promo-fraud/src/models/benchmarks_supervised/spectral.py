from utils.utils_func import verPrint

import torch
import dgl
import sympy
import scipy
import dgl.nn.pytorch.conv as dglnn
import dgl.function as fn

from torch import nn
from models.benchmarks_supervised.simple import MLP
from models.base_model import BaseModel

### BWGNN ###
## Polyconv Module
class PolyConv(nn.Module):
    def __init__(self, theta):
        """_summary_

        Args:
            theta (_type_): _description_
        """
        super().__init__()
        
        self._theta = theta
        self._k = len(self._theta)

    def forward(self, graph, feat):
        def unnLaplacian(feat, D_invsqrt, graph):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - graph.ndata.pop('h') * D_invsqrt

        with graph.local_scope():
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0]*feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, graph)
                h += self._theta[k]*feat
        return h
    
## Main Model
class BWGNN(BaseModel):
    def __init__(self, in_feats, num_classes, h_feats, num_layers, mlp_layers, dropout_rate=0, act_name='ReLU', verbose=0, device='cuda:0', **kwargs):
        super().__init__()
        self.verbose=verbose
        self.device=device   

        # Misc modules
        self.act = getattr(nn, act_name)()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # BW Filters
        self.thetas = self.calculate_theta(d=num_layers)
        self.conv = []
        for i in range(len(self.thetas)):
            self.conv.append(PolyConv(self.thetas[i]))
        
        # Linear and MLP
        self.linear = nn.Linear(in_feats, h_feats).to(device)
        self.linear2 = nn.Linear(h_feats, h_feats).to(device)
        self.mlp = MLP(h_feats * len(self.thetas), h_feats, num_classes, mlp_layers, dropout_rate, device=device)


    def forward(self, blocks, x, **kwargs):
        h_final = self.embed_nodes(blocks, x)
        h = self.mlp(h_final, False)
        
        return h, None, None # No loss returned
    
    def embed_nodes(self, graph, x):
        in_feat = x
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_final = torch.zeros([len(in_feat), 0], device=h.device)

        for conv in self.conv:
            h0 = conv(graph, h)
            h_final = torch.cat([h_final, h0], -1)
        h_final = self.dropout(h_final)

        return h_final
    
    def calculate_theta(self, d):
        thetas = []
        x = sympy.symbols('x')
        for i in range(d+1):
            f = sympy.poly((x/2) ** i * (1 - x/2) ** (d-i) / (scipy.special.beta(i+1, d+1-i)))
            coeff = f.all_coeffs()
            inv_coeff = []
            for i in range(d+1):
                inv_coeff.append(float(coeff[d-i]))
            thetas.append(inv_coeff)
        return thetas
    
### GHRN ###
class GHRN(BWGNN):
    def __init__(self, in_feats, num_classes, h_feats, num_layers, mlp_layers, drop_rate=0.1, dropout_rate=0, act_name='ReLU', verbose=0, device='cuda:0', **kwargs):
        super().__init__(in_feats, num_classes, h_feats, num_layers, mlp_layers, dropout_rate=dropout_rate, act_name=act_name, verbose=verbose, device=device, **kwargs)
        self.drop_rate = drop_rate # Sparsification droprate

    def random_walk_update(self):
        graph = self.graph

        edge_weight = torch.ones(graph.num_edges()).to(self.device)
        norm = dgl.nn.pytorch.conv.EdgeWeightNorm(norm='both')
        
        graph.edata['w'] = norm(graph, edge_weight)
        aggregate_fn = fn.u_mul_e('h', 'w', 'm')
        reduce_fn = fn.sum(msg='m', out='ay')

        graph.ndata['h'] = graph.ndata['feature']
        graph.update_all(aggregate_fn, reduce_fn)
        graph.ndata['ly'] = graph.ndata['feature'] - graph.ndata['ay']
        
        graph.apply_edges(self.inner_product_black)
        black = graph.edata['inner_black']
        
        threshold = int(self.drop_rate * graph.num_edges())
        edge_to_move = set(black.sort()[1][:threshold].tolist())
        graph_new = dgl.remove_edges(graph, list(edge_to_move))
        
        return graph_new

    def inner_product_black(self, edges):
        inner_black = (edges.src['ly'] * edges.dst['ly']).sum(axis=1)
        return {'inner_black': inner_black}
    
    def preprocess_graph(self, round_num):
        if self.drop_rate != 0.:
            new_graph = self.random_walk_update()
            new_graph = dgl.add_self_loop(dgl.remove_self_loop(new_graph))

            self.graph = new_graph