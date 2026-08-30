from torch import nn

import dgl
import dgl.nn.pytorch.conv as dglnn

from utils.utils_func import verPrint
from models.base_model import BaseModel

### Common Submodules ###
class MLP(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, dropout_rate=0, activation='ReLU', verbose=0, device='cuda:0', **kwargs):

        super().__init__()
        self.verbose=verbose

        # Linears
        self.layers = nn.ModuleList()       
        if num_layers == 0:
            return
        if num_layers == 1:
            self.layers.append(nn.Linear(in_feats, num_classes))
        else:
            self.layers.append(nn.Linear(in_feats, h_feats))
            for i in range(1, num_layers-1):
                self.layers.append(nn.Linear(h_feats, h_feats))
            self.layers.append(nn.Linear(h_feats, num_classes))

        # Other modules
        self.act = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, h, is_graph=True):
        if is_graph:
            h = h.ndata['feature']
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h)
            if i != len(self.layers)-1:
                h = self.act(h)
        return h

### GCN ###
class GCN(BaseModel):
    def __init__(self, in_feats, num_classes, h_feats, num_layers, mlp_feats, mlp_layers, dropout_rate=0, act_name='ReLU', verbose=0, **kwargs):

        super().__init__()
        self.verbose=verbose
        self.graph = None
        
        # Other modules
        self.act = getattr(nn, act_name)()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Layers
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GraphConv(in_feats, h_feats, activation=self.act))
        for i in range(num_layers-1):
            self.layers.append(dglnn.GraphConv(h_feats, h_feats, activation=self.act))
        self.mlp = MLP(h_feats, h_feats=mlp_feats, num_classes=num_classes, num_layers=mlp_layers, dropout_rate=dropout_rate)  

    def forward(self, blocks, x, **kwargs):
        h = self.embed_nodes(blocks, x)
        h = self.mlp(h, False)

        return h, None, None # No loss returned
    
    def embed_nodes(self, blocks, x, **kwargs):
        h = x
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(blocks, h)
        
        return h
    
### GCNII ###
class GCNII(BaseModel):
    def __init__(self, in_feats, num_classes, h_feats, num_layers, mlp_feats, mlp_layers, 
                alpha=0.1, lambda_=1,
                 dropout_rate=0, act_name='ReLU', verbose=0, **kwargs):

        super().__init__()
        self.verbose=verbose
        
        # Other modules
        self.act = getattr(nn, act_name)()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Layers
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GCN2Conv(in_feats, 1, activation=self.act, alpha=alpha, lambda_=lambda_))
        for i in range(num_layers-1):
            self.layers.append(dglnn.GCN2Conv(in_feats, i+2, activation=self.act, alpha=alpha, lambda_=lambda_))

        self.mlp = MLP(in_feats, h_feats=mlp_feats, num_classes=num_classes, num_layers=mlp_layers, dropout_rate=dropout_rate)  

    def forward(self, blocks, x, **kwargs):
        h = self.embed_nodes(blocks, x)
        h = self.mlp(h, False)

        return h, None, None # No loss returned
    
    def embed_nodes(self, blocks, x, **kwargs):
        h = x
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(blocks, h, x)
        
        return h
    
    def preprocess_graph(self, round_num):
        self.graph = dgl.add_self_loop(self.graph)

### GRAPHSAGE ###
class GraphSAGE(BaseModel):
    def __init__(self, in_feats, num_classes, h_feats, num_layers, mlp_feats, mlp_layers, agg='pool', dropout_rate=0, act_name='ReLU', verbose=0, **kwargs):

        super().__init__()
        self.verbose=verbose
        
        # Other modules
        self.act = getattr(nn, act_name)()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Layers
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, h_feats, agg, activation=self.act))
        for i in range(num_layers-1):
            self.layers.append(dglnn.SAGEConv(h_feats, h_feats, agg, activation=self.act))
        self.mlp = MLP(h_feats, h_feats=mlp_feats, num_classes=num_classes, num_layers=mlp_layers, dropout_rate=dropout_rate)  


    def forward(self, blocks, x, **kwargs):
        h = self.embed_nodes(blocks, x)
        h = self.mlp(h, False)

        return h, None, None # No loss returned

    def embed_nodes(self, blocks, x, **kwargs):
        h = x
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(blocks, h)
        
        return h
    
## GIN
class GIN(BaseModel):
    def __init__(self, in_feats, num_classes, h_feats, num_layers, agg='mean', dropout_rate=0, act_name='ReLU', verbose=0, **kwargs):

        super().__init__()
        self.verbose=verbose
        
        # Other modules
        self.act = getattr(nn, act_name)()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GINConv(nn.Linear(in_feats, h_feats), activation=self.act, aggregator_type=agg))
        for i in range(1, num_layers-1):
            self.layers.append(dglnn.GINConv(nn.Linear(h_feats, h_feats), activation=self.act, aggregator_type=agg))
        self.layers.append(dglnn.GINConv(nn.Linear(h_feats, num_classes),  activation=None, aggregator_type=agg))

    def forward(self, blocks, x, **kwargs):
        h = self.embed_nodes(blocks, x)
        return h, None, None # No loss returned

    def embed_nodes(self, blocks, x, **kwargs):
        h = x
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(blocks, h)
        
        return h

## GAT
class GAT(BaseModel):
    def __init__(self, in_feats, num_classes, h_feats, num_layers, mlp_feats, mlp_layers, 
                 att_heads=2, dropout_rate=0, act_name='ReLU', verbose=0, **kwargs):

        super().__init__()
        self.verbose=verbose
        
        self.act = getattr(nn, act_name)()
        
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GATConv(in_feats, h_feats, att_heads, feat_drop=dropout_rate, attn_drop=dropout_rate, activation=self.act))
        for i in range(1, num_layers-1):
            self.layers.append(dglnn.GATConv(h_feats * att_heads, h_feats, att_heads, feat_drop=dropout_rate, attn_drop=dropout_rate, activation=self.act))
        self.mlp = MLP(h_feats, h_feats=mlp_feats, num_classes=num_classes, num_layers=mlp_layers, dropout_rate=dropout_rate)  

    def forward(self, blocks, x, **kwargs):
        h = self.embed_nodes(blocks, x)
        h = self.mlp(h, False)

        return h, None, None
    
    def embed_nodes(self, blocks, x, **kwargs):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(blocks, h)
            if i != (len(self.layers) - 1):  # Not last layer
                h = h.flatten(1)     
            else:
                h = h.mean(1)
        
        return h