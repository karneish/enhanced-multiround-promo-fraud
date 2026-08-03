import torch
import gc

from torch import nn
from utils.utils_func import verPrint

class BaseModel(nn.Module):
    ### BASIC METHODS ###
    def __init__(self, verbose=0, device='cuda:0', **kwargs):
        super().__init__()
        self.verbose=verbose
        self.device=device
        self.graph = None

    def forward(self, graph, x):
        x = self.embed_nodes(graph, x)
        return x, None
    
    def embed_nodes(self, graph, x):
        return x, None
    
    ### SETTER ###
    def _set_device(self, device):
        self.device = device

    ### SET/RELEASE GRAPH ###
    def set_graph(self, graph, round_num, device, **kwargs):
        self.graph = graph

        if device != None: 
            self._set_device(device)
            self.graph = self.graph.to(self.device)

        # Initialize ps_label/ptrainmask
        self.graph.ndata['ps_label'] = self.graph.ndata['label'].clone()
        self.graph.ndata['ps_train_mask'] = self.graph.ndata['train_mask'].clone()
        
        self.preprocess_graph(round_num)

    def release_graph(self):
        if hasattr(self, 'graph'):
            del self.graph
            gc.collect()

    ### GRAPH PREPROCESSING ###
    def augment_graph(self, augment_strat=None, round_num=0, **kwargs):
        verPrint(self.verbose, 3, f'Resampling graph with strategy {augment_strat.__qualname__}...')

        if (augment_strat != None):
            augment_strat(self, round_num=round_num, **kwargs)
        
        verPrint(self.verbose, 3, f'Done resampling graph!')
    
    def preprocess_graph(self, round_num):
        return

    ### CHECKPOINTING ###
    def save_model(self, path):
        self.to('cpu')
        torch.save({'model_state_dict': self.state_dict()}, f'{path}.pt')
        self.to(self.device)
        
        gc.collect()

    def load_model(self, path):
        checkpoint = torch.load(f'{path}.pt')
        self.load_state_dict(checkpoint['model_state_dict'])

    ### OTHER HOOKS ###
    def postBackprop(self, **kwargs):
        return 
    
    def get_latest_trainlog(self, **kwargs):
        return {}