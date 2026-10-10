import torch
import random
import numpy as np

from collections import Counter
from adversary.choose.base_choose import BaseAdversarialChoice
from utils.utils_func import verPrint

#########################
### SIMPLE STRATEGIES ###
#########################
class RandomChoose(BaseAdversarialChoice):
    def __init__(self, replace=False, verbose=0, **kwargs):
        super().__init__()
        self.verbose=verbose       
        self.replace = replace

    def generate_seeds(self, graph, n_instances=10, label=1, return_id=False, prio_pool=torch.tensor([], dtype=torch.long), **kwargs):
        seed_ids = BaseAdversarialChoice.random_node_seeds(graph, n_instances=n_instances, label=label, prio_pool=prio_pool, replace=self.replace)
        return BaseAdversarialChoice.duplicate_nodes(graph, seed_ids, return_id=return_id)

class GreedyChoose(BaseAdversarialChoice):
    def __init__(self, replace=False, verbose=0, **kwargs):
        super().__init__()
        self.verbose=verbose       
        self.replace = replace

    def generate_seeds(self, graph, n_instances=10, label=1, return_id=False, **kwargs):
        # Prioritize undetected nodes
        prio_pool = ((graph.ndata['predicted'] == False) & (graph.ndata['label'] == label)).nonzero().flatten()
        seed_ids = BaseAdversarialChoice.random_node_seeds(graph, n_instances=n_instances, label=label, prio_pool=prio_pool, replace=self.replace)
    
        return BaseAdversarialChoice.duplicate_nodes(graph, seed_ids, return_id=return_id)

class OGGreedyChoose(BaseAdversarialChoice):
    def __init__(self, replace=False, verbose=0, **kwargs):
        super().__init__()
        self.verbose=verbose       
        self.replace = replace

    def generate_seeds(self, graph, n_instances=10, label=1, return_id=False, **kwargs):
        # Prioritize undetected nodes
        og_prio_pool = ((graph.ndata['predicted'] == False) & (graph.ndata['creation_round'] < 1) & (graph.ndata['label'] == label)).nonzero().flatten()
        seed_ids = BaseAdversarialChoice.random_node_seeds(graph, n_instances=n_instances, label=label, prio_pool=og_prio_pool, replace=self.replace)
    
        return BaseAdversarialChoice.duplicate_nodes(graph, seed_ids, return_id=return_id)