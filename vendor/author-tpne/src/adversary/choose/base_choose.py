import torch
import numpy as np

from numpy import random
from collections import Counter
from utils.utils_func import verPrint

#############################################
### BASE ADVERSARIAL SEED SELECTION CLASS ###
#############################################
class BaseAdversarialChoice():
    def __init__(self, verbose=0):
        pass

    def generate_seeds(self, graph, n_instances=1, label=1, return_id=False, **kwargs):
        return None   

    @staticmethod
    def random_node_seeds(
            graph, n_instances=1, label=None, 
            prio_pool=torch.tensor([], dtype=torch.long), replace=False, 
            verbose=0
        ):

        if len(prio_pool) > 0:
            # Split how many comes from prio pool how many comes from rest        
            if replace == False:
                prio_instances = min([prio_pool.shape[0], n_instances])
                rest_instances = max([0, n_instances - prio_pool.shape[0]])
            else:
                prio_instances = n_instances
                rest_instances = 0
        else:
            prio_instances = 0
            rest_instances = n_instances
        
        # Create pool of IDs to randomly choose from if prio not enough
        pool_ids = list(range(graph.num_nodes())) if label == None else (graph.ndata['label'] == label).nonzero().flatten().tolist()
        pool_ids = torch.tensor(list(set(pool_ids) - set(prio_pool.tolist())), dtype=torch.long)

        # Choose seeds from pool and map new + old Node ID
        seed_ids = torch.cat([prio_pool[random.choice(list(range(len(prio_pool))), size=prio_instances, replace=replace)], pool_ids[random.choice(list(range(len(pool_ids))), size=rest_instances, replace=replace)]])

        return seed_ids   
    
    @staticmethod
    def duplicate_nodes(graph, node_ids, return_id=False, verbose=0):
        
        node_ids_ctr = dict(Counter(sorted(node_ids.tolist())))
        new_ids = (torch.tensor(list(range(len(node_ids)))) + graph.num_nodes()).int() # New Node IDs sequentially generated from original biggest ID

        node_ids_stack = [torch.Tensor([k for k, v in node_ids_ctr.items() if v > i]).long() for i in range(max(node_ids_ctr.values()))]
        node_ids_stack_lens = [len(a) for a in node_ids_stack]

        new_node_features_list, new_edge_features_list, node_ids_list, new_ids_list = [], [], [], []
        for i, lst in enumerate(node_ids_stack):
            start = sum(node_ids_stack_lens[:i])
            end = start + len(lst)

            node_ids_part = lst
            new_ids_part = new_ids[start:end]

            id_dict = dict(zip(node_ids_part.tolist(), new_ids_part.tolist())) # Maps old ID to node ID

            # Copy node features
            new_node_features = { key: graph.ndata[key][node_ids_part] for key, _v in graph.node_attr_schemes().items() if key != '_ID' }
            new_edge_features = {}

            # Copy edge features
            for etype in graph.etypes:
                in_src, in_dst, in_ids = graph.in_edges(node_ids_part, form='all', etype=etype)
                out_src, out_dst, out_ids = graph.out_edges(node_ids_part, form='all', etype=etype)

                in_dst = torch.from_numpy(np.fromiter((id_dict[i] for i in in_dst.tolist()), int))
                out_src = torch.from_numpy(np.fromiter((id_dict[i] for i in out_src.tolist()), int))

                new_edge_features[etype] = {}
                new_edge_features[etype]['in'] = { key: graph.edges[etype].data[key][in_ids] for key, _v in graph.edge_attr_schemes(etype).items() if key != '_ID' }
                new_edge_features[etype]['in']['src'] = in_src
                new_edge_features[etype]['in']['dst'] = in_dst

                new_edge_features[etype]['out'] = { key: graph.edges[etype].data[key][out_ids] for key, _v in graph.edge_attr_schemes(etype).items() if key != '_ID' }
                new_edge_features[etype]['out']['src'] = out_src
                new_edge_features[etype]['out']['dst'] = out_dst
        
            new_node_features_list.append(new_node_features)
            new_edge_features_list.append(new_edge_features)
            node_ids_list.append(node_ids_part)
            new_ids_list.append(new_ids_part)
        
        # Merge node features
        node_features_fin = {}
        for key, _v in graph.node_attr_schemes().items(): 
            if key != '_ID':
                key_nf_list = [nf[key] for nf in new_node_features_list]
                ndata = torch.cat(key_nf_list)
                node_features_fin[key] = ndata

        # Merge edge features
        edge_features_fin = { key: {} for key in graph.etypes }
        for etype in graph.etypes:
            for direction in ['in', 'out']:
                edge_features_fin[etype][direction] = { key:torch.cat([ef[etype][direction][key] for ef in new_edge_features_list]) for key in (list(graph.edge_attr_schemes(etype).keys()) + ['src', 'dst']) if key != '_ID' }

        # Merge id list
        node_ids_fin = None if not return_id else torch.cat(node_ids_list)
        new_ids_fin = None if not return_id else torch.cat(new_ids_list)
        
        # Return
        return node_features_fin, edge_features_fin, node_ids_fin, new_ids_fin
    