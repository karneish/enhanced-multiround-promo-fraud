import torch
import random
import numpy as np

from collections import Counter
from adversary.modify.base_mod import BaseAdversarialMod
from utils.utils_func import verPrint

#####################
### SIMPLE REPLAY ###
#####################
class ReplayMod(BaseAdversarialMod):
    def __init__(self, verbose=0, **kwargs):
        super().__init__()
        self.verbose=verbose

    def modify_seeds(self, graph, node_data, edge_data, seed_ids, modified_ids, **kwargs):
        return node_data, edge_data, seed_ids, modified_ids # Do nothing

##########################
### PERTURBATION BASED ###
##########################
class BasePerturbMod(BaseAdversarialMod):
    def __init__(self, feat_coef=1.0, conn_coef=0.1, verbose=0, **kwargs):
        super().__init__()
        self.verbose=verbose              
        self.feat_coef = feat_coef
        self.conn_coef = conn_coef

    
    @staticmethod
    def get_rewires(todos, edge_data, relname, baseid, verbose=0):
        reduceds, addeds = [], [] # Container
        for id, min_count, plus_count in todos:
            current_index = (edge_data[relname]['in']['dst'] == id).nonzero().flatten().tolist() # All index of Node's edge
            verPrint(verbose, 4, f'Rewiring node id {id} with {len(current_index)} nodes| removing {min_count} edges and adding {plus_count} edges')

            # Get index after reduction and index for addition base
            reduced_index = sorted(np.random.choice(current_index, len(current_index) - min_count, replace=False)) # New list of index after reduction
            added_index = sorted(np.random.choice(current_index, plus_count, replace=True))

            # REDUCTION - Copy to new container
            reduced_data = {}
            reduced_data['in'] = { feat:edge_data[relname]['in'][feat][reduced_index].clone() for feat in edge_data[relname]['in'].keys() }
            reduced_data['out'] = { feat:reduced_data['in'][feat].clone() for feat in reduced_data['in'].keys() }
            reduced_data['out']['src'] = reduced_data['in']['dst'].clone()
            reduced_data['out']['dst'] = reduced_data['in']['src'].clone()

            # ADDITION - Rewire and add to new container
            added_data = {}
            added_data['in'] = { feat:edge_data[relname]['in'][feat][added_index].clone() for feat in edge_data[relname]['in'].keys() }
            added_data['in']['src'] = torch.randint(baseid, added_data['in']['src'].shape) # Rewiring
            added_data['out'] = { feat:added_data['in'][feat].clone() for feat in reduced_data['in'].keys() }
            added_data['out']['src'] = added_data['in']['dst'].clone()
            added_data['out']['dst'] = added_data['in']['src'].clone()

            # Append to container
            reduceds.append(reduced_data)
            addeds.append(added_data)

        return reduceds, addeds

    @staticmethod
    def split_connection_budget(degrees, perturb_amount, verbose=0):
        max_minus = torch.maximum(((degrees + perturb_amount) / 2).floor() - 1, torch.zeros(degrees.shape)) # Maximum deletion possible, - 1 to prevent 0 degree node
        perturb_minus = torch.minimum(((torch.rand(degrees.shape) * perturb_amount).round()), max_minus) # Deletion count capped by the max
        perturb_cancels = torch.minimum((degrees - perturb_minus), torch.zeros(degrees.shape)).abs() # Amount of plus and minus that cancels out if any

        perturb_minus = (perturb_minus - perturb_cancels).long()
        perturb_plus = (perturb_amount - perturb_minus - perturb_cancels).long()

        return perturb_minus, perturb_plus

class RelativePerturbMod(BasePerturbMod):    
    def modify_seeds(self, graph, node_data, edge_data, seed_ids, modified_ids, **kwargs):
        
        ## FEATURE PERTURBATION ##
        verPrint(self.verbose, 4, f'Perturbing each feature for {self.feat_coef} times its stdev...')
        feats = node_data['feature'].clone()
        perturb_final = torch.std(graph.ndata['feature'], dim=0) * torch.sign((torch.rand(feats.shape) - 0.5)) * self.feat_coef
        node_data['feature'] = feats + perturb_final

        ## STRUCTURAL PERTURBATION ## TODO: Directed version
        for val in [r for r in graph.etypes]:
            verPrint(self.verbose, 4, f'Perturbing structure for relation {val} at most for {self.conn_coef} times its degree')

            # Get perturbation amount
            counter = dict(Counter(edge_data[val]['in']['dst'].tolist()))
            degrees = torch.tensor([counter[id] for id in modified_ids.tolist()], dtype=torch.long) # Get degrees of each nodes
            perturb_amount = (degrees * self.conn_coef).round()
            perturb_minus, perturb_plus = BasePerturbMod.split_connection_budget(degrees, perturb_amount)
            
            # Get rewiring based on perturbation amount
            todos = list(zip(modified_ids.tolist(), perturb_minus.tolist(), perturb_plus.tolist()))
            reduceds, addeds = BasePerturbMod.get_rewires(todos, edge_data, val, min(modified_ids), verbose=self.verbose)

            # Replace data in seed container
            edge_data[val]['in'] = { feat: torch.cat([d['in'][feat] for d in reduceds + addeds]) for feat in reduceds[0]['in'].keys() }
            edge_data[val]['out'] = { feat: torch.cat([d['out'][feat] for d in reduceds + addeds]) for feat in reduceds[0]['out'].keys() }
        
        return node_data, edge_data, seed_ids, modified_ids

class AbsolutePerturbMod(BasePerturbMod):    
    def modify_seeds(self, graph, node_data, edge_data, seed_ids, modified_ids, **kwargs):
        
        ## FEATURE PERTURBATION ##
        verPrint(self.verbose, 4, f'Perturbing feature with absolute an budget of {self.feat_coef}...')
        feats = node_data['feature'].clone()
        perturb_weight = torch.rand(feats.shape)
        perturb_weight = perturb_weight / perturb_weight.sum(dim=1).unsqueeze(1) # This is the distribution for the noise over the entire feature dimension for each node
        perturb_amount = self.feat_coef # This is the amount of noise for each node
        perturb_final = (perturb_weight * perturb_amount.unsqueeze(1)) * (torch.rand(feats.shape) - 0.5).sign() # Randomize noise sign
        node_data['feature'] = feats + perturb_final

        ## STRUCTURAL PERTURBATION ## TODO: Directed version
        verPrint(self.verbose, 4, f'Perturbing structure with an absolute budget of {self.conn_coef}...')
        
        # Distribute budget over relations
        rels = [r for r in graph.etypes] # Exception for H2F
        rounding_error = 1
        while rounding_error != 0: # Split budget over all edge relations
            perturb_weight = torch.rand((1, len(rels)))
            perturb_weight = perturb_weight / perturb_weight.sum(dim=1).unsqueeze(1)
            rel_budgets = (perturb_weight * self.conn_coef).round().long()
            rounding_error = rel_budgets.sum() - self.conn_coef
        
        # Iterate over relation type
        for idx, val in enumerate(rels):
            verPrint(self.verbose, 4, f'Perturbing structure for relation {val} with budget {rel_budgets[idx]}')

            # Get randomized perturbation amount based on the max budget
            counter = dict(Counter(edge_data[val]['in']['dst'].tolist()))
            degrees = torch.tensor([counter[id] for id in modified_ids.tolist()], dtype=torch.long)
            perturb_amount = torch.full(rel_budgets[idx], degrees.shape)
            perturb_minus, perturb_plus = BasePerturbMod.split_connection_budget(degrees, rel_budgets[idx])

            # Final to-do list per node
            todos = list(zip(modified_ids.tolist(), perturb_minus.tolist(), perturb_plus.tolist()))
            reduceds, addeds = BasePerturbMod.get_rewires(todos, edge_data, val, min(modified_ids), verbose=self.verbose)

            # Replace data in seed container
            edge_data[val]['in'] = { feat: torch.cat([d['in'][feat] for d in reduceds + addeds]) for feat in reduceds[0]['in'].keys() }
            edge_data[val]['out'] = { feat: torch.cat([d['out'][feat] for d in reduceds + addeds]) for feat in reduceds[0]['out'].keys() }
    
        return node_data, edge_data, seed_ids, modified_ids

####################
### MIXING BASED ###
####################
class MixingMod(BaseAdversarialMod):
    def __init__(self, feat_coef=1.0, conn_coef=0.1, verbose=0, **kwargs):

        super().__init__()
        self.verbose=verbose              
        self.feat_coef = feat_coef
        self.conn_coef = conn_coef
    
    def modify_seeds(self, graph, node_data, edge_data, seed_ids, modified_ids, **kwargs):

        ## FEATURE MIXING ##
        verPrint(self.verbose, 4, f'Mixing {self.feat_coef} percent of all of the seeds features')
        
        # Get number of feature mutated and randomly get their index
        feats = node_data['feature'].clone()
        num_mutated = round(self.feat_coef  * feats.shape[1])
        idx_mutated = torch.randperm(feats.shape[1])[:num_mutated]

        # Shuffle the feature in each index among the seed
        for idx in idx_mutated:
            data_mutated = feats[:, idx].clone()
            feats[:, idx] = data_mutated[torch.randperm(data_mutated.shape[0])]
        node_data['feature'] = feats

        ## STRUCTURAL MIXING ## TODO: Directed version
        for val in [r for r in graph.etypes]:
            verPrint(self.verbose, 4, f'Mixing {self.conn_coef} percent of all of the seeds edges')

            init_idx = {id.item(): (edge_data['_E']['in']['dst'] == id).nonzero().flatten() for id in modified_ids} # Dict containing all original index of each node in the edge list
            permuted_idx_mask = {key: torch.randperm(val.shape[0]) for key, val in init_idx.items()} # Dict containing permutation mask for the index of each node

            node_idx, mixed, constant = zip(*[(key, init_idx[key][val[:round(val.shape[0] * self.conn_coef)]], init_idx[key][val[round(val.shape[0] * self.conn_coef):]]) for key, val in permuted_idx_mask.items()]) # Mixed is the list of edge index that will be swapped, constant is list of edge index that stays 
            final_idx = { i[0]: torch.cat(i[1:]) for i in list(zip(node_idx, random.sample(mixed, len(mixed)), constant)) } # Just concat mixed and stay then make into dictionary
            
            dst, src_idx = zip(*[(torch.full(value.shape, key), value) for key, value in final_idx.items()]) # Now we have edge list but remember that src is still idx
            dst = torch.cat(list(dst))
            src = edge_data['_E']['in']['src'][torch.cat(list(src_idx))].clone() # Use the idx on the original edge data to get actual src idx

            # Replace data in seed container
            edge_data[val] = {'in': { 'src': src, 'dst': dst }, 'out': { 'src': dst, 'dst': src }}
        
        return node_data, edge_data, seed_ids, modified_ids