import torch
from adversary.choose.simple_choose import RandomChoose
from utils.utils_func import add_generated_nodes

########################
# ADVERSARIAL SAMPLING #
########################

def NoneSampling(model, round_num=0, **kwargs):
    return

# Randomly replay a number of positive and negative samples with given ratio (relative to current graph size)
def RandomReplaySampling(model, round_num=0, augment_pos_ratio=0.5, augment_neg_ratio=0.5, augment_round_split=5, verbose=1, **kwargs):
    # Initialize
    adver_choose = RandomChoose(replace=True)
    pos_instance = int(augment_pos_ratio / augment_round_split * model.graph.num_nodes())
    neg_instance = int(augment_neg_ratio / augment_round_split * model.graph.num_nodes())

    for i in range(augment_round_split):
        pos_prio_pool = ((model.graph.ndata['train_mask'] == True) & (model.graph.ndata['label'] == 1)).nonzero().flatten()
        neg_prio_pool = ((model.graph.ndata['train_mask'] == True) & (model.graph.ndata['label'] == 0)).nonzero().flatten()

        # Positives
        pos_node_feats, pos_edge_feats, pos_seed_ids, pos_new_ids = adver_choose.generate_seeds(model.graph, n_instances=pos_instance, label=1, return_id=True, prio_pool=pos_prio_pool) # Generate
        add_generated_nodes(model.graph, (pos_node_feats, pos_edge_feats), round_num - i, set_name='train')
    
        # Negatives
        neg_node_feats, neg_edge_feats, neg_seed_ids, neg_new_ids = adver_choose.generate_seeds(model.graph, n_instances=neg_instance, label=0, return_id=True, prio_pool=neg_prio_pool) # Generate
        add_generated_nodes(model.graph, (neg_node_feats, neg_edge_feats), round_num - i, set_name='train')

# Always randomize age for all round 0 data 
def ReAge(model, round_num=0, augment_round_split=3, augment_thres=0.5, **kwargs):
    device = model.graph.device
    
    shuffled = ((model.graph.ndata['creation_round'] <= 0) & (model.graph.ndata['train_mask'])).nonzero().flatten()
    model.graph.ndata['creation_round'][shuffled] = torch.randint_like(shuffled, low=(round_num-augment_round_split+1), high=round_num+1).to(device)

# Always randomize age for all round 0 data 
def ReAgeFull(model, round_num=0, augment_round_split=3, augment_thres=0.5, **kwargs):
    device = model.graph.device
    
    shuffled = ((model.graph.ndata['creation_round'] <= 0)).nonzero().flatten()
    model.graph.ndata['creation_round'][shuffled] = torch.randint_like(shuffled, low=(round_num-augment_round_split+1), high=round_num+1).to(device)

# Same but duplicate for round 0 only
def ReplayAge(model, round_num=0, augment_round_split=3, augment_thres=0.5, **kwargs):
    device = model.graph.device
    
    if round_num == 0:
        # On the first round, do randomized aging
        prio_pool = model.graph.ndata['train_mask'].nonzero().flatten()
        num_instance = int(model.graph.num_nodes()/augment_round_split)
        
        adver_choose = RandomChoose(replace=True)

        for i in range(augment_round_split):
            node_feats, edge_feats, seed_ids, new_ids = adver_choose.generate_seeds(model.graph, n_instances=num_instance, prio_pool=prio_pool, return_id=True) # Generate
            add_generated_nodes(model.graph, (node_feats, edge_feats), round_num - i, set_name='train')
    else:
        shuffled = ((model.graph.ndata['creation_round'] <= 0) & (model.graph.ndata['train_mask'])).nonzero().flatten()
        model.graph.ndata['creation_round'][shuffled] = torch.randint_like(shuffled, low=(round_num-augment_round_split+1), high=round_num + 1).to(device)
