import sys, os
sys.path.append('../src')

import torch
import dgl

from adversary.choose.simple_choose import RandomChoose, GreedyChoose
from adversary.modify.simple_mod import ReplayMod
from adversary.modify.intelligent_mod import IntelligentMod
from utils.utils_func import add_generated_nodes

# ---- build a small homogeneous graph ----
n = 400
g = dgl.rand_graph(n, 1200)
g.ndata['label'] = torch.randint(0, 2, (n,))
g.ndata['feature'] = torch.rand(n, 10)
g.ndata['predicted'] = torch.zeros(n, dtype=torch.bool)
g.ndata['creation_round'] = torch.full((n,), -1, dtype=torch.long)

print('GRAPH:', g.num_nodes(), g.num_edges(), g.etypes, 'hom', g.is_homogeneous)

adver_config = dict(
    adver_gen_type='GAN',
    adver_gen_epochs=5,
    adver_gen_noise_dim=8,
    adver_gen_hidden=32,
    adver_gen_feat_coef=1.0,
    adver_gen_conn_coef=0.5,
    adver_gen_ring_ratio=0.5,
    adver_gen_round_window=5,
    verbose=4,
)

for gen_type in ['GAN', 'PROB']:
    adver_config['adver_gen_type'] = gen_type
    mod = IntelligentMod(**adver_config)
    choose = GreedyChoose(verbose=4)
    print(f'\n######## TESTING {gen_type} MODE ########')
    for r in range(1, 4):
        new_node_feats, new_edge_feats, seed_ids, new_ids = choose.generate_seeds(g, n_instances=20, label=1, return_id=True)
        missed = torch.rand(len(seed_ids)) * 0.5
        nn2, ne2, sid2, nid2 = mod.modify_seeds(g, new_node_feats, new_edge_feats, seed_ids, new_ids,
                                                round_num=r, missed_probs=missed)
        add_generated_nodes(g, (nn2, ne2), r)
        print('ROUND', r, '-> stats:', mod.get_generation_stats())
        print('  graph now:', g.num_nodes(), g.num_edges())

print('\nOK - IntelligentMod modify_seeds ran without errors in GAN + PROB modes')
