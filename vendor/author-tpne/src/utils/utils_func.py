import torch
import copy
import dgl
import gc
import numpy as np
from torch import nn

from collections import Counter
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix
from torch.functional import F

###############
### GENERAL ###
###############

# Print but only if currently above specified verbosity level
def verPrint(verbose_status, verbose_threshold, msg):
    if verbose_status >= verbose_threshold:
        print(msg)

################################
### GENERIC TRAINING RELATED ###
################################

def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    
    # To CPU to enable numpy calc
    labels = labels.cpu()
    probs = probs.cpu()

    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:,1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre

def eval_and_print(verbose_level, labels, preds, probs, msg):
    # To CPU to enable sklearn calc
    labels = labels.int().cpu()
    preds = preds.cpu()
    probs = None if probs is None else probs.cpu()

    rec = recall_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    auc = 0 if probs is None else roc_auc_score(labels, probs.detach().numpy()) if torch.unique(labels).shape[0] > 1 else -1

    tp, fn, fp, tn = confusion_matrix(labels, preds, labels=[1, 0]).ravel()
    verPrint(verbose_level, 1, f'{msg}: REC {rec*100:.2f} PRE {prec*100:.2f} MF1 {f1*100:.2f} AUC {auc*100:.2f} TP {tp} FP {fp} TN {tn} FN {fn} | {len(labels)} {dict(Counter(labels.tolist()))}')

    return (rec, prec, f1, auc), (tp, fp, tn, fn)

###############################
### GRAPH SPECIFIC FUNCTION ###
###############################

# Add generated new nodes to graph
def add_generated_nodes(graph, data, round_num, predicted_flag=False, set_name='test'):
    new_nodes, new_edges = data
    train_flag = (set_name == 'train') 
    val_flag = (set_name == 'val') 
    test_flag = (set_name == 'test') 

    # Initialize features
    new_nodes['creation_round'] = torch.full([len(new_nodes['label'])], round_num)
    new_nodes['predicted'] = torch.full([len(new_nodes['label'])], predicted_flag)
    new_nodes['train_mask'] = torch.full([len(new_nodes['label'])], train_flag).bool()
    new_nodes['val_mask'] = torch.full([len(new_nodes['label'])], val_flag).bool()
    new_nodes['test_mask'] = torch.full([len(new_nodes['label'])], test_flag).bool()

    # Add nodes
    graph.add_nodes(len(new_nodes['label']), new_nodes)
    
    # Add edges TODO: edge features?
    for etype in new_edges.keys():        
        for dir in new_edges[etype].keys(): # Incoming and outcoming edges
            edge_src = new_edges[etype][dir]['src'].long()
            edge_dst = new_edges[etype][dir]['dst'].long()
            del new_edges[etype][dir]['src'], new_edges[etype][dir]['dst'] 
            gc.collect()

            graph.add_edges(edge_src, edge_dst, etype=etype)

# Remove selected nodes and resulting isolateds
def remove_generated_nodes(graph, node_ids):
    new_graph = dgl.remove_nodes(copy.deepcopy(graph), node_ids, store_ids = True)
    
    # Additionally remove isolated nodes
    isolateds = ((new_graph.in_degrees() == 0) & (new_graph.out_degrees() == 0)).nonzero().squeeze(1)
    isolateds_ori_idx = new_graph.ndata['_ID'][isolateds]
    new_graph = dgl.remove_nodes(new_graph, isolateds, store_ids = False)
    
    node_ids = torch.cat([node_ids, isolateds_ori_idx])
    return new_graph, node_ids

# Subgraph from selected nodes and remove isolateds
# Remove selected nodes and resulting isolateds
def subgraph_from_nodes(graph, node_ids):
    new_graph = dgl.node_subgraph(graph, node_ids, store_ids = True)
    
    # Additionally remove isolated nodes
    isolateds = ((new_graph.in_degrees() == 0) & (new_graph.out_degrees() == 0)).nonzero().squeeze(1)
    isolateds_ori_idx = new_graph.ndata['_ID'][isolateds]
    new_graph = dgl.remove_nodes(new_graph, isolateds, store_ids = True)
    
    node_ids = torch.cat([node_ids, isolateds_ori_idx])
    return new_graph, node_ids