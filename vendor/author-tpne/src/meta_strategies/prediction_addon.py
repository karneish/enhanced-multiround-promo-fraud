import torch
import dgl

from sklearn.cluster import DBSCAN
EPS = 1e-10

###########################
# SIMPLE ADDON STRATEGIES #
###########################

def NoneAddon(model, round_num, state={}, **kwargs):
    return None, {}, None

def FeatureDistThreshold(model, round_num, state={}, addon_round_window=3, addon_perc=0.95, step=5, active_subgraph=None, **kwargs):
    graph = model.graph
    data = model.graph.ndata['feature']
    
    if round_num == 0:
        return torch.zeros_like(graph.ndata['label']), {}, None
    else:
        recent_nodes_mask = (graph.ndata['creation_round'] > 0) & ((round_num - graph.ndata['creation_round']) <= addon_round_window) # Get all nodes created in window
        if active_subgraph == None:
            active_subgraph = dgl.khop_in_subgraph(graph, recent_nodes_mask.nonzero().flatten(), 1, store_ids=True)[0] # Get the 1-hop neighbors of the nodes
        active_data = active_subgraph.ndata['feature']

        thres, stop = 0, False
        while not stop:
            thres = thres + step
            dbscan = DBSCAN(eps=EPS, min_samples=thres).fit(active_data.cpu().numpy())
            stop = (dbscan.core_sample_indices_.shape[0] / active_data.shape[0])  < (1-addon_perc)

            if stop:
                for i in range(step-1):
                    dbscan = DBSCAN(eps=EPS, min_samples=thres-1).fit(active_data.cpu().numpy())
                    stop_1 = (dbscan.core_sample_indices_.shape[0] / active_data.shape[0])  < (1-addon_perc)
                    thres = (thres - 1) if stop_1 else thres

        dbscan = DBSCAN(eps=EPS, min_samples=thres).fit(active_data.cpu().numpy())
        spams = active_subgraph.ndata['_ID'][dbscan.core_sample_indices_]

        final_preds = torch.zeros_like(graph.ndata['label'])
        final_preds[spams] = 1

        return final_preds.bool(), {}, active_subgraph

def AggFeatureDistThreshold(model, round_num, state={}, addon_round_window=3, addon_perc=0.95, step=5, active_subgraph=None, **kwargs):
    graph = model.graph
    data = model.embed_nodes(model.graph, model.graph.ndata['feature'])
    
    if round_num == 0:
        return torch.zeros_like(graph.ndata['label']), {}, None
    else:
        recent_nodes_mask = (graph.ndata['creation_round'] > 0) & ((round_num - graph.ndata['creation_round']) <= addon_round_window) # Get all nodes created in window
        if active_subgraph == None:
            active_subgraph = dgl.khop_in_subgraph(graph, recent_nodes_mask.nonzero().flatten(), 1, store_ids=True)[0] # Get the 1-hop neighbors of the nodes
        active_data = model.embed_nodes(active_subgraph, active_subgraph.ndata['feature']).detach().clone()

        thres, stop = 0, False
        while not stop:
            thres = thres + step
            dbscan = DBSCAN(eps=EPS, min_samples=thres).fit(active_data.cpu().numpy())
            stop = (dbscan.core_sample_indices_.shape[0] / active_data.shape[0])  < (1-addon_perc)

            if stop:
                for i in range(step-1):
                    dbscan = DBSCAN(eps=EPS, min_samples=thres-1).fit(active_data.cpu().numpy())
                    stop_1 = (dbscan.core_sample_indices_.shape[0] / active_data.shape[0])  < (1-addon_perc)
                    thres = (thres - 1) if stop_1 else thres

        dbscan = DBSCAN(eps=EPS, min_samples=thres).fit(active_data.cpu().numpy())
        spams = active_subgraph.ndata['_ID'][dbscan.core_sample_indices_]

        final_preds = torch.zeros_like(graph.ndata['label'])
        final_preds[spams] = 1

        return final_preds.bool(), {}, active_subgraph

def DegreeActivityThreshold(model, round_num, state={}, addon_round_window=3, addon_perc=0.95, active_subgraph=None, **kwargs):
    graph = model.graph
    data = model.embed_nodes(model.graph, model.graph.ndata['feature'])
    
    if round_num == 0:
        return torch.zeros_like(graph.ndata['label']), {}, None
    else:
        recent_nodes_mask = (graph.ndata['creation_round'] > 0) & ((round_num - graph.ndata['creation_round']) <= addon_round_window) # Get all nodes created in window
        if active_subgraph == None:
            active_subgraph = dgl.khop_in_subgraph(graph, recent_nodes_mask.nonzero().flatten(), 1, store_ids=True)[0] # Get the 1-hop neighbors of the nodes
        
        # Get top percentile of nodes in that 1 hop-neighbors
        thres = torch.quantile(graph.in_degrees().float(), addon_perc, dim=0).long().item()
        sus_centers_mask = (active_subgraph.in_degrees() > thres)

        # Spams are nodes connected to these top percentile
        sus_subgraph = dgl.khop_in_subgraph(graph, active_subgraph.ndata['_ID'][sus_centers_mask], 1, store_ids=True)[0]
        sus_nodes_mask = torch.zeros_like(graph.ndata['label'])
        sus_nodes_mask[sus_subgraph.ndata['_ID']] = 1
        
        final_preds = recent_nodes_mask.bool() & sus_nodes_mask.bool()
        return final_preds, {}, active_subgraph

def DegreeFeatureThreshold(model, round_num, state={}, addon_round_window=3, addon_perc=0.95, addon_internal_agg='OR', **kwargs):
    feat_pred, feat_state, active_subgraph = FeatureDistThreshold(model, round_num, state=state, addon_round_window=3, addon_perc=addon_perc, **kwargs)
    degree_pred, degree_state, active_subgraph  = DegreeActivityThreshold(model, round_num, state=state, addon_round_window=addon_round_window, addon_perc=addon_perc, active_subgraph=active_subgraph, **kwargs)
    final_preds = (feat_pred.bool() | degree_pred.bool()) if addon_internal_agg == 'OR' else (feat_pred.bool() & degree_pred.bool())
    
    return final_preds, feat_state | degree_state, None

def DegreeAggFeatureThreshold(model, round_num, state={}, addon_round_window=3, addon_perc=0.95, addon_internal_agg='OR', **kwargs):
    feat_pred, feat_state, active_subgraph = AggFeatureDistThreshold(model, round_num, state=state, addon_round_window=3, addon_perc=addon_perc, **kwargs)
    degree_pred, degree_state, active_subgraph = DegreeActivityThreshold(model, round_num, state=state, addon_round_window=addon_round_window, addon_perc=addon_perc, active_subgraph=active_subgraph, **kwargs)
    final_preds = (feat_pred.bool() | degree_pred.bool()) if addon_internal_agg == 'OR' else (feat_pred.bool() & degree_pred.bool())

    return final_preds, feat_state | degree_state, None

