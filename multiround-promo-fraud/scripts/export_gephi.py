"""Export tolokers_bid DGL graph to Gephi GEXF format.

Usage:
  python export_gephi.py              # sampled (500 nodes) for Gephi Lite web
  python export_gephi.py --full       # full graph for Gephi desktop
  python export_gephi.py --nodes 800  # custom sample size
"""
import os
import sys
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, '..', 'dataset')
INPUT_FILE = os.path.join(DATASET_DIR, 'tolokers_bid')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Export full graph (slow, desktop Gephi only)')
    parser.add_argument('--nodes', type=int, default=500, help='Number of nodes to sample (default: 500)')
    args = parser.parse_args()

    try:
        import dgl
        import torch
    except ImportError:
        print("ERROR: dgl/torch not installed. Run: pip install dgl torch")
        sys.exit(1)

    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Dataset not found at {INPUT_FILE}")
        sys.exit(1)

    print(f"Loading {INPUT_FILE} ...")
    graphs, _ = dgl.load_graphs(INPUT_FILE)
    g = graphs[0]
    print(f"Full graph: {g.num_nodes()} nodes, {g.num_edges()} edges")

    labels = g.ndata.get('label')
    features = g.ndata.get('feature')

    if labels is not None:
        if labels.dim() > 1:
            labels = labels.argmax(1)
        labels = labels.squeeze(-1)
    else:
        labels = torch.zeros(g.num_nodes(), dtype=torch.long)

    feat_names = []
    feature_names_map = {
        0: 'age', 1: 'email_disposable', 2: 'phone_verified',
        3: 'device_fresh', 4: 'ip_proxy', 5: 'loc_entropy',
        6: 'login_night', 7: 'amount_mean', 8: 'amount_std',
        9: 'txn_count', 10: 'txn_freq',
    }
    if features is not None:
        feat_dim = features.shape[1]
        for i in range(feat_dim):
            feat_names.append(feature_names_map.get(i, f'feature_{i}'))
        features_np = features.numpy()
    else:
        features_np = None

    if not args.full:
        node_set = _sample_nodes(g, labels, args.nodes)
        g = dgl.node_subgraph(g, node_set)
        labels = labels[node_set]
        if features_np is not None:
            features_np = features_np[node_set]
        remap = {int(old): i for i, old in enumerate(node_set.tolist())}
        print(f"Sampled: {g.num_nodes()} nodes, {g.num_edges()} edges")
    else:
        remap = {i: i for i in range(g.num_nodes())}

    labels_list = labels.tolist()
    features_list = features_np.tolist() if features_np is not None else []
    fraud_count = sum(1 for l in labels_list if l == 1)
    genuine_count = sum(1 for l in labels_list if l == 0)

    print(f"Labels: {fraud_count} fraud, {genuine_count} genuine")
    if feat_names:
        print(f"Features: {feat_names}")

    gexf = ET.Element('gexf', xmlns='http://www.gexf.net/1.2draft', version='1.2')
    graph_el = ET.SubElement(gexf, 'graph', defaultedgetype='directed', mode='static')

    attrs_el = ET.SubElement(graph_el, 'attributes', {'class': 'node', 'mode': 'static'})
    ET.SubElement(attrs_el, 'attribute', id='label_attr', title='Label', type='integer')
    ET.SubElement(attrs_el, 'attribute', id='is_fraud', title='Is Fraud', type='boolean')
    for i, fname in enumerate(feat_names):
        ET.SubElement(attrs_el, 'attribute', id=f'feat_{i}', title=fname, type='float')

    nodes_el = ET.SubElement(graph_el, 'nodes')
    for i in range(g.num_nodes()):
        node = ET.SubElement(nodes_el, 'node', id=str(i), label=str(i))
        vis = ET.SubElement(node, 'attvalues')
        _add_attval(vis, 'label_attr', str(labels_list[i]))
        _add_attval(vis, 'is_fraud', str(labels_list[i] == 1).lower())
        if features_list and i < len(features_list):
            for j, val in enumerate(features_list[i]):
                _add_attval(vis, f'feat_{j}', f'{val:.4f}')

    edges_el = ET.SubElement(graph_el, 'edges')
    src, dst = g.edges()
    src_list = src.tolist()
    dst_list = dst.tolist()
    for idx in range(len(src_list)):
        ET.SubElement(edges_el, 'edge', id=str(idx), source=str(src_list[idx]), target=str(dst_list[idx]))

    raw = ET.tostring(gexf, encoding='unicode')
    xml_str = minidom.parseString(raw).toprettyxml(indent='  ', encoding=None)
    if not xml_str.startswith('<?xml'):
        xml_str = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_str

    suffix = '' if args.full else '_sample'
    output = os.path.join(DATASET_DIR, f'tolokers_bid{suffix}.gexf')
    with open(output, 'w', encoding='utf-8') as f:
        f.write(xml_str)

    print(f"\nExported: {output}")
    print(f"Open in lite.gephi.org: File -> Open -> select tolokers_bid{'_sample' if not args.full else ''}.gexf")
    print(f"\nGephi tips:")
    print(f"  - Color nodes by 'Label' partition (0=genuine, 1=fraud)")
    print(f"  - Run 'Force Atlas 2' layout for spatial clustering")
    print(f"  - Use 'age' or 'amount_mean' for node size")


def _sample_nodes(g, labels, target_count):
    import torch
    n = g.num_nodes()
    if n <= target_count:
        return torch.arange(n)

    degrees = (g.in_degrees() + g.out_degrees()).float()

    fraud_mask = labels == 1
    fraud_idx = torch.where(fraud_mask)[0]
    genuine_idx = torch.where(~fraud_mask)[0]

    fraud_degrees = degrees[fraud_idx]
    genuine_degrees = degrees[genuine_idx]

    fraud_target = min(len(fraud_idx), max(target_count // 3, 50))
    genuine_target = min(len(genuine_idx), target_count - fraud_target)

    _, fraud_top = fraud_degrees.topk(min(fraud_target, len(fraud_idx)))
    _, genuine_top = genuine_degrees.topk(min(genuine_target, len(genuine_idx)))

    selected = torch.cat([fraud_idx[fraud_top], genuine_idx[genuine_top]])

    neighbors = set(selected.tolist())
    frontier = selected.tolist()
    while len(neighbors) < target_count and frontier:
        nxt = []
        for u in frontier[:200]:
            for v in g.successors(u).tolist():
                if v not in neighbors:
                    neighbors.add(v)
                    nxt.append(v)
                    if len(neighbors) >= target_count:
                        break
            if len(neighbors) >= target_count:
                break
        frontier = nxt

    return torch.tensor(sorted(neighbors), dtype=torch.long)


def _add_attval(parent, attr_id, value):
    ET.SubElement(parent, 'attvalue', {'for': attr_id, 'value': value})


if __name__ == '__main__':
    main()
