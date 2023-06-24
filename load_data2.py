import numpy as np
import torch
import networkx as nx

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0
        self.max_neighbor = 0

def load_data2(dataset, degree_as_tag):
    '''
        dataset: name of dataset
        degree_as_tag: boolean flag to indicate whether to use degree as a tag

        Returns:
        g_list: list of S2VGraph objects
        num_classes: number of unique labels/classes
    '''

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        lines = f.readlines()

    for line in lines:
        row = line.strip().split()
        n, l = [int(w) for w in row]
        if l not in label_dict:
            mapped = len(label_dict)
            label_dict[l] = mapped
        g = nx.Graph()
        node_tags = []
        node_features = []
        n_edges = 0
        for _ in range(n):
            row = lines.pop(0).strip().split()
            tmp = int(row[1]) + 2
            if tmp == len(row):
                # no node attributes
                row = [int(w) for w in row]
                attr = None
            else:
                row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
            if row[0] not in feat_dict:
                mapped = len(feat_dict)
                feat_dict[row[0]] = mapped
            node_tags.append(feat_dict[row[0]])

            if tmp > len(row):
                node_features.append(attr)

            n_edges += row[1]
            for k in range(2, len(row)):
                g.add_edge(_, row[k])

        if node_features:
            node_features = np.stack(node_features)
            node_feature_flag = True
        else:
            node_features = None
            node_feature_flag = False

        assert len(g) == n

        g_list.append(S2VGraph(g, l, node_tags))

    # Add labels and edge_mat
    for g in g_list:
        g.neighbors = [[] for _ in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0, 1)

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    # Extracting unique tag labels
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]: i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1
    print('# classes: %d' % num_classes)
    print('# maximum node tag: %d' % len(tagset))
    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)
