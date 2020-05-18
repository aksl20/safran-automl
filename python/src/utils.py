import numpy as np
import re
import scipy.sparse as sp
import networkx as nx


def get_parameters(node, char_to_remove=','):
    node = ''.join(list(filter(lambda char: char not in char_to_remove, node.lower())))
    parameters = node.replace('(', ' ', 1).split()
    parameters = ' '.join(list(filter(lambda el: not el.isdigit(), parameters)))
    name = parameters.split()[0]
    parameters = parameters.replace('(', '').replace(')', '').split()[1:]
    parameters = list(filter(lambda parameter: '=' in parameter, parameters))
    return [name] + parameters


def get_graph_attribut(graph, params=False, word2vec=False):
    nodes = []
    names = {}
    edges = []
    skipcons = []

    for idx_operation, operation in graph.items():
        if 'skipcon' not in operation:
            nodes.append((idx_operation, operation[1]))
            names[operation[0]] = idx_operation
        else:
            layer_in, layer_out = re.findall(r'(?<=\().*?(?=\))', operation)[0].split('->')
            skipcons.append((layer_in, layer_out))

    for current_node, next_node in zip(nodes, nodes[1:]):
        edges.append((current_node[0], next_node[0]))

    for layer_in, layer_out in skipcons:
        edges.append((names[layer_in], names[layer_out] + 1))
    
    nodes = {idx: node for idx, node in nodes}
    edges = np.array(edges)

    if params :
        features = np.eye(len(nodes))
    elif word2vec:
        features = None
    else:
        nodes = {idx: get_parameters(node) for idx, node in nodes.items()}
        features = np.zeros((len(params), len(nodes)))
        for idx_node, parameters in nodes.items():
            parameters = [parameter.split('=') for parameter in parameters]
            for parameter in parameters:
                if parameter[0] in params:
                    if parameter[1] == 'true':
                        features[params[parameter[0]], idx_node] += 1
                    elif parameter[1] == 'false':
                        features[params[parameter[0]], idx_node] += 0
                    else:
                        features[params[parameter[0]], idx_node] += float(parameter[1])
    
    G = nx.DiGraph()
    G.add_edges_from(edges)

    return G, nodes, features


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx