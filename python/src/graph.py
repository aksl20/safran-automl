import torch
import itertools
import numpy as np
import re
import scipy.sparse as sp


def parse_modules(modules):
    """Fonction to parse the module list return by the pytorch model.

    Parameters
    ----------
    modules : list
        List of all modules of the model

    Return
    ------
    dict
        Dictionnary which contains as keys the idex of the layer and as values
        the name and parameters of the layer.
    """

    search_space = {}
    idx_layer = 0
    nb_choice = 0

    # Iterate other all modules of the model
    for module in modules:
        name, module = module
        parameters = str(module)

        if 'LayerChoice' in parameters:
            # For each layerchoice, there is two modules in the list. The first one
            # is not usefull.
            continue
        elif 'skipcon' in parameters:
            if nb_choice > 0:
                # Each choice are store in the list shortly after the ModuleList,
                # We need to skip them.
                nb_choice -= 1
                continue
            else:
                search_space[name] = parameters
        elif isinstance(module, torch.nn.modules.container.ModuleList):
            # Iterate over each choice in the layerchoice and store them in a the
            # Dictionnary
            choices = []
            for choice in module:
                choices.append((name, str(choice)))
            nb_choice = len(choices)
            search_space[idx_layer] = choices
            idx_layer += 1
        else:
            if nb_choice > 0:
                # Each choice are store in the list shortly after the ModuleList,
                # We need to skip them.
                nb_choice -= 1
                continue
            else:
                search_space[idx_layer] = (name, parameters)
                idx_layer += 1
    return search_space


def get_graphs(graph):
    graphs = []
    layers_choice = [layer for layer in graph.values() if isinstance(layer, list)]
    choices = [y for x in layers_choice for y in x]
    nb_graph = len(list(itertools.product(choices)))

    for i in range(nb_graph):
        graph_i = {}
        for idx_layer, layer in graph.items():
            if isinstance(layer, list):
                choice = layer.pop()
                graph[idx_layer] = layer
                graph_i[idx_layer] = choice
            else:
                graph_i[idx_layer] = layer
        graphs.append(graph_i)
    return graphs


def get_parameters(node, char_to_remove=','):
    node = ''.join(list(filter(lambda char: char not in char_to_remove, node.lower())))
    parameters = node.replace('(', ' ', 1).split()
    parameters = ' '.join(list(filter(lambda el: not el.isdigit(), parameters)))
    name = parameters.split()[0]
    parameters = parameters.replace('(', '').replace(')', '').split()[1:]
    parameters = list(filter(lambda parameter: '=' in parameter, parameters))
    return [name] + parameters


def get_graph_attribut(graph, symetric=True, dict_params=None):
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

    nodes = {idx: get_parameters(node) for idx, node in nodes}
  
    edges = np.array(edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                    shape=(len(nodes), len(nodes)),
                    dtype=np.float32)

    if dict_params is None:
        features = np.eye(len(nodes))
    else:
        features = np.zeros((len(dict_params), len(nodes)))
        for idx_node, parameters in nodes.items():
            parameters = [parameter.split('=') for parameter in parameters]
            for parameter in parameters:
                if parameter[0] in dict_params:
                    if parameter[1] == 'true':
                        features[dict_params[parameter[0]], idx_node] += 1
                    elif parameter[1] == 'false':
                        features[dict_params[parameter[0]], idx_node] += 0
                    else:
                        features[dict_params[parameter[0]], idx_node] += float(parameter[1])

    if symetric:
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return nodes, edges, features, adj


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx