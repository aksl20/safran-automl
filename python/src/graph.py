import torch
import itertools
import numpy as np
import re


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

    graph = {}
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
                graph[name] = parameters
        elif isinstance(module, torch.nn.modules.container.ModuleList):
            # Iterate over each choice in the layerchoice and store them in a the
            # Dictionnary
            choices = []
            for choice in module:
                choices.append((name, str(choice)))
            nb_choice = len(choices)
            graph[idx_layer] = choices
            idx_layer += 1
        else:
            if nb_choice > 0:
                # Each choice are store in the list shortly after the ModuleList,
                # We need to skip them.
                nb_choice -= 1
                continue
            else:
                graph[idx_layer] = (name, parameters)
                idx_layer += 1
    return graph


def get_search_space(graph):
    search_space = []
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
        search_space.append(graph_i)
    return search_space


def get_graph_attribut(graph):
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

    n = len(nodes)
    adjacency = np.zeros((n, n))

    for edge in edges:
        adjacency[edge[0], edge[1]] += 1

    return nodes, edges, adjacency
