#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:51:46 2020

@author: btayart
"""

import torch
import torch.nn as nn

class GraphModel(nn.Module):
    def __init__(self, graph):
        super(GraphModel, self).__init__()
        self.graph_model = graph
        self.graph_model.check_consistency()
        self.graph_model.order_nodes()

        self.layers = nn.ModuleList([
                        nn.ModuleList([
                            layer_obj.to_torch_module()
                            for layer_obj, u
                            in node_backward_layers
                        ])
                        for node_backward_layers
                        in self.graph_model.backward_layer_list
                      ])

    def forward(self, X):
        node_data = [X]
        for inode in range(1,self.graph_model.output_node+1):
            previous_layers = self.layers[inode]
            previous_data = self.graph_model.backward_layer_list[inode]
            port_data = [L(node_data[n]) for L,(_,n) in zip(previous_layers, previous_data)]
            node = self.graph_model.nodes[inode]
            if node.join_type is node.JoinType.CONCATENATE:
                node_data.append(torch.cat(port_data, dim=1))
            elif node.join_type is node.JoinType.ADD:
                node_data.append(sum(port_data))
        return node_data[self.graph_model.output_node]

    def weights_to_modules(self):
        for u, v, port, layer_obj in self.graph_model.layers:
            layer_obj.save_weights_to_module(self.layers[v][port])

    def weights_to_graph(self):
        for u, v, port, layer_obj in self.graph_model.layers:
            layer_obj.load_weights_from_module(self.layers[v][port])

    def reset_weights(self):
        for u, v, port, layer_obj in self.graph_model.layers:
            layer_obj.reset_weights(self.layers[v][port])

        self.layers.apply(lambda m: m.reset_parameters()
                          if hasattr(m, "reset_parameters") else None)
