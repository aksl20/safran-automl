#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 23:06:18 2020

@author: btayart
"""

import numpy as np
from model import GraphModel
from enum import Enum



class HashMixin(object):
    def __hash__(self):
        return hash(self.__class__.__name__) + hash(self.id)


class NNNode(HashMixin):
    """A node is a data blob in the graph"""

    class JoinType(Enum):
        ADD = 1
        CONCATENATE = 2
    
    def __init__(self, size=None, join_type="concatenate"):
        self.id = None
        self.size = size
        self.mutable = (size is None)
        if join_type in ("c", "concatenate", "concat"):
            self.join_type = self.JoinType.CONCATENATE
        elif join_type in ("add",):
            self.join_type = self.JoinType.ADD
        else:
            raise ValueError("join_type: expected 'add' or 'concatenate'")

class NNGraph:
    r"""
    Graph the maintains:
        * a list of nodes, the ID of which is their index in the list
        * a list of layers (=directed edges) in the form
            (from_node, to_node, to_node_port, layer)
        * a list of cell connection (= pair of directed edges) in the form
            (node in lower cell, node in upper cell)

    Note: this format is not optimized compared to storing the parents/children
    as object properties, but simple to implement
    """

    def __init__(self, input_size, output_size):
        self.nodes = []
        self.layers = []
        self.cell_links = []
        self.input_size = input_size
        self.output_size = output_size
        self.input_node = self.add_node(input_size)
        self.output_node = self.add_node(output_size)

        self._reset_layer_lists()


    def __str__(self):
        out = f"NNGraph object with {len(self.nodes)} nodes"
        out += f"\nInput node {self.input_node:2d}, Output node {self.output_node:2d}"
        out += f"\nLayers:"
        for u,v,port, layer_obj in sorted(self.layers, key=lambda t:t[:3]):
            out += f"\n From {u:2d} to {v:2d} ({port}) : {layer_obj}"
        return out
    # property functions    
    @property
    def forward_layer_list(self):
        self.update_layer_lists()
        return self._forward_layer_list
    
    @property
    def backward_layer_list(self):
        self.update_layer_lists()
        return self._backward_layer_list   

    def update_layer_lists(self, force_update=False):
        if not any((self._forward_layer_list is None,
                    self._backward_layer_list is  None,
                    force_update)) :
            return
        self._forward_layer_list = [[] for _ in self.nodes]
        self._backward_layer_list = [[] for _ in self.nodes]

        for u, v, port, layer_obj in self.layers:
            self._forward_layer_list[u].append((layer_obj, v, port))
            while not port < len(self._backward_layer_list[v]):
                self._backward_layer_list[v].append(None)
            self._backward_layer_list[v][port] = (layer_obj, u)    

    def _reset_layer_lists(self):
        self._forward_layer_list = None
        self._backward_layer_list = None        

    # Graph edition methods
    def add_node(self, size=None):
        # Add a node in the list
        node = NNNode(size)
        node.id = len(self.nodes)
        self.nodes.append(node)
        self._reset_layer_lists()
        return node.id

    def add_layer(self, layer_obj, u, v, port=0):
        r"""Adds a layer"""
        # Adds a layer beteen node u and v
        layer_id = len(self.layers)
        new_layer = (u, v, port, layer_obj)
        self.layers.append(new_layer)
        self._reset_layer_lists()
        return layer_id

    def add_child(self, node_id, new_layer_obj):
        r"""Add a layer and a new node after a node"""
        v = self.add_node()
        self.add_layer(new_layer_obj, node_id, v)
        return v

    def insert_after(self, layer_id, new_layer_obj):
        """Insert a new node and a new layer after an edge"""
        u = self.add_node()
        l = self.add_layer(new_layer_obj, u, u)
        self._swap_connections(layer_id, l)
        return l

    def _swap_connections(self, layer_id1, layer_id2):
        """Swap two edges"""
        from1, to1, port1, obj1 = self.layers[layer_id1]
        from2, to2, port2, obj2 = self.layers[layer_id2]
        self.layers[layer_id1] = (from1, to2, port2, obj1)
        self.layers[layer_id2] = (from2, to1, port1, obj2)

    # Cell edition methods
    # Same as above, but applies to whole cells
    def make_cell_connection(self, node_id_list):
        new_connections = [lnk for lnk in zip(
            node_id_list[:-1], node_id_list[1:])]
        self.cell_links.extend(new_connections)

    @staticmethod
    def group_apply(fcn, *args):
        return [fcn(*arg) for arg in zip(*args)]

    def cell_add_nodes(self, n):
        new_nodes = [self.add_node() for _ in range(n)]
        self.make_cell_connection(new_nodes)
        return new_nodes

    def cell_add_layers(self, *args):
        return self.group_apply(self.add_layer, *args)

    def cell_add_children(self, node_ids, new_layer_objects):
        r"""Add a layer and a new node after a node"""
        v = self.cell_add_nodes(len(node_ids))
        self.cell_add_layers(new_layer_objects, node_ids, v)
        return v

    def cell_insert_after(self, layer_ids, new_layer_objects):
        """Insert a new node and a new layer after an edge"""
        u = self.cell_add_nodes(len(layer_ids))
        l = self.add_layer(new_layer_objects, u, u)
        self.group_apply(self._swap_connections, layer_ids, l)
        return l

    # Looking across cells
    def get_upper(self, node_id):
        upper = [up for low, up in self.cell_links if low == node_id]
        return upper[0] if upper else None

    def get_lower(self, node_id):
        lower = [low for low, up in self.cell_links if up == node_id]
        return lower[0] if lower else None

    # Ordering the nodes
    def order_nodes(self):
        """Find an order for the nodes so that a node ancestors have a smaller
        index, and its descendants a larger index

        Not optimized!
        """
        ordered_nodes = []
        explored_nodes = set()
        open_nodes = {self.input_node}
        ancestors = [set() for _ in self.nodes]
        while self.output_node not in explored_nodes:
            for node in open_nodes:
                parents = set([ u for layer_obj, u in self.backward_layer_list[node]])
                if parents <= explored_nodes:
                    break
            else:
                raise(RuntimeError("Error ordering nodes in forward pass " +
                                   "cycle detected"))
            ancestors[node] = parents.copy()
            for p in parents:
                ancestors[node] |= ancestors[p]
            ordered_nodes.append(node)
            explored_nodes.add(node)
            open_nodes.remove(node)
            open_nodes |= set([ v for layer_obj, v, port in self.forward_layer_list[node]])

        # append disconnected nodes
        disconnected_nodes = set(range(len(self.nodes))) - set(ordered_nodes)
        while disconnected_nodes:
            ordered_nodes.append(disconnected_nodes.pop())

        # update lists
        orderer = {node:ii for ii,node in enumerate(ordered_nodes)}

        self.nodes=[self.nodes[orderer[ii]] for ii in range(len(self.nodes))]       
        self.input_node = orderer[self.input_node]
        self.output_node = orderer[self.output_node]
        self.layers=[(orderer[u], orderer[v], port, layer_obj) for
                     u, v, port, layer_obj in self.layers]
        self.update_layer_lists(force_update=True)
        
    # utility functions
    def check_consistency(self):
        """
        Check the consistency of connections between nodes and layers
        """
        err = ""
        self.order_nodes()
        for ii,back_layers in enumerate(self.backward_layer_list):
            if None in back_layers:
                err += f"Port numbering is inconsistent on node {ii}"
            if not back_layers and ii != self.input_node:
                err += f"Node {ii} has no input layer"

        for ii,fwd_layers in enumerate(self.forward_layer_list):
            if not fwd_layers and ii != self.output_node:
                err += f"Node {ii} has no output layer"

        # TODO Check size of data
        if err:
            raise(RuntimeError(err))

    def torch_model(self):
        return GraphModel(self)