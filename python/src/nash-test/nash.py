from nni.networkmorphism_tuner.networkmorphism_tuner import NetworkMorphismTuner
from nni.networkmorphism_tuner.graph import graph_to_json

from copy import deepcopy

from nni.networkmorphism_tuner.graph_transformer import \
    to_deeper_graph, \
    to_wider_graph, \
    to_skip_connection_graph
from nni.networkmorphism_tuner.utils import Constant
from nni.networkmorphism_tuner.bayesian import SearchTree, contain
import numpy as np
import os

from torch import load as torch_load, save as torch_save, zeros as torch_zeros

GEN_SIZE = 8
N_MORPH = 5

def mutate_graph(graph):
    if graph is None:
        return None
    random_num = np.random.choice(3)
    if random_num == 0:
        graph = to_deeper_graph(graph)
    elif random_num == 1:
        graph = to_wider_graph(graph)
    elif random_num == 2:
        graph = to_skip_connection_graph(graph)

    if graph.size() > Constant.MAX_MODEL_SIZE:
        return None
    else:
        return graph


def get_neighbor(graph, n_morph):
    candidate = deepcopy(graph)
    for _ in range(n_morph):
        candidate = mutate_graph(candidate)
    return candidate


def get_neighbors(graph, descriptors, n_neighbors=GEN_SIZE, n_morph=N_MORPH, max_tries=50):
    neighbors = [deepcopy(graph)] # include a copy of best graph so far, to be trained further
    for _ in range(max_tries):
        candidate = get_neighbor(graph, n_morph)
        if candidate is None or contain(descriptors, candidate.extract_descriptor()):
            continue

#        # Check if the model will crash        
#        candidate_model=candidate.produce_torch_model()
#        dummy=torch_zeros((2,1,64,64))
#        try:
#            candidate_model(dummy)
#        except:
#            continue
        
        descriptors.append(candidate.extract_descriptor())
        neighbors.append(candidate)
        if len(neighbors) >= n_neighbors+1:
            break
    return neighbors


class HillClimbingOptimizer:
    """ A Bayesian optimizer for neural architectures.
    Attributes:
        gen_size
    """

    def __init__(self, tuner, gen_size=GEN_SIZE):
        self.gen_size = gen_size
        self.searcher = tuner
        self.search_tree = SearchTree()
        self.generation_queue = []

    def fit(self, x_queue, y_queue):
        pass

    def generate(self, descriptors):
        """Generate new architecture.
        Args:
            descriptors: All the searched neural architectures.
        Returns:
            graph: An instance of Graph. A morphed neural network with weights.
            father_id: The father node ID in the search tree.
        """
        father_id = self.searcher.get_best_model_id()
        graph = self.searcher.load_model_by_id(father_id, load_weights=True)
        new_queue = get_neighbors(graph, descriptors)

        if not new_queue:
            return None, None
        else:
            return new_queue, father_id

    def add_child(self, father_id, model_id):
        ''' add child to the search tree
        Arguments:
            father_id {int} -- father id
            model_id {int} -- model id
        '''

        self.search_tree.add_child(father_id, model_id)


class HillClimbingTuner(NetworkMorphismTuner):
    '''Overwrite the Bayesian optimizer in NetworkMorphismTuner
       with the Hill Climbing algorithm
    '''

    def __init__(self, **kwargs):
        gen_size = kwargs.pop("gen_size", GEN_SIZE)
        max_model_size = kwargs.pop("gen_size", None)
        super(HillClimbingTuner, self).__init__(**kwargs)
        self.bo = HillClimbingOptimizer(self, gen_size)
        if max_model_size is not None:
            Constant.MAX_MODEL_SIZE = max_model_size

    def generate_parameters(self, parameter_id, **kwargs):
        """
        Returns a set of trial neural architecture, as a serializable object.
        Parameters
        ----------
        parameter_id : int
        """
        if not self.history:
            self.init_search()

        new_father_id = None
        generated_graph = None
        if not self.training_queue:
            new_father_id, graph_list = self.generate()
            if type(graph_list) is not list:
                graph_list = [graph_list]
            for generated_graph in graph_list:
                new_model_id = self.model_count
                self.model_count += 1
                self.training_queue.append(
                    (generated_graph, new_father_id, new_model_id))
                self.descriptors.append(generated_graph.extract_descriptor())

        graph, father_id, model_id = self.training_queue.pop(0)

        # from graph to json
        json_out = self.save_weighted_graph(graph, model_id)
        self.total_data[parameter_id] = (json_out, father_id, model_id)

        return graph

    def load_model_by_id(self, father_id, load_weights=False):
        """Overload to work with weights"""
        graph = super(HillClimbingTuner, self).load_model_by_id(father_id)

        if load_weights and graph.weighted:
            weight_path = os.path.join(self.path, str(father_id) + ".torch")
            state_dict = torch_load(weight_path)
            graph.weighted = False
            model = graph.produce_torch_model()
            model.load_state_dict(state_dict)
            model.set_weight_to_graph()
            graph = model.graph
        else:
            graph.weighted = False            
        return graph

    def update_from_model(self, model, parameter_id):
        _, father_id, model_id = self.total_data[parameter_id]
        json_out = self.save_weighted_graph(model.graph, model_id)
        self.total_data[parameter_id] = (json_out, father_id, model_id)

    def save_weighted_graph(self, graph, model_id):
        json_model_path = os.path.join(self.path, str(model_id) + ".json")
        json_out = graph_to_json(graph, json_model_path)
        if graph.weighted:
            weight_path = os.path.join(self.path, str(model_id) + ".torch")
            model = graph.produce_torch_model()
            state_dict = model.state_dict()
            torch_save(state_dict, weight_path)
        return json_out

    def receive_trial_result(self, parameter_id, value, model=None, **kwargs):
        if model is not None:
            self.update_from_model(model, parameter_id)
        super(HillClimbingTuner, self).receive_trial_result(
            parameter_id, None, value, **kwargs)
