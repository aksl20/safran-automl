import numpy as np
import torch


class GraphGenerator:
    def __init__(self):
        self.search_space = {}
        self.choices_given = {}

    def parse_modules(self, modules):
        """Fonction to parse the module list returned by the pytorch model.

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
                    self.search_space[name] = parameters
            elif isinstance(module, torch.nn.modules.container.ModuleList):
                # Iterate over each choice in the layerchoice and store them in a the
                # Dictionnary
                choices = []
                for choice in module:
                    choices.append((name, str(choice)))
                nb_choice = len(choices)
                self.search_space[idx_layer] = np.array(choices)
                idx_layer += 1
            else:
                if nb_choice > 0:
                    # Each choice are store in the list shortly after the ModuleList,
                    # We need to skip them.
                    nb_choice -= 1
                    continue
                else:
                    self.search_space[idx_layer] = (name, parameters)
                    idx_layer += 1

    
    def _add_layer(self, graph, idx_layer, layer, indices_choice, skipcon=False):
        if not self.choices_given.get(idx_layer) or len(self.choices_given[idx_layer]) == len(layer):
            self.choices_given[idx_layer] = []
            choice = np.random.choice(indices_choice)
        else:
            if not skipcon:
                indices_choice = [choice for choice in indices_choice if choice not in self.choices_given[idx_layer]]
            choice = np.random.choice(indices_choice)
        
        if skipcon and choice:
            graph[idx_layer] = layer
        elif not skipcon:
            graph[idx_layer] = layer[choice]
        self.choices_given[idx_layer].append(choice)
        return graph

    def clear_cache(self):
        self.choices_given.clear()

    def generate(self):
        """Generate child network pick from the search space. 
        """
        graph = {}

        for idx_layer, layer in self.search_space.items():
            # Check if layer is a mutable from LayerChoice
            if isinstance(layer, np.ndarray):
                indices_choice = list(range(len(layer)))
                graph = self._add_layer(graph, idx_layer, layer, indices_choice)
            # Check if layer is a mutable from InputChoice (skipconnect)
            elif isinstance(idx_layer, str):
                indices_choice = [0, 1]
                graph = self._add_layer(graph, idx_layer, layer, indices_choice, skipcon=True)
            else:
                graph[idx_layer] = layer

        return graph