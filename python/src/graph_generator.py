import numpy as np
import torch


class GraphGenerator:
    def __init__(self):
        self.search_space = {}

    def parse_modules(self, modules):
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

    def generate(self):
        graph = {}

        for idx_layer, layer in self.search_space.items():
            if isinstance(layer, np.ndarray):
                nb_choices = len(layer)
                choice = np.random.randint(low=0, high=nb_choices)
                graph[idx_layer] = layer[choice]
            elif isinstance(idx_layer, str):
                if np.random.choice([True, False]):
                    graph[idx_layer] = layer
            else:
                graph[idx_layer] = layer

        return graph