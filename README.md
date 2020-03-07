# safran-automl

## Prerequisite

You will need following dependencies on your system:

* [Git](http://git-scm.com/)
* [Anaconda](https://www.anaconda.com/)

## Installation

__Python__

Open a linux terminal and run the following commands:

```sh
  $ git clone <repository-url>
  $ cd safran-automl && conda env create -f env.yml
```

Wait until the creation of the environment finish then execute the following commands
to activate the environment and launch the jupyter lab interface.

```shell script
  $ conda activate automl
  $ jupyter lab
```

## Extract computational deep learning graph

You will find the notebook `demo-extract_graph_from_pytorch.ipynb` in the folder `python/notebook` 
with an example of processing to extract a list of nodes, edges and the adjacency matrix
from a pytorch deeep learning model.