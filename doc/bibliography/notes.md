# AutoML method

## 1. hyperparameter optimisation
### a. Blackbox HPO
#### Concept
"Black Box" optimization refers to a problem setup in which an optimization algorithm is supposed to optimize (e.g., minimize) an objective function through a so-called black-box interface: the algorithm may query the value f(x) for a point x, but it does not obtain gradient information, and in particular it cannot make any assumptions on the analytic form of f (e.g., being linear or quadratic). see [blackbox challenge](https://bbcomp.ini.rub.de/)

#### Strategies
__Population based method__
* genetic algorithms
* evolutionary algorithms
* evolutionary strategies
* particle swarm optimization
* covariance matrix adaption evolutionary strategy (best poppulation-based method)

### b. Baysean based method

state-of-the-art optimization framework for the global optimization of expensive blackbox functions, which recently gained traction in HPO by obtaining new state-of-the-art results in tuning deep neural networks
for image classification

## 2. Neural Architecture Search (NAS) - _Nacim talk a lot about it_

Deep Learning has enabled remarkable progress over the last years on a variety of tasks, such as image recognition, speech recognition, and machine translation. One crucial aspect for this progress are novel neural architectures. Currently employed architectures have mostly been developed manually by human experts, which is a time-consuming and error-prone process. Because of this, there is growing interest in automated neural architecture search methods. We provide
an overview of existing work in this field of research and categorize them according to three dimensions: search space, search strategy, and performance estimation strategy.

![nas illustration](https://github.com/aksl20/safran-automl/blob/documentation/doc/image/illustration-nas.PNG)

- __Search Space__: The search space defines which architectures can be represented in principle. Incorporating prior knowledge about properties wellsuited for a task can reduce the size of the search space and simplify the search. However, this also introduces a human bias, which may preventfinding novel architectural building blocks that go beyond the current human knowledge.
- __Search Strategy__: The search strategy details how to explore the search space. It encompasses the classical exploration-exploitation trade-off since, on the one hand, it is desirable to find well-performing architectures quickly, while on the other hand, premature convergence to a region of suboptimal architectures should be avoided.
- __Performance Estimation Strategy__ The objective of NAS is typically to find architectures that achieve high predictive performance on unseen data. Performance Estimation refers to the process of estimating this performance: the simplest option is to perform a standard training and validation of the architecture on data, but this is unfortunately computationally expensive and limits the number of architectures that can be
explored. Much recent research therefore focuses on developing methods
that reduce the cost of these performance estimations.

### Search space

A relatively simple search space is the space of chain-structured neural networks. A chain-structured neural network
architecture _A_ can be written as a sequence of _n_ layers where the i’th layer _L_i_ receives its input from layer _i − 1_ and its output serves as the input for layer _i + 1_. The search space is then parametrized by :
- The (maximum) number of layers _n_
- The type of operation every layer can execute , e.g., pooling, convolution, or more advanced layer types like depthwise separable convolutions or dilated convolutions
- Hyperparameters associated with the operation, e.g., number of filters, kernel size and strides for a convolutional layer, or simply
number of units for fully-connected networks.

Recent works on NAS incorporate modern design elements known from hand-crafted architectures such as skip connections, which
allow to build complex, multi-branch networks. Below some example of works: 
- [SMASH](https://paperswithcode.com/paper/smash-one-shot-model-architecture-search)
- [Learning transferable architecture for scalable image recognition](https://paperswithcode.com/paper/learning-transferable-architectures-for)
- [Path-Level Network Transformation for Efficient Architecture Search](https://paperswithcode.com/paper/path-level-network-transformation-for)

 In this case the input of layer i can be formally described as a function _g_i(L_out_i-1, ..., L_out_0)_ combining previous layers output : 
 - Chaine-structured network : _g_i(L_out_i-1, ..., L_out_0) = L_out_i-1_
 - ResidualNetwork : _g_i(L_out_i-1, ..., L_out_0) = L_out_i-1 + L_out_j avec j < i_
 - DenseNet : _g_i(L_out_i-1, ..., L_out_0) = concat(L_out_i-1, ..., L_out_0)_

Motivated by hand-crafted architectures consisting of repeated motifs [Zoph et al.](https://paperswithcode.com/paper/learning-transferable-architectures-for) and [Zhong et al.](https://paperswithcode.com/paper/practical-block-wise-neural-network) propose to search for such motifs, dubbed cells or blocks, respectively, rather than for whole architectures. Zoph et al. optimize two different kind of cells: a normal cell that preservers the dimensionality of the input and a reduction cell which reduces the spatial dimension.

![illustration-cells](https://github.com/aksl20/safran-automl/blob/documentation/doc/image/cell-block.PNG)

The final architecture is then built by stacking these cells in a predefined manner, as illustrated in Figure 3.3. This search space has two major advantages compared to the ones discussed above:
1. The size of the search space is drastically reduced since cells can be comparably small
2. Cells can more easily be transferred to other datasets by adapting the number of cells used within a model.

However, a new design-choice arises when using a cell-based search space, namely how to choose __the meta-architecture__: how many cells shall be used and how should they be connected to build the actual model? One step in the direction of optimizing meta-architectures is the hierarchical search space introduced by [Liu et al.](https://paperswithcode.com/paper/hierarchical-representations-for-efficient)

### State of the art : Three different method 

People offers different type of algorithm to try to answer to the question of NAS. We need to explore this three
article to better understand the problem :

- [Efficient Neural Architecture Search - Pham et al;](https://arxiv.org/pdf/1802.03268.pdf)
- [Differentiable architecture search (DARTS) - ](https://arxiv.org/pdf/1806.09055.pdf)
- Evolution - Real et al.; Liu et al.

Source:

