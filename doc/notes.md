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

Source:

