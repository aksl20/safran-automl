__1. Concepts and goals of the project__
- _DNN_: How to truly define a neural network architecture ? For basic neural nets it's easy (number of hidden layer, number of neurons and activation function...) but the state of the art for DNN is some complex neural nets as Resnet, Unet, etc with some relations and configurations that we couldn't represent easily. Those configurations are not numeric values but relations and technical combinations of layers inside the network represented through a graph. How to transform a graph space into a numeric space for autoML ?
- _AutoML_: There is a lot of resources and implementations about AutoML. Many of the tools to build autoML pipeline use a specific range of values given for each parameter that we want to test. By using those range, the autoML tools are able to estimate the true function to optimize and offers an estimate area of search for the task that we want to optimize. However, there is not such tools for model based on relational graphs as neural networks. How can we represent a graph so that it could be plug in classical autoML tools to better understand the search area of a network ? The optimization algorithm are divided into three main principles:
- _graph theory_: The goal of the project turn around this graph representation of a neural network, try to find a better technique to represent all meta-information stored inside a network and explore different technique for autoML based on graph comprehension.
- _Business problem_: Some example of task at Safran tech that need some autoML to save engineers' time : Control airplane's pieces produced by the factories with computer vision techniques. For example, try to automatically detect on a piece of carbon fiber a twisted wire to avoid a early break during the life of the piece.

__2. Ressources__
- [add link to Nacim's document]
- [paperwithcode](https://paperswithcode.com/)

__3. Roadmap__
- December to do the bibliography part (read a maximum of papers to better understand the problem)
- 20/12 - Meeting with Safran
- 10/01 - Meeting with Safran


