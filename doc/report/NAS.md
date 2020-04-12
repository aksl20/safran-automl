## Goal of the project

### Graph embedding for a continuous search space

Outstanding results have been achieved using NAS with the reinforcement learning search strategy. Here, a recurrent network was used to generate a string to form a child network. However, such a type of network exhibits two problems: non continuous and high-dimensional search space. The frequent large strings of action from the recurrent network and the discrete space result in difficulty in optimization. Other NAS methods use continuous search space that enable the use of gradient based optimization techniques. These kinds of methods reduce significantly the amount of computational resources.  For example, NAO (Neural Architecture Optimization) use three key components in the proposed approach to benefits of the continuous space :

- An encoder embeds/maps neural network architectures into a continuous space.
- A predictor takes the continuous representation of a network as input and predicts its accuracy
- A decoder maps a continuous representation of a network back to its architecture.

![](C:\Users\AxelC\Documents\workspace\safran-automl\doc\report\img\nao.png)

In this part, we will focus on the techniques to mapping a set of graphs with a continuous space. We think that we could achieve good results with less computational resources by using a continuous optimization approach.



#### Graph theory

Graphs represent the main mathematical object in this section. For better understanding of the concept, we begin by introducing some basic concepts on graph theory. 

**Definition 1**.  In one restricted but very common sense of the term, a graph is an ordered pair $G = (V, E)$ comprising:

- $V$ a set of vertices (also called nodes or points)
- $E \subseteq \{\{x, y\} | (x, y) \in V^2 \wedge x \ne y\}$ a set of edges (also called links or lines), which are unordered pairs of vertices (i.e. an edge is associated with two distinct vertices).

This type of object may be called precisely an **undirected simple graph**.

In our case, graph will be representation of neural network architectures.  The data flow inside a network go through the first node to the final node, so we are able to represent networks by **directed simple graph**.

__Definition 2.__ A directed graph or digraph is a graph in which edges have orientations i.e. edges $(x, y)$ and $(y, x)$ are different objects.

We also introduce another important tool in graph theory , the adjacency matrix. 

__Definition 3.__ For a simple graph with vertex set $V$, the adjacency matrix is a square $|V| \times |V|$ matrix A such that its element $A_{ij}$ is one when there is an edge from vertex $i$ to vertex $j$, and zero when there is no edge.