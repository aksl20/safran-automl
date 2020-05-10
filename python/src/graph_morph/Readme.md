# Graph morphisms
## Representing a neural network as a graph
Nodes represent data blobs and edges represent layers.

At this stage of development, the graph has one input node, one output node and intermediate nodes.

Several layers may originate from the same node. This simply means that the same daya is used as an input for several layers,
for instance because the is a some branching, or a skip connection in the network.

Conversely, several layers may point, with some restriction upon the data they convey, to the same node. This means that the
data from several layers is to be aggregated into a single data blob. If all layers output data of the same size, the
aggregation may be done by addition. If the layers output data of compatible size, the tensors may be concatenated. Because
concatenation is sensitive to the order, it has to be tracked. For this reason, layers do not only point to a node, but also
to a port number within the node.

Finally, nodes and layers may be regrouped in cell blocks. Any operation done on one node or one layer in the block must also
be performed on the other nodes or layers of the block.


## Build a simple network for MNIST
The code below shows how to create a simple classifier network for MNIST. We first create a graph object that will take
data a 28x28 image with 1 channel and will output  vector of size 10. The input and output nodes are created
automatically. Their index is stored in `graph.input_node` and `graph.output_node`
```python
from graph_morph import NNGraph
from layers import ConvBNPReLU, DepthSepConvBNPReLU, DensePReLU, Identity, \
                    Flatten, MaxPoolStride, Dummy
import torch
graph = NNGraph((1,28,28), (10,))
```

The network will consist in a first 5x5 convolution layer that shrinks the image size to 24x24. Then,
a 3x3 depthwise separable convolution followed by a max pooling with stride 2 is performed three times. Finally,
the output is fed into a multi-layer perceptron with one hidden layer.

First, we set three nodes and build the cells. `cell_in` and `cell_out` contain the index of the nodes created
by each operation within the `graph` object.
```python
cell_in = graph.cell_add_nodes(3)
cell_out = graph.cell_add_children(cell_in,[
    DepthSepConvBNPReLU(chn_in, chn_out, 3)
    for chn_in, chn_out in [(16,16), (16,32), (32,32)]
    ])
cell_out = graph.cell_add_children(cell_out,
    [MaxPoolStride() for _ in cell_out])
```

Now, connect the input to the first cell input with a 5x5 conv layer , and connect the cells together with identity.
```python
graph.add_layer(ConvBNPReLU(1,16,5,padding=0), graph.input_node, cell_in[0])
for ii,jj in [(0,1),(1,2)]:
    graph.add_layer(Identity(),cell_out[ii], cell_in[jj])
```

Finally, add the classifier. The last layer is connected to the output node.
```python
n = graph.add_child(cell_out[2], Flatten())
n = graph.add_child(n, LinearPReLU(3*3*32,256)) 
graph.add_layer(Linear(256,10),n,graph.output_node)
print(graph)
```
``` 
NNGraph object with 13 nodes
Input node  0, Output node 12
Layers:
 From  0 to  1 (0) : ConvBNPReLU[in_channels=1,out_channels=16,kernel_size=5,stride=1,padding=2]
 From  1 to  2 (0) : DepthSepConvBNPReLU[in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1]
 From  2 to  3 (0) : MaxPoolStride[]
 From  3 to  4 (0) : Identity[]
 From  4 to  5 (0) : DepthSepConvBNPReLU[in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1]
 From  5 to  6 (0) : MaxPoolStride[]
 From  6 to  7 (0) : Identity[]
 From  7 to  8 (0) : DepthSepConvBNPReLU[in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1]
 From  8 to  9 (0) : MaxPoolStride[]
 From  9 to 10 (0) : Flatten[]
 From 10 to 11 (0) : LinearPReLU[in_features=288,out_features=256]
 From 11 to 12 (0) : Linear[in_features=256,out_features=10]
```

Now, the graph can be converted into a model:
```python
model = graph.torch_model()
dummy_picture = torch.rand((10,1,28,28),dtype=torch.float32)
output = model(dummy_picture)
```

The graph is stored in the `graph_model` attribute of the `model` obejct. Model parameters weights need to be explicitely
transfered between the graph object (where they are stored as `numpy.ndarray` and used in function-preserving morphisms) and
the various `torch.nn.Module` objects within the model. For instance, the following code will load weights into the graph,
reset the weights of the last 3x3 convolution layer (that goes to port 0 of node 8) and update the weight of the modules.
```python
model.weights_to_graph()
layer_obj, from__node = model.graph_model.backward_layer_list[8][0]
layer_obj.to_identity()
model.weights_to_modules()
```
