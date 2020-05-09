#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:17:48 2020

@author: btayart
"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

class ParametricModule(nn.Module):
    def __init__(self,module):
        super(ParametricModule,self).__init__()
        self.add_module("param_module", module)
        self.add_parameter("weight", torch.FloatTensor([0.]))
    def forward(self,X):
        lambda_ = self.weight
        return lambda_*self.param_module(X) + (1-lambda_)*X
        
class NNLayer(object):

    def __init__(self, *args, **kwargs):
        self.parameters = {}  # parameters (kernel size, etc)
        self.w = None  # weights

    def out_size(self, input_size):
        raise(NotImplementedError())

    def to_torch_module(self):
        raise(NotImplementedError())

    def save_weights_to_module(self, module):
        if self.w is None:
            raise(RuntimeError(
                "weights not initialized, cannot save to module"))
        module.load_state_dict({key: torch.tensor(val) for key, val
                                in self.w.items()})

    def load_weights_from_module(self, module):
        self.w = {key: tensor.cpu().numpy() for key, tensor
                  in module.state_dict().items()}

    def reset_weights(self):
        self.w = None

    def size(self):
        return 0

    def __str__(self):
        name = self.__class__.__name__
        paramstr = ""
        for key, val in self.parameters.items():
            paramstr += "%s=%s," % (key, str(val))
        if paramstr:
            paramstr = paramstr[:-1]
        return name+"["+paramstr+"]"


class InsertableMixin(object):
    def __init__(self, *args, **kwargs):
        self.insertable = True
        super(InsertableMixin, self).__init__(*args, **kwargs)

    def to_identity(self):
        self.w = self._to_identity()

    def _to_identity(self):
        raise(NotImplementedError())


class Dummy(NNLayer):
    def __init__(self,name="",**kwargs):
        self.name = name
        self.parameters = kwargs
    def __str__(self):
        if self.name:
            prefix = self.name+"-"
        else:
            prefix = ""
        return prefix+super(Dummy,self).__str__()
    
class Identity(NNLayer):
    """ Identity layer"""

    def out_size(self, input_size):
        return input_size

    def to_torch_module(self):
        return nn.Identity()


class Flatten(NNLayer):
    """ Flatten layer"""

    def out_size(self, input_size):
        return (input_size.prod(), )

    def to_torch_module(self):
        return nn.Flatten()

class Linear(InsertableMixin, NNLayer):
    """ Linear without activation (use for the last layer)"""
    def __init__(self, input_size, output_size=None, do_identity=False):
        super(Linear,self).__init__()
        if output_size is None:
            output_size = input_size
        self.parameters = {"in_features": int(input_size),
                           "out_features": int(output_size)}
        if do_identity:
            self.to_identity()
       

    def out_size(self, input_size):
        expected = self.parameters["in_features"]
        if input_size != (self.input_size, expected):
            raise RuntimeError("Input size given is " +
                               f"{input_size}, expected {(expected,)}")
        return (self.output_size,)

    def to_torch_module(self):
        return nn.Linear(bias=True, **self.parameters)
    
    def size(self):
        input_size = self.parameters["in_features"]
        output_size = self.parameters["out_features"]
        return 1 + (1+input_size)*output_size
    
    
class LinearPReLU(Linear):
    """ Dense + PReLU"""
    def to_torch_module(self):
        modules = OrderedDict([
            ("linear", nn.Linear(bias=True, **self.parameters)),
            ("prelu", nn.PReLU())
        ])
        return nn.Sequential(modules)

    def size(self):
        return 1 + super(LinearPReLU,self).size()
    
    def _to_identity(self):
        w = super(LinearPReLU,self)._to_identity()
        w2 = {"linear."+k:v for k,v in w.items()}
        w2["prelu.weight"] = np.array([1], dtype=np.single)
        return w2


class ConvLayer(InsertableMixin, NNLayer):
    def __init__(self,
                 in_channels, out_channels=None, kernel_size=1,
                 stride=1, padding=None, do_identity=False, **kwargs):
        super(ConvLayer,self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.parameters = kwargs.copy()
        self.parameters.update({"in_channels": int(in_channels),
                                "out_channels": int(out_channels),
                                "kernel_size": int(kernel_size),
                                "stride": int(stride)})
        self.parameters["padding"] = padding if padding else (kernel_size-1)//2

    def out_size(self, input_size):
        if len(input_size) != 3 or \
                input_size[0] != self.input_chn:
            raise ValueError("Expected input_size with 3 dimensions and  " +
                             f"input_size[0] = {self.input_size}")
        _, h, w = input_size
        c = self.parameters["out_channels"]
        k = self.parameters["kernel"]
        p = self.parameters["padding"]
        s = self.parameters["stride"]
        def new_size(l): return 1+(l+2*p-k)//s
        return c, new_size(h), new_size(w)

    def _to_identity(self):
        chn = self.parameters["in_channels"]
        if chn != self.parameters["out_channels"]:
            raise RuntimeError("weighting to identity can be set done only " +
                               "if input and output channel counts are equal")
        if self.parameters["stride"] != 1 or \
                self.parameters["padding"]*2+1 != \
                    self.parameters["kernel_size"] or \
                self.parameters.get("groups",1) != 1:
            raise RuntimeError("weighting to identity can be set done only " +
                               "if stride is 1, groups is 1 and padding "+
                               "matches kernel size")
        return self._to_identity_conv()
        
    def _to_identity_conv(self):
        raise NotImplementedError()


class ConvBNPReLU(ConvLayer):
    """ Conv2d + BatchNorm + PReLU"""

    def to_torch_module(self):
        modules = OrderedDict([
            ("conv", nn.Conv2d(bias=False, **self.parameters)),
            ("batch_norm", nn.BatchNorm2d(
                num_features=self.parameters["out_channels"])),
            ("prelu", nn.PReLU())
        ])
        return nn.Sequential(modules)

    def _to_identity_conv(self):
        chn = self.parameters("in_channels")
        ker = self.parameters["kernel_size"]
        midpoint = self.parameters["padding"]

        rng = np.arange(chn)
        conv_weight = np.zeros((chn, chn, ker, ker), dtype=np.single)
        conv_weight[rng, rng, midpoint, midpoint] = 1.0
        w = {
            "conv.weight": conv_weight,
            "batch_norm.weight": np.ones((chn,), dtype=np.single),
            "batch_norm.bias": np.zeros((chn,), dtype=np.single),
            "batch_norm.running_mean": np.zeros((chn,), dtype=np.single),
            "batch_norm.running_var": np.ones((chn,), dtype=np.single),
            "batch_norm.num_batches_tracked": np.array(0, dtype=np.int64),
            "prelu.weight ": np.ones((1,), dtype=np.single)
            }
        return w
    
    def size(self):
        ker = self.parameters["kernel_size"]
        in_chn = self.parameters("in_channels")
        out_chn = self.parameters("out_channels")
        grp = self.parameters.get("groups", 1)

        conv_size = (ker*ker)*in_chn*out_chn/grp
        bn_size = 4*out_chn+1
        prelu_size = 1
        return conv_size + bn_size + prelu_size


class DepthSepConvBNPReLU(ConvLayer):
    """ Depthwise separable Conv2d + BatchNorm + PReLU"""

    def to_torch_module(self):
        in_chn = self.parameters["in_channels"]
        out_chn = self.parameters["out_channels"]

        param_space = self.parameters.copy()
        param_space["out_channels"] = in_chn
        param_space["groups"] = in_chn

        param_depth = self.parameters.copy()
        param_depth["kernel_size"] = 1
        param_depth["stride"] = 1
        param_depth["padding"] = 0

        modules = OrderedDict([
            ("conv_space", nn.Conv2d(bias=False, **param_space)),
            ("conv_depth", nn.Conv2d(bias=False, **param_depth)),
            ("batch_norm", nn.BatchNorm2d(num_features=out_chn)),
            ("prelu", nn.PReLU())
        ])
        return nn.Sequential(modules)

    def _to_identity_conv(self):
        chn = self.parameters["in_channels"]
        ker = self.parameters["kernel_size"]
        midpoint = self.parameters["padding"]
        rng = np.arange(chn)
        conv_space_weight = np.zeros((chn, 1, ker, ker), dtype=np.single)
        conv_space_weight[rng, 0, midpoint, midpoint] = 1.0
        conv_depth_weight = np.zeros((chn, chn, 1, 1), dtype=np.single)
        conv_depth_weight[rng, rng, 0, 0] = 1.0
        w={"conv_space.weight": conv_space_weight,
           "conv_depth.weight": conv_depth_weight,
           "batch_norm.weight": np.ones((chn,), dtype=np.single),
           "batch_norm.bias": np.zeros((chn,), dtype=np.single),
           "batch_norm.running_mean": np.zeros((chn,), dtype=np.single),
           "batch_norm.running_var": np.ones((chn,), dtype=np.single),
           "batch_norm.num_batches_tracked": np.array(0, dtype=np.int64),
           "prelu.weight": np.ones((1,), dtype=np.single)
           }
        return w
    
    def size(self):
        ker = self.parameters["kernel_size"]
        in_chn = self.parameters("in_channels")
        out_chn = self.parameters("out_channels")
        grp = self.parameters.get("groups", 1)

        conv_space_size = (ker*ker)*in_chn
        conv_depth_size = in_chn*out_chn/grp
        bn_size = 4*out_chn+1
        prelu_size = 1
        return conv_space_size + conv_depth_size + bn_size + prelu_size


class MaxPoolStride(NNLayer):
    
    def to_torch_module(self):
        return nn.MaxPool2d(kernel_size=2,stride=2)
    def out_size(self, input_size):
        if len(input_size) != 3:
            raise ValueError("Expected input_size with 3 dimensions")
        c, h, w = input_size
        k = 2
        p = 0
        s = 2
        def new_size(l): return 1+(l+2*p-k)//s
        return c, new_size(h), new_size(w)
