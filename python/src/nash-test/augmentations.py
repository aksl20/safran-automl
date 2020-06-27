#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 09:36:51 2020
@author: btayart

The module implements the Cutout and RandAugment transforms
"""
import torch
import torchvision.transforms.functional as TF
import numpy as np
import PIL

class CONST:
    average_color = torch.tensor([0.4914, 0.4822, 0.4465])
    average_color_PIL = tuple(np.uint8(rgb) for rgb in [125,122,113])
    
  
class Cutout(object):
    """
    Transform object that applies a Cutout on torch Tensor or PIL Image objects
    Cutout consists in erasing a patch centered around a random pixel of the
    image
    
    REFERENCE: Terrance DeVries and Graham W Taylor. Improved regularization of
    convolutional neural networks with cutout. preprint arXiv:1708.04552, 2017
    """

    def __init__(self, patch_size=16,
                 default_color=(0,0,0),
                 input_type="tensor",
                 channel_drop_proba=1.0,
                 inplace=False):
        """
        Parameters
        ----------
        patch_size : int
            size of the square patch to be removed
        default_color : array-like, optional
            RGB triplet with default color. For PIL Image, channel values
            should be integers in the [0 255] range, for Torch Tensor, values
            should be floats in the [0.0 1.0] range.
        channel_drop_proba : float, optional
            Probability that a channel will be affected by cutout. If different
            from 1, only some channels may be erased. Value shoudl be in the
            [0. 1.] range.
            The default ia 1.0
        input_type : string, optional
            'tensor' for Torch.Tensor
            'image' for PIL.Image
            The default is 'tensor'
        inplace : bool, optional
            True to create an inplace transform.
            False to return a copy of the input.
            The default is False.

        Returns
        -------
        Cutout object, callable upon a PIL Image or Torch Tensor.

        """        
        self.patch_size = int(patch_size)
        self.channel_drop_proba = float(channel_drop_proba)
        if channel_drop_proba < 0. or channel_drop_proba > 1.:
            raise ValueError(
                "channel_drop_proba: expected float value in [0. 1.] range")
        self.inplace = bool(inplace)
        if input_type == "tensor":
            self._call_fcn = self._tensor_cutout
            self.default_values = torch.FloatTensor(
                default_color).reshape((len(default_color), 1, 1))
        elif input_type == "image":
            self._call_fcn = self._image_cutout
            self.default_values = tuple(default_color)
        else:
            raise ValueError("input_type: expected 'tensor' or 'image'")

    def __call__(self, image_tensor):
        """
        Parameters
        ----------
        image_tensor : PIL image or torch.FloatTensor
            image to be transformed

        Returns
        -------
        image_tensor : PIL image or torch.FloatTensor (same type as input)
        """
        return self._call_fcn(image_tensor)

    def get_range(self, w):
        """Takes a size, samples a range of pixels to be zeroed along that dimension"""
        w1 = np.random.choice(w) - self.patch_size//2
        w2 = np.minimum(w1 + self.patch_size, w)
        w1 = np.maximum(w1, 0)
        return w1, w2

    def _tensor_cutout(self, tensor):
        c, w, h = tensor.size()
        w1, w2 = self.get_range(w)
        h1, h2 = self.get_range(h)
        if self.channel_drop_proba == 1:
            out = TF.erase(tensor,
                          h1, w1,
                          h2-h1, w2-w1,
                          v=self.default_values,
                          inplace=self.inplace)
        else:
            out = tensor if self.inplace else tensor.clone().detach()
            for channel, val in enumerate(self.default_values):
                if np.random.rand() <= self.channel_drop_proba:
                    out[channel, w1:w2, h1:h2] = val
        return out

    def _image_cutout(self, img):
        w, h = img.size
        w1, w2 = self.get_range(w)
        h1, h2 = self.get_range(h)
        out = img if self.inplace else img.copy()
        box = (w1, h1, w2, h2)

        if self.channel_drop_proba == 1:
            to_paste = self.default_values
        else:
            npimg = np.array(img.copy().crop(box))
            for channel, val in enumerate(self.default_values):
                if np.random.rand() <= self.channel_drop_proba:
                    npimg[:, :, channel] = val
            to_paste = TF.to_pil_image(npimg)
        out.paste(to_paste, box)
        return out
    
#%% RandAumgment
# We first define some callable wrapper classes to be able to call the
# transform functions with only an image and an optional magnitude
# sampled from a range. Some transforms are functions, other are PIL.Image
# class methods, and some PIL.ImageEnhance subclass methods
# - functions are defined to call PIL.Image class methods
# - two wrapper classes are defines to call 
# - two 
class EnhancerMixin:
    """Upon call, applies transform through a PIL.ImageEngance object"""
    def __call__(self, image, magnitude=None):
        enhancer = self._fcn(image)
        return enhancer.enhance(self._get_param(magnitude))


class FunctionMixin:
    """Upon call, applies transform through a function"""
    def __call__(self, image, magnitude=None):
        return self._fcn(image, self._get_param(magnitude))


class NoArgFcn():
    """Wrapper for a function with no argument"""
    def __init__(self, fcn):
        self._fcn = fcn

    def __call__(self, img, magnitude=None):
        return self._fcn(img)


class ParamTransform():
    """Wrapper for a function with a range of parameters"""
    def __init__(self, fcn, range_1, range_2, dtype = None):
        """
        Parameters
        ----------
        fcn : callable
            callable that takes two arguments: an image and a parameter.
        range_1 : scalar
            Parameter value that will transform the image least. For transforms
            having some symmetry regarding the parameter, this is the middle of
            the range.
        range_2 : scalar
            Parameter value that will transform the image most. For transforms
            having some symmetry regarding the parameter, this is the largest 
            parameter value (minimum value will be inferred as
            range_min = 2*range_1 - range_2).
        dtype : TYPE, optional
            Type of the parameter (e.g. PL.ImageOps.posterize requires an
            uint8). None to leave as float. The default is None.

        Returns
        -------
        None.

        """
        self._fcn = fcn
        self.a = range_2-range_1
        self.b = range_1
        self.dtype = dtype

    def _get_param(self, magnitude):
        if magnitude is None:
            magnitude = np.random.rand()
        param = magnitude*self.a + self.b
        if self.dtype is not None:
            param = self.dtype(param)
        return param

class SymmetricTransform(ParamTransform):
    """Wrapper for a function with a range of parameters. Extreme parameters
    of the range cause the strongest augmentation (e.g. rotation is maximal
    for lage positive and large negative angles)"""

    def _get_param(self, magnitude):
        result = super(SymmetricTransform, self)._get_param(magnitude)
        if np.random.rand() < 0.5:
            result = 2*self.b - result
        return result


class ParamFcn(FunctionMixin, ParamTransform):
    pass


class ParamEnhancer(EnhancerMixin, ParamTransform):
    pass


class SymmetricFcn(FunctionMixin, SymmetricTransform):
    pass


class SymmetricEnhancer(EnhancerMixin, SymmetricTransform):
    pass


# Set some transforms as functions of an image and a single parameter

def img_rotate(img, angle):
    return img.rotate(angle, fillcolor=CONST.average_color_PIL)


def pil_transform(img, data):
    return img.transform(img.size,
                         method=PIL.Image.AFFINE,
                         data=data,
                         fillcolor=CONST.average_color_PIL)


def shear_x(img, sx):
    return pil_transform(img, (1, sx, 0, 0, 1, 0))


def shear_y(img, sy):
    return pil_transform(img, (1, 0, 0, sy, 1, 0))


def translate_x(img, tx):
    tx = tx * img.size[0]
    return pil_transform(img, (1, 0, tx, 0, 1, 0))


def translate_y(img, ty):
    ty = ty * img.size[1]
    return pil_transform(img, (1, 0, 0, 0, 1, ty))

def symmetric_solarize(img, level):
    """
    Same as solarize, but:
        - parameter is opposite
            -> level = 0 is the smallest transform
            -> level = 255 inverts completely the image
        - with 50% chance, dark pixels are inverted rather than bright ones
    """
    if np.random.rand() < 0.5:
        img = PIL.ImageOps.solarize(img, 255-level)
    else:
        img = PIL.ImageOps.solarize(img,level)
        img = PIL.ImageOps.solarize(img,0)
    return img

# RandAugment class
class RandAugment(object):
    """ Rand augment: applies a successively transforms sampled from a 
    set of 15 transforms (14 from the reference paper + hue shift)
    
    
    REFERENCE: Ekin D. Cubuk, Barret Zoph, Jonathon Shlens, Quoc V. Le
    RandAugment: Practical automated data augmentation with a reduced search space
    arXiv preprint  arXiv:1909.13719
    """
    transforms = {
        "autocontrast": NoArgFcn(PIL.ImageOps.autocontrast),
        "brightness": ParamEnhancer(PIL.ImageEnhance.Brightness, .95, .05),
        "color": ParamEnhancer(PIL.ImageEnhance.Color, .95, .05),
        "contrast": ParamEnhancer(PIL.ImageEnhance.Contrast, .95, .05),
        "equalize": NoArgFcn(PIL.ImageOps.equalize),
        "identity": NoArgFcn(lambda x:x),
        "posterize": ParamFcn(PIL.ImageOps.posterize, 8, 4, np.uint8),
        "img_rotate": SymmetricFcn(img_rotate, 0, 30),
        "sharpness": SymmetricEnhancer(PIL.ImageEnhance.Sharpness, 2.0, 0.),
        "shear_x": SymmetricFcn(shear_x, 0, .3),
        "shear_y": SymmetricFcn(shear_y, 0, .3),
        "solarize": ParamFcn(symmetric_solarize, 0, 255),
        "translate_x": SymmetricFcn(translate_x, 0, .3),
        "translate_y": SymmetricFcn(translate_y, 0, .3),
        "hue_shift": SymmetricFcn(TF.adjust_hue, 0, .2)
    }

    def __init__(self, n_transforms, magnitudes=None):
        """
        Parameters
        ----------
        n_transforms : int or array of int
            int: number of successive transforms to be applied. If an array of,
            int, the number of transforms will be sampled wih uniform  
            probability from the list
        magnitudes : array of float, optional
            None to sample the magnitude of each transform across their full
            range. Array of floats in the [0. 1.] range to sample the magnitude
            from the array, with uniform probability. All transforms will be 
            applied with the sampled magnitude.
        Returns
        -------
        RandAugment object.

        """
        self.n_transforms = n_transforms
        self.magnitudes = magnitudes

    def sample_magnitude(self):
        """Returns a magnitude sampled according to parameters"""
        if self.magnitudes is None:
            return None
        else:
            return np.random.choice(self.magnitudes)

    def sample_n_transforms(self):
        """Returns a number of transforms sampled according to parameters"""
        if np.isscalar(self.n_transforms):
            return self.n_transforms
        else:
            return np.random.choice(self.n_transforms)
        
    def __call__(self, image):
        mag = self.sample_magnitude()
        n_tr = self.sample_n_transforms()
        order = np.random.choice(len(self.transforms), n_tr)
        transform_list = [t for t in self.transforms.values()]
        for ii in order:
            image = transform_list[ii](image, mag)
        return image