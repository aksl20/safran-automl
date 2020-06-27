#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 09 21:15:22 2020

@author: btayart
"""
from torch.utils.data import Dataset, random_split
import os
import h5py
import numpy as np

class MapDataset(Dataset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.
    """
    def __init__(self, dataset, map_func):
        self.dataset = dataset
        self.map_func = map_func
    
    def mapping(self,d):
        return self.map_func(d)
    
    def __getitem__(self, index):
        return self.mapping(self.dataset[index])

    def __len__(self):
        return len(self.dataset)


class AugmentedDataset(MapDataset):
    """
    Given a labeled dataset, applies the transform to the image only while
    leaving the label untouched
    """
    def mapping(self, d):
        img, y = d
        return self.map_func(img), y

def train_test_split(dataset,
                     train_transform=None,
                     test_transform=None,
                     ratio=.9):
    """
    Split a dataset in two subsets with their own transforms applied

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset to be split
    train_augments : callable, optional
        Transform to be called on the train subset. The default is None.
    test_augments : callable, optional
        Transform to be called on the test subset. The default is None.
    ratio : float, optional
        Size of the train set relative to the dataset. The default is .9.

    Returns
    -------
    out_train : torch.utils.data.Dataset
        Train dataset sampled from input dataset
    out_test : torch.utils.data.Dataset
        Test dataset sampled from input dataset
    """
    
    n_train = int(len(dataset)*ratio)
    n_test = len(dataset) - n_train
    sub_train, sub_test = random_split(dataset, [n_train, n_test])
    if train_transform is None:
        out_train = sub_train
    else:
        out_train = AugmentedDataset(sub_train, train_transform)
        
    if test_transform is None:
        out_test = sub_test
    else:
        out_test = AugmentedDataset(sub_test, test_transform)
        
    return out_train, out_test

class TextureDataset(Dataset):
    def __init__(self,
                 dataset='train64',
                 label_type='defects',
                 transform=None,
                 directory="./textures",
                 normalized=False):
        """
        Parameters
        ----------
        label_type : string, optional
            'angles' for clean images labeled with angle
            'defects' for images labeled with a default
            'all-angles' for images labeled with angle
            'full' for images with a label containing the angle and default
            The default is 'defects'
        dataset : string, optional
            'train64' or 'test64' for 64x64 images
            'train32' or 'test32' for 32x32 images
            The default is 'train64'
        image_size : int, optional
            size of the images, 32 or 64 (for 32x32 or 64x64 images)
            The default is 64
        inplace : bool, optional
            True to create an inplace transform.
            False to return a copy of the input.
            The default is False.
        transform : callable
            transform called on the image (see torchvision transforms)
            The default is None
        directory : str
            Path of the directory where the data is stored
            The default is "./textures"
        normalized : bool
            True to get data with mean 0 and variance 1. 
            False to get data in the [0, 1] range.
            Normalization is done according to the train set (i.e. the test set
            will be normalized with the mean and variance from the test set)
            The default is False        
        Returns
        -------
        Torch Dataset object.

        """
        super(TextureDataset, self).__init__()
        
        if label_type not in  ['angles', 'defects', 'all-angles', 'full']:
            raise ValueError(
                "label_type: expected 'angles', 'defects', 'all-angles' or 'full', got "
                + str(label_type))

        if dataset not in ['train64', 'test64', 'train32', 'test32']:
            raise dataset(
                "label_type: expected 'train64', 'test64', 'train32' or 'test32', got "
                + str(dataset))

        self.transform = transform
        self.labelfile = os.path.join(directory,dataset+".csv")
        self.imagefile = os.path.join(directory,dataset+".h5")

        h5f = h5py.File(self.imagefile,'r')
        self.data = h5f['images'][:]
        h5f.close()

        # Label input is somewhat hardcoded, but we won't import Pandas just for a csv
        raw_labels = np.genfromtxt(self.labelfile,
                         dtype=int, usecols=[1,3], delimiter=",",skip_header=1)
        angle_labels = raw_labels[:,0]//20
        defects_labels = raw_labels[:,1]

        if dataset[-2:]=="32":
            mean, std = 0.3288534, 0.1411142
        elif dataset[-2:]=="64":
            mean, std = 0.3291129, 0.1431457
            
        if label_type == 'angles':
            selection = (defects_labels==0)
            self.labels = angle_labels[selection]
            self.data = self.data[selection]
            self.classes = ["%d degrees"%(i*20) for i in np.unique(angle_labels)]
            if dataset[-2:]=="32":
                mean, std = 0.3532409, 0.1373712    
            elif dataset[-2:]=="64":
                mean, std = 0.3541475, 0.1375972

        elif label_type == 'defects':
            self.labels = defects_labels
            self.classes = ['good', 'color', 'cut', 'hole', 'thread', 'metal_contamination']

        elif label_type == 'all-angles':
            self.labels = angle_labels
            self.classes = ["%d degrees"%(i*20) for i in np.unique(angle_labels)]

        elif label_type == 'full':
            m = defects_labels.max()
            self.labels = angle_labels*m + defects_labels
            defects = ['good', 'color', 'cut', 'hole', 'thread', 'metal_contamination']
            self.classes=[]
            for ang in ["%d deg"%(i*20) for i in np.unique(angle_labels)]:
                self.classes += [d+" | "+ang for d in defects]

        if normalized:
            self.data = (self.data-mean)/std
            self.mean, self.std = 0., 1.
        else:
            self.mean, self.std = mean, std
            

    def __getitem__(self,key):
        img, label = self.data[key], self.labels[key]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    
    def __len__(self):
        return self.labels.size