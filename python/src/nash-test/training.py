#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 00:15:10 2020

@author: btayart
"""

import time
import torch
from torch.nn.functional import cross_entropy
import numpy as np

def train_net(net, dataloader, optimizer, n_epoch=10, criterion=cross_entropy, display_interval=100,
             verbose=3, lr_scheduler=None):
    net.train(True)
        
    t0 = time.time()
    t00 = t0

    for epoch in range(n_epoch):
        cumul_loss = 0.0
        running_loss = 0.0
        for batch_idx, (X, y) in enumerate(dataloader):
            X = X.cuda()
            y = y.cuda()

            y_pred = net(X)
            
            optimizer.zero_grad()

            loss = criterion(y_pred, y)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            cumul_loss += loss.item()

            batch_idx += 1
            if batch_idx % display_interval == 0:
                running_loss /= display_interval
                t1 = time.time()
                if verbose >=3:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tt={:.1f}s'.format(
                        epoch, batch_idx * len(X), len(dataloader.dataset),
                        100. * batch_idx / len(dataloader), running_loss, t1-t0))
                running_loss = 0
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        cumul_loss /= batch_idx
        t1 = time.time()
        if verbose>=2:
            print('====> Epoch: {} done in {:.1f}s, Average loss: {:.4f}'.format(
                epoch, t1-t0, cumul_loss))
        t0 = t1
    if verbose>=1:
        print('Training done, {} epochs in {:.1f}s, loss at last epoch: {:.4f}'.format(
            n_epoch, t1-t00, cumul_loss))
        

def test_net(model, dataloader, criterion=cross_entropy):
    """
    Check the performance of a model on a validation dataset

    Parameters
    ----------
    model : torch.nn.Module
        Model. Calculation is performed on the same device as the model
    dataloader : torch.utils.data.Dataloader
        Dataloader for the validation set

    Returns
    -------
    avg_loss : float
        Average of the loss function across the validation set
    accuracy : float
        Proportion of accurately predicted labels across the validation set
        (top-1 error)
    confusion_matrix : numpy.ndarray
        Confustion matrix, with predicted labels as rows and ground truth
        labels as columns
        confusion_matrix[ii][jj] contains the number of images of the validation
        set classified as *ii* while their actual label is *jj*

    """
    model.train(False)
    device = next(model.parameters()).device
    n_in_dataset = len(dataloader.dataset)
    cumul_loss = 0.0
    class_ok = 0
    
    all_gt = torch.zeros(n_in_dataset, dtype=torch.long)
    all_pred = torch.zeros(n_in_dataset, dtype=torch.long)
    with torch.no_grad():
        for batch_idx, (x, gt) in enumerate(dataloader):
            siz, = gt.size()
            i0 = batch_idx * dataloader.batch_size
            all_gt[i0:i0+siz] = gt
            
            x, gt = x.to(device), gt.to(device)
            output = model(x)
            cumul_loss += criterion(output, gt, reduction="sum").item()
            predicted_class = output.argmax(dim=1)
            class_ok += (predicted_class == gt).sum()
            
            predicted_class = predicted_class.cpu()
            all_pred[i0:i0+siz] = predicted_class

    avg_loss = cumul_loss /len(dataloader.dataset)
    accuracy = float(class_ok) / n_in_dataset
    print("Well classified %5d / %5d, (%5.2f%% accuracy)" % (
        class_ok, n_in_dataset, 100*accuracy))
    print("Average loss : %f" % avg_loss)
    
    all_gt = all_gt.detach().numpy()
    all_pred = all_pred.detach().numpy()
    max_label = all_gt.max()
    
    cm_size = max_label+1 
    confusion_matrix = np.zeros((cm_size, cm_size), dtype=np.int64)
    for i_pred in range(cm_size):
        sel_gt = all_gt[all_pred==i_pred]
        for j_gt in range(cm_size):
            confusion_matrix[i_pred,j_gt] = (sel_gt==j_gt).sum()
            
    return avg_loss, accuracy, confusion_matrix