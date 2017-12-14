#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 12:31:09 2017

@author: patfol
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 16:23:47 2017
proxym-inter.aphp.fr:8080
@author: Interne
"""

import numpy as np

#==============================================================================
# batch iter
#==============================================================================
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


#==============================================================================
# batch iter : balanced batch with replacement and variable batch size
#==============================================================================
def batch_balanced(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a balanced batch iterator for a dataset.
    With a sample of positive examples and batch size as param
    """
    data = np.array(data)
    pos_idx = np.where(np.array(list(data[:,1]))[:,1]==1)[0]
    neg_idx = np.where(np.array(list(data[:,1]))[:,1]==0)[0]
    neg_data = data[neg_idx,:]  
    data_size = len(neg_data)
    batch_size = batch_size//2
    num_batches_per_epoch = int((data_size - 1) / (batch_size)) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = neg_data[shuffle_indices]
        else:
            shuffled_data = neg_data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            pos_data = data[np.random.choice(pos_idx, size=batch_size//2, replace = True),:]
            mini_batch = np.vstack((shuffled_data[start_index:end_index], pos_data))
            shuffle_indices = np.random.permutation(np.arange(len(mini_batch)))
            yield mini_batch[shuffle_indices]

