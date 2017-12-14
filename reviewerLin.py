#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 12:13:44 2017

@author: patfol
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 17:33:33 2017

@author: Biostat84
"""

import tensorflow as tf

class TextLin(object):
    """
    A linear model for text classification.
    Uses an embedding layer, followed by logistic regression
    """
    def __init__(
      self, num_classes, embedding_size, ratio, l2_reg_lambda=0.):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        # Create  operation
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([embedding_size, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.input_x, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            class_weight = tf.constant([[1.0, ratio]])
            weight_per_label = tf.transpose(tf.matmul(self.input_y, tf.transpose(class_weight))) #shape [1, batch_size]
            losses = tf.losses.softmax_cross_entropy(self.input_y, self.scores,
                                                        weights = tf.reshape(weight_per_label, [-1]))
            self.loss = losses + l2_reg_lambda * l2_loss  
        # Metrics : Accuracy, recall, specificity 
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
      
        with tf.name_scope("recall"):
            tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(tf.argmax(self.input_y, 1), True),
                                                      tf.equal(self.predictions, True)), "float"))
            fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(tf.argmax(self.input_y, 1), True),
                                                      tf.equal(self.predictions, False)), "float"))
            recall = tp / (tp + fn + 1e-7)
            self.recall = tf.identity(recall, name="recall")

        with tf.name_scope("specificity"):
            tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(tf.argmax(self.input_y, 1), False),
                                                      tf.equal(self.predictions, False)), "float"))
            fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(tf.argmax(self.input_y, 1), False),
                                                      tf.equal(self.predictions, True)), "float"))
            specificity = tn / (tn + fp + 1e-7)
            self.specificity = tf.identity(specificity, name="specificity")