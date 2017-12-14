#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 12:20:40 2017

@author: patfol
"""


import tensorflow as tf
import numpy as np
import os
import time
import datetime
import dataHelpers
from reviewerLin import TextLin
import pickle

# Parameters
# ==================================================

# Data loading and writing params
tf.flags.DEFINE_float("dev_sample_percentage", .2, "Percentage of the training data to use for validation")
tf.flags.DEFINE_integer("test_year", 1, "Duration in year from last date of search for test set")
tf.flags.DEFINE_string("data_file", "error", "Data source (default: None).")
tf.flags.DEFINE_integer("model_number", 0, "number of the model")



# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding (default: 200)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate_power", 0.95, "(absolute ) Learning rate power (default: 0.5)")
tf.flags.DEFINE_float("learning_rate", 1., "Learning rate (default: 1)")
tf.flags.DEFINE_float("pos_weight", 1., "positive weight for loss (default: 0.)")


# Training parameters
tf.flags.DEFINE_integer("batch_size", 200, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 50)")
tf.flags.DEFINE_integer("num_checkpoints", 1, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")



# Data Preparation
# ==================================================
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

#year limit for test set
if FLAGS.data_file == 'chen':
    yearLimit = 2012 - FLAGS.test_year
elif FLAGS.data_file == 'crequit':
    yearLimit = 2015 - FLAGS.test_year
elif FLAGS.data_file == 'khoo':
    yearLimit = 2015 - FLAGS.test_year
elif FLAGS.data_file == 'bateman':
    yearLimit = 2013 - FLAGS.test_year

# Load data
print("Loading data...")
arts, _, _, _ = pickle.load(open('./data/' + FLAGS.data_file + 'Vectorized', "rb"))
X = np.array([x['vectorized'] for x in arts if x['year'] <= yearLimit])
y = np.array([x['y'] for x in arts if x['year'] <= yearLimit])
y = np.vstack((1-y, y)).T
del arts

# shuffle data
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = X[shuffle_indices]
y_shuffled = y[shuffle_indices]
del y, X

##########
### Split train/test set AND make sure sufficient amount of pos example in dev set
############
#find pos and neg indices
pos_ind = np.where(y_shuffled[:,1]==1)[0]
neg_ind = np.where(y_shuffled[:,1]==0)[0]

#cut indices with a desired dev percentage (20% usually here)
dev_pos_index = -1 * int(FLAGS.dev_sample_percentage * float(len(pos_ind)))
dev_neg_index = -1 * int(FLAGS.dev_sample_percentage * float(len(neg_ind)))

pos_ind_train, pos_ind_dev = pos_ind[:dev_pos_index], pos_ind[dev_pos_index:]
neg_ind_train, neg_ind_dev = neg_ind[:dev_neg_index], neg_ind[dev_neg_index:]


train_ind = np.sort(np.concatenate((pos_ind_train, neg_ind_train)))
dev_ind = np.sort(np.concatenate((pos_ind_dev, neg_ind_dev)))

#build data set from indices
x_train, x_dev = x_shuffled[train_ind], x_shuffled[dev_ind]
y_train, y_dev = y_shuffled[train_ind], y_shuffled[dev_ind]
del x_shuffled, y_shuffled
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))          

###Save and evaluate model after this many steps , corresponding to that number of epochs
checkpoint_every = int(len(neg_ind_train) * FLAGS.num_epochs / (FLAGS.batch_size//2))
evaluate_every = int(len(neg_ind_train) * FLAGS.num_epochs / (FLAGS.batch_size//2))

#parameters to save
params_to_save = [FLAGS.data_file, FLAGS.test_year, FLAGS.learning_rate, FLAGS.learning_rate_power, FLAGS.pos_weight,
                  FLAGS.l2_reg_lambda, len(x_train), len(pos_ind_train), len(neg_ind_train)]
params_to_save = ' '.join([str(x) for x in params_to_save])
# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        lin = TextLin(
            num_classes=y_train.shape[1],
            embedding_size=FLAGS.embedding_dim,
            ratio = FLAGS.pos_weight,
            l2_reg_lambda = FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, 100,
                                                   FLAGS.learning_rate_power, staircase=True)

        # Passing global_step to minimize() will increment it at each step.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(lin.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, './results/',FLAGS.data_file))
        print("Writing to {}\n".format(out_dir))
        # Summaries for loss and youden, precision and recall
        loss_summary = tf.summary.scalar("loss", lin.loss)
        ydn_summary = tf.summary.scalar("youden", lin.recall + lin.specificity -1)
        rcl_summary = tf.summary.scalar("recall", lin.recall)
        sp_summary = tf.summary.scalar("specificity", lin.specificity)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, ydn_summary, rcl_summary, sp_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train", timestamp)
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, ydn_summary,  rcl_summary, sp_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev", timestamp)
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints", 
                                                      'year_'+str(FLAGS.test_year)+'_weight_'+str(FLAGS.pos_weight)+'_model_'+str(FLAGS.model_number)))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        #assign pre-trained word vector matrix
        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              lin.input_x: x_batch,
              lin.input_y: y_batch
            }
            _, step, summaries, loss, youden, recall, specificity = sess.run(
                [train_op, global_step, train_summary_op, lin.loss,
                 lin.recall + lin.specificity - 1,lin.recall, lin.specificity],
                feed_dict)
            #time_str = datetime.datetime.now().isoformat()
            #print("{}: step {}, loss {:g}, ydn {:g}, rcl {:g}, sp {:g}".format(time_str,
            #      step, loss, youden, recall, specificity))
            #train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """          
            feed_dict = {
              lin.input_x: x_batch,
              lin.input_y: y_batch
            }
            step, summaries, loss, youden, recall, specificity = sess.run(
                [global_step, dev_summary_op, lin.loss,
                 lin.recall + lin.specificity -1, lin.recall, lin.specificity],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            #saving parameters of runs and results
            with open(out_dir + '.txt', 'a') as out:
                results = "{:g} {:g} {:g} {:g} {:g}".format(step, loss, youden, recall, specificity)
                params_results = ' '.join([params_to_save, results])
                out.write(params_results +'\n')
            print("{}: step {}, loss {:g}, ydn {:g}, rcl {:g}, sp{:g}".format(time_str,
                  step, loss, youden, recall, specificity))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = dataHelpers.batch_balanced(list(zip(x_train, y_train)), FLAGS.batch_size, num_epochs = FLAGS.num_epochs)
 
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
