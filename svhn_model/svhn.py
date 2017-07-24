# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the SVHN network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import svhn_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '../../svhn_data/',
                           """Path to the SVHN data directory.""")

# Global constants describing the SVHN data set.
IMAGE_SIZE = svhn_input.IMAGE_SIZE
NUM_CLASSES = svhn_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = FLAGS.num_examples_per_epoch_for_train
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = FLAGS.num_examples_per_epoch_for_test
NUM_EXAMPLES_PER_EPOCH_FOR_VALID = FLAGS.num_examples_per_epoch_for_valid


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average. 
NUM_EPOCHS_PER_DECAY = 100.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.
MOMENTUM = 0.5

# If a model is trained with multiple GPU's prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/gpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  if wd:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs():
  """Construct distorted input for SVHN training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'svhn_data_recordes_with_extra')
  return svhn_input.distorted_inputs(data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)


def inputs(eval_data):
  """Construct input for SVHN evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'svhn_data_recordes_with_extra')
  return svhn_input.inputs(eval_data=eval_data, data_dir=data_dir,
                              batch_size=FLAGS.batch_size)


def inference(images, _dropout=1.):
  """Build the SVHN model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  
#######################################################################################
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 64],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')
    
  # Apply Dropout
  norm1 = tf.nn.dropout(norm1, _dropout, name='dropout1')
#######################################################################################

#######################################################################################    
  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 128],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    
  # Apply Dropout
  pool2 = tf.nn.dropout(pool2, _dropout, name='dropout2')
#######################################################################################

#######################################################################################    
  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[5, 5, 128, 160],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [160], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv3)

  # pool3
  pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool3')    
    
  # norm3
  norm3 = tf.nn.lrn(pool3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm3')
    
  # Apply Dropout
  norm3 = tf.nn.dropout(norm3, _dropout, name='dropout3')
#######################################################################################

#######################################################################################    
  # conv4
  with tf.variable_scope('conv4') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[5, 5, 160, 192],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(norm3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv4)

  # norm4
  norm4 = tf.nn.lrn(conv4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm4')    
    
  # pool4
  pool4 = tf.nn.max_pool(norm4, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool4')    
    
  # Apply Dropout
  pool4 = tf.nn.dropout(pool4, _dropout, name='dropout4')
#######################################################################################

#######################################################################################
  # Move everything into depth so we can perform a single matrix multiply.
  dim = 1
  for d in pool4.get_shape()[1:].as_list():
    dim *= d    
  reshape = tf.reshape(pool4, [FLAGS.batch_size, dim])  
#######################################################################################
    
#######################################################################################    
  # length
  with tf.variable_scope('length_local1') as scope:
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    length_local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(length_local1)

  with tf.variable_scope('length_local2') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    length_local2 = tf.nn.relu(tf.matmul(length_local1, weights) + biases, name=scope.name)
    _activation_summary(length_local2)

  # softmax_linear, i.e. (WX + b)
  with tf.variable_scope('softmax_length') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_length = tf.add(tf.matmul(length_local2, weights), biases, name=scope.name)
    _activation_summary(softmax_length)
#######################################################################################
    
#######################################################################################    
  # first digit
  with tf.variable_scope('digit1_local1') as scope:
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    digit1_local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(digit1_local1)

  with tf.variable_scope('digit1_local2') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    digit1_local2 = tf.nn.relu(tf.matmul(digit1_local1, weights) + biases, name=scope.name)
    _activation_summary(digit1_local2)

  with tf.variable_scope('digit1') as scope:
    weights = _variable_with_weight_decay('weights', [192, 11],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [11],
                              tf.constant_initializer(0.0))
    digit1 = tf.add(tf.matmul(digit1_local2, weights), biases, name=scope.name)
    _activation_summary(digit1)
#######################################################################################
    
#######################################################################################    
  # second digit
  with tf.variable_scope('digit2_local1') as scope:
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    digit2_local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(digit2_local1)

  with tf.variable_scope('digit2_local2') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    digit2_local2 = tf.nn.relu(tf.matmul(digit2_local1, weights) + biases, name=scope.name)
    _activation_summary(digit2_local2)

  with tf.variable_scope('digit2') as scope:
    weights = _variable_with_weight_decay('weights', [192, 11],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [11],
                              tf.constant_initializer(0.0))
    digit2 = tf.add(tf.matmul(digit2_local2, weights), biases, name=scope.name)
    _activation_summary(digit2)
#######################################################################################

#######################################################################################    
  # third digit
  with tf.variable_scope('digit3_local1') as scope:
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    digit3_local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(digit3_local1)

  with tf.variable_scope('digit3_local2') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    digit3_local2 = tf.nn.relu(tf.matmul(digit3_local1, weights) + biases, name=scope.name)
    _activation_summary(digit3_local2)

  with tf.variable_scope('digit3') as scope:
    weights = _variable_with_weight_decay('weights', [192, 11],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [11],
                              tf.constant_initializer(0.0))
    digit3 = tf.add(tf.matmul(digit3_local2, weights), biases, name=scope.name)
    _activation_summary(digit3)
#######################################################################################

#######################################################################################    
  # fourth digit
  with tf.variable_scope('digit4_local1') as scope:
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    digit4_local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(digit4_local1)

  with tf.variable_scope('digit4_local2') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    digit4_local2 = tf.nn.relu(tf.matmul(digit4_local1, weights) + biases, name=scope.name)
    _activation_summary(digit4_local2)

  with tf.variable_scope('digit4') as scope:
    weights = _variable_with_weight_decay('weights', [192, 11],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [11],
                              tf.constant_initializer(0.0))
    digit4 = tf.add(tf.matmul(digit4_local2, weights), biases, name=scope.name)
    _activation_summary(digit4)
#######################################################################################

#######################################################################################    
  # fifth digit
  with tf.variable_scope('digit5_local1') as scope:
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    digit5_local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(digit5_local1)

  with tf.variable_scope('digit5_local2') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    digit5_local2 = tf.nn.relu(tf.matmul(digit5_local1, weights) + biases, name=scope.name)
    _activation_summary(digit5_local2)

  with tf.variable_scope('digit5') as scope:
    weights = _variable_with_weight_decay('weights', [192, 11],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [11],
                              tf.constant_initializer(0.0))
    digit5 = tf.add(tf.matmul(digit5_local2, weights), biases, name=scope.name)
    _activation_summary(digit5)
#######################################################################################
    
  return softmax_length, digit1, digit2, digit3, digit4, digit5 



def loss(logits_length, logits_digit1, logits_digit2, logits_digit3, logits_digit4, logits_digit5, length, digits):
  """Add L2Loss to all the trainable variables.

  Add summary for for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Split the labels
  digit1_temp, digit2_temp, digit3_temp, digit4_temp, digit5_temp = tf.split(1, 5, digits)
  digit1 = tf.reshape(digit1_temp, [FLAGS.batch_size, 11])
  digit2 = tf.reshape(digit2_temp, [FLAGS.batch_size, 11])
  digit3 = tf.reshape(digit3_temp, [FLAGS.batch_size, 11])
  digit4 = tf.reshape(digit4_temp, [FLAGS.batch_size, 11])
  digit5 = tf.reshape(digit5_temp, [FLAGS.batch_size, 11])

  # Calculate the average cross entropy loss across the batch.
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits_length, length, name='cross_entropy_length')
  cross_entropy += tf.nn.softmax_cross_entropy_with_logits(logits_digit1, digit1, name='cross_entropy_digit1')
  cross_entropy += tf.nn.softmax_cross_entropy_with_logits(logits_digit2, digit2, name='cross_entropy_digit2')
  cross_entropy += tf.nn.softmax_cross_entropy_with_logits(logits_digit3, digit3, name='cross_entropy_digit3')
  cross_entropy += tf.nn.softmax_cross_entropy_with_logits(logits_digit4, digit4, name='cross_entropy_digit4')
  cross_entropy += tf.nn.softmax_cross_entropy_with_logits(logits_digit5, digit5, name='cross_entropy_digit5')
           
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    # opt = tf.train.MomentumOptimizer(lr, 0.5, use_locking=False, name='Momentum')
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op

