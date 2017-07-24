# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import gzip
import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import svhn
import svhn_localizer_input

FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_integer('localizer_batch_size', 64,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('localizer_data_dir', '../../svhn_data/',
                           """Path to the SVHN data directory.""")


# Global constants describing the SVHN data set.
IMAGE_SIZE = svhn_localizer_input.IMAGE_SIZE
NUM_CLASSES = svhn_localizer_input.NUM_CLASSES
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
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.localizer_data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.localizer_data_dir, 'svhn_data_recordes')
  return svhn_localizer_input.distorted_inputs(data_dir=data_dir,
                                        batch_size=FLAGS.localizer_batch_size)


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
  if not FLAGS.localizer_data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.localizer_data_dir, 'svhn_data_recordes')
  return svhn_localizer_input.inputs(eval_data=eval_data, data_dir=data_dir,
                              batch_size=FLAGS.localizer_batch_size)


def load_parameters():  
    
  class SVHNRecord(object):
    pass
  result = SVHNRecord()  
    
  """Train SVHN for a number of steps."""
  g_1 = tf.Graph()
  with g_1.as_default():
    global_step = tf.Variable(0, trainable=False)
    start_step = 0

    # Get images and labels for SVHN.
    print('train-svhn.distorted_inputs')
    images, length, digits = svhn.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits_length, logits1, logits2, logits3, logits4, logits5 = svhn.inference(images,_dropout=0.9)

    # Calculate loss.
    # loss = svhn.loss(logits_length, logits1, logits2, logits3, logits4, logits5, length, digits)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    # train_op = svhn.train(loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    # summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess_1 = tf.Session(config=tf.ConfigProto(
      log_device_placement=FLAGS.localizer_log_device_placement,
      intra_op_parallelism_threads=FLAGS.localizer_num_threads))
    sess_1.run(init)
    
    ckpt = tf.train.get_checkpoint_state(FLAGS.localizer_checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess_1, ckpt.model_checkpoint_path)
      
      # for op in tf.all_variables():
      #   print(op.name)
       
      # Fetch variables
      with tf.variable_scope("conv1", reuse=True):
        _k_conv1 = tf.get_variable( "weights", shape=[5, 5, 3, 64] )
        result._k_conv1 = _k_conv1.eval(sess_1)
        _b_conv1 = tf.get_variable( "biases", shape=[64] )
        result._b_conv1 = _b_conv1.eval(sess_1)

      with tf.variable_scope("conv2", reuse=True):
        _k_conv2 = tf.get_variable( "weights", shape=[5, 5, 64, 128] )
        result._k_conv2 = _k_conv2.eval(sess_1)
        _b_conv2 = tf.get_variable( "biases", shape=[128] )
        result._b_conv2 = _b_conv2.eval(sess_1)
        
      with tf.variable_scope("conv3", reuse=True):
        _k_conv3 = tf.get_variable( "weights", shape=[5, 5, 128, 160] )
        result._k_conv3 = _k_conv3.eval(sess_1)
        _b_conv3 = tf.get_variable( "biases", shape=[160] )
        result._b_conv3 = _b_conv3.eval(sess_1)  
        
      with tf.variable_scope("conv4", reuse=True):
        _k_conv4 = tf.get_variable( "weights", shape=[5, 5, 160, 192] )
        result._k_conv4 = _k_conv4.eval(sess_1)
        _b_conv4 = tf.get_variable( "biases", shape=[192] )
        result._b_conv4 = _b_conv4.eval(sess_1)  
#############################################################################
#########    digits for length   ############
#############################################################################
      with tf.variable_scope("length_local1", reuse=True):
        _w_local3 = tf.get_variable( "weights", shape=[3072, 384] )
        result._w_local3 = _w_local3.eval(sess_1)
        _b_local3 = tf.get_variable( "biases", shape=[384] )
        result._b_local3 = _b_local3.eval(sess_1)
        
      with tf.variable_scope("length_local2", reuse=True):
        _w_local4 = tf.get_variable( "weights", shape=[384, 192] )
        result._w_local4 = _w_local4.eval(sess_1)
        _b_local4 = tf.get_variable( "biases", shape=[192] )
        result._b_local4 = _b_local4.eval(sess_1)
        
      # with tf.variable_scope("softmax_linear", reuse=True):
      #   _w_local5 = tf.get_variable( "weights", shape=[192, 7] )
      #   _w_local5 = _w_local5.eval(sess_1)
      #   _b_local5 = tf.get_variable( "biases", shape=[7] )
      #   _b_local5 = _b_local5.eval(sess_1)
#############################################################################
#########    digit1   ############
#############################################################################
      with tf.variable_scope("digit1_local1", reuse=True):
        _w_digit1_local1 = tf.get_variable( "weights", shape=[3072, 384] )
        result._w_digit1_local1 = _w_digit1_local1.eval(sess_1)
        _b_digit1_local1 = tf.get_variable( "biases", shape=[384] )
        result._b_digit1_local1 = _b_digit1_local1.eval(sess_1)
        
      with tf.variable_scope("digit1_local2", reuse=True):
        _w_digit1_local2 = tf.get_variable( "weights", shape=[384, 192] )
        result._w_digit1_local2 = _w_digit1_local2.eval(sess_1)
        _b_digit1_local2 = tf.get_variable( "biases", shape=[192] )
        result._b_digit1_local2 = _b_digit1_local2.eval(sess_1)
        
      # with tf.variable_scope("digit1", reuse=True):
      #   _w_digit1 = tf.get_variable( "weights", shape=[192, 11] )
      #   _w_digit1 = _w_digit1.eval(sess_1)
      #   _b_digit1 = tf.get_variable( "biases", shape=[11] )
      #   _b_digit1 = _b_digit1.eval(sess_1)
#############################################################################
#########    digit2   ############
#############################################################################
      with tf.variable_scope("digit2_local1", reuse=True):
        _w_digit2_local1 = tf.get_variable( "weights", shape=[3072, 384] )
        result._w_digit2_local1 = _w_digit2_local1.eval(sess_1)
        _b_digit2_local1 = tf.get_variable( "biases", shape=[384] )
        result._b_digit2_local1 = _b_digit2_local1.eval(sess_1)
        
      with tf.variable_scope("digit2_local2", reuse=True):
        _w_digit2_local2 = tf.get_variable( "weights", shape=[384, 192] )
        result._w_digit2_local2 = _w_digit2_local2.eval(sess_1)
        _b_digit2_local2 = tf.get_variable( "biases", shape=[192] )
        result._b_digit2_local2 = _b_digit2_local2.eval(sess_1)
        
      # with tf.variable_scope("digit2", reuse=True):
      #   _w_digit2 = tf.get_variable( "weights", shape=[192, 11] )
      #   _w_digit2 = _w_digit2.eval(sess_1)
      #   _b_digit2 = tf.get_variable( "biases", shape=[11] )
      #   _b_digit2 = _b_digit2.eval(sess_1)
#############################################################################
#########    digit3   ############
#############################################################################
      with tf.variable_scope("digit3_local1", reuse=True):
        _w_digit3_local1 = tf.get_variable( "weights", shape=[3072, 384] )
        result._w_digit3_local1 = _w_digit3_local1.eval(sess_1)
        _b_digit3_local1 = tf.get_variable( "biases", shape=[384] )
        result._b_digit3_local1 = _b_digit3_local1.eval(sess_1)
        
      with tf.variable_scope("digit3_local2", reuse=True):
        _w_digit3_local2 = tf.get_variable( "weights", shape=[384, 192] )
        result._w_digit3_local2 = _w_digit3_local2.eval(sess_1)
        _b_digit3_local2 = tf.get_variable( "biases", shape=[192] )
        result._b_digit3_local2 = _b_digit3_local2.eval(sess_1)
        
      # with tf.variable_scope("digit3", reuse=True):
      #   _w_digit3 = tf.get_variable( "weights", shape=[192, 11] )
      #   _w_digit3 = _w_digit3.eval(sess_1)
      #   _b_digit3 = tf.get_variable( "biases", shape=[11] )
      #   _b_digit3 = _b_digit3.eval(sess_1)
#############################################################################
#########    digit4   ############
#############################################################################
      with tf.variable_scope("digit4_local1", reuse=True):
        _w_digit4_local1 = tf.get_variable( "weights", shape=[3072, 384] )
        result._w_digit4_local1 = _w_digit4_local1.eval(sess_1)
        _b_digit4_local1 = tf.get_variable( "biases", shape=[384] )
        result._b_digit4_local1 = _b_digit4_local1.eval(sess_1)
        
      with tf.variable_scope("digit4_local2", reuse=True):
        _w_digit4_local2 = tf.get_variable( "weights", shape=[384, 192] )
        result._w_digit4_local2 = _w_digit4_local2.eval(sess_1)
        _b_digit4_local2 = tf.get_variable( "biases", shape=[192] )
        result._b_digit4_local2 = _b_digit4_local2.eval(sess_1)
        
      # with tf.variable_scope("digit4", reuse=True):
      #   _w_digit4 = tf.get_variable( "weights", shape=[192, 11] )
      #   _w_digit4 = _w_digit4.eval(sess_1)
      #   _b_digit4 = tf.get_variable( "biases", shape=[11] )
      #   _b_digit4 = _b_digit4.eval(sess_1)
#############################################################################
#########    digit5   ############
#############################################################################
      with tf.variable_scope("digit5_local1", reuse=True):
        _w_digit5_local1 = tf.get_variable( "weights", shape=[3072, 384] )
        result._w_digit5_local1 = _w_digit5_local1.eval(sess_1)
        _b_digit5_local1 = tf.get_variable( "biases", shape=[384] )
        result._b_digit5_local1 = _b_digit5_local1.eval(sess_1)
        
      with tf.variable_scope("digit5_local2", reuse=True):
        _w_digit5_local2 = tf.get_variable( "weights", shape=[384, 192] )
        result._w_digit5_local2 = _w_digit5_local2.eval(sess_1)
        _b_digit5_local2 = tf.get_variable( "biases", shape=[192] )
        result._b_digit5_local2 = _b_digit5_local2.eval(sess_1)
        
      # with tf.variable_scope("digit5", reuse=True):
      #   _w_digit5 = tf.get_variable( "weights", shape=[192, 11] )
      #   _w_digit5 = _w_digit5.eval(sess_1)
      #   _b_digit5 = tf.get_variable( "biases", shape=[11] )
      #   _b_digit5 = _b_digit5.eval(sess_1)
#############################################################################
    else:
      print('No checkpoint file found')
      return
         
  sess_1.close()

  return result



def inference(images, result):
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
    kernel = tf.constant(result._k_conv1)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.constant(result._b_conv1)
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')
#######################################################################################

#######################################################################################    
  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = tf.constant(result._k_conv2)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.constant(result._b_conv2)
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')
#######################################################################################

#######################################################################################    
  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = tf.constant(result._k_conv3)
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.constant(result._b_conv3)
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope.name)
    
  # pool3
  pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool3')    
    
  # norm3
  norm3 = tf.nn.lrn(pool3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm3')
#######################################################################################

#######################################################################################    
  # conv4
  with tf.variable_scope('conv4') as scope:
    kernel = tf.constant(result._k_conv4)
    conv = tf.nn.conv2d(norm3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.constant(result._b_conv4)
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope.name)

  # norm4
  norm4 = tf.nn.lrn(conv4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm4')    
    
  # pool4
  pool4 = tf.nn.max_pool(norm4, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool4') 
#######################################################################################

#######################################################################################
  # Move everything into depth so we can perform a single matrix multiply.
  dim = 1
  for d in pool4.get_shape()[1:].as_list():
    dim *= d    
  reshape = tf.reshape(pool4, [FLAGS.localizer_batch_size, dim])  
#######################################################################################
    
#######################################################################################    
  # digit1
  with tf.variable_scope('digit1_local1') as scope:
    weights = tf.constant(result._w_digit1_local1)
    biases = tf.constant(result._b_digit1_local1)
    digit1_local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

  with tf.variable_scope('digit1_local2') as scope:
    weights = tf.constant(result._w_digit1_local2)
    biases = tf.constant(result._b_digit1_local2)
    digit1_local2 = tf.nn.relu(tf.matmul(digit1_local1, weights) + biases, name=scope.name)
    
  with tf.variable_scope('digit1_box') as scope:
    weights = _variable_with_weight_decay('weights', [192, 4],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [4],
                              tf.constant_initializer(0.0))
    digit1_box = tf.add(tf.matmul(digit1_local2, weights), biases, name=scope.name)
    _activation_summary(digit1_box)
    # tf.add_to_collection('digit_boxes', weights)
    # tf.add_to_collection('digit_boxes', biases)
#######################################################################################
    
#######################################################################################    
  # digit2
  with tf.variable_scope('digit2_local1') as scope:
    weights = tf.constant(result._w_digit2_local1)
    biases = tf.constant(result._b_digit2_local1)
    digit2_local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

  with tf.variable_scope('digit2_local2') as scope:
    weights = tf.constant(result._w_digit2_local2)
    biases = tf.constant(result._b_digit2_local2)
    digit2_local2 = tf.nn.relu(tf.matmul(digit2_local1, weights) + biases, name=scope.name)
    
  with tf.variable_scope('digit2_box') as scope:
    weights = _variable_with_weight_decay('weights', [192, 4],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [4],
                              tf.constant_initializer(0.0))
    digit2_box = tf.add(tf.matmul(digit2_local2, weights), biases, name=scope.name)
    _activation_summary(digit2_box)
    # tf.add_to_collection('digit_boxes', weights)
    # tf.add_to_collection('digit_boxes', biases)
#######################################################################################

#######################################################################################    
  # digit3
  with tf.variable_scope('digit3_local1') as scope:
    weights = tf.constant(result._w_digit3_local1)
    biases = tf.constant(result._b_digit3_local1)
    digit3_local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

  with tf.variable_scope('digit3_local2') as scope:
    weights = tf.constant(result._w_digit3_local2)
    biases = tf.constant(result._b_digit3_local2)
    digit3_local2 = tf.nn.relu(tf.matmul(digit3_local1, weights) + biases, name=scope.name)
    
  with tf.variable_scope('digit3_box') as scope:
    weights = _variable_with_weight_decay('weights', [192, 4],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [4],
                              tf.constant_initializer(0.0))
    digit3_box = tf.add(tf.matmul(digit3_local2, weights), biases, name=scope.name)
    _activation_summary(digit3_box)
    # tf.add_to_collection('digit_boxes', weights)
    # tf.add_to_collection('digit_boxes', biases)
#######################################################################################

#######################################################################################    
  # digit4
  with tf.variable_scope('digit4_local1') as scope:
    weights = tf.constant(result._w_digit4_local1)
    biases = tf.constant(result._b_digit4_local1)
    digit4_local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

  with tf.variable_scope('digit4_local2') as scope:
    weights = tf.constant(result._w_digit4_local2)
    biases = tf.constant(result._b_digit4_local2)
    digit4_local2 = tf.nn.relu(tf.matmul(digit4_local1, weights) + biases, name=scope.name)
    
  with tf.variable_scope('digit4_box') as scope:
    weights = _variable_with_weight_decay('weights', [192, 4],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [4],
                              tf.constant_initializer(0.0))
    digit4_box = tf.add(tf.matmul(digit4_local2, weights), biases, name=scope.name)
    _activation_summary(digit4_box)
    # tf.add_to_collection('digit_boxes', weights)
    # tf.add_to_collection('digit_boxes', biases)
#######################################################################################

#######################################################################################    
  # digit5
  with tf.variable_scope('digit5_local1') as scope:
    weights = tf.constant(result._w_digit5_local1)
    biases = tf.constant(result._b_digit5_local1)
    digit5_local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

  with tf.variable_scope('digit5_local2') as scope:
    weights = tf.constant(result._w_digit5_local2)
    biases = tf.constant(result._b_digit5_local2)
    digit5_local2 = tf.nn.relu(tf.matmul(digit5_local1, weights) + biases, name=scope.name)
    
  with tf.variable_scope('digit5_box') as scope:
    weights = _variable_with_weight_decay('weights', [192, 4],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [4],
                              tf.constant_initializer(0.0))
    digit5_box = tf.add(tf.matmul(digit5_local2, weights), biases, name=scope.name)
    _activation_summary(digit5_box)
    # tf.add_to_collection('digit_boxes', weights)
    # tf.add_to_collection('digit_boxes', biases)
#######################################################################################
    
  return digit1_box, digit2_box, digit3_box, digit4_box, digit5_box 



def loss(digit1_box, digit2_box, digit3_box, digit4_box, digit5_box, boxes):
  """Add L2Loss to all the trainable variables.

  Add summary for for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  box1 = tf.reshape(digit1_box, [FLAGS.localizer_batch_size, 1, 4])
  box2 = tf.reshape(digit2_box, [FLAGS.localizer_batch_size, 1, 4])
  box3 = tf.reshape(digit3_box, [FLAGS.localizer_batch_size, 1, 4])
  box4 = tf.reshape(digit4_box, [FLAGS.localizer_batch_size, 1, 4])
  box5 = tf.reshape(digit5_box, [FLAGS.localizer_batch_size, 1, 4])
  logits_boxes = tf.concat(1, [box1, box2, box3, box4, box5])


  # Calculate the average cross entropy loss across the batch.
  regerssion_loss_box = tf.nn.sigmoid_cross_entropy_with_logits(logits_boxes, boxes, name='regerssion_loss_box')
           
  regerssion_loss_box_mean = tf.reduce_mean(regerssion_loss_box, name='regerssion_loss_box_mean')
  tf.add_to_collection('regerssion_loss_box', regerssion_loss_box_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('regerssion_loss_box'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in SVHN model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('regerssion_loss_box')
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
  """Train SVHN model.

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
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.localizer_batch_size
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

