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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# from tensorflow.python.tools import freeze_graph

import svhn
import svhn_input

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('num_threads', 16,
                            """number of threads.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '../svhn_results/svhn_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('output_graph_dir', '../svhn_results/svhn_graph_weights',
                           """Where to save the trained graph.""")
tf.app.flags.DEFINE_string('final_result', 'final_result',
                           """The name of the output classification layer in"""
                           """ the retrained graph.""")

        

def main(argv=None):  # pylint: disable=unused-argument
    
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
    loss = svhn.loss(logits_length, logits1, logits2, logits3, logits4, logits5, length, digits)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = svhn.train(loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess_1 = tf.Session(config=tf.ConfigProto(
      log_device_placement=FLAGS.log_device_placement,
      intra_op_parallelism_threads=FLAGS.num_threads))
    sess_1.run(init)
    
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess_1, ckpt.model_checkpoint_path)
      
      # for op in tf.all_variables():
      #   print(op.name)
       
      # Store variables
      with tf.variable_scope("conv1", reuse=True):
        _k_conv1 = tf.get_variable( "weights", shape=[5, 5, 3, 64] )
        _k_conv1 = _k_conv1.eval(sess_1)
        _b_conv1 = tf.get_variable( "biases", shape=[64] )
        _b_conv1 = _b_conv1.eval(sess_1)

      with tf.variable_scope("conv2", reuse=True):
        _k_conv2 = tf.get_variable( "weights", shape=[5, 5, 64, 128] )
        _k_conv2 = _k_conv2.eval(sess_1)
        _b_conv2 = tf.get_variable( "biases", shape=[128] )
        _b_conv2 = _b_conv2.eval(sess_1)
        
      with tf.variable_scope("conv3", reuse=True):
        _k_conv3 = tf.get_variable( "weights", shape=[5, 5, 128, 160] )
        _k_conv3 = _k_conv3.eval(sess_1)
        _b_conv3 = tf.get_variable( "biases", shape=[160] )
        _b_conv3 = _b_conv3.eval(sess_1)  
        
      with tf.variable_scope("conv4", reuse=True):
        _k_conv4 = tf.get_variable( "weights", shape=[5, 5, 160, 192] )
        _k_conv4 = _k_conv4.eval(sess_1)
        _b_conv4 = tf.get_variable( "biases", shape=[192] )
        _b_conv4 = _b_conv4.eval(sess_1)  
#############################################################################
#########    digits for length   ############
#############################################################################
      with tf.variable_scope("length_local1", reuse=True):
        _w_local3 = tf.get_variable( "weights", shape=[3072, 384] )
        _w_local3 = _w_local3.eval(sess_1)
        _b_local3 = tf.get_variable( "biases", shape=[384] )
        _b_local3 = _b_local3.eval(sess_1)
        
      with tf.variable_scope("length_local2", reuse=True):
        _w_local4 = tf.get_variable( "weights", shape=[384, 192] )
        _w_local4 = _w_local4.eval(sess_1)
        _b_local4 = tf.get_variable( "biases", shape=[192] )
        _b_local4 = _b_local4.eval(sess_1)
        
      with tf.variable_scope("softmax_length", reuse=True):
        _w_local5 = tf.get_variable( "weights", shape=[192, 7] )
        _w_local5 = _w_local5.eval(sess_1)
        _b_local5 = tf.get_variable( "biases", shape=[7] )
        _b_local5 = _b_local5.eval(sess_1)
#############################################################################
#########    digit1   ############
#############################################################################
      with tf.variable_scope("digit1_local1", reuse=True):
        _w_digit1_local1 = tf.get_variable( "weights", shape=[3072, 384] )
        _w_digit1_local1 = _w_digit1_local1.eval(sess_1)
        _b_digit1_local1 = tf.get_variable( "biases", shape=[384] )
        _b_digit1_local1 = _b_digit1_local1.eval(sess_1)
        
      with tf.variable_scope("digit1_local2", reuse=True):
        _w_digit1_local2 = tf.get_variable( "weights", shape=[384, 192] )
        _w_digit1_local2 = _w_digit1_local2.eval(sess_1)
        _b_digit1_local2 = tf.get_variable( "biases", shape=[192] )
        _b_digit1_local2 = _b_digit1_local2.eval(sess_1)
        
      with tf.variable_scope("digit1", reuse=True):
        _w_digit1 = tf.get_variable( "weights", shape=[192, 11] )
        _w_digit1 = _w_digit1.eval(sess_1)
        _b_digit1 = tf.get_variable( "biases", shape=[11] )
        _b_digit1 = _b_digit1.eval(sess_1)
#############################################################################
#########    digit2   ############
#############################################################################
      with tf.variable_scope("digit2_local1", reuse=True):
        _w_digit2_local1 = tf.get_variable( "weights", shape=[3072, 384] )
        _w_digit2_local1 = _w_digit2_local1.eval(sess_1)
        _b_digit2_local1 = tf.get_variable( "biases", shape=[384] )
        _b_digit2_local1 = _b_digit2_local1.eval(sess_1)
        
      with tf.variable_scope("digit2_local2", reuse=True):
        _w_digit2_local2 = tf.get_variable( "weights", shape=[384, 192] )
        _w_digit2_local2 = _w_digit2_local2.eval(sess_1)
        _b_digit2_local2 = tf.get_variable( "biases", shape=[192] )
        _b_digit2_local2 = _b_digit2_local2.eval(sess_1)
        
      with tf.variable_scope("digit2", reuse=True):
        _w_digit2 = tf.get_variable( "weights", shape=[192, 11] )
        _w_digit2 = _w_digit2.eval(sess_1)
        _b_digit2 = tf.get_variable( "biases", shape=[11] )
        _b_digit2 = _b_digit2.eval(sess_1)
#############################################################################
#########    digit3   ############
#############################################################################
      with tf.variable_scope("digit3_local1", reuse=True):
        _w_digit3_local1 = tf.get_variable( "weights", shape=[3072, 384] )
        _w_digit3_local1 = _w_digit3_local1.eval(sess_1)
        _b_digit3_local1 = tf.get_variable( "biases", shape=[384] )
        _b_digit3_local1 = _b_digit3_local1.eval(sess_1)
        
      with tf.variable_scope("digit3_local2", reuse=True):
        _w_digit3_local2 = tf.get_variable( "weights", shape=[384, 192] )
        _w_digit3_local2 = _w_digit3_local2.eval(sess_1)
        _b_digit3_local2 = tf.get_variable( "biases", shape=[192] )
        _b_digit3_local2 = _b_digit3_local2.eval(sess_1)
        
      with tf.variable_scope("digit3", reuse=True):
        _w_digit3 = tf.get_variable( "weights", shape=[192, 11] )
        _w_digit3 = _w_digit3.eval(sess_1)
        _b_digit3 = tf.get_variable( "biases", shape=[11] )
        _b_digit3 = _b_digit3.eval(sess_1)
#############################################################################
#########    digit4   ############
#############################################################################
      with tf.variable_scope("digit4_local1", reuse=True):
        _w_digit4_local1 = tf.get_variable( "weights", shape=[3072, 384] )
        _w_digit4_local1 = _w_digit4_local1.eval(sess_1)
        _b_digit4_local1 = tf.get_variable( "biases", shape=[384] )
        _b_digit4_local1 = _b_digit4_local1.eval(sess_1)
        
      with tf.variable_scope("digit4_local2", reuse=True):
        _w_digit4_local2 = tf.get_variable( "weights", shape=[384, 192] )
        _w_digit4_local2 = _w_digit4_local2.eval(sess_1)
        _b_digit4_local2 = tf.get_variable( "biases", shape=[192] )
        _b_digit4_local2 = _b_digit4_local2.eval(sess_1)
        
      with tf.variable_scope("digit4", reuse=True):
        _w_digit4 = tf.get_variable( "weights", shape=[192, 11] )
        _w_digit4 = _w_digit4.eval(sess_1)
        _b_digit4 = tf.get_variable( "biases", shape=[11] )
        _b_digit4 = _b_digit4.eval(sess_1)
#############################################################################
#########    digit5   ############
#############################################################################
      with tf.variable_scope("digit5_local1", reuse=True):
        _w_digit5_local1 = tf.get_variable( "weights", shape=[3072, 384] )
        _w_digit5_local1 = _w_digit5_local1.eval(sess_1)
        _b_digit5_local1 = tf.get_variable( "biases", shape=[384] )
        _b_digit5_local1 = _b_digit5_local1.eval(sess_1)
        
      with tf.variable_scope("digit5_local2", reuse=True):
        _w_digit5_local2 = tf.get_variable( "weights", shape=[384, 192] )
        _w_digit5_local2 = _w_digit5_local2.eval(sess_1)
        _b_digit5_local2 = tf.get_variable( "biases", shape=[192] )
        _b_digit5_local2 = _b_digit5_local2.eval(sess_1)
        
      with tf.variable_scope("digit5", reuse=True):
        _w_digit5 = tf.get_variable( "weights", shape=[192, 11] )
        _w_digit5 = _w_digit5.eval(sess_1)
        _b_digit5 = tf.get_variable( "biases", shape=[11] )
        _b_digit5 = _b_digit5.eval(sess_1)
#############################################################################
    else:
      print('No checkpoint file found')
      return
         
  sess_1.close()





  g_2 = tf.Graph()
  with g_2.as_default():
    
    images_input = tf.placeholder("float", shape=[64, 64, 3], name="input")
    images = tf.reshape(images_input, [1, 64, 64, 3]) 

#######################################################################################
    # conv1
    with tf.variable_scope('conv1') as scope:
      kernel = tf.constant(_k_conv1)
      conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
      biases = tf.constant(_b_conv1)
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
      kernel = tf.constant(_k_conv2)
      conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
      biases = tf.constant(_b_conv2)
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
      kernel = tf.constant(_k_conv3)
      conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
      biases = tf.constant(_b_conv3)
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
      kernel = tf.constant(_k_conv4)
      conv = tf.nn.conv2d(norm3, kernel, [1, 1, 1, 1], padding='SAME')
      biases = tf.constant(_b_conv4)
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
    dim = 3072
    reshape = tf.reshape(pool4, [1, dim]) 
#######################################################################################

#######################################################################################    
    # local3
    with tf.variable_scope('length_local1') as scope:
      weights = tf.constant(_w_local3)
      biases = tf.constant(_b_local3)
      local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # local4
    with tf.variable_scope('length_local2') as scope:
      weights = tf.constant(_w_local4)
      biases = tf.constant(_b_local4)
      local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

    # softmax_linear, i.e. (WX + b)
    with tf.variable_scope('softmax_length') as scope:
      weights = tf.constant(_w_local5)
      biases = tf.constant(_b_local5)
      softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    
    final_length = tf.nn.softmax( softmax_linear, name='output_length' )
####################################################################################### 
    
#######################################################################################    
    # digit1
    with tf.variable_scope('digit1_local1') as scope:
      weights = tf.constant(_w_digit1_local1)
      biases = tf.constant(_b_digit1_local1)
      digit1_local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    with tf.variable_scope('digit1_local2') as scope:
      weights = tf.constant(_w_digit1_local2)
      biases = tf.constant(_b_digit1_local2)
      digit1_local2 = tf.nn.relu(tf.matmul(digit1_local1, weights) + biases, name=scope.name)

    with tf.variable_scope('digit1') as scope:
      weights = tf.constant(_w_digit1)
      biases = tf.constant(_b_digit1)
      digit1 = tf.add(tf.matmul(digit1_local2, weights), biases, name=scope.name)
    
    final_digit1 = tf.nn.softmax( digit1, name='output_digit1' )
#######################################################################################    

#######################################################################################    
    # digit2
    with tf.variable_scope('digit2_local1') as scope:
      weights = tf.constant(_w_digit2_local1)
      biases = tf.constant(_b_digit2_local1)
      digit2_local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    with tf.variable_scope('digit2_local2') as scope:
      weights = tf.constant(_w_digit2_local2)
      biases = tf.constant(_b_digit2_local2)
      digit2_local2 = tf.nn.relu(tf.matmul(digit2_local1, weights) + biases, name=scope.name)

    with tf.variable_scope('digit2') as scope:
      weights = tf.constant(_w_digit2)
      biases = tf.constant(_b_digit2)
      digit2 = tf.add(tf.matmul(digit2_local2, weights), biases, name=scope.name)
    
    final_digit2 = tf.nn.softmax( digit2, name='output_digit2' )
####################################################################################### 

#######################################################################################    
    # digit3
    with tf.variable_scope('digit3_local1') as scope:
      weights = tf.constant(_w_digit3_local1)
      biases = tf.constant(_b_digit3_local1)
      digit3_local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    with tf.variable_scope('digit3_local2') as scope:
      weights = tf.constant(_w_digit3_local2)
      biases = tf.constant(_b_digit3_local2)
      digit3_local2 = tf.nn.relu(tf.matmul(digit3_local1, weights) + biases, name=scope.name)

    with tf.variable_scope('digit3') as scope:
      weights = tf.constant(_w_digit3)
      biases = tf.constant(_b_digit3)
      digit3 = tf.add(tf.matmul(digit3_local2, weights), biases, name=scope.name)
    
    final_digit3 = tf.nn.softmax( digit3, name='output_digit3' )
####################################################################################### 

#######################################################################################    
    # digit4
    with tf.variable_scope('digit4_local1') as scope:
      weights = tf.constant(_w_digit4_local1)
      biases = tf.constant(_b_digit4_local1)
      digit4_local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    with tf.variable_scope('digit4_local2') as scope:
      weights = tf.constant(_w_digit4_local2)
      biases = tf.constant(_b_digit4_local2)
      digit4_local2 = tf.nn.relu(tf.matmul(digit4_local1, weights) + biases, name=scope.name)

    with tf.variable_scope('digit4') as scope:
      weights = tf.constant(_w_digit4)
      biases = tf.constant(_b_digit4)
      digit4 = tf.add(tf.matmul(digit4_local2, weights), biases, name=scope.name)
    
    final_digit4 = tf.nn.softmax( digit4, name='output_digit4' )
####################################################################################### 

#######################################################################################    
    # digit5
    with tf.variable_scope('digit5_local1') as scope:
      weights = tf.constant(_w_digit5_local1)
      biases = tf.constant(_b_digit5_local1)
      digit5_local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    with tf.variable_scope('digit5_local2') as scope:
      weights = tf.constant(_w_digit5_local2)
      biases = tf.constant(_b_digit5_local2)
      digit5_local2 = tf.nn.relu(tf.matmul(digit5_local1, weights) + biases, name=scope.name)

    with tf.variable_scope('digit5') as scope:
      weights = tf.constant(_w_digit5)
      biases = tf.constant(_b_digit5)
      digit5 = tf.add(tf.matmul(digit5_local2, weights), biases, name=scope.name)
    
    final_digit5 = tf.nn.softmax( digit5, name='output_digit5' )
####################################################################################### 
    
    sess_2 = tf.Session()
    init_2 = tf.initialize_all_variables();
    sess_2.run(init_2)

    graph_def = g_2.as_graph_def()
    tf.train.write_graph(graph_def, FLAGS.output_graph_dir, 'output.pb', as_text=False)
    
        


if __name__ == '__main__':
  tf.app.run()
