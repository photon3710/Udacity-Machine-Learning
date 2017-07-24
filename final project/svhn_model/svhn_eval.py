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

"""Evaluation for SVHN.

Usage:
Please see the tutorial and website for how to download the svhn
data set, compile the program and train the model.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import svhn

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '../svhn_results/svhn_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'valid'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '../svhn_results/svhn_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")


def eval_once(saver, summary_writer, top_k_op, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/svhn_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))
      if FLAGS.eval_data == 'test':  
        num_iter = int(math.ceil(FLAGS.num_examples_per_epoch_for_test / FLAGS.batch_size))
      else:
        num_iter = int(math.ceil(FLAGS.num_examples_per_epoch_for_valid / FLAGS.batch_size))
        
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval SVHN for a number of steps."""
  with tf.Graph().as_default():
    # Get images and labels for svhn.
    eval_data = FLAGS.eval_data == 'test'
    images, length, digits = svhn.inputs(eval_data=eval_data)

    # Split the labels
    digit1_temp, digit2_temp, digit3_temp, digit4_temp, digit5_temp = tf.split(1, 5, digits)
    digit1 = tf.reshape(digit1_temp, [FLAGS.batch_size, 11])
    digit2 = tf.reshape(digit2_temp, [FLAGS.batch_size, 11])
    digit3 = tf.reshape(digit3_temp, [FLAGS.batch_size, 11])
    digit4 = tf.reshape(digit4_temp, [FLAGS.batch_size, 11])
    digit5 = tf.reshape(digit5_temp, [FLAGS.batch_size, 11])
    
    
    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits_length, logits1, logits2, logits3, logits4, logits5 = svhn.inference(images)

    # Get the targets class from one-hot lables.
    top_1_length = tf.nn.in_top_k(logits_length, tf.argmax(length, 1), 1)
    top_1_digit1 = tf.nn.in_top_k(logits1, tf.argmax(digit1, 1), 1)
    top_1_digit2 = tf.nn.in_top_k(logits2, tf.argmax(digit2, 1), 1)
    top_1_digit3 = tf.nn.in_top_k(logits3, tf.argmax(digit3, 1), 1)
    top_1_digit4 = tf.nn.in_top_k(logits4, tf.argmax(digit4, 1), 1)
    top_1_digit5 = tf.nn.in_top_k(logits5, tf.argmax(digit5, 1), 1)
       
    top_k_op = top_1_length & top_1_digit1 & top_1_digit2 & top_1_digit3 & top_1_digit4 & top_1_digit5
    
    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        svhn.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    graph = tf.get_default_graph()
    summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir,
                                            graph=graph)

    while True:
      eval_once(saver, summary_writer, top_k_op, summary_op)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
