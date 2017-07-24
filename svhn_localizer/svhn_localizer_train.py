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

import svhn_localizer

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('localizer_train_dir', '../svhn_results/svhn_localizer_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('localizer_max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('localizer_log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('localizer_num_threads', 16,
                            """number of threads.""")
tf.app.flags.DEFINE_boolean('localizer_restore_network', False,
                            """restore network.""")
tf.app.flags.DEFINE_string('localizer_checkpoint_dir', '../svhn_results/svhn_train',
                           """Directory where to read model checkpoints.""")


def train():
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)
    start_step = 0

    # Get images and labels for SVHN.
    print('train-svhn.distorted_inputs')
    images, boxes = svhn_localizer.distorted_inputs()
    
    loading_parameters = svhn_localizer.load_parameters()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    digit1_box, digit2_box, digit3_box, digit4_box, digit5_box = svhn_localizer.inference(images, loading_parameters)
    
    # Calculate loss.
    loss = svhn_localizer.loss(digit1_box, digit2_box, digit3_box, digit4_box, digit5_box, boxes)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = svhn_localizer.train(loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
      log_device_placement=FLAGS.localizer_log_device_placement,
      intra_op_parallelism_threads=FLAGS.localizer_num_threads))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(FLAGS.localizer_train_dir,
                                            graph=sess.graph)
    
    # whether to restore last time data
    if FLAGS.localizer_restore_network:
      ckpt = tf.train.get_checkpoint_state(FLAGS.localizer_checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        start_step = int(global_step)
      else:
        print('No checkpoint file found')
        return

    for step in xrange(start_step, FLAGS.localizer_max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.localizer_batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.localizer_max_steps:
        checkpoint_path = os.path.join(FLAGS.localizer_train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
                

def main(argv=None):  # pylint: disable=unused-argument
  if not FLAGS.localizer_restore_network:
    if tf.gfile.Exists(FLAGS.localizer_train_dir):
      tf.gfile.DeleteRecursively(FLAGS.localizer_train_dir)
    tf.gfile.MakeDirs(FLAGS.localizer_train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
